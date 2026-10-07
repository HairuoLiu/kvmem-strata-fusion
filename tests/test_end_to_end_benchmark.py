"""
Unit and Integration Tests for Track A End-to-End Evaluation & Needle-in-a-Haystack Benchmark.

Validates:
1. SyntheticTransformerAttentionRunner context generation & dispersion properties.
2. Full-Context attention baseline: rank 1, 100% recall, baseline byte footprint.
3. Naive Flat Compression baseline: Mean-K needle dilution & Jensen's logit theft failure modes.
4. KVMem-Strata-Fusion pipeline (CASA + IFR + UBBA + LADDER):
   - Preserved needle recall (100%, rank 1).
   - High cosine similarity (> 0.98).
   - Low perplexity drift (<= 0.05).
   - 4x-8x byte footprint reduction.
   - GPUDirect Storage Big-Tile coalescing.
   - U-E-F-C Four-Dimensional Statistical Evaluation Gate pass.
5. Needle depth invariance (10%, 50%, 90% depth).
6. Context length scaling (4K, 8K, 16K).
"""

import numpy as np
import pytest

from benchmarks.eval_end_to_end import (
    SyntheticTransformerAttentionRunner,
    run_needle_in_haystack_benchmark,
)
from kvmem_fusion.core import StorageSuperTile


def test_synthetic_transformer_runner_generation():
    """Verify context session generation, shapes, and dispersion properties."""
    runner = SyntheticTransformerAttentionRunner(seed=42)
    ctx = runner.generate_needle_session(context_length=4096, needle_depth=0.5)

    assert ctx.context_length == 4096
    assert ctx.num_blocks == 128
    assert ctx.raw_keys.shape == (4096, 64)
    assert ctx.raw_values.shape == (4096, 64)
    assert ctx.query_raw.shape == (64,)
    assert ctx.query_semantic.shape == (64,)
    assert ctx.w_vocab.shape == (64, 1000)

    # Check needle block location
    expected_block_idx = int(0.5 * 128)
    assert ctx.needle_block_idx == expected_block_idx
    assert ctx.needle_token_idx == 16
    assert ctx.needle_pos == expected_block_idx * 32 + 16

    # Needle block must have high intra-block dispersion due to the outlier needle key
    needle_b_keys = ctx.raw_keys[ctx.needle_block_idx * 32 : (ctx.needle_block_idx + 1) * 32]
    needle_mean = np.mean(needle_b_keys, axis=0)
    needle_dispersion = float(np.max(np.linalg.norm(needle_b_keys - needle_mean, axis=-1)))
    assert needle_dispersion > 0.85, f"Expected needle dispersion > 0.85, got {needle_dispersion}"

    # Typical background blocks must have small dispersion
    bg_b_keys = ctx.raw_keys[0 : 32]
    bg_mean = np.mean(bg_b_keys, axis=0)
    bg_dispersion = float(np.max(np.linalg.norm(bg_b_keys - bg_mean, axis=-1)))
    assert bg_dispersion < 0.50, f"Expected bg dispersion < 0.50, got {bg_dispersion}"


def test_full_context_attention_accuracy():
    """Verify ground truth full-context attention achieves rank 1 and 100% recall."""
    runner = SyntheticTransformerAttentionRunner(seed=42)
    ctx = runner.generate_needle_session(context_length=4096, needle_depth=0.5)
    res = runner.evaluate_full_context(ctx)

    assert res.needle_recalled is True
    assert res.needle_rank == 1
    assert res.needle_score > 0.0
    assert res.cosine_similarity == 1.0
    assert res.perplexity_drift == 0.0
    assert res.total_bytes == 128 * 1024  # 128 KB
    assert res.compression_ratio == 1.0


def test_naive_compression_failure_modes():
    """Verify that naive flat compression suffers from needle dilution and rank loss."""
    runner = SyntheticTransformerAttentionRunner(seed=42)
    ctx = runner.generate_needle_session(context_length=8192, needle_depth=0.5)
    full_res = runner.evaluate_full_context(ctx)
    naive_res = runner.evaluate_naive_compression(ctx, full_res, target_top_k_blocks=16)

    # In naive Mean-K without Anti-Collapse, the needle is diluted by 1/32 and dropped
    assert naive_res.needle_recalled is False
    assert naive_res.needle_rank > 16
    assert naive_res.cosine_similarity < 0.50
    assert naive_res.perplexity_drift > 50.0  # Catastrophic target token perplexity drift
    assert naive_res.uefc_gate_passed is False


def test_kvmem_strata_fusion_end_to_end():
    """
    Verify complete KVMem-Strata-Fusion pipeline:
    - 100% needle recall (rank 1).
    - Preserved attention fidelity (cosine similarity >= 0.98).
    - Low perplexity drift (<= 0.05).
    - 4x-8x byte footprint reduction.
    - GPUDirect Storage Big-Tile coalescing.
    - U-E-F-C evaluation gate PASS.
    """
    runner = SyntheticTransformerAttentionRunner(seed=42)
    ctx = runner.generate_needle_session(context_length=8192, needle_depth=0.5)
    full_res = runner.evaluate_full_context(ctx)
    fusion_res = runner.evaluate_kvmem_strata_fusion(
        ctx,
        full_res,
        target_top_k=16,
        target_coverage=0.85,
        rho_floor=0.35
    )

    # 1. Needle recall & rank
    assert fusion_res.needle_recalled is True
    assert fusion_res.needle_rank == 1

    # 2. Attention fidelity
    assert fusion_res.cosine_similarity >= 0.98
    assert fusion_res.perplexity_drift <= 0.05

    # 3. Compression ratio in ~4x-8x target range
    assert 4.0 <= fusion_res.compression_ratio <= 8.5
    assert fusion_res.total_bytes < full_res.total_bytes

    # 4. Big-Tile coalescing
    assert "num_super_tiles" in fusion_res.metadata
    assert fusion_res.metadata["num_super_tiles"] >= 1

    # 5. U-E-F-C evaluation gate
    assert fusion_res.uefc_gate_passed is True
    gate_res = fusion_res.metadata["gate_res"]
    assert gate_res["gate_f"]["passed"] is True
    assert gate_res["gate_e"]["passed"] is True
    assert gate_res["gate_c"]["passed"] is True
    assert gate_res["gate_u"]["passed"] is True


def test_needle_depth_invariance():
    """Verify that KVMem-Strata-Fusion reliably recovers needles across depths (10%, 50%, 90%)."""
    runner = SyntheticTransformerAttentionRunner(seed=123)
    c_len = 8192

    for depth in [0.10, 0.50, 0.90]:
        ctx = runner.generate_needle_session(context_length=c_len, needle_depth=depth)
        full_res = runner.evaluate_full_context(ctx)
        fusion_res = runner.evaluate_kvmem_strata_fusion(ctx, full_res)

        assert fusion_res.needle_recalled is True, f"Failed recall at depth {depth}"
        assert fusion_res.needle_rank == 1, f"Failed rank 1 at depth {depth}"
        assert fusion_res.cosine_similarity >= 0.98, f"Degraded cos sim at depth {depth}"
        assert fusion_res.compression_ratio >= 4.0, f"Low compression at depth {depth}"


def test_context_length_scaling():
    """Verify scaling across context lengths (4K, 8K, 16K)."""
    runner = SyntheticTransformerAttentionRunner(seed=999)

    for c_len in [4096, 8192, 16384]:
        ctx = runner.generate_needle_session(context_length=c_len, needle_depth=0.5)
        full_res = runner.evaluate_full_context(ctx)
        fusion_res = runner.evaluate_kvmem_strata_fusion(ctx, full_res)

        assert fusion_res.needle_recalled is True
        assert fusion_res.needle_rank == 1
        assert fusion_res.cosine_similarity >= 0.98
        assert fusion_res.compression_ratio >= 4.0
        assert fusion_res.uefc_gate_passed is True


def test_benchmark_harness_aggregate():
    """Verify the benchmark harness aggregate summary generation."""
    bench_data = run_needle_in_haystack_benchmark(
        context_lengths=[4096],
        needle_depths=[0.25, 0.75],
        seed=42,
        verbose=False
    )

    summary = bench_data["summary"]
    assert "Full-Context" in summary
    assert "Naive-Compression" in summary
    assert "KVMem-Strata-Fusion" in summary

    # Full context recall must be 100%
    assert summary["Full-Context"]["avg_recall"] == 1.0
    # Naive compression recall must collapse
    assert summary["Naive-Compression"]["avg_recall"] == 0.0
    # KVMem-Strata-Fusion recall must be 100%
    assert summary["KVMem-Strata-Fusion"]["avg_recall"] == 1.0
    # Compression ratio must be >= 4.0x
    assert summary["KVMem-Strata-Fusion"]["avg_compression_ratio"] >= 4.0
    # UEFC gate pass rate must be 100%
    assert summary["KVMem-Strata-Fusion"]["gate_pass_rate"] == 1.0
