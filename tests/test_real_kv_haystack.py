"""
tests/test_real_kv_haystack.py
------------------------------
Pytest Verification Suite for IFR Anti-Collapse under Real Qwen3 KV Dispersion.

Addresses OG Critique R5:
  1. Dispersion is verified non-zero, matching empirical Qwen3-0.6B layer-0 K (~0.7388).
  2. IFR anti-collapse doublet splitting triggers when intra-block cluster diameter /
     normalized dispersion exceeds threshold theta = 0.868.
  3. Retrieval recall reaches >= 98% across 32K, 64K, and 128K context scales.
"""

import numpy as np
import pytest

from benchmarks.eval_qwen3_real_kv import (
    DEFAULT_DISPERSION_THRESHOLD,
    QWEN3_EMPIRICAL_DISPERSION_MEAN,
    QWEN3_EMPIRICAL_DISPERSION_P95,
    generate_qwen3_empirical_haystack,
    inject_needles_into_haystack,
    run_scale_evaluation,
)
from kvmem_fusion.ifr import AntiCollapseSplitter, IFRBlock, IFRRetriever


def test_empirical_dispersion_matches_qwen3():
    """
    OG Critique R5 Verification:
    Assert that the empirical KV generator produces non-uniform KV activations
    matching real Qwen3 empirical dispersion (~0.7388), rejecting the degenerate
    uniform dispersion (0.0000) seen in naive np.tile synthetic benchmarks.
    """
    num_blocks = 512  # 16K context
    dim = 128
    raw_blocks, _, _ = generate_qwen3_empirical_haystack(
        num_blocks=num_blocks,
        dim=dim,
        block_size=32,
        seed=42,
    )

    dispersions = []
    for b_idx, k_mat in enumerate(raw_blocks):
        blk = IFRBlock(
            block_id=f"blk_{b_idx}",
            tokens=list(range(b_idx * 32, (b_idx + 1) * 32)),
            keys=k_mat,
            values=np.zeros_like(k_mat),
            orig_pos_start=b_idx * 32,
        )
        dispersions.append(blk.norm_dispersion)

    disp_arr = np.array(dispersions)
    mean_disp = float(disp_arr.mean())
    median_disp = float(np.median(disp_arr))
    p95_disp = float(np.percentile(disp_arr, 95))

    # Assert dispersion is strictly non-zero
    assert mean_disp > 0.65, f"Dispersion collapsed to near-zero: {mean_disp}"

    # Assert close calibration with real Qwen3 layer-0 K (~0.7388 mean, ~0.8332 p95)
    assert abs(mean_disp - QWEN3_EMPIRICAL_DISPERSION_MEAN) < 0.03, (
        f"Mean dispersion {mean_disp:.4f} diverges from Qwen3 empirical {QWEN3_EMPIRICAL_DISPERSION_MEAN:.4f}"
    )
    assert abs(p95_disp - QWEN3_EMPIRICAL_DISPERSION_P95) < 0.03, (
        f"P95 dispersion {p95_disp:.4f} diverges from Qwen3 empirical {QWEN3_EMPIRICAL_DISPERSION_P95:.4f}"
    )
    assert abs(median_disp - 0.7408) < 0.03, (
        f"Median dispersion {median_disp:.4f} diverges from expected median"
    )

    # In background blocks, only a tiny fraction of outliers trigger anti-collapse (< 5%)
    splitter = AntiCollapseSplitter(dispersion_threshold=DEFAULT_DISPERSION_THRESHOLD, use_normalized=True)
    trigger_count = sum(1 for d in disp_arr if d > DEFAULT_DISPERSION_THRESHOLD)
    trigger_rate = trigger_count / num_blocks
    assert trigger_rate < 0.05, f"Unintended excessive background trigger rate: {trigger_rate:.4f}"


def test_anti_collapse_trigger_on_needle_cluster_diameter():
    """
    Verify that IFR anti-collapse doublet splitting triggers when cluster diameter
    exceeds threshold theta = 0.868 upon injecting needle tokens.
    """
    dim = 128
    splitter = AntiCollapseSplitter(dispersion_threshold=DEFAULT_DISPERSION_THRESHOLD, use_normalized=True)

    # 1. Background block with empirical Qwen3 dispersion (~0.74)
    raw_blocks, _, _ = generate_qwen3_empirical_haystack(num_blocks=1, dim=dim, seed=99)
    bg_block = IFRBlock(
        block_id="bg_block",
        tokens=list(range(32)),
        keys=raw_blocks[0],
        values=np.zeros_like(raw_blocks[0]),
        orig_pos_start=0,
    )

    assert bg_block.norm_dispersion < DEFAULT_DISPERSION_THRESHOLD
    is_split_bg, doublets_bg = splitter.inspect_and_split(bg_block)
    assert not is_split_bg, "Background block should not trigger anti-collapse splitting"
    assert len(doublets_bg) == 1, "Background block must remain a single centroid"

    # 2. Inject needle token with subtle semantic perturbation
    rng = np.random.RandomState(123)
    needle_core = rng.randn(dim).astype(np.float32)
    needle_core /= np.linalg.norm(needle_core)

    needle_token = needle_core + rng.randn(dim).astype(np.float32) * 0.02
    needle_token /= np.linalg.norm(needle_token)

    needle_keys = raw_blocks[0].copy()
    needle_keys[15] = needle_token  # Outlier needle placed at position 15

    needle_block = IFRBlock(
        block_id="needle_block",
        tokens=list(range(32)),
        keys=needle_keys,
        values=np.zeros_like(needle_keys),
        orig_pos_start=0,
    )

    # Needle creates an outlier that inflates cluster diameter beyond threshold theta (typically ~1.18 - 1.22)
    assert needle_block.norm_dispersion > DEFAULT_DISPERSION_THRESHOLD, (
        f"Needle block dispersion {needle_block.norm_dispersion:.4f} failed to exceed theta {DEFAULT_DISPERSION_THRESHOLD}"
    )

    is_split_needle, doublets_needle = splitter.inspect_and_split(needle_block)
    assert is_split_needle, "Needle block must trigger anti-collapse doublet splitting"
    assert len(doublets_needle) == 2, "Doublet splitting must produce exactly 2 sub-centroids"

    needle_centroid, residual_centroid = doublets_needle
    needle_query = needle_core + rng.randn(dim).astype(np.float32) * 0.02
    needle_query /= np.linalg.norm(needle_query)

    needle_sim = float(np.dot(needle_centroid, needle_query))
    residual_sim = float(np.dot(residual_centroid, needle_query))
    naive_mean_sim = float(np.dot(needle_block.mean_k, needle_query))

    # Anti-collapse needle centroid retains high fidelity (> 0.95)
    # Naive mean centroid is heavily diluted (< 0.20)
    assert needle_sim > 0.95, f"Needle centroid dot product degraded: {needle_sim:.4f}"
    assert needle_sim > naive_mean_sim * 3.0, (
        f"Needle centroid {needle_sim:.4f} does not significantly dominate naive mean {naive_mean_sim:.4f}"
    )
    assert residual_sim < needle_sim, "Residual centroid score should be lower than needle centroid"


@pytest.mark.parametrize("context_tokens", [32768, 65536, 131072])
def test_retrieval_recall_32k_to_128k(context_tokens: int):
    """
    End-to-End Scale Verification:
    Verify that IFR achieves >= 98% retrieval recall across 32K~128K context scales
    under realistic non-uniform empirical KV activations, while naive mean-pooling
    collapses to poor recall.
    """
    res = run_scale_evaluation(
        context_tokens=context_tokens,
        num_needles=25,
        top_k_blocks=56,
        dim=128,
        n_clusters=16,
        n_probe=8,
        dispersion_threshold=DEFAULT_DISPERSION_THRESHOLD,
        seed=1000 + context_tokens,
    )

    # 1. Background dispersion matches Qwen3 empirical range
    assert abs(res["empirical_dispersion_mean"] - QWEN3_EMPIRICAL_DISPERSION_MEAN) < 0.03
    assert res["empirical_dispersion_mean"] > 0.70

    # 2. Anti-collapse triggers reliably on 100% of needle blocks
    assert res["anti_collapse_trigger_rate_needle"] == 1.0

    # 3. Retrieval recall with IFR reaches >= 98% across 32K ~ 128K
    assert res["recall_with_ifr_pct"] >= 98.0, (
        f"IFR recall {res['recall_with_ifr_pct']}% below 98% gate at scale {context_tokens}"
    )

    # 4. Without IFR anti-collapse, naive mean-pooling suffers severe rank inversion
    assert res["recall_without_ifr_pct"] < 50.0, (
        f"Naive recall did not demonstrate rank collapse: {res['recall_without_ifr_pct']}%"
    )

    # 5. Latency conforms to real-time interactive requirements
    assert res["mean_retrieval_ms_with_ifr"] < 5.0, (
        f"Retrieval latency excessive: {res['mean_retrieval_ms_with_ifr']} ms"
    )
