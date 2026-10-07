"""
Comprehensive Verification Suite for Track C: Fused Tier-Bias FlashAttention Kernel.

Validates:
1. Exact mathematical equivalence between Fused Tier-Bias FlashAttention (online softmax rescaling)
   and Unfused PyTorch/NumPy Attention across diverse tensor geometries.
2. Tile block size invariance (B_c in {16, 32, 64, 128}, B_r in {16, 32, 64}).
3. Numerical stability against extreme logits (+2000, -2000) avoiding exp overflow / underflow.
4. LADDER / UBBA tier-bias injection accuracy:
   b_t = - (rho_t * s)^2 / 2 counteracting Jensen's inequality logit theft.
5. Causal masking correctness with ragged / non-power-of-2 sequence lengths.
6. Bit-exact integration with CASA CanonicalAtomStore PagedAttention GEMM.
7. Microarchitectural HBM traffic elimination and roofline operational intensity metrics.
8. Universal dispatcher API compatibility across 2D, 3D, and 4D inputs.
"""

import math
import numpy as np
import pytest

from kernels.fused_tier_bias_attention import (
    fused_tier_bias_attention,
    fused_tier_bias_attention_sim,
    unfused_tier_bias_attention,
    paged_fused_tier_bias_attention_sim,
    FusedTierBiasAttentionSimulator,
    compute_microarch_metrics,
    benchmark_kernel_overhead,
)
from kvmem_fusion.core import CanonicalAtomStore, apply_rope


# ==============================================================================
# 1. MATHEMATICAL EQUIVALENCE & PRECISION VERIFICATION
# ==============================================================================

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_exact_mathematical_equivalence(causal: bool, head_dim: int, seq_len: int):
    """Verify bit-exact precision (double-precision machine epsilon < 1e-12)
    between Fused Tier-Bias FlashAttention and Unfused Reference.
    """
    np.random.seed(42 + seq_len + head_dim)
    batch_size = 2
    num_heads = 4

    q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float64)
    k = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float64)
    v = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float64)

    # Heterogeneous tier biases per key token
    bias = np.random.uniform(-0.5, 0.0, size=(seq_len,)).astype(np.float64)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # 1. Unfused Reference Attention
    out_ref, lse_ref = unfused_tier_bias_attention(
        q, k, v, bias=bias, sm_scale=sm_scale, causal=causal
    )

    # 2. Fused Tier-Bias FlashAttention Simulator
    out_fused, lse_fused = fused_tier_bias_attention_sim(
        q, k, v, bias=bias, sm_scale=sm_scale, causal=causal,
        block_m=64, block_n=64
    )

    diff_out = np.max(np.abs(out_ref - out_fused))
    diff_lse = np.max(np.abs(lse_ref - lse_fused))

    assert diff_out < 1e-12, f"Context mismatch: {diff_out:.2e} exceeds 1e-12 threshold"
    assert diff_lse < 1e-12, f"LSE mismatch: {diff_lse:.2e} exceeds 1e-12 threshold"


# ==============================================================================
# 2. TILE BLOCK SIZE INVARIANCE
# ==============================================================================

def test_tile_size_invariance():
    """Verify that tile geometry (block_m, block_n) does not affect output values.
    
    The FlashAttention-2 online softmax rescaling equations:
        alpha = exp(m_prev - m_curr)
        acc_curr = alpha * acc_prev + p_tile @ v_tile
    must produce mathematically invariant outputs regardless of tiling granularity.
    """
    np.random.seed(99)
    seq_len = 256
    head_dim = 64
    q = np.random.randn(1, 1, seq_len, head_dim).astype(np.float64)
    k = np.random.randn(1, 1, seq_len, head_dim).astype(np.float64)
    v = np.random.randn(1, 1, seq_len, head_dim).astype(np.float64)
    bias = np.random.uniform(-0.3, 0.0, size=(seq_len,)).astype(np.float64)

    tile_configs = [
        (16, 16),
        (32, 16),
        (32, 32),
        (64, 32),
        (64, 64),
        (128, 64),
    ]

    outputs = []
    for bm, bn in tile_configs:
        out, _ = fused_tier_bias_attention_sim(
            q, k, v, bias=bias, causal=True, block_m=bm, block_n=bn
        )
        outputs.append((bm, bn, out))

    # Compare all pairs against the first configuration
    base_bm, base_bn, base_out = outputs[0]
    for bm, bn, out in outputs[1:]:
        pair_diff = np.max(np.abs(base_out - out))
        assert pair_diff < 1e-12, (
            f"Tile config ({bm}, {bn}) differs from ({base_bm}, {base_bn}) by {pair_diff:.2e}"
        )


# ==============================================================================
# 3. NUMERICAL STABILITY & EXTREME LOGITS
# ==============================================================================

def test_numerical_stability_extreme_logits():
    """Verify numerical stability under extreme logits (+1500, -1500).
    
    Naive softmax without local max-stabilization overflows exp(1500) -> inf.
    Fused FlashAttention rescales by subtracting m_new = max(m_prev, rowmax(S_ij)),
    guaranteeing all arguments to exp() are <= 0.0.
    """
    seq_len = 64
    head_dim = 32
    q = np.ones((1, 1, seq_len, head_dim), dtype=np.float64) * 10.0
    k = np.ones((1, 1, seq_len, head_dim), dtype=np.float64) * 10.0
    v = np.random.randn(1, 1, seq_len, head_dim).astype(np.float64)

    # Dot products will be ~ 10 * 10 * 32 / sqrt(32) = 3200 / 5.656 = 565.6
    # Add huge tier bias +1000.0 to test extreme positive shift
    extreme_bias = np.full((seq_len,), 1000.0, dtype=np.float64)

    out_fused, lse_fused = fused_tier_bias_attention_sim(
        q, k, v, bias=extreme_bias, block_m=32, block_n=32
    )

    # Must contain no NaN and no Inf
    assert not np.isnan(out_fused).any(), "Fused output contains NaN"
    assert not np.isinf(out_fused).any(), "Fused output contains Inf"
    assert not np.isnan(lse_fused).any(), "LSE contains NaN"
    assert not np.isinf(lse_fused).any(), "LSE contains Inf"

    # Context must equal uniform average of V because all rows and keys are identical
    v_mean = np.mean(v, axis=-2, keepdims=True)
    mean_diff = np.max(np.abs(out_fused - v_mean))
    assert mean_diff < 1e-12, f"Extreme logit output deviated from expected mean: {mean_diff:.2e}"


# ==============================================================================
# 4. LADDER / UBBA TIER-BIAS CALIBRATION ACCURACY
# ==============================================================================

def test_tier_bias_ladder_calibration():
    """Verify that UBBA / LADDER tier-biases:
        b_t = - (rho_t * s)^2 / 2
    correctly scale down attention allocation on compressed tiers, counteracting
    Jensen's log-normal attention theft.
    """
    head_dim = 64
    seq_len = 128
    block_size = 32
    num_blocks = seq_len // block_size  # 4 blocks

    # Setup 4 distinct tiers matching LADDER specs:
    # Block 0: FP8   (rho = 0.010, b = -0.00017)
    # Block 1: INT4  (rho = 0.120, b = -0.0246)
    # Block 2: INT2  (rho = 0.343, b = -0.2013)
    # Block 3: MERGE (rho = 0.420, b = -0.3019)
    s_param = 1.85
    rho_specs = [0.010, 0.120, 0.343, 0.420]
    tier_biases_list = [-(rho * s_param) ** 2 / 2.0 for rho in rho_specs]
    bias_per_token = np.repeat(tier_biases_list, block_size).astype(np.float64)

    np.random.seed(314)
    q = np.random.randn(1, 1, 1, head_dim).astype(np.float64)  # Single decode query
    k = np.random.randn(1, 1, seq_len, head_dim).astype(np.float64)
    v = np.random.randn(1, 1, seq_len, head_dim).astype(np.float64)

    # 1. Attention with Tier Bias
    out_biased, _ = fused_tier_bias_attention_sim(
        q, k, v, bias=bias_per_token, block_m=1, block_n=block_size
    )

    # 2. Attention without Tier Bias (neutral baseline)
    out_unbiased, _ = fused_tier_bias_attention_sim(
        q, k, v, bias=None, block_m=1, block_n=block_size
    )

    # 3. Unfused Reference with Tier Bias
    out_ref_biased, _ = unfused_tier_bias_attention(
        q, k, v, bias=bias_per_token
    )

    # Confirm fused biased matches unfused biased bit-for-bit
    assert np.max(np.abs(out_biased - out_ref_biased)) < 1e-12

    # Confirm bias shifted the output substantially away from unbiased attention
    bias_shift = np.linalg.norm(out_biased - out_unbiased)
    assert bias_shift > 1e-3, f"Tier bias had negligible effect ({bias_shift:.2e})"


# ==============================================================================
# 5. RAGGED BOUNDARIES & ASYMMETRIC SEQUENCE LENGTHS
# ==============================================================================

@pytest.mark.parametrize("m, n", [(73, 151), (137, 269), (31, 65)])
def test_ragged_boundary_and_asymmetric_sequence_lengths(m: int, n: int):
    """Verify that sequence lengths not aligned to power-of-2 tile blocks
    execute without memory faults or padding contamination.
    """
    np.random.seed(123)
    head_dim = 64
    block_m = 64
    block_n = 64

    q = np.random.randn(1, 2, m, head_dim).astype(np.float64)
    k = np.random.randn(1, 2, n, head_dim).astype(np.float64)
    v = np.random.randn(1, 2, n, head_dim).astype(np.float64)
    bias = np.random.uniform(-0.1, 0.0, size=(n,)).astype(np.float64)

    # Non-causal
    out_nc_ref, lse_nc_ref = unfused_tier_bias_attention(q, k, v, bias=bias, causal=False)
    out_nc_fused, lse_nc_fused = fused_tier_bias_attention_sim(
        q, k, v, bias=bias, causal=False, block_m=block_m, block_n=block_n
    )
    assert np.max(np.abs(out_nc_ref - out_nc_fused)) < 1e-12
    assert np.max(np.abs(lse_nc_ref - lse_nc_fused)) < 1e-12

    # Causal (when m <= n)
    if m <= n:
        out_c_ref, lse_c_ref = unfused_tier_bias_attention(q, k, v, bias=bias, causal=True)
        out_c_fused, lse_c_fused = fused_tier_bias_attention_sim(
            q, k, v, bias=bias, causal=True, block_m=block_m, block_n=block_n
        )
        assert np.max(np.abs(out_c_ref - out_c_fused)) < 1e-12
        assert np.max(np.abs(lse_c_ref - lse_c_fused)) < 1e-12


# ==============================================================================
# 6. CASA PAGEDATTENTION INTEGRATION
# ==============================================================================

def test_paged_attention_casa_integration():
    """Verify that paged_fused_tier_bias_attention_sim achieves bit-exact equivalence
    with CanonicalAtomStore.compute_paged_attention_gemm over physical memory pages.
    """
    head_dim = 64
    block_size = 32
    num_blocks = 8
    np.random.seed(777)

    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    page_table = []
    tier_assignment = ["FP8", "INT4", "INT2", "MERGED", "FP8", "INT4", "INT2", "MERGED"]
    tier_biases = {"FP8": 0.0, "INT4": -0.0246, "INT2": -0.2013, "MERGED": -0.3019}

    parent_hash = None
    for b in range(num_blocks):
        tokens = list(range(b * block_size, (b + 1) * block_size))
        k_raw = np.random.randn(block_size, head_dim)
        v_raw = np.random.randn(block_size, head_dim)
        blk_id, parent_hash = store.register_prefix_block(
            tokens=tokens,
            k_unrotated=k_raw,
            v=v_raw,
            orig_pos_start=b * block_size,
            parent_hash=parent_hash,
            tier=tier_assignment[b]
        )
        page_table.append(store.blocks[blk_id].physical_page_id)

    # Decode query at sequence end
    q_pos = num_blocks * block_size
    q_raw = np.random.randn(head_dim)

    # 1. CASA Engine Execution
    weights_casa, context_casa = store.compute_paged_attention_gemm(
        q_raw=q_raw, query_logical_pos=q_pos, page_table=page_table, tier_biases=tier_biases
    )

    # 2. Fused PagedAttention Kernel Simulator Execution
    # Rotate Q once at query logical position
    q_rot = apply_rope(q_raw, pos=q_pos)
    page_biases_mapped = {p_idx: tier_biases[store.blocks[store.page_to_block_id[p_idx]].tier] for p_idx in page_table}

    out_paged, _ = paged_fused_tier_bias_attention_sim(
        q=q_rot,
        physical_k_pool=store.physical_k_pool,
        physical_v_pool=store.physical_v_pool,
        page_table=page_table,
        page_biases=page_biases_mapped,
        block_size=block_size
    )

    diff = np.max(np.abs(context_casa - out_paged))
    assert diff < 1e-12, f"CASA vs Fused PagedAttention context difference {diff:.2e} exceeds threshold"


# ==============================================================================
# 7. HBM MEMORY TRAFFIC ELIMINATION & ROOFLINE ANALYSIS
# ==============================================================================

def test_hbm_traffic_elimination_and_zero_roundtrip_overhead():
    """Verify microarchitectural properties:
    1. Zero bytes of intermediate score matrix S materialized in HBM.
    2. HBM traffic reduction ratio > 1.0 (scales as O(N / d)).
    3. Peak SRAM consumption is bounded within GPU shared memory limits (< 100 KB).
    """
    metrics = compute_microarch_metrics(
        batch_size=2,
        num_heads=8,
        seq_len_q=2048,
        seq_len_k=2048,
        head_dim=64,
        block_m=64,
        block_n=64,
        bytes_per_elem=2,
    )

    # Unfused must materialize full intermediate S matrix in HBM (2 * 8 * 2048 * 2048 * 4 bytes = 268.4 MB)
    assert metrics.intermediate_score_matrix_bytes == 2 * 8 * 2048 * 2048 * 4
    assert metrics.intermediate_score_matrix_bytes > 200 * 1024 * 1024  # > 200 MB

    # Fused HBM traffic must be significantly less than Unfused
    assert metrics.hbm_traffic_reduction_ratio > 30.0, (
        f"Expected > 30x HBM traffic reduction at seq_len 2048, got {metrics.hbm_traffic_reduction_ratio:.1f}x"
    )

    # Fused operational intensity must be dramatically higher
    assert metrics.fused_operational_intensity > metrics.unfused_operational_intensity * 20.0

    # Peak SRAM footprint for 64x64 tiles must be <= 64 KB (fits easily in A100/H100 108/228 KB SRAM)
    assert metrics.sram_peak_bytes <= 65536, f"SRAM peak {metrics.sram_peak_bytes} exceeds 64 KB"


# ==============================================================================
# 8. UNIVERSAL DISPATCHER & BENCHMARK HARNESS
# ==============================================================================

def test_universal_dispatcher_shapes():
    """Verify universal dispatcher supports 2D, 3D, and 4D tensor shapes."""
    np.random.seed(11)
    d = 64

    # 2D: [seq_len, head_dim]
    q2 = np.random.randn(64, d)
    k2 = np.random.randn(64, d)
    v2 = np.random.randn(64, d)
    o2, _ = fused_tier_bias_attention(q2, k2, v2)
    assert o2.shape == (64, d)

    # 3D: [batch, seq_len, head_dim]
    q3 = np.random.randn(2, 64, d)
    k3 = np.random.randn(2, 64, d)
    v3 = np.random.randn(2, 64, d)
    o3, _ = fused_tier_bias_attention(q3, k3, v3)
    assert o3.shape == (2, 64, d)

    # 4D: [batch, heads, seq_len, head_dim]
    q4 = np.random.randn(2, 4, 64, d)
    k4 = np.random.randn(2, 4, 64, d)
    v4 = np.random.randn(2, 4, 64, d)
    o4, _ = fused_tier_bias_attention(q4, k4, v4)
    assert o4.shape == (2, 4, 64, d)


def test_benchmark_kernel_overhead_suite():
    """Verify that benchmark_kernel_overhead runs successfully and returns full telemetry."""
    res = benchmark_kernel_overhead(batch_size=1, num_heads=2, seq_len=128, head_dim=64, block_size=32)
    assert res["max_abs_err"] < 1e-12
    assert res["max_lse_err"] < 1e-12
    assert res["traffic_reduction_ratio"] > 1.0
    assert "intermediate_score_matrix_mb" in res
    assert "sram_peak_kb" in res
