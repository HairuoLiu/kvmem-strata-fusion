"""
Tests for Plan 02 (LADDER: The In-KV Fidelity Ladder).
Verifies:
1. De-RoPE phase preservation & high-frequency mode protection (Section 3.2, M0 Gate).
2. Analytical Moment Generating Function bias verification: E[exp(l + e)] = exp(l) * exp(sigma_t^2 / 2).
3. Mixed-precision softmax tier-bias calibration across FP8, INT4, INT2, and MERGED tiers.
4. Mitigation of attention mass theft by low-bit blocks.
5. Quad-Merge (G=4) centroid clustering, delta position encoding, rank-r residual reconstruction, and HKVD prioritization.
6. P0 Gate Criteria falsification benchmarks.
"""

import numpy as np
import pytest
from kvmem_fusion.ladder import (
    apply_rope,
    derope,
    get_tier_biases,
    LADDER_TIERS,
    LadderTierConfig,
    mixed_tier_lse,
    mixed_tier_softmax,
    quantize_simulate,
    quad_merge_blocks,
)


def test_derope_phase_preservation_high_freq():
    """Verify LADDER Section 3.2 and P0 M0 Gate:
    Direct RoPE averaging suffers severe phase cancellation in high-frequency modes (decay ~82%),
    while de-RoPE -> centroid -> re-RoPE retains the full semantic direction (>70%).
    """
    dim = 128
    block_size = 32
    np.random.seed(42)

    # Coherent semantic cluster representing tokens in a block or queryable concept
    semantic_center = np.random.randn(dim)
    semantic_center /= np.linalg.norm(semantic_center)

    raw_keys = semantic_center + 0.05 * np.random.randn(block_size, dim)
    raw_keys /= np.linalg.norm(raw_keys, axis=-1, keepdims=True)

    positions = np.arange(block_size)
    rotated_keys = np.zeros_like(raw_keys)
    for i in range(block_size):
        rotated_keys[i] = apply_rope(raw_keys[i], pos=positions[i])

    # Method A: Naive direct averaging on RoPE-rotated keys
    naive_avg = np.mean(rotated_keys, axis=0)

    # Method B: LADDER de-RoPE -> average -> re-RoPE
    deroped_keys = np.zeros_like(raw_keys)
    for i in range(block_size):
        deroped_keys[i] = derope(rotated_keys[i], pos=positions[i])
    deroped_avg = np.mean(deroped_keys, axis=0)
    ladder_re_roped = apply_rope(deroped_avg, pos=0)

    # High-frequency channels: lowest indices [0..7] with highest omega_j
    high_freq_slice = slice(0, 8)
    norm_gt_high = np.linalg.norm(semantic_center[high_freq_slice])
    norm_naive_high = np.linalg.norm(naive_avg[high_freq_slice])
    norm_ladder_high = np.linalg.norm(ladder_re_roped[high_freq_slice])

    retention_naive = norm_naive_high / norm_gt_high
    retention_ladder = norm_ladder_high / norm_gt_high

    print(f"\n[Test de-RoPE] Naive High-Freq Retention: {retention_naive:.4f}")
    print(f"[Test de-RoPE] LADDER High-Freq Retention: {retention_ladder:.4f}")

    # P0 Gate Criteria: Naive decays under destructive interference (<0.25), LADDER retains (>0.70)
    assert retention_naive < 0.25, f"Expected naive retention < 0.25, got {retention_naive}"
    assert retention_ladder > 0.70, f"Expected LADDER retention > 0.70, got {retention_ladder}"


def test_mixed_precision_tier_bias_analytical_mgf():
    """Verify LADDER Section 3.5:
    For any tier t with relative error rho_t, logit noise variance is sigma_t^2 = (rho_t * s)^2.
    Uncorrected expectation ratio is exp(sigma_t^2 / 2).
    Tier-bias b_t = -sigma_t^2 / 2 cancels this factor exactly.
    """
    np.random.seed(1234)
    n_samples = 200000
    base_logit = 1.5
    s = 1.85

    for tier_name, cfg in LADDER_TIERS.items():
        sigma = cfg.get_sigma(s)
        b_t = cfg.get_tier_bias(s)
        expected_theft_factor = np.exp((sigma ** 2) / 2.0)

        noise = np.random.randn(n_samples) * sigma
        noisy_logits = base_logit + noise

        # Uncorrected expectation
        uncorrected_exp = np.mean(np.exp(noisy_logits))
        gt_exp = np.exp(base_logit)
        emp_ratio = uncorrected_exp / gt_exp

        # Corrected expectation
        corrected_logits = noisy_logits + b_t
        corrected_exp = np.mean(np.exp(corrected_logits))
        corrected_ratio = corrected_exp / gt_exp

        print(
            f"Tier {tier_name:6s} (rho={cfg.rho:.3f}, sigma={sigma:.4f}): "
            f"Theory Factor={expected_theft_factor:.4f}, Emp={emp_ratio:.4f}, Corrected={corrected_ratio:.4f}"
        )

        assert abs(emp_ratio - expected_theft_factor) < 0.04
        assert abs(corrected_ratio - 1.0) < 0.02


def test_mixed_tier_softmax_attention_theft_mitigation():
    """Verify that uncorrected mixed-tier softmax skews attention mass towards noisy low-bit blocks,
    and tier-bias correction b_t restores balanced ground-truth attention mass.
    """
    np.random.seed(777)
    n_trials = 2000
    head_dim = 64

    # Setup: 4 blocks in active attention view:
    # Block 0: FP8 (the true ground-truth needle block with higher logit)
    # Block 1: INT4 (distractor)
    # Block 2: INT2 (noisy distractor)
    # Block 3: MERGED (coarse distractor)
    tiers = ["FP8", "INT4", "INT2", "MERGED"]
    clean_logits = np.array([3.0, 1.8, 1.5, 1.2])

    clean_probs = np.exp(clean_logits - np.max(clean_logits))
    clean_probs /= np.sum(clean_probs)
    gt_needle_prob = clean_probs[0]

    uncorrected_needle_probs = []
    corrected_needle_probs = []

    s = 1.85
    sigmas = [LADDER_TIERS[t].get_sigma(s) for t in tiers]

    for _ in range(n_trials):
        noise = np.array([np.random.randn() * sig for sig in sigmas])
        noisy_logits = clean_logits + noise

        # 1. Uncorrected mixed softmax
        p_uncorrected = mixed_tier_softmax(noisy_logits, tiers, apply_correction=False, s=s)
        uncorrected_needle_probs.append(p_uncorrected[0])

        # 2. Corrected mixed softmax with tier bias
        p_corrected = mixed_tier_softmax(noisy_logits, tiers, apply_correction=True, s=s)
        corrected_needle_probs.append(p_corrected[0])

    mean_uncorrected_needle = np.mean(uncorrected_needle_probs)
    mean_corrected_needle = np.mean(corrected_needle_probs)

    print(f"\n[Softmax Attention Theft]")
    print(f"Ground Truth Needle Attention: {gt_needle_prob:.4f}")
    print(f"Uncorrected Mean Needle Attention: {mean_uncorrected_needle:.4f} (Under-allocation due to low-tier theft)")
    print(f"Corrected Mean Needle Attention: {mean_corrected_needle:.4f}")

    # Low-bit noisy blocks steal probability mass, reducing needle allocation
    assert mean_uncorrected_needle < gt_needle_prob - 0.02
    # Tier-bias correction brings the needle attention mass back to ground truth
    assert abs(mean_corrected_needle - gt_needle_prob) < 0.015


def test_quad_merge_reconstruction_and_hkvd():
    """Verify Section 3.3 and 3.4:
    G=4 Quad-Merge in unrotated space:
    1. Exact reversibility when full residuals are preserved.
    2. Controlled low rank truncation (rank-r SVD).
    3. Proper alignment and HKVD priority scoring for selective recomputation.
    """
    dim = 64
    block_size = 32
    G = 4
    np.random.seed(999)

    # 4 blocks, each at logical start positions 0, 100, 200, 300
    block_positions = np.zeros((G, block_size), dtype=int)
    for g in range(G):
        block_positions[g] = np.arange(g * 100, g * 100 + block_size)

    # Shared semantic cluster with slight token variations
    cluster_center = np.random.randn(block_size, dim)
    cluster_center /= np.linalg.norm(cluster_center, axis=-1, keepdims=True)

    unrotated_k = np.zeros((G, block_size, dim))
    blocks_k = np.zeros((G, block_size, dim))

    for g in range(G):
        noise = 0.05 * np.random.randn(block_size, dim)
        unrotated_k[g] = cluster_center + noise
        for b in range(block_size):
            blocks_k[g, b] = apply_rope(unrotated_k[g, b], pos=block_positions[g, b])

    # Case 1: Exact residual retention
    result_exact = quad_merge_blocks(blocks_k, block_positions, rank_r=None)
    assert result_exact.reconstruction_err < 1e-12, (
        f"Expected exact reconstruction, got error {result_exact.reconstruction_err}"
    )

    # Verify position deltas delta = pos - canonical_pos
    for g in range(G):
        reconstructed_pos = result_exact.canonical_positions + result_exact.pos_deltas[g]
        np.testing.assert_array_equal(reconstructed_pos, block_positions[g])

    # Case 2: Rank-r low rank residual compression (r=4)
    # Theory: rho_merge falls within [0.215, 0.42]
    result_rank_r = quad_merge_blocks(blocks_k, block_positions, rank_r=4)
    print(f"\n[Quad-Merge] Rank-4 Relative Reconstruction Error: {result_rank_r.reconstruction_err:.4f}")
    assert 0.15 <= result_rank_r.reconstruction_err <= 0.35

    # Case 3: CacheBlend HKVD priority scoring
    # Introduce one anomalous outlier token in block 0 token 5
    blocks_k_outlier = blocks_k.copy()
    outlier_vec = np.random.randn(dim) * 3.0
    blocks_k_outlier[0, 5] = apply_rope(outlier_vec, pos=block_positions[0, 5])
    result_outlier = quad_merge_blocks(blocks_k_outlier, block_positions, rank_r=4)

    # Outlier should have the highest HKVD residual score
    hkvd_scores = result_outlier.hkvd_scores
    max_idx = np.unravel_index(np.argmax(hkvd_scores), hkvd_scores.shape)
    assert max_idx == (0, 5), f"Expected outlier at (0, 5) to have highest HKVD score, got {max_idx}"


def test_p0_gate_criteria_falsification():
    """Verify P0 M0 Gate criteria:
    - High-frequency norm retention > 0.70 for LADDER vs < 0.25 for Naive.
    - Softmax KL divergence: expected attention distribution under tier-bias correction
      achieves > 10x lower KL divergence to ground truth than uncorrected mixed softmax.
    """
    dim = 64
    block_size = 32
    np.random.seed(555)

    # 1. Frequency retention gate
    semantic_center = np.random.randn(dim)
    semantic_center /= np.linalg.norm(semantic_center)
    raw_keys = np.tile(semantic_center, (block_size, 1))

    rotated_keys = np.zeros_like(raw_keys)
    for i in range(block_size):
        rotated_keys[i] = apply_rope(raw_keys[i], pos=i)

    naive_avg = np.mean(rotated_keys, axis=0)
    deroped_avg = np.mean([derope(rotated_keys[i], pos=i) for i in range(block_size)], axis=0)

    hf_slice = slice(0, 4)
    naive_retention = np.linalg.norm(naive_avg[hf_slice]) / np.linalg.norm(semantic_center[hf_slice])
    ladder_retention = np.linalg.norm(deroped_avg[hf_slice]) / np.linalg.norm(semantic_center[hf_slice])

    # M0 Gates
    assert naive_retention < 0.25
    assert ladder_retention > 0.95

    # 2. Softmax Expected Attention KL divergence Gate
    clean_logits = np.array([2.5, 2.0, 1.8, 1.2, 0.9, 0.5])
    tiers = ["FP8", "INT4", "INT4", "INT2", "INT2", "MERGED"]
    p_gt = np.exp(clean_logits - np.max(clean_logits))
    p_gt /= np.sum(p_gt)

    s = 1.85
    sigmas = [LADDER_TIERS[t].get_sigma(s) for t in tiers]

    p_uncorr_sum = np.zeros_like(p_gt)
    p_corr_sum = np.zeros_like(p_gt)
    n_eval = 20000

    for _ in range(n_eval):
        noise = np.array([np.random.randn() * sig for sig in sigmas])
        noisy_logits = clean_logits + noise

        p_uncorr_sum += mixed_tier_softmax(noisy_logits, tiers, apply_correction=False, s=s)
        p_corr_sum += mixed_tier_softmax(noisy_logits, tiers, apply_correction=True, s=s)

    p_uncorr_avg = p_uncorr_sum / n_eval
    p_corr_avg = p_corr_sum / n_eval

    kl_uncorr = np.sum(p_gt * np.log(p_gt / p_uncorr_avg))
    kl_corr = np.sum(p_gt * np.log(p_gt / p_corr_avg))

    print(f"\n[P0 Gate Expected Softmax KL Divergence]")
    print(f"Uncorrected Mixed Softmax Expected KL: {kl_uncorr:.6f}")
    print(f"Tier-Bias Corrected Mixed Softmax Expected KL: {kl_corr:.6f}")
    print(f"KL Reduction Ratio: {kl_corr / kl_uncorr:.4f}")

    # Corrected softmax eliminates systematic bias, achieving > 10x KL reduction
    assert kl_corr < kl_uncorr * 0.10, (
        f"Expected tier-bias to reduce expected KL divergence by > 90%, got {kl_corr} vs {kl_uncorr}"
    )


def test_apply_rope_vectorized_broadcasting_shapes():
    """Verify that apply_rope and derope seamlessly handle scalar, 1D, and 2D pos shapes
    without broadcast errors across varied batch dimensions.
    """
    dim = 64
    np.random.seed(42)

    # 1. Scalar pos with 1D vector
    x_1d = np.random.randn(dim)
    r_1d = apply_rope(x_1d, pos=7)
    assert r_1d.shape == (dim,)
    recon_1d = derope(r_1d, pos=7)
    np.testing.assert_allclose(recon_1d, x_1d, atol=1e-12)

    # 2. 1D pos with 2D tensor (seq, dim)
    seq_len = 32
    x_2d = np.random.randn(seq_len, dim)
    pos_1d = np.arange(seq_len)
    r_2d = apply_rope(x_2d, pos=pos_1d)
    assert r_2d.shape == (seq_len, dim)
    recon_2d = derope(r_2d, pos=pos_1d)
    np.testing.assert_allclose(recon_2d, x_2d, atol=1e-12)

    # 3. 1D pos with 3D batched tensor (batch, seq, dim)
    batch_size = 4
    x_3d = np.random.randn(batch_size, seq_len, dim)
    r_3d = apply_rope(x_3d, pos=pos_1d)
    assert r_3d.shape == (batch_size, seq_len, dim)
    recon_3d = derope(r_3d, pos=pos_1d)
    np.testing.assert_allclose(recon_3d, x_3d, atol=1e-12)

    # 4. 2D pos matching 3D batched tensor (batch, seq)
    pos_2d = np.arange(batch_size * seq_len).reshape(batch_size, seq_len)
    r_3d_per_batch = apply_rope(x_3d, pos=pos_2d)
    assert r_3d_per_batch.shape == (batch_size, seq_len, dim)
    recon_3d_per_batch = derope(r_3d_per_batch, pos=pos_2d)
    np.testing.assert_allclose(recon_3d_per_batch, x_3d, atol=1e-12)


def test_ifr_lse_caching_compatibility_and_ranking_invariance():
    """Verify LADDER interface with IFR retrieval & LSE Caching:
    1. get_tier_biases returns valid negative biases for all tiers.
    2. Ranking by unnormalized corrected logits (s_j + b_{t_j}) strictly equals
       ranking by normalized mixed softmax probabilities.
    3. Expected partition function Z under tier bias cancels noise inflation.
    """
    np.random.seed(888)
    s = 1.85
    biases = get_tier_biases(s=s)
    assert "FP8" in biases and "INT4" in biases and "INT2" in biases and "MERGED" in biases
    assert biases["FP8"] > biases["INT4"] > biases["INT2"] > biases["MERGED"]

    # Candidate set of mixed-tier blocks (e.g. from IFR candidate routing)
    clean_logits = np.array([4.0, 3.2, 2.8, 1.9, 1.5, 0.8])
    tiers = ["FP8", "INT4", "INT4", "INT2", "INT2", "MERGED"]
    noisy_logits = clean_logits.copy()

    # 1. Unnormalized ranking equivalence
    corrected_logits = noisy_logits.copy()
    for i, t in enumerate(tiers):
        corrected_logits[i] += biases[t]

    probs = mixed_tier_softmax(noisy_logits, tiers, apply_correction=True, s=s)
    lse, z = mixed_tier_lse(noisy_logits, tiers, apply_correction=True, s=s)

    # Check softmax matches exp(corrected_logits - lse)
    expected_probs = np.exp(corrected_logits - lse)
    np.testing.assert_allclose(probs, expected_probs, atol=1e-12)

    # Ranking by unnormalized logits exactly matches ranking by normalized softmax
    rank_unnorm = np.argsort(-corrected_logits)
    rank_softmax = np.argsort(-probs)
    np.testing.assert_array_equal(rank_unnorm, rank_softmax)

    # 2. Partition function Z preservation under Monte Carlo noise
    gt_z = np.sum(np.exp(clean_logits))
    sigmas = [LADDER_TIERS[t].get_sigma(s) for t in tiers]
    n_mc = 15000
    z_uncorr_samples = []
    z_corr_samples = []

    for _ in range(n_mc):
        noise = np.array([np.random.randn() * sig for sig in sigmas])
        sample_logits = clean_logits + noise

        # Uncorrected Z
        _, z_uncorr = mixed_tier_lse(sample_logits, tiers, apply_correction=False, s=s)
        z_uncorr_samples.append(z_uncorr)

        # Corrected Z
        _, z_corr = mixed_tier_lse(sample_logits, tiers, apply_correction=True, s=s)
        z_corr_samples.append(z_corr)

    mean_z_uncorr = np.mean(z_uncorr_samples)
    mean_z_corr = np.mean(z_corr_samples)

    print(f"\n[IFR LSE Caching Compatibility]")
    print(f"Ground Truth Z:         {gt_z:.4f}")
    print(f"Uncorrected Noisy E[Z]: {mean_z_uncorr:.4f} (Inflated by ~{mean_z_uncorr / gt_z - 1.0:.2%})")
    print(f"Tier-Corrected E[Z]:    {mean_z_corr:.4f} (Ratio: {mean_z_corr / gt_z:.4f})")

    # Uncorrected Z is inflated due to Jensen / log-normal MGF
    assert mean_z_uncorr > gt_z * 1.03
    # Tier-corrected Z recovers the ground-truth partition function within 0.5%
    assert abs(mean_z_corr - gt_z) / gt_z < 0.005


def test_quad_merge_position_delta_int8_bounds():
    """Verify Section 4 & Table 3.3 signed 8-bit integer bounds for position deltas:
    - Contiguous local blocks (span <= 128 tokens) strictly fit inside [-128, 127].
    - Non-local blocks (span > 256 tokens) exceed int8 and require 32-bit block base pointers.
    """
    dim = 64
    block_size = 32
    G = 4
    np.random.seed(101)

    # Case A: Local contiguous blocks (total span = 128 tokens)
    local_positions = np.zeros((G, block_size), dtype=int)
    for g in range(G):
        local_positions[g] = np.arange(g * block_size, (g + 1) * block_size)

    blocks_k_local = np.random.randn(G, block_size, dim)
    res_local = quad_merge_blocks(blocks_k_local, local_positions)

    assert res_local.fits_int8, f"Expected contiguous blocks to fit in int8, max delta: {np.max(np.abs(res_local.pos_deltas))}"
    # Verify int8 conversion is lossless
    deltas_int8 = res_local.pos_deltas.astype(np.int8)
    np.testing.assert_array_equal(deltas_int8, res_local.pos_deltas)

    # Case B: Widely separated blocks across non-local sequence (span = 600 tokens)
    distant_positions = np.zeros((G, block_size), dtype=int)
    for g, start_pos in enumerate([0, 200, 400, 600]):
        distant_positions[g] = np.arange(start_pos, start_pos + block_size)

    res_distant = quad_merge_blocks(blocks_k_local, distant_positions)
    assert not res_distant.fits_int8, "Expected distant blocks to exceed int8 range"
    max_delta = np.max(np.abs(res_distant.pos_deltas))
    assert max_delta > 128, f"Expected delta > 128, got {max_delta}"

