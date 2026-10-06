"""
Phase 0 Mathematical & Mechanism Falsification Benchmark
Validating:
1. P0 RoPE Phase Cancellation: direct RoPE averaging vs de-RoPE -> average -> re-RoPE.
2. CASA K-Freeze & Transient Q-remap: algebraic equivalence of relative RoPE.
3. LADDER Tier-Bias Correction in Mixed-Precision Softmax: b_t = -sigma_t^2 / 2.
"""

import numpy as np


def apply_rope(x: np.ndarray, pos: int, base: float = 10000.0) -> np.ndarray:
    """Apply standard rotary position embedding (RoPE) to vector or batch of vectors.
    Uses 2D rotary pairs on adjacent channels [2*i, 2*i+1].
    """
    dim = x.shape[-1]
    assert dim % 2 == 0
    idx = np.arange(0, dim // 2)
    theta = pos * (base ** (-2.0 * idx / dim))
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    out = np.zeros_like(x)
    out[..., 0::2] = x_even * cos_t - x_odd * sin_t
    out[..., 1::2] = x_even * sin_t + x_odd * cos_t
    return out


def test_p0_rope_phase_cancellation():
    """Verify LADDER Section 3.2:

    Direct averaging of 32 tokens in a block suffers massive amplitude decay in high-frequency
    modes (~82% loss), whereas de-RoPE -> average -> re-RoPE preserves signal representation.
    """
    dim = 128
    block_size = 32
    np.random.seed(42)

    # 1. Test case: tokens with shared semantic cluster (e.g. repeated prompt or cluster in quad-merge)
    semantic_center = np.random.randn(dim)
    semantic_center /= np.linalg.norm(semantic_center)
    raw_keys = semantic_center + 0.05 * np.random.randn(block_size, dim)
    raw_keys /= np.linalg.norm(raw_keys, axis=-1, keepdims=True)

    # Rotate tokens to their respective logical positions 0..31
    rotated_keys = np.zeros_like(raw_keys)
    for i in range(block_size):
        rotated_keys[i] = apply_rope(raw_keys[i], pos=i)

    # Method A: Direct naive averaging on rotated keys
    naive_avg = np.mean(rotated_keys, axis=0)

    # Method B: de-RoPE -> average -> re-RoPE (LADDER approach)
    deroped_keys = np.zeros_like(raw_keys)
    for i in range(block_size):
        deroped_keys[i] = apply_rope(rotated_keys[i], pos=-i)
    deroped_avg = np.mean(deroped_keys, axis=0)
    ladder_centroid = apply_rope(deroped_avg, pos=0)

    # High frequency modes (first 8 dims) vs low frequency modes (last 8 dims)
    high_freq_idx = slice(0, 8)
    norm_raw_high = np.linalg.norm(semantic_center[high_freq_idx])
    norm_naive_high = np.linalg.norm(naive_avg[high_freq_idx])
    norm_ladder_high = np.linalg.norm(ladder_centroid[high_freq_idx])

    high_retention_naive = norm_naive_high / norm_raw_high
    high_retention_ladder = norm_ladder_high / norm_raw_high

    print("\n--- P0 RoPE Phase Cancellation Falsification ---")
    print(
        f"High-frequency mode norm retention: Naive={high_retention_naive:.4f}, LADDER (de-RoPE)={high_retention_ladder:.4f}"
    )

    # Naive loses ~80-90% of high-frequency amplitude due to destructive interference (Naïve ≈ 0.16)
    # LADDER Section 5 (M0 Gate): Expected 0.60–1.0 vs ≈0.18
    assert high_retention_naive < 0.25, f"Expected high frequency decay, got {high_retention_naive}"
    assert high_retention_ladder > 0.70, (
        f"Expected de-RoPE to preserve high frequency, got {high_retention_ladder}"
    )
    print("✓ P0 RoPE Phase Cancellation theorem CONFIRMED.")


def test_casa_kfreeze_qremap():
    """Verify CASA core foundation:

    Instead of re-rotating K to new compact slot c, we freeze K at its original slot n
    and remap transient query q from m to (m - n).
    Attention score: q_m^T R_{m-n} k_n is mathematically identical.
    """
    dim = 64
    np.random.seed(123)

    q_raw = np.random.randn(dim)
    k_raw = np.random.randn(dim)

    m = 1050  # Query logical position
    n = 200  # Key original logical position

    # 1. Ground truth attention logit: (R_m q)^T (R_n k)
    q_rot = apply_rope(q_raw, pos=m)
    k_rot = apply_rope(k_raw, pos=n)
    logit_gt = np.dot(q_rot, k_rot)

    # 2. CASA relative rotation: (R_{m-n} q_raw)^T k_raw
    q_rel = apply_rope(q_raw, pos=m - n)
    logit_casa_raw = np.dot(q_rel, k_raw)

    print("\n--- CASA K-Freeze & Q-remap Invariance ---")
    print(f"Ground Truth Logit: {logit_gt:.8f}")
    print(f"CASA Q-remap Logit: {logit_casa_raw:.8f}")
    diff = abs(logit_gt - logit_casa_raw)
    print(f"Absolute Numerical Error: {diff:.2e}")
    assert diff < 1e-12, f"Q-remap invariance failed, error={diff}"
    print("✓ CASA K-Freeze / Q-remap algebraic equivalence CONFIRMED.")


def test_tier_bias_correction():
    """Verify LADDER Section 3.5:

    When lower-tier noisy keys (variance sigma_t^2) enter mixed softmax,
    the uncorrected expected exp(logit + e) is scaled by exp(sigma_t^2 / 2),
    systematically stealing ~22% attention mass.
    b_t = -sigma_t^2 / 2 exactly restores unbiased expectation.
    """
    np.random.seed(999)
    n_samples = 250000
    base_logit = 2.0  # true clean logit
    sigma_noise = 0.634  # corresponding to ~0.343 relative error * s=1.85

    noise = np.random.randn(n_samples) * sigma_noise
    noisy_logits = base_logit + noise

    # Uncorrected expectation of exp(logit)
    exp_noisy_uncorrected = np.mean(np.exp(noisy_logits))
    exp_clean = np.exp(base_logit)
    bias_ratio = exp_noisy_uncorrected / exp_clean
    theoretical_bias = np.exp((sigma_noise**2) / 2.0)

    # Corrected with tier bias: b_t = -sigma^2 / 2
    b_t = -(sigma_noise**2) / 2.0
    corrected_logits = noisy_logits + b_t
    exp_noisy_corrected = np.mean(np.exp(corrected_logits))
    corrected_ratio = exp_noisy_corrected / exp_clean

    print("\n--- LADDER Mixed-Tier Softmax Bias Falsification ---")
    print(f"Clean exp(logit):                     {exp_clean:.4f}")
    print(
        f"Uncorrected noisy exp(logit):         {exp_noisy_uncorrected:.4f} (Ratio: {bias_ratio:.4f}, Theory: {theoretical_bias:.4f})"
    )
    print(
        f"Tier-bias corrected exp(logit + b_t): {exp_noisy_corrected:.4f} (Ratio: {corrected_ratio:.4f})"
    )

    assert abs(bias_ratio - theoretical_bias) < 0.05
    assert abs(corrected_ratio - 1.0) < 0.02
    print("✓ LADDER Tier-Bias Mathematical Derivation CONFIRMED.")


if __name__ == "__main__":
    test_p0_rope_phase_cancellation()
    test_casa_kfreeze_qremap()
    test_tier_bias_correction()
    print("\nALL PHASE 0 MATHEMATICAL THEOREMS PASS WITH FLYING COLORS!\n")
