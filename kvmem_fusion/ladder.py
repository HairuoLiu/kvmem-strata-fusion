"""
LADDER: The In-KV Fidelity Ladder Engine & Quantization Module

Implements:
1. De-RoPE phase recovery & unrotated manifold transformations.
2. Multi-tier mixed-precision simulation (FP8, INT4, INT2, Quad-Merged).
3. Softmax tier-bias correction: b_t = -sigma_t^2 / 2 = -(rho_t * s)^2 / 2.
4. Quad-Merge (G=4) centroid clustering, delta position encoding, and rank-r residual decomposition.
5. CacheBlend HKVD (High-KV-Deviation) priority scoring for selective recomputation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def apply_rope(x: np.ndarray, pos: Union[int, np.ndarray], base: float = 10000.0) -> np.ndarray:
    """Standard 2D Rotary Position Embedding (RoPE) for (..., dim) vectors."""
    dim = x.shape[-1]
    assert dim % 2 == 0, f"Head dim must be even, got {dim}"
    idx = np.arange(0, dim // 2)
    inv_freq = base ** (-2.0 * idx / dim)

    # Support vectorized or scalar pos
    if isinstance(pos, np.ndarray):
        theta = np.outer(pos, inv_freq)  # [N, dim//2]
        while theta.ndim < x.ndim:
            theta = np.expand_dims(theta, axis=-2)
    else:
        theta = pos * inv_freq

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    out = np.zeros_like(x)
    out[..., 0::2] = x_even * cos_t - x_odd * sin_t
    out[..., 1::2] = x_even * sin_t + x_odd * cos_t
    return out


def derope(x: np.ndarray, pos: Union[int, np.ndarray], base: float = 10000.0) -> np.ndarray:
    """Apply inverse RoPE rotation to map vectors back to unrotated semantic space."""
    return apply_rope(x, pos=-pos, base=base)


@dataclass
class LadderTierConfig:
    name: str
    bits: float
    rho: float          # Relative error: ||e|| / ||k||
    description: str

    def get_sigma(self, s: float = 1.85) -> float:
        """Compute standard deviation of logit noise: sigma_t = rho_t * s."""
        return self.rho * s

    def get_tier_bias(self, s: float = 1.85) -> float:
        """Compute softmax tier-bias correction: b_t = -sigma_t^2 / 2."""
        sigma = self.get_sigma(s)
        return -(sigma ** 2) / 2.0


# Standard 4-Tier LADDER Configuration
LADDER_TIERS: Dict[str, LadderTierConfig] = {
    "FP8": LadderTierConfig(name="FP8", bits=8.0, rho=0.010, description="L0 Raw FP8 (Authority, reversible)"),
    "INT4": LadderTierConfig(name="INT4", bits=4.0, rho=0.120, description="L1 High-fidelity INT4 per-channel"),
    "INT2": LadderTierConfig(name="INT2", bits=2.0, rho=0.343, description="L2 KIVI-style 2-bit asymmetric"),
    "MERGED": LadderTierConfig(name="MERGED", bits=1.0, rho=0.420, description="L2' Quad-Merge G=4 Centroid 4-bit"),
}


def quantize_simulate(k: np.ndarray, tier_name: str, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate quantization noise corresponding to the target tier.
    Returns: (k_quantized, noise)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    tier = LADDER_TIERS[tier_name]
    signal_std = np.std(k)
    noise_std = tier.rho * signal_std
    noise = rng.randn(*k.shape) * noise_std
    k_quant = k + noise
    return k_quant, noise


def mixed_tier_softmax(
    logits: np.ndarray,
    tier_names: List[str],
    apply_correction: bool = True,
    s: float = 1.85
) -> np.ndarray:
    """Compute softmax over logits originating from mixed fidelity tiers.

    Args:
        logits: [N] array of attention logits
        tier_names: List of tier strings of length N
        apply_correction: If True, applies b_t = -sigma_t^2 / 2 per tier
        s: logit scale factor (default 1.85)
    Returns:
        softmax probability distribution [N]
    """
    assert len(logits) == len(tier_names)
    corrected_logits = logits.copy()

    if apply_correction:
        for idx, t_name in enumerate(tier_names):
            if t_name in LADDER_TIERS:
                b_t = LADDER_TIERS[t_name].get_tier_bias(s)
                corrected_logits[idx] += b_t

    # Numerically stable softmax
    max_l = np.max(corrected_logits)
    exp_l = np.exp(corrected_logits - max_l)
    return exp_l / np.sum(exp_l)


@dataclass
class QuadMergeResult:
    centroids_unrotated: np.ndarray     # [32, dim] Centroids in unrotated space
    canonical_positions: np.ndarray     # [32] Canonical reference positions
    pos_deltas: np.ndarray              # [4, 32] Position delta relative to canonical pos
    residuals_unrotated: np.ndarray     # [4, 32, dim] Residuals in unrotated space
    reconstructed_keys: np.ndarray      # [4, 32, dim] Reconstructed RoPE keys
    reconstruction_err: float           # Relative Frobenius reconstruction error
    hkvd_scores: np.ndarray             # [4, 32] Residual norm priority scores


def quad_merge_blocks(
    blocks_k: np.ndarray,
    block_positions: np.ndarray,
    base: float = 10000.0,
    rank_r: Optional[int] = None
) -> QuadMergeResult:
    """Execute G=4 Quad-Merge in unrotated space.

    Args:
        blocks_k: [4, 32, dim] RoPE-rotated keys of 4 blocks
        block_positions: [4, 32] Logical token positions of the 4 blocks
        base: RoPE base frequency
        rank_r: SVD truncation rank for residuals (if None, keeps full residual)
    """
    G, B, dim = blocks_k.shape
    assert G == 4, f"Quad-merge expects G=4 blocks, got {G}"

    # Step 1: De-RoPE all keys back to unrotated semantic space
    unrotated_k = np.zeros_like(blocks_k)
    for g in range(G):
        for b in range(B):
            unrotated_k[g, b] = derope(blocks_k[g, b], pos=block_positions[g, b], base=base)

    # Step 2: Centroid in unrotated space
    centroids_unrotated = np.mean(unrotated_k, axis=0)  # [32, dim]

    # Step 3: Canonical logical positions (median / mean rounded)
    canonical_positions = np.round(np.mean(block_positions, axis=0)).astype(int)  # [32]

    # Step 4: Position deltas delta_{g, b} = pos_{g, b} - canonical_pos_{b}
    pos_deltas = np.zeros((G, B), dtype=int)
    for g in range(G):
        pos_deltas[g] = block_positions[g] - canonical_positions

    # Step 5: Content residuals in unrotated space
    residuals_unrotated = np.zeros_like(unrotated_k)
    for g in range(G):
        residuals_unrotated[g] = unrotated_k[g] - centroids_unrotated

    # Optional Low-rank SVD compression on residuals across the 4 blocks
    approx_residuals = residuals_unrotated.copy()
    if rank_r is not None and rank_r < min(G * B, dim):
        flat_res = residuals_unrotated.reshape(G * B, dim)
        U, S, Vt = np.linalg.svd(flat_res, full_matrices=False)
        # Use np.dot to avoid macOS Accelerate BLAS matmul warning
        flat_approx = np.dot(U[:, :rank_r] * S[:rank_r], Vt[:rank_r, :])
        approx_residuals = flat_approx.reshape(G, B, dim)

    # Step 6: Reconstruction into RoPE space
    reconstructed_keys = np.zeros_like(blocks_k)
    for g in range(G):
        for b in range(B):
            # Reconstruct unrotated key: centroid + residual
            k_unrot_recon = centroids_unrotated[b] + approx_residuals[g, b]
            # Re-RoPE back to token's original logical position
            reconstructed_keys[g, b] = apply_rope(k_unrot_recon, pos=block_positions[g, b], base=base)

    # Calculate Frobenius relative error
    frob_norm_orig = np.linalg.norm(blocks_k)
    frob_norm_diff = np.linalg.norm(blocks_k - reconstructed_keys)
    rel_error = frob_norm_diff / frob_norm_orig

    # Step 7: CacheBlend HKVD (High-KV-Deviation) priority scores: norm of content residual
    hkvd_scores = np.linalg.norm(approx_residuals, axis=-1)  # [4, 32]

    return QuadMergeResult(
        centroids_unrotated=centroids_unrotated,
        canonical_positions=canonical_positions,
        pos_deltas=pos_deltas,
        residuals_unrotated=approx_residuals,
        reconstructed_keys=reconstructed_keys,
        reconstruction_err=rel_error,
        hkvd_scores=hkvd_scores
    )
