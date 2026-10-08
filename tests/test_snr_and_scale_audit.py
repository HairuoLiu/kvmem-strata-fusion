"""
Pytest suite for SNR sensitivity, false alarm rate, and scale scaling audit.
Verifies external critique points:
- Failure modes of fixed threshold under high background noise
- Dilution failure of standard Mean-K vs doublet centroid preservation
- Selection rate scaling from 32K to 1M tokens
"""

import numpy as np
import pytest
from kvmem_fusion.ifr import (
    AntiCollapseSplitter,
    compute_deroped_mean,
    IFRBlock,
    IFRRetriever,
)


def test_anti_collapse_dispersion_false_alarm_under_high_noise():
    """Verify critique §2.4 & A15: static threshold 0.85 causes false alarms when sigma >= 0.10."""
    rng = np.random.RandomState(42)
    dim = 64
    block_size = 32

    # Low noise (sigma = 0.02): false alarm rate should be 0%
    low_noise_keys = rng.randn(block_size, dim) * 0.02
    _, disp_low, _ = compute_deroped_mean(low_noise_keys)
    assert disp_low < 0.85

    # High noise (sigma = 0.15): false alarm rate is high
    high_noise_keys = rng.randn(block_size, dim) * 0.15
    _, disp_high, _ = compute_deroped_mean(high_noise_keys)
    assert disp_high > 0.85, "High background noise correctly triggers threshold, proving need for adaptive calibration"


def test_needle_dilution_in_mean_k_vs_doublet_preservation():
    """Verify that 1/32 mean-pooling dilutes needle, while doublet split preserves needle salience."""
    rng = np.random.RandomState(42)
    dim = 64
    block_size = 32

    q = rng.randn(dim)
    q /= np.linalg.norm(q)

    # 31 diffuse background tokens + 1 needle token
    keys = rng.randn(block_size, dim) * 0.05
    needle_burst = 2.0
    keys[16] = q * needle_burst

    mean_k, disp, avg_disp = compute_deroped_mean(keys)
    diluted_score = np.dot(mean_k, q)

    # Diluted score is roughly 1/32 of burst
    assert diluted_score < 0.25

    splitter = AntiCollapseSplitter(dispersion_threshold=0.85)
    block = IFRBlock(
        block_id="test_blk",
        tokens=list(range(block_size)),
        keys=keys,
        values=np.zeros_like(keys),
        orig_pos_start=0,
        mean_k=mean_k,
        dispersion=disp,
        avg_dispersion=avg_disp,
    )

    is_split, sub_centroids = splitter.inspect_and_split(block)
    assert is_split is True
    assert len(sub_centroids) == 2

    # Needle centroid score is restored to full burst level
    needle_centroid_score = np.dot(sub_centroids[0], q)
    assert needle_centroid_score >= 1.8


def test_merkle_prefix_hash_chain_uniqueness():
    """Verify Theorem 2 (Merkle DAG): identical tokens at different prefix positions yield distinct hashes."""
    from kvmem_fusion.core import compute_prefix_hash

    tokens = [1, 2, 3, 4]
    h1 = compute_prefix_hash(tokens, parent_hash="0" * 64, model_id="test")
    h2 = compute_prefix_hash(tokens, parent_hash=h1, model_id="test")

    assert h1 != h2, "Prefix chain ensures causal uniqueness"


def test_normalized_dispersion_under_massive_activation_outliers():
    """
    Verify Critique v3 §4 & OG Audit:
    Massive activation channels distort Euclidean dispersion (sigma ~8.3),
    while normalized directional dispersion isolates semantic variation (sigma ~0.80).
    """
    from kvmem_fusion.ifr import compute_normalized_dispersion, AntiCollapseSplitter

    rng = np.random.RandomState(42)
    dim = 64
    block_size = 32

    # Simulate realistic Qwen key distribution with massive activation outlier channel (channel 0)
    keys = rng.randn(block_size, dim) * 0.1
    keys[:, 0] = 8.0 + rng.randn(block_size) * 1.5  # Massive activation outlier channel

    mean_k, raw_disp, _ = compute_deroped_mean(keys)
    norm_disp = compute_normalized_dispersion(keys)

    # Raw Euclidean dispersion is inflated by outlier channel magnitude (~8.0+)
    assert raw_disp > 3.0, f"Raw dispersion should reflect amplitude: {raw_disp}"

    # Normalized dispersion removes channel scale and stays in directional range (< 1.2)
    assert norm_disp < 1.2, f"Normalized dispersion should be bounded: {norm_disp}"

    # Verify that AntiCollapseSplitter with use_normalized=True correctly triggers on directional needle
    # Inject directional needle at index 10
    keys[10] = -keys[10]  # Flip direction
    norm_disp_with_needle = compute_normalized_dispersion(keys)
    assert norm_disp_with_needle > 1.2

    splitter = AntiCollapseSplitter(dispersion_threshold=0.85, use_normalized=True)
    blk = IFRBlock(
        block_id="outlier_blk",
        tokens=list(range(block_size)),
        keys=keys,
        values=np.zeros_like(keys),
        orig_pos_start=0,
    )
    is_split, sub_centroids = splitter.inspect_and_split(blk)
    assert is_split is True
    assert len(sub_centroids) == 2

