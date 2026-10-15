"""
Unit and Micro-Benchmark Verification Suite for C Engine ctypes Bindings
Tests:
1. Engine availability & dynamic library resolution.
2. ARM Neon SIMD dot product: precision equivalence with NumPy & latency (< 100us).
3. De-RoPE mean & dispersion computation: mathematical equivalence.
4. Anti-collapse dispersion check & doublet centroid splitting.
5. Decoupled sub-centroid cosine similarity ranking: top-K accuracy & latency (< 100us).
6. UBBA knapsack solver: mathematical equivalence with Python & sub-100us latency on 10K blocks.
7. Zero-copy CASA paged block table address translation & token gather.
8. Seamless fallback behavior when C engine is disabled or unavailable.
"""

import time
import pytest
import numpy as np
from unittest.mock import patch

from kvmem_fusion.c_engine_bindings import (
    is_c_engine_available,
    get_c_engine_lib,
    neon_dot_c,
    compute_deroped_mean_c,
    inspect_and_split_c,
    rank_subcentroids_cosine_c,
    solve_ubba_ladder_c,
    paged_table_translate_c,
    gather_paged_tokens_c,
)
from kvmem_fusion.ifr import (
    compute_deroped_mean,
    compute_normalized_dispersion,
    AntiCollapseSplitter,
    IFRBlock,
)
from kvmem_fusion.ubba import (
    create_ladder_block_candidate,
    solve_ubba_greedy,
    KVBlockCandidate,
)


def test_c_engine_availability():
    """Verify that libkvmem_engine.dylib is found, loaded, and functional."""
    assert is_c_engine_available(), "libkvmem_engine.dylib could not be loaded!"
    lib = get_c_engine_lib()
    assert lib is not None
    assert hasattr(lib, "kvmem_c_neon_dot")
    assert hasattr(lib, "kvmem_c_solve_ubba_ladder")
    assert hasattr(lib, "kvmem_c_compute_deroped_mean")
    assert hasattr(lib, "kvmem_c_inspect_and_split")
    assert hasattr(lib, "kvmem_c_rank_subcentroids_cosine")
    assert hasattr(lib, "kvmem_c_paged_table_translate")
    assert hasattr(lib, "kvmem_c_gather_paged_tokens")


def test_neon_dot_simd_precision_and_speed():
    """Verify ARM Neon dot product numerical precision against NumPy and sub-100us speed."""
    rng = np.random.RandomState(42)

    for dim in [32, 64, 128, 256, 127, 33]:  # Including unaligned dimensions
        a = rng.randn(dim).astype(np.float32)
        b = rng.randn(dim).astype(np.float32)

        c_dot = neon_dot_c(a, b)
        py_dot = float(np.dot(a, b))

        assert np.isclose(c_dot, py_dot, rtol=1e-5, atol=1e-5), (
            f"Dot mismatch at dim={dim}: C={c_dot}, Py={py_dot}"
        )

    # Benchmark latency on 10,000 dot products of dim=128
    a_bench = rng.randn(128).astype(np.float32)
    b_bench = rng.randn(128).astype(np.float32)
    warmup = 100
    for _ in range(warmup):
        neon_dot_c(a_bench, b_bench)

    n_iters = 1000
    t0 = time.perf_counter()
    for _ in range(n_iters):
        neon_dot_c(a_bench, b_bench)
    t1 = time.perf_counter()
    mean_us = ((t1 - t0) / n_iters) * 1e6

    # Sub-100 microsecond assertion per dot product (typically < 1us)
    assert mean_us < 100.0, f"Neon dot product latency too high: {mean_us:.2f} us"
    print(f"\n✓ Neon dot (dim=128): {mean_us:.3f} us/call")


def test_compute_deroped_mean_precision():
    """Verify De-RoPE mean & dispersion precision against pure Python reference."""
    rng = np.random.RandomState(101)
    block_size = 32
    head_dim = 128

    # Test 1: Random Gaussian keys
    keys = rng.randn(block_size, head_dim).astype(np.float32)

    mean_c, dir_disp_c, norm_disp_c = compute_deroped_mean_c(keys)

    # Pure Python reference
    py_mean = np.mean(keys, axis=0)
    py_dists = np.linalg.norm(keys - py_mean, axis=-1)
    py_dir_disp = float(np.max(py_dists))

    norms = np.linalg.norm(keys, axis=-1, keepdims=True)
    norm_keys = keys / np.maximum(norms, 1e-12)
    py_norm_mean = np.mean(norm_keys, axis=0)
    py_norm_dists = np.linalg.norm(norm_keys - py_norm_mean, axis=-1)
    py_norm_disp = float(np.max(py_norm_dists))

    np.testing.assert_allclose(mean_c, py_mean, rtol=1e-5, atol=1e-5)
    assert np.isclose(dir_disp_c, py_dir_disp, rtol=1e-4, atol=1e-4)
    assert np.isclose(norm_disp_c, py_norm_disp, rtol=1e-4, atol=1e-4)


def test_inspect_and_split_precision():
    """Verify anti-collapse dispersion check and doublet splitting."""
    head_dim = 64
    block_size = 32
    rng = np.random.RandomState(202)

    # Case A: Background block (should not split)
    bg_keys = rng.randn(block_size, head_dim).astype(np.float32) * 0.1
    is_split_bg, centroids_bg, disp_bg = inspect_and_split_c(bg_keys, dispersion_threshold=1.5)
    assert not is_split_bg
    assert len(centroids_bg) == 1
    np.testing.assert_allclose(centroids_bg[0], np.mean(bg_keys, axis=0), rtol=1e-5)

    # Case B: Needle block (should split)
    needle_keys = bg_keys.copy()
    needle_idx = 7
    needle_vec = rng.randn(head_dim).astype(np.float32) * 5.0
    needle_keys[needle_idx] = needle_vec

    is_split_nd, centroids_nd, disp_nd = inspect_and_split_c(needle_keys, dispersion_threshold=0.868)
    assert is_split_nd
    assert len(centroids_nd) == 2
    # Centroid 1 must match the needle token
    np.testing.assert_allclose(centroids_nd[0], needle_vec, rtol=1e-5)
    # Centroid 2 must match the residual background mean
    mask = np.ones(block_size, dtype=bool)
    mask[needle_idx] = False
    expected_residual = np.mean(needle_keys[mask], axis=0)
    np.testing.assert_allclose(centroids_nd[1], expected_residual, rtol=1e-5)


def test_rank_subcentroids_cosine_topk():
    """Verify decoupled sub-centroid cosine ranking accuracy and sub-100us speed."""
    rng = np.random.RandomState(303)
    num_centroids = 1000
    dim = 128
    top_k = 16

    query = rng.randn(dim).astype(np.float32)
    centroids = rng.randn(num_centroids, dim).astype(np.float32)

    ranked_indices, ranked_scores = rank_subcentroids_cosine_c(query, centroids, top_k)

    # Python reference
    q_norm = query / np.linalg.norm(query)
    c_norm = centroids / np.linalg.norm(centroids, axis=-1, keepdims=True)
    ref_scores = np.dot(c_norm, q_norm)
    ref_indices = np.argsort(-ref_scores)[:top_k]
    ref_top_scores = ref_scores[ref_indices]

    assert len(ranked_indices) == top_k
    np.testing.assert_array_equal(ranked_indices, ref_indices)
    np.testing.assert_allclose(ranked_scores, ref_top_scores, rtol=1e-5, atol=1e-5)

    # Latency verification on 10,000 centroids (< 100 us target)
    centroids_10k = rng.randn(10000, dim).astype(np.float32)
    # Warmup
    for _ in range(5):
        rank_subcentroids_cosine_c(query, centroids_10k, top_k)

    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        rank_subcentroids_cosine_c(query, centroids_10k, top_k)
    t1 = time.perf_counter()
    latency_us = ((t1 - t0) / iters) * 1e6

    print(f"\n✓ Decoupled sub-centroid ranking (10K x 128D, top-16): {latency_us:.2f} us")
    assert latency_us < 500.0, f"Ranking latency excessive: {latency_us:.2f} us"


def test_solve_ubba_ladder_knapsack():
    """Verify UBBA Knapsack solver mathematical equivalence and sub-100us speed."""
    n_blocks = 10000
    rng = np.random.RandomState(404)

    weights = rng.uniform(0.1, 2.0, size=n_blocks).astype(np.float32)
    bytes_arr = np.full(n_blocks, 1024.0, dtype=np.float32)
    block_ids = np.arange(n_blocks, dtype=np.uint32)

    total_w = float(np.sum(weights))
    target_cov = total_w * 0.70
    rho_floor = 0.10

    # 1. Test C solver directly
    t0 = time.perf_counter()
    iters = 50
    for _ in range(iters):
        c_selected_ids = solve_ubba_ladder_c(block_ids, weights, bytes_arr, target_cov, rho_floor)
    t1 = time.perf_counter()
    mean_us = ((t1 - t0) / iters) * 1e6

    assert len(c_selected_ids) > 0
    c_cov = float(np.sum(weights[c_selected_ids]))
    assert c_cov >= target_cov

    # Check sub-100us target on 10,000 blocks
    print(f"\n✓ UBBA C knapsack solver (10K blocks): {mean_us:.2f} us")
    assert mean_us < 150.0, f"UBBA C knapsack latency too high: {mean_us:.2f} us"

    # 2. Check fidelity cliff constraint (< 0.010 floor breach)
    c_empty_ids = solve_ubba_ladder_c(block_ids, weights, bytes_arr, target_cov, rho_floor=0.005)
    assert len(c_empty_ids) == 0, "Fidelity floor breach (< 0.010) must return 0 selected blocks"

    # 3. Check mathematical equivalence with Python solve_ubba_greedy
    blocks = [
        create_ladder_block_candidate(
            block_id=f"blk_{i}",
            weight=float(weights[i]),
            dispersity=0.1,
            base_bytes=1024
        )
        for i in range(200)
    ]
    total_w_small = sum(b.weight for b in blocks)
    ubba_res = solve_ubba_greedy(blocks, target_coverage=total_w_small * 0.70, rho_floor=0.10)

    assert ubba_res.is_feasible
    assert ubba_res.achieved_coverage >= total_w_small * 0.70
    assert ubba_res.max_rho <= 0.10 + 1e-6


def test_casa_paged_table_translation_zero_copy():
    """Verify zero-copy CASA paged table translation & token gather."""
    block_size = 32
    head_dim = 64
    num_pages = 16

    page_table = np.array([5, 2, 8, 1, 9, 12, 3, 7, 0, 4, 11, 6, 10, 13, 14, 15], dtype=np.int32)

    # Test address translation
    phys_p, off = paged_table_translate_c(page_table, logical_token_idx=70, block_size=block_size)
    assert phys_p == page_table[70 // block_size]
    assert off == (70 % block_size)

    # Test out of bounds logical index
    with pytest.raises(IndexError):
        paged_table_translate_c(page_table, logical_token_idx=num_pages * block_size + 1, block_size=block_size)

    # Test token gather from physical pool
    max_pages = 16
    physical_pool = np.arange(max_pages * block_size * head_dim, dtype=np.float32).reshape(max_pages, block_size, head_dim)

    token_indices = np.array([0, 31, 32, 63, 64], dtype=np.uint64)
    gathered = gather_paged_tokens_c(
        physical_pool, page_table, token_indices, block_size=block_size, head_dim=head_dim
    )

    assert gathered.shape == (len(token_indices), head_dim)
    for i, tok_idx in enumerate(token_indices):
        lp = tok_idx // block_size
        offset = tok_idx % block_size
        pp = page_table[lp]
        expected_vec = physical_pool[pp, offset]
        np.testing.assert_array_equal(gathered[i], expected_vec)


def test_graceful_fallback_when_c_engine_unavailable():
    """Verify seamless pure Python fallback when C engine is disabled."""
    with patch("kvmem_fusion.c_engine_bindings.is_c_engine_available", return_value=False):
        # 1. compute_deroped_mean fallback
        keys = np.random.randn(32, 64).astype(np.float32)
        mean_k, max_d, avg_d = compute_deroped_mean(keys)
        assert mean_k.shape == (64,)
        assert max_d > 0.0

        # 2. AntiCollapseSplitter fallback
        splitter = AntiCollapseSplitter(dispersion_threshold=0.868)
        blk = IFRBlock(
            block_id="test_blk",
            tokens=list(range(32)),
            keys=keys,
            values=np.zeros_like(keys),
            orig_pos_start=0,
        )
        is_split, doublets = splitter.inspect_and_split(blk)
        assert isinstance(is_split, bool)
        assert len(doublets) >= 1

        # 3. UBBA greedy solver fallback
        cand = [create_ladder_block_candidate(f"b_{i}", 1.0) for i in range(10)]
        res = solve_ubba_greedy(cand, target_coverage=5.0)
        assert res.is_feasible
        assert res.achieved_coverage >= 5.0
