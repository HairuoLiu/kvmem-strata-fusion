"""
C Engine ctypes Bindings for KVMem-Strata-Fusion
Exposes low-level ARM Neon SIMD and native C++ entry points:
1. UBBA minimum-cost demand-covering knapsack solver
2. Vectorized De-RoPE mean & dispersion computation
3. Anti-collapse dispersion check & doublet centroid splitting
4. Decoupled sub-centroid cosine similarity ranking
5. ARM Neon SIMD dot product
6. Zero-copy CASA paged block table translation & token gather
"""

import os
import sys
import ctypes
from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np


# Dynamic library resolution
_LIB_INSTANCE: Optional[ctypes.CDLL] = None
_LIB_PATH: Optional[Path] = None


def _find_library() -> Optional[Path]:
    """Search for libkvmem_engine.dylib in expected locations."""
    custom_path = os.environ.get("KVMEM_DYLIB_PATH")
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "build" / "libkvmem_engine.dylib",
        repo_root / "build" / "libkvmem_engine.so",
        Path(__file__).resolve().parent / "libkvmem_engine.dylib",
        Path(__file__).resolve().parent / "libkvmem_engine.so",
        Path("/usr/local/lib/libkvmem_engine.dylib"),
    ]

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _init_bindings() -> Optional[ctypes.CDLL]:
    global _LIB_INSTANCE, _LIB_PATH
    if _LIB_INSTANCE is not None:
        return _LIB_INSTANCE

    lib_path = _find_library()
    if lib_path is None:
        return None

    try:
        lib = ctypes.CDLL(str(lib_path))
    except Exception:
        return None

    # 1. float kvmem_c_neon_dot(const float* a, const float* b, size_t dim)
    if hasattr(lib, "kvmem_c_neon_dot"):
        lib.kvmem_c_neon_dot.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.kvmem_c_neon_dot.restype = ctypes.c_float

    # 2. int kvmem_c_solve_ubba_ladder(...)
    if hasattr(lib, "kvmem_c_solve_ubba_ladder"):
        lib.kvmem_c_solve_ubba_ladder.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.kvmem_c_solve_ubba_ladder.restype = ctypes.c_int

    # 3. int kvmem_c_compute_deroped_mean(...)
    if hasattr(lib, "kvmem_c_compute_deroped_mean"):
        lib.kvmem_c_compute_deroped_mean.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.kvmem_c_compute_deroped_mean.restype = ctypes.c_int

    # 4. int kvmem_c_inspect_and_split(...)
    if hasattr(lib, "kvmem_c_inspect_and_split"):
        lib.kvmem_c_inspect_and_split.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.kvmem_c_inspect_and_split.restype = ctypes.c_int

    # 5. int kvmem_c_rank_subcentroids_cosine(...)
    if hasattr(lib, "kvmem_c_rank_subcentroids_cosine"):
        lib.kvmem_c_rank_subcentroids_cosine.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.kvmem_c_rank_subcentroids_cosine.restype = ctypes.c_int

    # 6. int kvmem_c_paged_table_translate(...)
    if hasattr(lib, "kvmem_c_paged_table_translate"):
        lib.kvmem_c_paged_table_translate.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.kvmem_c_paged_table_translate.restype = ctypes.c_int

    # 7. int kvmem_c_gather_paged_tokens(...)
    if hasattr(lib, "kvmem_c_gather_paged_tokens"):
        lib.kvmem_c_gather_paged_tokens.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.kvmem_c_gather_paged_tokens.restype = ctypes.c_int

    _LIB_PATH = lib_path
    _LIB_INSTANCE = lib
    return lib


def is_c_engine_available() -> bool:
    """Return True if libkvmem_engine.dylib is loaded and functional."""
    lib = _init_bindings()
    return lib is not None


def get_c_engine_lib() -> Optional[ctypes.CDLL]:
    """Retrieve raw CDLL handle if available."""
    return _init_bindings()


# -----------------------------------------------------------------------------
# High-Performance Zero-Copy Python Wrappers
# -----------------------------------------------------------------------------

def neon_dot_c(a: np.ndarray, b: np.ndarray) -> float:
    """ARM Neon SIMD accelerated dot product of two 1D float32 arrays."""
    lib = _init_bindings()
    if lib is None or not hasattr(lib, "kvmem_c_neon_dot"):
        return float(np.dot(a, b))

    a_c = np.ascontiguousarray(a, dtype=np.float32)
    b_c = np.ascontiguousarray(b, dtype=np.float32)
    dim = a_c.shape[0]
    if b_c.shape[0] != dim:
        raise ValueError(f"Shape mismatch: {a_c.shape} vs {b_c.shape}")

    a_ptr = a_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return float(lib.kvmem_c_neon_dot(a_ptr, b_ptr, ctypes.c_size_t(dim)))


def compute_deroped_mean_c(keys: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Compute mean centroid, max Euclidean dispersion, and normalized directional dispersion.
    
    Returns:
        (mean_k, dir_dispersion, norm_dispersion)
    """
    lib = _init_bindings()
    if lib is None or not hasattr(lib, "kvmem_c_compute_deroped_mean"):
        # Pure Python fallback
        mean_k = np.mean(keys, axis=0)
        dists = np.linalg.norm(keys - mean_k, axis=-1)
        dir_disp = float(np.max(dists)) if len(dists) > 0 else 0.0
        norms = np.linalg.norm(keys, axis=-1, keepdims=True)
        norm_keys = keys / np.maximum(norms, 1e-12)
        norm_mean = np.mean(norm_keys, axis=0)
        norm_dists = np.linalg.norm(norm_keys - norm_mean, axis=-1)
        norm_disp = float(np.max(norm_dists)) if len(norm_dists) > 0 else 0.0
        return mean_k.astype(np.float32), dir_disp, norm_disp

    keys_c = np.ascontiguousarray(keys, dtype=np.float32)
    if keys_c.ndim == 1:
        keys_c = keys_c.reshape(1, -1)
    block_size, head_dim = keys_c.shape

    out_mean = np.empty(head_dim, dtype=np.float32)
    out_dir_disp = ctypes.c_float(0.0)
    out_norm_disp = ctypes.c_float(0.0)

    ret = lib.kvmem_c_compute_deroped_mean(
        keys_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(block_size),
        ctypes.c_size_t(head_dim),
        out_mean.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(out_dir_disp),
        ctypes.byref(out_norm_disp),
    )
    if ret != 0:
        raise RuntimeError(f"kvmem_c_compute_deroped_mean failed with status {ret}")

    return out_mean, float(out_dir_disp.value), float(out_norm_disp.value)


def inspect_and_split_c(
    keys: np.ndarray,
    dispersion_threshold: float = 0.868
) -> Tuple[bool, List[np.ndarray], float]:
    """Anti-collapse dispersion check and doublet splitting.
    
    Returns:
        (is_split, [c1, c2] if is_split else [c1], dispersion)
    """
    lib = _init_bindings()
    if lib is None or not hasattr(lib, "kvmem_c_inspect_and_split"):
        # Pure Python fallback
        mean_k = np.mean(keys, axis=0)
        norms = np.linalg.norm(keys, axis=-1, keepdims=True)
        norm_keys = keys / np.maximum(norms, 1e-12)
        norm_mean = np.mean(norm_keys, axis=0)
        norm_dists = np.linalg.norm(norm_keys - norm_mean, axis=-1)
        metric = float(np.max(norm_dists)) if len(norm_dists) > 0 else 0.0

        if metric <= dispersion_threshold or len(keys) <= 1:
            return False, [mean_k.astype(np.float32)], metric

        dists = np.linalg.norm(keys - mean_k, axis=-1)
        needle_idx = int(np.argmax(dists))
        needle_vec = keys[needle_idx].astype(np.float32)
        mask = np.ones(len(keys), dtype=bool)
        mask[needle_idx] = False
        res_mean = np.mean(keys[mask], axis=0).astype(np.float32)
        return True, [needle_vec, res_mean], metric

    keys_c = np.ascontiguousarray(keys, dtype=np.float32)
    if keys_c.ndim == 1:
        keys_c = keys_c.reshape(1, -1)
    block_size, head_dim = keys_c.shape

    out_should_split = ctypes.c_int(0)
    out_c1 = np.empty(head_dim, dtype=np.float32)
    out_c2 = np.empty(head_dim, dtype=np.float32)
    out_dispersion = ctypes.c_float(0.0)

    ret = lib.kvmem_c_inspect_and_split(
        keys_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(block_size),
        ctypes.c_size_t(head_dim),
        ctypes.c_float(dispersion_threshold),
        ctypes.byref(out_should_split),
        out_c1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_c2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(out_dispersion),
    )
    if ret != 0:
        raise RuntimeError(f"kvmem_c_inspect_and_split failed with status {ret}")

    is_split = bool(out_should_split.value == 1)
    disp = float(out_dispersion.value)
    centroids = [out_c1, out_c2] if is_split else [out_c1]
    return is_split, centroids, disp


def rank_subcentroids_cosine_c(
    query: np.ndarray,
    centroids: np.ndarray,
    top_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Decoupled sub-centroid cosine similarity ranking with top-K selection.
    
    Returns:
        (ranked_indices, ranked_scores)
    """
    lib = _init_bindings()
    if lib is None or not hasattr(lib, "kvmem_c_rank_subcentroids_cosine"):
        # Pure Python fallback
        q_norm = query / np.maximum(np.linalg.norm(query), 1e-12)
        c_norms = centroids / np.maximum(np.linalg.norm(centroids, axis=-1, keepdims=True), 1e-12)
        scores = np.dot(c_norms, q_norm)
        k = min(top_k, len(scores))
        if k >= len(scores):
            indices = np.argsort(-scores)
        else:
            cand_idx = np.argpartition(-scores, k)[:k]
            indices = cand_idx[np.argsort(-scores[cand_idx])]
        return indices[:k], scores[indices[:k]]

    q_c = np.ascontiguousarray(query, dtype=np.float32)
    cent_c = np.ascontiguousarray(centroids, dtype=np.float32)
    num_centroids, dim = cent_c.shape
    if q_c.shape[0] != dim:
        raise ValueError(f"Dimension mismatch: query {q_c.shape} vs centroids {cent_c.shape}")

    k = min(top_k, num_centroids)
    out_indices = np.empty(k, dtype=np.uint64)
    out_scores = np.empty(k, dtype=np.float32)

    ret = lib.kvmem_c_rank_subcentroids_cosine(
        q_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cent_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(num_centroids),
        ctypes.c_size_t(dim),
        ctypes.c_size_t(top_k),
        out_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if ret != 0:
        raise RuntimeError(f"kvmem_c_rank_subcentroids_cosine failed with status {ret}")

    return out_indices, out_scores


def solve_ubba_ladder_c(
    block_ids: np.ndarray,
    fidelities: np.ndarray,
    bytes_arr: np.ndarray,
    target_cov: float,
    rho_floor: float = 0.10
) -> np.ndarray:
    """Solve UBBA minimum-cost demand-covering knapsack using native C engine.
    
    Returns:
        selected_block_ids: np.ndarray of uint32
    """
    lib = _init_bindings()
    if lib is None or not hasattr(lib, "kvmem_c_solve_ubba_ladder"):
        # Pure Python fallback
        eff = fidelities / np.maximum(bytes_arr, 1.0)
        order = np.argsort(-eff)
        cum_w = 0.0
        selected = []
        for idx in order:
            selected.append(block_ids[idx])
            cum_w += float(fidelities[idx])
            if cum_w >= target_cov:
                break
        return np.array(selected, dtype=np.uint32)

    b_ids_c = np.ascontiguousarray(block_ids, dtype=np.uint32)
    fid_c = np.ascontiguousarray(fidelities, dtype=np.float32)
    bytes_c = np.ascontiguousarray(bytes_arr, dtype=np.float32)
    n = len(b_ids_c)

    out_selected_ids = np.empty(n, dtype=np.uint32)
    out_selected_count = ctypes.c_uint32(0)

    ret = lib.kvmem_c_solve_ubba_ladder(
        b_ids_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        fid_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bytes_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(n),
        ctypes.c_double(target_cov),
        ctypes.c_double(rho_floor),
        out_selected_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(out_selected_count),
    )
    if ret != 0:
        raise RuntimeError(f"kvmem_c_solve_ubba_ladder failed with status {ret}")

    count = out_selected_count.value
    return out_selected_ids[:count].copy()


def paged_table_translate_c(
    page_table: np.ndarray,
    logical_token_idx: int,
    block_size: int = 32
) -> Tuple[int, int]:
    """Translate logical token index to (physical_page, offset_in_page) in zero-copy CASA store."""
    lib = _init_bindings()
    if lib is None or not hasattr(lib, "kvmem_c_paged_table_translate"):
        logical_page = logical_token_idx // block_size
        offset_in_page = logical_token_idx % block_size
        if logical_page >= len(page_table):
            raise IndexError("Logical token index out of bounds in page table")
        return int(page_table[logical_page]), int(offset_in_page)

    pt_c = np.ascontiguousarray(page_table, dtype=np.int32)
    out_phys_page = ctypes.c_int32(0)
    out_offset = ctypes.c_size_t(0)

    ret = lib.kvmem_c_paged_table_translate(
        pt_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_size_t(len(pt_c)),
        ctypes.c_size_t(logical_token_idx),
        ctypes.c_size_t(block_size),
        ctypes.byref(out_phys_page),
        ctypes.byref(out_offset),
    )
    if ret != 0:
        raise IndexError(f"Logical token index {logical_token_idx} out of bounds")

    return int(out_phys_page.value), int(out_offset.value)


def gather_paged_tokens_c(
    physical_pool: np.ndarray,
    page_table: np.ndarray,
    token_indices: np.ndarray,
    block_size: int = 32,
    head_dim: int = 64
) -> np.ndarray:
    """Gather token vectors directly from contiguous physical pool using zero-copy page table."""
    lib = _init_bindings()
    tokens_c = np.ascontiguousarray(token_indices, dtype=np.uint64)
    num_tokens = len(tokens_c)
    out = np.empty((num_tokens, head_dim), dtype=np.float32)

    if lib is None or not hasattr(lib, "kvmem_c_gather_paged_tokens"):
        page_elements = block_size * head_dim
        pool_flat = physical_pool.reshape(-1)
        for i, tok_idx in enumerate(tokens_c):
            lp = tok_idx // block_size
            offset = tok_idx % block_size
            pp = page_table[lp]
            start = pp * page_elements + offset * head_dim
            out[i] = pool_flat[start:start + head_dim]
        return out

    pool_c = np.ascontiguousarray(physical_pool, dtype=np.float32)
    pt_c = np.ascontiguousarray(page_table, dtype=np.int32)

    ret = lib.kvmem_c_gather_paged_tokens(
        pool_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pt_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_size_t(len(pt_c)),
        tokens_c.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(num_tokens),
        ctypes.c_size_t(block_size),
        ctypes.c_size_t(head_dim),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if ret != 0:
        raise RuntimeError("kvmem_c_gather_paged_tokens failed")

    return out
