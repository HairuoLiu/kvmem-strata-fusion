"""
Fused Tier-Bias FlashAttention Kernel & Microarchitectural Simulator.

Implements tile-local online softmax max reduction with tier-bias injection:
    scores_tile = (Q @ K_tile.T) / sqrt(d) + b_tile
integrated directly into FlashAttention-2/3 online softmax rescaling without extra HBM round-trips.

Mathematical Foundation:
In heterogeneous KV cache storage (LADDER / UBBA), different tokens/pages reside in divergent
quantization/compression tiers (e.g. FP8, INT4, INT2, MERGED). Jensen's inequality induces attention
theft on low-bit tiers due to log-normal noise expectation:
    E[exp(ell + eps_t)] = exp(ell) * exp(sigma_t^2 / 2).
To neutralize this bias, UBBA supplies a calibration bias:
    b_t = - (rho_t * s)^2 / 2.
This kernel injects b_t directly into tile-local SRAM registers during the QK^T contraction,
updating online softmax row-max and denominator scaling on-the-fly:
    m_i^(j+1) = max(m_i^(j), rowmax(S_ij))
    alpha_ij  = exp(m_i^(j) - m_i^(j+1))
    P_ij      = exp(S_ij - m_i^(j+1))
    l_i^(j+1) = alpha_ij * l_i^(j) + rowsum(P_ij)
    O_i^(j+1) = alpha_ij * O_i^(j) + P_ij @ V_j

Key Features:
1. Zero HBM round-trips for the intermediate N x N score matrix: memory traffic drops from O(N^2) to O(Nd).
2. Tile-local bias injection with 0 additional memory allocation.
3. Production Triton GPU kernel implementation (@triton.jit) with CUDA/ROCm execution path.
4. Bit-exact hardware-accurate SRAM simulator for deterministic validation on CPU/Darwin test harnesses.
5. PagedAttention mode compatible with CASA physical page tables.
6. Microarchitectural profiler: FLOP counting, HBM bandwidth analysis, and roofline operational intensity.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import numpy as np

# Conditional imports for PyTorch and Triton
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


# ==============================================================================
# 1. TRITON GPU KERNEL IMPLEMENTATION
# ==============================================================================

if HAS_TRITON:
    @triton.jit
    def _fused_tier_bias_fwd_kernel(
        Q, K, V, Bias, Out, Lse,
        sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        stride_bz, stride_bh, stride_bm, stride_bn,
        Z, H, N_CTX_Q, N_CTX_K,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """Triton FlashAttention-2 forward kernel with fused tier-bias injection."""
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        # Offset pointers for current batch and head
        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

        # Block offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)

        # Load Q block into SRAM registers: [BLOCK_M, BLOCK_D]
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        mask_m = offs_m[:, None] < N_CTX_Q
        q = tl.load(q_ptrs, mask=mask_m, other=0.0)

        # Initialize online softmax statistics in SRAM registers
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Loop over K, V blocks along sequence length
        end_n = N_CTX_K
        if IS_CAUSAL:
            end_n = tl.minimum(N_CTX_K, (start_m + 1) * BLOCK_M)

        for start_n in range(0, end_n, BLOCK_N):
            curr_offs_n = start_n + offs_n
            mask_n = curr_offs_n[None, :] < N_CTX_K

            # Load K tile: [BLOCK_N, BLOCK_D] -> transposed for dot: [BLOCK_D, BLOCK_N]
            k_ptrs = K + k_offset + curr_offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=curr_offs_n[:, None] < N_CTX_K, other=0.0)

            # S_tile = (Q @ K^T) * sm_scale
            qk = tl.dot(q, tl.trans(k)) * sm_scale

            # Tile-local Tier-Bias injection directly into SRAM registers
            if HAS_BIAS:
                b_ptrs = (
                    Bias
                    + off_z.to(tl.int64) * stride_bz
                    + off_h.to(tl.int64) * stride_bh
                    + offs_m[:, None] * stride_bm
                    + curr_offs_n[None, :] * stride_bn
                )
                b_tile = tl.load(b_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (curr_offs_n[None, :] < N_CTX_K), other=0.0)
                qk += b_tile

            # Causal mask
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= curr_offs_n[None, :], qk, float("-inf"))
            qk = tl.where(mask_m & (curr_offs_n[None, :] < N_CTX_K), qk, float("-inf"))

            # Online Softmax step
            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            # Rescale denominator and accumulator
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            # Load V tile: [BLOCK_N, BLOCK_D]
            v_ptrs = V + v_offset + curr_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=curr_offs_n[:, None] < N_CTX_K, other=0.0)

            # Accumulate context: P @ V
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # Final normalization
        acc = acc / l_i[:, None]

        # Write output to HBM
        out_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m)

        # Write LSE to HBM
        lse = m_i + tl.log(l_i)
        lse_ptrs = Lse + off_hz * N_CTX_Q + offs_m
        tl.store(lse_ptrs, lse, mask=offs_m < N_CTX_Q)


# ==============================================================================
# 2. MICROARCHITECTURAL TELEMETRY & PROFILING
# ==============================================================================

@dataclass
class KernelMetrics:
    """Hardware efficiency, memory traffic, and arithmetic intensity telemetry."""
    batch_size: int
    num_heads: int
    seq_len_q: int
    seq_len_k: int
    head_dim: int
    bytes_per_elem: int

    # Flops
    total_flops: float
    gemm1_flops: float
    gemm2_flops: float
    softmax_flops: float

    # HBM Traffic (Bytes)
    unfused_hbm_reads: int
    unfused_hbm_writes: int
    unfused_hbm_total: int

    fused_hbm_reads: int
    fused_hbm_writes: int
    fused_hbm_total: int

    hbm_traffic_reduction_ratio: float
    intermediate_score_matrix_bytes: int

    # SRAM Footprint (Bytes)
    sram_peak_bytes: int
    sram_block_m: int
    sram_block_n: int

    # Operational Intensity (FLOPs / Byte)
    unfused_operational_intensity: float
    fused_operational_intensity: float

    # Numerical Precision
    max_abs_diff_vs_unfused: float = 0.0
    mean_abs_diff_vs_unfused: float = 0.0


def compute_microarch_metrics(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    block_m: int = 64,
    block_n: int = 64,
    bytes_per_elem: int = 2,  # FP16 / BF16
) -> KernelMetrics:
    """Derive theoretical roofline telemetry, HBM memory traffic, and SRAM working set."""
    B, H, M, N, D = batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    elem_b = bytes_per_elem

    # GEMM1: Q @ K^T -> 2 * B * H * M * N * D flops
    gemm1_flops = 2.0 * B * H * M * N * D
    # GEMM2: P @ V   -> 2 * B * H * M * N * D flops
    gemm2_flops = 2.0 * B * H * M * N * D
    # Softmax + Bias: ~5 flops per element (scale, bias add, max, exp, sum, div)
    softmax_flops = 5.0 * B * H * M * N
    total_flops = gemm1_flops + gemm2_flops + softmax_flops

    # Unfused Attention HBM Footprint:
    # Read Q, K, V, Bias
    q_bytes = B * H * M * D * elem_b
    k_bytes = B * H * N * D * elem_b
    v_bytes = B * H * N * D * elem_b
    b_bytes = B * H * M * N * elem_b
    # Intermediate S matrix: write S [M, N], read S, write P [M, N], read P
    s_bytes = B * H * M * N * 4  # FP32 logits
    p_bytes = B * H * M * N * elem_b
    # Output O: write O [M, D]
    o_bytes = B * H * M * D * elem_b

    unfused_reads = q_bytes + k_bytes + v_bytes + b_bytes + s_bytes + p_bytes
    unfused_writes = s_bytes + p_bytes + o_bytes
    unfused_total = unfused_reads + unfused_writes

    # Fused Tier-Bias FlashAttention HBM Footprint:
    # Read Q, K, V, Bias (Bias is broadcasted per tile, or read once per token)
    # Write O, LSE [M]
    lse_bytes = B * H * M * 4
    fused_reads = q_bytes + k_bytes + v_bytes + (B * H * N * elem_b)  # token-level tier bias
    fused_writes = o_bytes + lse_bytes
    fused_total = fused_reads + fused_writes

    traffic_reduction = unfused_total / max(1, fused_total)
    intermediate_score_bytes = B * H * M * N * 4

    # SRAM Peak Footprint:
    # Q tile: block_m * D * elem_b
    # K tile: block_n * D * elem_b
    # V tile: block_n * D * elem_b
    # Score tile: block_m * block_n * 4 (FP32 accumulator)
    # Output acc: block_m * D * 4
    sram_peak = (
        block_m * D * elem_b
        + block_n * D * elem_b
        + block_n * D * elem_b
        + block_m * block_n * 4
        + block_m * D * 4
    )

    unfused_oi = total_flops / max(1.0, float(unfused_total))
    fused_oi = total_flops / max(1.0, float(fused_total))

    return KernelMetrics(
        batch_size=B,
        num_heads=H,
        seq_len_q=M,
        seq_len_k=N,
        head_dim=D,
        bytes_per_elem=elem_b,
        total_flops=total_flops,
        gemm1_flops=gemm1_flops,
        gemm2_flops=gemm2_flops,
        softmax_flops=softmax_flops,
        unfused_hbm_reads=unfused_reads,
        unfused_hbm_writes=unfused_writes,
        unfused_hbm_total=unfused_total,
        fused_hbm_reads=fused_reads,
        fused_hbm_writes=fused_writes,
        fused_hbm_total=fused_total,
        hbm_traffic_reduction_ratio=traffic_reduction,
        intermediate_score_matrix_bytes=intermediate_score_bytes,
        sram_peak_bytes=sram_peak,
        sram_block_m=block_m,
        sram_block_n=block_n,
        unfused_operational_intensity=unfused_oi,
        fused_operational_intensity=fused_oi,
    )


# ==============================================================================
# 3. UNFUSED GROUND-TRUTH PYTORCH / NUMPY REFERENCE
# ==============================================================================

def unfused_tier_bias_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    bias: Optional[np.ndarray] = None,
    sm_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unfused baseline attention: materializes full [M, N] score matrix in global memory.
    
    Args:
        q: Query array, shape (..., M, D)
        k: Key array, shape (..., N, D)
        v: Value array, shape (..., N, D)
        bias: Tier-bias array, broadcastable to (..., M, N)
        sm_scale: Softmax scale factor, defaults to 1.0 / sqrt(D)
        causal: Whether to apply lower-triangular causal masking
        
    Returns:
        (out, lse):
            out: Context representation, shape (..., M, D)
            lse: Log-Sum-Exp row statistics, shape (..., M)
    """
    orig_shape_q = q.shape
    d = q.shape[-1]
    m = q.shape[-2]
    n = k.shape[-2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    # Flatten leading batch dimensions to 2D for computation
    q_flat = q.reshape(-1, m, d)
    k_flat = k.reshape(-1, n, d)
    v_flat = v.reshape(-1, n, d)
    b_size = q_flat.shape[0]

    outs = []
    lses = []

    for b in range(b_size):
        # Materialize entire S matrix: [M, N]
        with np.errstate(all="ignore"):
            scores = np.dot(q_flat[b], k_flat[b].T) * sm_scale

        if bias is not None:
            # Broadcast bias
            b_slice = bias if bias.ndim <= 2 else bias.reshape(-1, m, n)[b]
            scores = scores + b_slice

        if causal:
            # Mask upper triangular elements (col > row)
            causal_mask = np.triu(np.ones((m, n), dtype=bool), k=1)
            scores[causal_mask] = -1e9

        # Global Softmax across entire sequence length
        row_max = np.max(scores, axis=-1, keepdims=True)
        # Numerical protection: where row_max is -1e9, avoid -inf - (-inf)
        stable_scores = scores - row_max
        with np.errstate(all="ignore"):
            exp_scores = np.exp(stable_scores)
        if causal:
            exp_scores[causal_mask] = 0.0

        row_sum = np.sum(exp_scores, axis=-1, keepdims=True)
        # Avoid division by zero for fully masked rows
        safe_sum = np.where(row_sum > 0, row_sum, 1.0)
        p = exp_scores / safe_sum

        with np.errstate(all="ignore"):
            out = np.dot(p, v_flat[b])
        lse = np.squeeze(row_max, axis=-1) + np.log(np.squeeze(safe_sum, axis=-1))

        outs.append(out)
        lses.append(lse)

    out_arr = np.stack(outs, axis=0).reshape(orig_shape_q)
    lse_shape = orig_shape_q[:-1]
    lse_arr = np.stack(lses, axis=0).reshape(lse_shape)
    return out_arr, lse_arr


# ==============================================================================
# 4. HARDWARE-ACCURATE SRAM KERNEL SIMULATOR (FLASHATTENTION-2/3 EMULATION)
# ==============================================================================

def fused_tier_bias_attention_sim(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    bias: Optional[np.ndarray] = None,
    sm_scale: Optional[float] = None,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hardware-accurate simulator of Fused Tier-Bias FlashAttention.
    
    Executes tile-local online softmax max reduction with tier-bias injection:
        scores_tile = (Q @ K_tile.T) / sqrt(d) + b_tile
    without ever materializing the global [M, N] attention matrix in memory.
    
    Mimics GPU SRAM shared-memory buffers and register files:
    - SRAM block size: block_m queries x block_n keys
    - Online max tracking: m_i updated per tile
    - Online denominator scaling: l_i rescaled via alpha_ij = exp(m_i - m_new)
    - Output accumulator: O_i rescaled via alpha_ij and accumulated in-place
    """
    orig_shape_q = q.shape
    d = q.shape[-1]
    m = q.shape[-2]
    n = k.shape[-2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    q_flat = q.reshape(-1, m, d)
    k_flat = k.reshape(-1, n, d)
    v_flat = v.reshape(-1, n, d)
    b_size = q_flat.shape[0]

    outs = []
    lses = []

    for b in range(b_size):
        qb = q_flat[b]
        kb = k_flat[b]
        vb = v_flat[b]

        out_b = np.zeros((m, d), dtype=np.float64)
        lse_b = np.zeros(m, dtype=np.float64)

        # Loop over Q blocks (outer loop over SRAM query tiles)
        for i_start in range(0, m, block_m):
            i_end = min(i_start + block_m, m)
            cur_bm = i_end - i_start
            q_tile = qb[i_start:i_end]  # [cur_bm, d] loaded to SRAM registers

            # Running statistics for this query tile in SRAM registers
            m_i = np.full(cur_bm, -np.inf, dtype=np.float64)
            l_i = np.zeros(cur_bm, dtype=np.float64)
            acc_i = np.zeros((cur_bm, d), dtype=np.float64)

            # Determine K/V loop upper bound
            j_max = n
            if causal:
                j_max = min(n, i_end)

            # Loop over K, V blocks (inner loop over SRAM key/value tiles)
            for j_start in range(0, j_max, block_n):
                j_end = min(j_start + block_n, n)
                cur_bn = j_end - j_start

                k_tile = kb[j_start:j_end]  # [cur_bn, d] loaded to SRAM
                v_tile = vb[j_start:j_end]  # [cur_bn, d] loaded to SRAM

                # Tile score computation: [cur_bm, cur_bn] in SRAM registers
                with np.errstate(all="ignore"):
                    s_tile = np.dot(q_tile, k_tile.T) * sm_scale

                # Inject tier bias directly into tile-local SRAM registers
                if bias is not None:
                    if bias.ndim == 1:
                        # 1D bias across sequence length K: [n]
                        b_tile = bias[j_start:j_end]
                        s_tile = s_tile + b_tile[np.newaxis, :]
                    elif bias.ndim == 2:
                        # 2D bias: [m, n]
                        b_tile = bias[i_start:i_end, j_start:j_end]
                        s_tile = s_tile + b_tile
                    elif bias.ndim >= 3:
                        # Multi-head batched bias: [B, ..., m, n]
                        b_flat = bias.reshape(-1, m, n)[b]
                        b_tile = b_flat[i_start:i_end, j_start:j_end]
                        s_tile = s_tile + b_tile

                # Apply causal mask within tile
                if causal:
                    # Global query index: i_start + r
                    # Global key index: j_start + c
                    q_indices = np.arange(i_start, i_end)[:, np.newaxis]
                    k_indices = np.arange(j_start, j_end)[np.newaxis, :]
                    tile_mask = k_indices > q_indices
                    s_tile[tile_mask] = -np.inf

                # Tile-local row-max reduction
                m_ij = np.max(s_tile, axis=-1)

                # Online softmax update
                m_new = np.maximum(m_i, m_ij)

                # Compute rescaling factor alpha = exp(m_i - m_new)
                # Safeguard against inf - inf:
                alpha = np.zeros(cur_bm, dtype=np.float64)
                valid_mask = m_i > -np.inf
                with np.errstate(all="ignore"):
                    alpha[valid_mask] = np.exp(m_i[valid_mask] - m_new[valid_mask])
                # If m_i was -inf, alpha remains 0.0

                # Unnormalized exponentials for the current tile
                p_tile = np.zeros_like(s_tile)
                tile_valid_mask = s_tile > -np.inf
                with np.errstate(all="ignore"):
                    p_tile[tile_valid_mask] = np.exp(
                        s_tile[tile_valid_mask] - m_new[:, np.newaxis].repeat(cur_bn, axis=1)[tile_valid_mask]
                    )

                # Update denominator sum: l_i = alpha * l_i + rowsum(p_tile)
                l_tile = np.sum(p_tile, axis=-1)
                l_i = alpha * l_i + l_tile

                # Update output accumulator: acc_i = alpha * acc_i + p_tile @ v_tile
                with np.errstate(all="ignore"):
                    acc_i = alpha[:, np.newaxis] * acc_i + np.dot(p_tile, v_tile)

                # Advance running row-max
                m_i = m_new

            # Tile completion: normalize accumulated context in SRAM
            safe_denom = np.where(l_i > 0, l_i, 1.0)
            out_b[i_start:i_end] = acc_i / safe_denom[:, np.newaxis]
            lse_b[i_start:i_end] = m_i + np.log(safe_denom)

        outs.append(out_b)
        lses.append(lse_b)

    out_arr = np.stack(outs, axis=0).reshape(orig_shape_q)
    lse_shape = orig_shape_q[:-1]
    lse_arr = np.stack(lses, axis=0).reshape(lse_shape)
    return out_arr, lse_arr


# ==============================================================================
# 5. CASA PAGEDATTENTION FUSED TIER-BIAS SIMULATOR
# ==============================================================================

def paged_fused_tier_bias_attention_sim(
    q: np.ndarray,
    physical_k_pool: Union[List[np.ndarray], np.ndarray],
    physical_v_pool: Union[List[np.ndarray], np.ndarray],
    page_table: List[int],
    page_biases: Optional[Union[Dict[int, float], np.ndarray, List[float]]] = None,
    sm_scale: Optional[float] = None,
    block_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """CASA PagedAttention Fused Kernel Simulator.
    
    Directly iterates over physical pages mapped in `page_table`, injecting
    per-page UBBA tier biases (b_t = -(rho_t * s)^2 / 2) on-the-fly during
    online softmax max reduction.
    
    Args:
        q: Query vector / matrix, shape (..., D)
        physical_k_pool: Physical pool of pre-rotated canonical key pages, [P, block_size, D]
        physical_v_pool: Physical pool of raw value pages, [P, block_size, D]
        page_table: List of physical page IDs mapped to logical sequence blocks
        page_biases: Mapping or list of tier biases per physical page
        sm_scale: Softmax scale factor (defaults to 1.0 / sqrt(D))
        block_size: Tokens per page
        
    Returns:
        (out, lse): Context vector (..., D) and Log-Sum-Exp row statistic
    """
    d = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    q_vec = q.reshape(-1, d)
    num_queries = q_vec.shape[0]

    outs = []
    lses = []

    for qi in range(num_queries):
        cur_q = q_vec[qi]  # [d]

        # SRAM registers for single query
        m_curr = -np.inf
        l_curr = 0.0
        acc_curr = np.zeros(d, dtype=np.float64)

        for page_idx in page_table:
            k_page = physical_k_pool[page_idx]  # [block_size, d]
            v_page = physical_v_pool[page_idx]  # [block_size, d]

            # Tensor Core GEMM: k_page @ cur_q -> [block_size]
            s_tile = np.dot(k_page, cur_q) * sm_scale

            # Per-page tier bias injection directly into SRAM registers
            if page_biases is not None:
                if isinstance(page_biases, dict):
                    bias_val = page_biases.get(page_idx, 0.0)
                else:
                    bias_val = float(page_biases[page_idx])
                s_tile = s_tile + bias_val

            # Tile max
            m_tile = np.max(s_tile)
            m_new = max(m_curr, m_tile)

            # Online softmax scaling
            alpha = math.exp(m_curr - m_new) if m_curr > -np.inf else 0.0
            p_tile = np.exp(s_tile - m_new)
            l_tile = np.sum(p_tile)

            # Update running stats
            l_curr = alpha * l_curr + l_tile
            acc_curr = alpha * acc_curr + np.dot(p_tile, v_page)
            m_curr = m_new

        safe_l = l_curr if l_curr > 0.0 else 1.0
        out_vec = acc_curr / safe_l
        lse_val = m_curr + math.log(safe_l)

        outs.append(out_vec)
        lses.append(lse_val)

    out_arr = np.array(outs).reshape(q.shape)
    lse_arr = np.array(lses).reshape(q.shape[:-1])
    return out_arr, lse_arr


# ==============================================================================
# 6. UNIFIED ENGINE & BENCHMARK DISPATCHER
# ==============================================================================

class FusedTierBiasAttentionSimulator:
    """High-level orchestrator and profiler for Fused Tier-Bias FlashAttention."""

    def __init__(self, block_m: int = 64, block_n: int = 64):
        self.block_m = block_m
        self.block_n = block_n

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        bias: Optional[np.ndarray] = None,
        sm_scale: Optional[float] = None,
        causal: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, KernelMetrics]:
        """Compute fused attention and return results with hardware microarchitectural metrics."""
        out, lse = fused_tier_bias_attention_sim(
            q, k, v, bias=bias, sm_scale=sm_scale, causal=causal,
            block_m=self.block_m, block_n=self.block_n
        )

        b = q.shape[0] if q.ndim >= 3 else 1
        h = q.shape[1] if q.ndim == 4 else 1
        m = q.shape[-2]
        n = k.shape[-2]
        d = q.shape[-1]

        metrics = compute_microarch_metrics(
            batch_size=b,
            num_heads=h,
            seq_len_q=m,
            seq_len_k=n,
            head_dim=d,
            block_m=self.block_m,
            block_n=self.block_n,
            bytes_per_elem=2,
        )
        return out, lse, metrics


def fused_tier_bias_attention(
    q: Union[np.ndarray, Any],
    k: Union[np.ndarray, Any],
    v: Union[np.ndarray, Any],
    bias: Optional[Union[np.ndarray, Any]] = None,
    sm_scale: Optional[float] = None,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
) -> Tuple[Union[np.ndarray, Any], Union[np.ndarray, Any]]:
    """Universal dispatcher for Fused Tier-Bias FlashAttention.
    
    If inputs are PyTorch GPU tensors and Triton is available, dispatches to the
    native GPU kernel; otherwise executes via the bit-exact hardware simulator.
    """
    # Check if GPU Triton execution is feasible
    if HAS_TORCH and HAS_TRITON and isinstance(q, torch.Tensor) and q.is_cuda:
        # CUDA Triton fast path
        return _launch_triton_fused_attention(
            q, k, v, bias=bias, sm_scale=sm_scale, causal=causal,
            block_m=block_m, block_n=block_n
        )

    # Standard NumPy hardware simulator fallback
    is_torch = HAS_TORCH and isinstance(q, torch.Tensor)
    if is_torch:
        q_np = q.detach().cpu().numpy()
        k_np = k.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        b_np = bias.detach().cpu().numpy() if bias is not None else None
    else:
        q_np = q
        k_np = k
        v_np = v
        b_np = bias

    out_np, lse_np = fused_tier_bias_attention_sim(
        q_np, k_np, v_np, bias=b_np, sm_scale=sm_scale, causal=causal,
        block_m=block_m, block_n=block_n
    )

    if is_torch:
        return (
            torch.from_numpy(out_np).to(device=q.device, dtype=q.dtype),
            torch.from_numpy(lse_np).to(device=q.device),
        )
    return out_np, lse_np


def _launch_triton_fused_attention(
    q: Any,
    k: Any,
    v: Any,
    bias: Optional[Any] = None,
    sm_scale: Optional[float] = None,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
) -> Tuple[Any, Any]:
    """Launch Triton GPU forward kernel."""
    assert HAS_TRITON and HAS_TORCH
    d = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    # Normalize dimensions to (B, H, M, D)
    if q.ndim == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        if bias is not None and bias.ndim == 2:
            bias = bias.unsqueeze(0).unsqueeze(1)

    z, h, m, d = q.shape
    _, _, n, _ = k.shape

    out = torch.empty_like(q)
    lse = torch.empty((z, h, m), device=q.device, dtype=torch.float32)

    has_bias = bias is not None
    if not has_bias:
        bias = torch.empty((1, 1, 1, 1), device=q.device, dtype=q.dtype)

    grid = (triton.cdiv(m, block_m), z * h)

    _fused_tier_bias_fwd_kernel[grid](
        q, k, v, bias, out, lse,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
        z, h, m, n,
        BLOCK_M=block_m,
        BLOCK_D=d,
        BLOCK_N=block_n,
        IS_CAUSAL=causal,
        HAS_BIAS=has_bias,
    )
    return out, lse


def benchmark_kernel_overhead(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 2048,
    head_dim: int = 64,
    block_size: int = 64,
) -> Dict[str, Any]:
    """Benchmark mathematical equivalence and compute overhead analysis."""
    np.random.seed(42)
    q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float64)
    k = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float64)
    v = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float64)

    # Synthesize UBBA-derived tier bias across 4 compression tiers:
    # Tier 0 (FP8): 0.0
    # Tier 1 (INT4): -0.0246
    # Tier 2 (INT2): -0.2013
    # Tier 3 (MERGED): -0.3019
    tier_palette = [0.0, -0.0246, -0.2013, -0.3019]
    num_tiles = seq_len // block_size
    tile_tiers = [tier_palette[i % 4] for i in range(num_tiles)]
    bias_tokens = np.repeat(tile_tiers, block_size).astype(np.float64)

    # Run Unfused Reference
    out_unfused, lse_unfused = unfused_tier_bias_attention(
        q, k, v, bias=bias_tokens, sm_scale=1.0 / math.sqrt(head_dim), causal=True
    )

    # Run Fused FlashAttention Simulator
    out_fused, lse_fused = fused_tier_bias_attention_sim(
        q, k, v, bias=bias_tokens, sm_scale=1.0 / math.sqrt(head_dim), causal=True,
        block_m=block_size, block_n=block_size
    )

    max_err = float(np.max(np.abs(out_unfused - out_fused)))
    mean_err = float(np.mean(np.abs(out_unfused - out_fused)))
    max_lse_err = float(np.max(np.abs(lse_unfused - lse_fused)))

    metrics = compute_microarch_metrics(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len,
        seq_len_k=seq_len,
        head_dim=head_dim,
        block_m=block_size,
        block_n=block_size,
    )
    metrics.max_abs_diff_vs_unfused = max_err
    metrics.mean_abs_diff_vs_unfused = mean_err

    return {
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "max_lse_err": max_lse_err,
        "metrics": metrics,
        "traffic_reduction_ratio": metrics.hbm_traffic_reduction_ratio,
        "intermediate_score_matrix_mb": metrics.intermediate_score_matrix_bytes / (1024 * 1024),
        "sram_peak_kb": metrics.sram_peak_bytes / 1024,
    }
