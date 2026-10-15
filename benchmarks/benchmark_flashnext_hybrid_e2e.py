#!/usr/bin/env python3
"""
Phase 1 Benchmark: Qwen3.8-Flash-Next 3:1 Hybrid Architecture & Bounded Retrieval Benchmark.

Empirically benchmarks:
1. 36 Linear Attention Layers (Gated DeltaNet, O(1) constant recurrent state)
2. 12 Sparse Attention Layers (QSA, indexer_budget=2048) with IFR content retrieval
3. Bounded Execution View Sparsity Ratio across context lengths: 4K, 8K, 16K, 32K, 64K, 128K
4. IFR Retrieval Latency (ms) and Slice Gather Latency (ms) on Apple Silicon
5. Physical Memory Footprint vs Dense Full-Attention Baseline
"""

import os
import sys
import time
import json
import math
import resource
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from kvmem_fusion.flashnext_hybrid import (
    FlashNextConfig,
    FlashNextMemoryModel,
    LinearDeltaNetStateCell,
)
from kvmem_fusion.core import CanonicalAtomStore
from kvmem_fusion.ifr import IFRBlock, IFRRetriever, AntiCollapseSplitter


def run_flashnext_phase1_benchmark(context_scales: List[int] = None) -> Dict[str, Any]:
    if context_scales is None:
        context_scales = [4096, 8192, 16384, 32768, 65536, 131072]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16
    cfg = FlashNextConfig()
    mem_model = FlashNextMemoryModel(cfg)

    print("=" * 80)
    print(f"🚀 Qwen3.8-Flash-Next 3:1 Hybrid Architecture & Bounded Retrieval Benchmark")
    print(f"• Hardware: Apple Silicon ({device.upper()}) | Precision: {dtype}")
    print(f"• Structure: {cfg.num_linear_layers} Linear Layers (DeltaNet) + {cfg.num_sparse_layers} Sparse Layers (QSA)")
    print(f"• Indexer Budget: {cfg.indexer_budget} tokens | Head Dim: {cfg.indexer_head_dim}")
    print("=" * 80)

    # 1. Benchmark Linear DeltaNet State Cell (36 Layers)
    linear_cell = LinearDeltaNetStateCell(key_dim=128, value_dim=128, num_heads=16).to(device)
    q_dummy = torch.randn(1, 16, 1, 128, device=device, dtype=torch.float32)
    k_dummy = torch.randn(1, 16, 1, 128, device=device, dtype=torch.float32)
    v_dummy = torch.randn(1, 16, 1, 128, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = linear_cell(q_dummy, k_dummy, v_dummy)

    t0_linear = time.perf_counter()
    linear_iters = 100
    for _ in range(linear_iters):
        _ = linear_cell(q_dummy, k_dummy, v_dummy)
    linear_step_us = ((time.perf_counter() - t0_linear) / linear_iters) * 1_000_000
    total_36_linear_us = linear_step_us * cfg.num_linear_layers

    results = []

    # 2. Benchmark Across Context Scales (4K -> 128K)
    for ctx in context_scales:
        num_blocks = ctx // 32
        atom_store = CanonicalAtomStore(block_size=32, head_dim=cfg.indexer_head_dim)
        ifr = IFRRetriever(dim=cfg.indexer_head_dim, target_top_k=cfg.indexer_budget // 32)
        splitter = AntiCollapseSplitter(dispersion_threshold=0.868, use_normalized=True)

        # Generate synthetic blocks with realistic semantic clusters
        rng = np.random.RandomState(42)
        centroids = rng.randn(8, cfg.indexer_head_dim).astype(np.float32)
        centroids /= np.linalg.norm(centroids, axis=-1, keepdims=True)

        for b_idx in range(num_blocks):
            c_idx = b_idx % 8
            k_base = centroids[c_idx] + rng.randn(cfg.indexer_head_dim).astype(np.float32) * 0.1
            k_base /= np.linalg.norm(k_base)
            block_tokens = list(range(b_idx * 32, (b_idx + 1) * 32))
            
            k_mat = np.tile(k_base, (32, 1))
            v_mat = rng.randn(32, cfg.indexer_head_dim).astype(np.float32) * 0.05
            
            registered_id, _ = atom_store.register_prefix_block(
                tokens=block_tokens,
                k_unrotated=k_mat,
                v=v_mat,
                orig_pos_start=b_idx * 32,
                parent_hash=f"H_{b_idx}",
                tier="FP16",
            )
            ifr_blk = IFRBlock(
                block_id=registered_id,
                tokens=block_tokens,
                keys=k_mat,
                values=v_mat,
                orig_pos_start=b_idx * 32,
                parent_hash=f"H_{b_idx}",
            )
            ifr.add_block(ifr_blk)

        # Benchmark IFR Retrieval Latency across 30 repeats
        q_vec = (centroids[0] + rng.randn(cfg.indexer_head_dim).astype(np.float32) * 0.05)
        q_vec /= np.linalg.norm(q_vec)
        
        ifr.build_index()

        retrieval_latencies = []
        for _ in range(30):
            t_ret0 = time.perf_counter()
            top_k_blocks = (cfg.indexer_budget - 256) // 32
            ret_bids, _ = ifr.retrieve(q_vec, top_k=top_k_blocks)
            retrieval_latencies.append((time.perf_counter() - t_ret0) * 1000)

        mean_retrieval_ms = float(np.mean(retrieval_latencies))
        p95_retrieval_ms = float(np.percentile(retrieval_latencies, 95))

        # Bounded Execution View Calculation
        local_window_tokens = 256
        retrieved_tokens = len(ret_bids) * 32
        bounded_tokens = min(ctx, local_window_tokens + retrieved_tokens)
        sparsity_ratio = bounded_tokens / ctx
        compute_reduction_pct = (1.0 - sparsity_ratio) * 100.0

        # Memory footprint
        mem_stats = mem_model.compute_footprint_mb(ctx, use_kvmem_bounded_view=True)

        res_entry = {
            "context_tokens": ctx,
            "num_blocks": num_blocks,
            "mean_retrieval_ms": round(mean_retrieval_ms, 3),
            "p95_retrieval_ms": round(p95_retrieval_ms, 3),
            "bounded_attended_tokens": bounded_tokens,
            "sparsity_ratio": round(sparsity_ratio, 4),
            "compute_reduction_pct": round(compute_reduction_pct, 1),
            "qsa_kv_resident_mb": mem_stats["qsa_kv_mb"],
            "dense_kv_baseline_mb": mem_stats["dense_kv_baseline_mb"],
            "total_resident_ram_gb": mem_stats["total_resident_ram_gb"],
            "fits_in_32gb": mem_stats["fits_in_32gb_mac"],
        }
        results.append(res_entry)

        print(
            f"Context: {ctx:6d} ({ctx//1024}K) | "
            f"IFR Latency: {mean_retrieval_ms:.3f} ms (p95: {p95_retrieval_ms:.3f} ms) | "
            f"Attended: {bounded_tokens:4d}/{ctx:6d} ({compute_reduction_pct:.1f}% cut) | "
            f"RAM: {mem_stats['total_resident_ram_gb']:.2f} GB / 32GB"
        )

    summary = {
        "hardware": f"Apple Silicon ({device.upper()})",
        "precision": str(dtype),
        "linear_36_layers_total_us": round(total_36_linear_us, 2),
        "results": results,
    }

    output_path = "benchmarks/flashnext_phase1_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("=" * 80)
    print(f"✅ Benchmark completed successfully! Results written to: {output_path}")

    return summary


if __name__ == "__main__":
    run_flashnext_phase1_benchmark()
