"""
kernels package: Fused Tier-Bias FlashAttention kernel and hardware-accurate microarchitectural simulator.
"""

from kernels.fused_tier_bias_attention import (
    fused_tier_bias_attention,
    fused_tier_bias_attention_sim,
    unfused_tier_bias_attention,
    paged_fused_tier_bias_attention_sim,
    FusedTierBiasAttentionSimulator,
    KernelMetrics,
    benchmark_kernel_overhead,
)

__all__ = [
    "fused_tier_bias_attention",
    "fused_tier_bias_attention_sim",
    "unfused_tier_bias_attention",
    "paged_fused_tier_bias_attention_sim",
    "FusedTierBiasAttentionSimulator",
    "KernelMetrics",
    "benchmark_kernel_overhead",
]
