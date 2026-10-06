"""
kvmem_fusion package
"""
from kvmem_fusion.core import CanonicalAtomStore, KVBlock, apply_rope
from kvmem_fusion.ifr import (
    IFRBlock,
    LSECacheTracker,
    AntiCollapseSplitter,
    IVFMeanKIndex,
    IFRRetriever,
    UEFCEvaluationGate,
    compute_deroped_mean,
)

__all__ = [
    "CanonicalAtomStore",
    "KVBlock",
    "apply_rope",
    "IFRBlock",
    "LSECacheTracker",
    "AntiCollapseSplitter",
    "IVFMeanKIndex",
    "IFRRetriever",
    "UEFCEvaluationGate",
    "compute_deroped_mean",
]
