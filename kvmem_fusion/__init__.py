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
from kvmem_fusion.spill_manager import (
    SuperTileSpillManager,
    DiskSpillManager,
    SuperTileManifest,
    BlockLocation,
)
from kvmem_fusion.agent_protocol import (
    negotiate_token_budget,
    format_tools_chatml,
    format_messages_with_tools,
    format_chatml,
    parse_single_tool_call_json,
    extract_tool_calls,
    create_initial_sse_chunk,
    create_content_delta_chunk,
    create_tool_call_chunks,
    create_final_sse_chunk,
    format_sse_event,
    StreamingToolProcessor,
    format_chat_completion_response,
)

from kvmem_fusion.metal_attention_hook import (
    MetalKVCacheInterceptor,
    metal_sdpa_prefill,
    metal_sdpa_attention_forward,
    apply_metal_attention_hooks,
    remove_metal_attention_hooks,
    benchmark_sdpa_vs_eager,
)

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
    "SuperTileSpillManager",
    "DiskSpillManager",
    "SuperTileManifest",
    "BlockLocation",
    "negotiate_token_budget",
    "format_tools_chatml",
    "format_messages_with_tools",
    "format_chatml",
    "parse_single_tool_call_json",
    "extract_tool_calls",
    "create_initial_sse_chunk",
    "create_content_delta_chunk",
    "create_tool_call_chunks",
    "create_final_sse_chunk",
    "format_sse_event",
    "StreamingToolProcessor",
    "format_chat_completion_response",
    "MetalKVCacheInterceptor",
    "metal_sdpa_prefill",
    "metal_sdpa_attention_forward",
    "apply_metal_attention_hooks",
    "remove_metal_attention_hooks",
    "benchmark_sdpa_vs_eager",
    "is_c_engine_available",
    "get_c_engine_lib",
    "neon_dot_c",
    "compute_deroped_mean_c",
    "inspect_and_split_c",
    "rank_subcentroids_cosine_c",
    "solve_ubba_ladder_c",
    "paged_table_translate_c",
    "gather_paged_tokens_c",
]


