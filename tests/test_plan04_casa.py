"""
Verification Suite for Plan 04 CASA (Canonical Atom Store Architecture):
1. Prefix-Chained Causal Deduplication (RadixTree / Merkle DAG Invariant vs Flawed Flat Chunking).
2. PagedAttention Tensor Core-Compatible GEMM vs CASA Q-Remap vs Ground Truth Bit-Exact Equivalence.
3. Big-Tile Coalescing for GPUDirect Storage (cuFile / NVMe DMA Bandwidth Saturation).
4. Physical Page Table Mapping & Zero-Copy Execution Invariant.
"""

import numpy as np
import pytest
from kvmem_fusion.core import (
    CanonicalAtomStore,
    compute_prefix_hash,
    apply_rope,
    StorageSuperTile,
)


def test_prefix_hash_chain_causal_invariance():
    """Verify that Prefix Hash Chains correctly prevent false deduplication across divergent histories,
    while guaranteeing exact deduplication for truly identical causal prefixes.
    """
    store = CanonicalAtomStore(block_size=32, head_dim=64, model_id="llama-3-70b")
    np.random.seed(42)

    # 1. System Prompt Prefix Block 0
    sys_prompt_tokens = list(range(100, 132))
    k0 = np.random.randn(32, 64)
    v0 = np.random.randn(32, 64)

    # Request A registers Prefix Block 0
    id_a0, hash_a0 = store.register_prefix_block(
        tokens=sys_prompt_tokens,
        k_unrotated=k0,
        v=v0,
        orig_pos_start=0,
        parent_hash=None
    )

    # Request B shares the EXACT same System Prompt Prefix Block 0
    id_b0, hash_b0 = store.register_prefix_block(
        tokens=sys_prompt_tokens,
        k_unrotated=k0,
        v=v0,
        orig_pos_start=0,
        parent_hash=None
    )

    # Exact causal prefix must deduplicate
    assert id_a0 == id_b0
    assert hash_a0 == hash_b0
    assert store.blocks[id_a0].ref_count == 2
    assert len(store.blocks) == 1

    # 2. Shared Sub-chunk with DIVERGENT Prefixes:
    # Chunk X tokens are identical in both conversations:
    subchunk_tokens = [999] * 32
    kx = np.random.randn(32, 64)
    vx = np.random.randn(32, 64)

    # In Request A: preceded by hash_a0
    id_a1, hash_a1 = store.register_prefix_block(
        tokens=subchunk_tokens,
        k_unrotated=kx,
        v=vx,
        orig_pos_start=32,
        parent_hash=hash_a0
    )

    # In Request C: preceded by a DIFFERENT context
    different_prompt_tokens = list(range(200, 232))
    k_diff = np.random.randn(32, 64)
    v_diff = np.random.randn(32, 64)
    id_c0, hash_c0 = store.register_prefix_block(
        tokens=different_prompt_tokens,
        k_unrotated=k_diff,
        v=v_diff,
        orig_pos_start=0,
        parent_hash=None
    )

    # Now Request C registers Chunk X with parent_hash = hash_c0
    id_c1, hash_c1 = store.register_prefix_block(
        tokens=subchunk_tokens,
        k_unrotated=kx,
        v=vx,
        orig_pos_start=32,
        parent_hash=hash_c0
    )

    # CRITICAL INVARIANT:
    # Even though subchunk_tokens are identical ([999]*32), their causal histories differ!
    # A flawed flat SHA-256 chunker would have collided (id_a1 == id_c1).
    # Prefix Hash Chain MUST produce distinct hashes and distinct storage entries!
    assert hash_a1 != hash_c1
    assert id_a1 != id_c1
    assert store.blocks[id_a1].ref_count == 1
    assert store.blocks[id_c1].ref_count == 1
    print("\n✓ Prefix Hash Chain correctly isolates divergent causal histories.")


def test_paged_attention_tensor_core_equivalence():
    """Verify that Tensor Core PagedAttention (GEMM over physical pages) achieves
    bit-exact equivalence with Ground Truth Attention and CASA Q-remap.
    
    This mathematically demonstrates why Paged logical addressing naturally avoids re-RoPE:
    Keys pre-rotated at canonical positions can be multiplied by (R_m q) directly via
    standard matrix multiplication tiles without per-token vector branches.
    """
    head_dim = 64
    block_size = 32
    num_blocks = 8
    total_tokens = num_blocks * block_size
    np.random.seed(101)

    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)

    # Generate synthetic sequence tokens & KV cache
    all_raw_keys = []
    all_raw_values = []
    block_ids = []
    page_table = []
    parent_hash = None

    for b in range(num_blocks):
        tokens = list(range(b * block_size, (b + 1) * block_size))
        k_raw = np.random.randn(block_size, head_dim)
        v_raw = np.random.randn(block_size, head_dim)
        all_raw_keys.append(k_raw)
        all_raw_values.append(v_raw)

        blk_id, parent_hash = store.register_prefix_block(
            tokens=tokens,
            k_unrotated=k_raw,
            v=v_raw,
            orig_pos_start=b * block_size,
            parent_hash=parent_hash
        )
        block_ids.append(blk_id)
        # In PagedAttention, logical page b maps to block's physical page
        page_table.append(store.blocks[blk_id].physical_page_id)

    concat_keys = np.concatenate(all_raw_keys, axis=0)
    concat_values = np.concatenate(all_raw_values, axis=0)

    # Single Query at decode step pos = 256
    q_pos = total_tokens
    q_raw = np.random.randn(head_dim)

    # 1. Ground Truth Standard Attention (Dense RoPE)
    q_gt = apply_rope(q_raw, pos=q_pos)
    gt_scores = np.zeros(total_tokens)
    for i in range(total_tokens):
        k_gt = apply_rope(concat_keys[i], pos=i)
        gt_scores[i] = np.dot(q_gt, k_gt) / np.sqrt(head_dim)
    exp_gt = np.exp(gt_scores - np.max(gt_scores))
    weights_gt = exp_gt / np.sum(exp_gt)
    context_gt = np.dot(weights_gt, concat_values)

    # 2. CASA Q-remap Attention (Relative RoPE Formulation)
    weights_casa, context_casa = store.compute_attention(
        q_raw=q_raw,
        query_logical_pos=q_pos,
        block_ids=block_ids
    )

    # 3. PagedAttention Tensor Core GEMM (Physical Page Table addressing)
    weights_paged, context_paged = store.compute_paged_attention_gemm(
        q_raw=q_raw,
        query_logical_pos=q_pos,
        page_table=page_table
    )

    # Precision Comparisons
    diff_qremap_gt = np.max(np.abs(context_gt - context_casa))
    diff_paged_gt = np.max(np.abs(context_gt - context_paged))
    diff_paged_qremap = np.max(np.abs(context_casa - context_paged))

    print(f"\nContext Diff (Ground Truth vs CASA Q-remap): {diff_qremap_gt:.2e}")
    print(f"Context Diff (Ground Truth vs Paged GEMM):    {diff_paged_gt:.2e}")
    print(f"Context Diff (CASA Q-remap vs Paged GEMM):    {diff_paged_qremap:.2e}")

    # All three must be bit-exact within double-precision machine epsilon
    assert diff_qremap_gt < 1e-12
    assert diff_paged_gt < 1e-12
    assert diff_paged_qremap < 1e-12
    print("✓ Bit-exact equivalence between Paged GEMM, Q-remap, and Ground Truth confirmed.")


def test_big_tile_coalescing_for_gpudirect_storage():
    """Verify that logical atom blocks (32 tokens, ~8 KB) coalesce into big-tile
    macro-chunks exceeding the 64 KB GPUDirect Storage / cuFile saturation threshold.
    """
    store = CanonicalAtomStore(block_size=32, head_dim=64)
    # Register 32 blocks (1024 tokens)
    block_ids = []
    parent_hash = None
    for b in range(32):
        tokens = list(range(b * 32, (b + 1) * 32))
        k = np.random.randn(32, 64)
        v = np.random.randn(32, 64)
        b_id, parent_hash = store.register_prefix_block(
            tokens=tokens,
            k_unrotated=k,
            v=v,
            orig_pos_start=b * 32,
            parent_hash=parent_hash
        )
        block_ids.append(b_id)

    # Coalesce into Big-Tiles of 16 blocks each
    super_tiles = store.coalesce_big_tiles(block_ids, pages_per_tile=16)

    assert len(super_tiles) == 2
    for tile in super_tiles:
        # 16 blocks * 32 tokens = 512 tokens
        assert tile.total_tokens == 512
        # 512 tokens * 64 dim * 2 bytes * 2 (K & V) = 131,072 bytes = 128 KB
        assert tile.size_bytes == 131072
        # 128 KB is comfortably above 64 KB (cuFile / GDS saturation boundary)
        assert tile.size_bytes >= 65536

    print(f"\n✓ Big-Tile coalescing generated {len(super_tiles)} tiles, each {super_tiles[0].size_bytes / 1024:.1f} KB (>= 64 KB).")


def test_paged_attention_mixed_precision_tier_bias():
    """Verify that CASA PagedAttention Tensor Core GEMM correctly honors UBBA mixed-precision
    tier-biases, achieving bit-exact numerical agreement with CASA Q-remap.
    """
    head_dim = 64
    block_size = 32
    num_blocks = 4
    np.random.seed(2024)

    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    tier_assignment = ["FP8", "INT4", "INT2", "MERGED"]
    tier_biases = {"FP8": 0.0, "INT4": -0.08, "INT2": -0.22, "MERGED": -0.45}

    block_ids = []
    page_table = []
    parent_hash = None

    for b in range(num_blocks):
        tokens = list(range(b * block_size, (b + 1) * block_size))
        k_raw = np.random.randn(block_size, head_dim)
        v_raw = np.random.randn(block_size, head_dim)
        b_id, parent_hash = store.register_prefix_block(
            tokens=tokens,
            k_unrotated=k_raw,
            v=v_raw,
            orig_pos_start=b * block_size,
            parent_hash=parent_hash,
            tier=tier_assignment[b]
        )
        block_ids.append(b_id)
        page_table.append(store.blocks[b_id].physical_page_id)

    q_pos = num_blocks * block_size
    q_raw = np.random.randn(head_dim)

    # Attention without tier-biases
    w_unbiased, ctx_unbiased = store.compute_paged_attention_gemm(
        q_raw=q_raw, query_logical_pos=q_pos, page_table=page_table
    )

    # Attention with UBBA tier-biases
    w_casa, ctx_casa = store.compute_attention(
        q_raw=q_raw, query_logical_pos=q_pos, block_ids=block_ids, tier_biases=tier_biases
    )
    w_paged, ctx_paged = store.compute_paged_attention_gemm(
        q_raw=q_raw, query_logical_pos=q_pos, page_table=page_table, tier_biases=tier_biases
    )

    diff_weights = np.max(np.abs(w_casa - w_paged))
    diff_context = np.max(np.abs(ctx_casa - ctx_paged))
    shift_magnitude = np.max(np.abs(w_unbiased - w_paged))

    assert diff_weights < 1e-12
    assert diff_context < 1e-12
    assert shift_magnitude > 1e-4  # Proves tier biases significantly and correctly calibrated attention


def test_prefix_hash_chain_ubba_tier_allocation_divergence():
    """Verify that non-identical suffixes branching from the same prefix:
    1. Are isolated with distinct prefix hashes (zero causal cross-contamination).
    2. Can be assigned divergent compression tiers by UBBA dynamic allocation.
    3. Execute cleanly without interfering with shared prefix atom references.
    """
    store = CanonicalAtomStore(block_size=32, head_dim=64)
    np.random.seed(777)

    # 1. Common Prefix (System Prompt)
    prefix_tokens = list(range(1000, 1032))
    k_pre = np.random.randn(32, 64)
    v_pre = np.random.randn(32, 64)

    # Request A & Request B share Prefix Block 0
    id_pre_a, hash_pre_a = store.register_prefix_block(
        tokens=prefix_tokens, k_unrotated=k_pre, v=v_pre, orig_pos_start=0, parent_hash=None
    )
    id_pre_b, hash_pre_b = store.register_prefix_block(
        tokens=prefix_tokens, k_unrotated=k_pre, v=v_pre, orig_pos_start=0, parent_hash=None
    )

    assert id_pre_a == id_pre_b
    assert store.blocks[id_pre_a].ref_count == 2

    # 2. Divergent Suffixes
    suffix_tokens_a = list(range(2000, 2032))  # Critical needle prompt
    suffix_tokens_b = list(range(3000, 3032))  # Generic background prompt

    k_a = np.random.randn(32, 64)
    v_a = np.random.randn(32, 64)
    k_b = np.random.randn(32, 64)
    v_b = np.random.randn(32, 64)

    id_suf_a, hash_suf_a = store.register_prefix_block(
        tokens=suffix_tokens_a, k_unrotated=k_a, v=v_a, orig_pos_start=32, parent_hash=hash_pre_a, tier="FP8"
    )
    id_suf_b, hash_suf_b = store.register_prefix_block(
        tokens=suffix_tokens_b, k_unrotated=k_b, v=v_b, orig_pos_start=32, parent_hash=hash_pre_b, tier="INT4"
    )

    assert hash_suf_a != hash_suf_b
    assert id_suf_a != id_suf_b

    # Verify UBBA dynamic allocation: Suffix A is FP8, Suffix B is INT4
    assert store.blocks[id_suf_a].tier == "FP8"
    assert store.blocks[id_suf_b].tier == "INT4"

    # Verify both can execute PagedAttention independently without contamination
    q_raw = np.random.randn(64)
    page_tbl_a = [store.blocks[id_pre_a].physical_page_id, store.blocks[id_suf_a].physical_page_id]
    page_tbl_b = [store.blocks[id_pre_b].physical_page_id, store.blocks[id_suf_b].physical_page_id]

    tier_biases = {"FP8": 0.0, "INT4": -0.10}
    w_a, ctx_a = store.compute_paged_attention_gemm(q_raw, 64, page_tbl_a, tier_biases)
    w_b, ctx_b = store.compute_paged_attention_gemm(q_raw, 64, page_tbl_b, tier_biases)

    # Outputs must differ due to divergent suffixes and divergent tiers
    assert not np.allclose(ctx_a, ctx_b)

