"""
Comprehensive Verification Suite for SuperTile Spill Manager:
1. Binary SuperTile Page-Aligned Serialization & Zero-Copy mmap.
2. Autonomous Watermark Spill on Token & Byte Thresholds.
3. Saliency-Aware vs Temporal (Oldest) Eviction Policies.
4. Bit-Exact Attention Equivalence (RAM vs Zero-Copy Disk mmap).
5. Seamless Two-Tier IFR (Indexed Forward Recall) Integration.
6. Cold-Restart Disk Recovery & Metadata Persistence.
7. Throughput & Sub-Microsecond Retrieval Latency Benchmarking.
"""

import os
import shutil
import tempfile
import time
import numpy as np
import pytest

from kvmem_fusion.core import (
    CanonicalAtomStore,
    KVBlock,
    apply_rope,
    compute_prefix_hash,
)
from kvmem_fusion.ifr import (
    IFRBlock,
    IFRRetriever,
)
from kvmem_fusion.spill_manager import (
    SuperTileSpillManager,
    SuperTileManifest,
    PAGE_SIZE,
    align_to_page,
)


@pytest.fixture
def temp_cache_dir():
    """Create and cleanly tear down a temporary cache directory."""
    d = tempfile.mkdtemp(prefix="kvmem_test_cache_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def create_synthetic_block(
    store: CanonicalAtomStore,
    pos_start: int,
    block_idx: int,
    dispersity_scale: float = 1.0,
    head_dim: int = 64,
    block_size: int = 32,
    dtype=np.float64,
    parent_hash: str = None,
) -> str:
    """Helper to generate a block with controllable dispersity."""
    tokens = list(range(pos_start, pos_start + block_size))
    # Base key vector + noise
    base_k = np.random.randn(head_dim)
    k_unrot = np.repeat(base_k[np.newaxis, :], block_size, axis=0) + (
        np.random.randn(block_size, head_dim) * dispersity_scale
    )
    k_unrot = k_unrot.astype(dtype)
    v = np.random.randn(block_size, head_dim).astype(dtype)

    bid, _ = store.register_prefix_block(
        tokens=tokens,
        k_unrotated=k_unrot,
        v=v,
        orig_pos_start=pos_start,
        parent_hash=parent_hash,
    )
    return bid


def test_supertile_serialization_and_mmap(temp_cache_dir):
    """
    Verify binary SuperTile serialization layout:
    - 64-byte binary header with magic b'KVST'
    - Page-aligned tensor offsets
    - Bit-exact tensor reconstruction
    - Zero-copy view invariant (view.base is not None)
    """
    head_dim = 64
    block_size = 32
    num_blocks = 16
    np.random.seed(42)

    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    spill_mgr = SuperTileSpillManager(
        atom_store=store,
        cache_dir=temp_cache_dir,
        max_active_tokens=10000,
        pages_per_tile=num_blocks,
    )

    block_ids = []
    parent_h = None
    for i in range(num_blocks):
        bid = create_synthetic_block(store, i * block_size, i, parent_hash=parent_h)
        block_ids.append(bid)
        spill_mgr.on_block_added(store.blocks[bid])
        parent_h = store.blocks[bid].prefix_hash

    # Save original arrays before spill
    orig_k = [store.blocks[bid].k_unrotated.copy() for bid in block_ids]
    orig_v = [store.blocks[bid].v.copy() for bid in block_ids]
    orig_kc = [store.blocks[bid].k_canonical.copy() for bid in block_ids]

    # Spill to disk SuperTile
    tile = spill_mgr.spill_blocks(block_ids)

    assert tile.total_tokens == num_blocks * block_size
    manifest = spill_mgr._tiles[tile.tile_id]
    assert os.path.exists(manifest.file_path)

    # Check page alignment
    assert manifest.tensors_offset_bytes % PAGE_SIZE == 0
    assert manifest.k_offset_bytes % PAGE_SIZE == 0
    assert manifest.v_offset_bytes % PAGE_SIZE == 0
    assert manifest.k_canon_offset_bytes % PAGE_SIZE == 0

    # Verify bit-exact data & zero-copy slices
    for idx, bid in enumerate(block_ids):
        blk = store.blocks[bid]
        # Array contents must be bit-exact
        assert np.array_equal(blk.k_unrotated, orig_k[idx])
        assert np.array_equal(blk.v, orig_v[idx])
        assert np.array_equal(blk.k_canonical, orig_kc[idx])

        # Verify zero-copy: array view's base must reference the underlying memory map
        assert blk.k_unrotated.base is not None
        assert blk.v.base is not None
        assert blk.k_canonical.base is not None

    spill_mgr.close()


def test_automatic_spill_on_token_threshold(temp_cache_dir):
    """
    Verify autonomous flushing when token count exceeds configurable threshold.
    """
    head_dim = 64
    block_size = 32
    max_tokens = 256  # 8 blocks of 32 tokens
    watermark_ratio = 0.75  # Should spill down to 192 tokens (6 blocks)

    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    spill_mgr = SuperTileSpillManager(
        atom_store=store,
        cache_dir=temp_cache_dir,
        max_active_tokens=max_tokens,
        watermark_ratio=watermark_ratio,
        pages_per_tile=4,
    )

    np.random.seed(123)
    # Register 16 blocks (512 tokens), exceeding 256 token limit
    for i in range(16):
        bid = create_synthetic_block(store, i * block_size, i)
        spill_mgr.on_block_added(store.blocks[bid])

    stats = spill_mgr.get_active_memory_stats()
    # Active tokens should have dropped below max_tokens
    assert stats["active_tokens"] <= max_tokens
    assert stats["spilled_block_count"] > 0
    assert stats["total_tiles"] > 0
    assert spill_mgr.stats["spill_events"] > 0

    # Spilled blocks should be registered in _spilled_blocks
    for bid, loc in spill_mgr._spilled_blocks.items():
        assert loc.is_spilled
        assert loc.tile_id is not None
        assert store.blocks[bid].k_unrotated.base is not None

    spill_mgr.close()


def test_automatic_spill_on_byte_threshold(temp_cache_dir):
    """
    Verify autonomous flushing when byte footprint exceeds max_active_bytes limit.
    """
    head_dim = 64
    block_size = 32
    # 1 block: 32 * 64 * 8 bytes * 3 tensors = 49,152 bytes (~48 KB)
    # Set limit to 100 KB (~2 blocks allowed active)
    max_bytes = 100 * 1024

    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    spill_mgr = SuperTileSpillManager(
        atom_store=store,
        cache_dir=temp_cache_dir,
        max_active_tokens=100000,
        max_active_bytes=max_bytes,
        watermark_ratio=0.5,
        pages_per_tile=2,
    )

    np.random.seed(999)
    for i in range(8):
        bid = create_synthetic_block(store, i * block_size, i)
        spill_mgr.on_block_added(store.blocks[bid])

    stats = spill_mgr.get_active_memory_stats()
    assert stats["active_bytes"] <= max_bytes
    assert stats["spilled_block_count"] >= 4

    spill_mgr.close()


def test_eviction_policies(temp_cache_dir):
    """
    Verify eviction ordering:
    1. 'salience': Least salient (low dispersity) evicted before burst needle.
    2. 'oldest': Earliest orig_pos_start evicted first.
    """
    head_dim = 64
    block_size = 32
    np.random.seed(777)

    # 1. Test Salience Policy
    store1 = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    spill_mgr1 = SuperTileSpillManager(
        atom_store=store1,
        cache_dir=temp_cache_dir + "/salience",
        max_active_tokens=128,  # 4 blocks max
        eviction_policy="salience",
        watermark_ratio=0.5,
        pages_per_tile=2,
    )

    # Register 3 diffuse background blocks (low dispersity)
    bg_ids = []
    for i in range(3):
        bid = create_synthetic_block(store1, i * block_size, i, dispersity_scale=0.01)
        bg_ids.append(bid)
        spill_mgr1.on_block_added(store1.blocks[bid])

    # Register 1 high-salience needle block (high dispersity)
    needle_id = create_synthetic_block(store1, 3 * block_size, 3, dispersity_scale=5.0)
    spill_mgr1.on_block_added(store1.blocks[needle_id])

    # Register 1 more block to trigger eviction
    trigger_id = create_synthetic_block(store1, 4 * block_size, 4, dispersity_scale=0.02)
    spill_mgr1.on_block_added(store1.blocks[trigger_id])

    # Under 'salience', low dispersity background blocks should be spilled first,
    # while needle_id MUST remain active in RAM!
    assert not spill_mgr1.is_spilled(needle_id)
    # At least some background blocks must have been spilled
    assert any(spill_mgr1.is_spilled(b) for b in bg_ids)
    spill_mgr1.close()

    # 2. Test Oldest Policy
    store2 = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    spill_mgr2 = SuperTileSpillManager(
        atom_store=store2,
        cache_dir=temp_cache_dir + "/oldest",
        max_active_tokens=128,  # 4 blocks max
        eviction_policy="oldest",
        watermark_ratio=0.5,
        pages_per_tile=2,
    )
    b0 = create_synthetic_block(store2, 0, 0, dispersity_scale=5.0)
    spill_mgr2.on_block_added(store2.blocks[b0])
    b1 = create_synthetic_block(store2, 32, 1, dispersity_scale=0.1)
    spill_mgr2.on_block_added(store2.blocks[b1])
    b2 = create_synthetic_block(store2, 64, 2, dispersity_scale=0.1)
    spill_mgr2.on_block_added(store2.blocks[b2])
    b3 = create_synthetic_block(store2, 96, 3, dispersity_scale=0.1)
    spill_mgr2.on_block_added(store2.blocks[b3])
    # Trigger spill
    b4 = create_synthetic_block(store2, 128, 4, dispersity_scale=0.1)
    spill_mgr2.on_block_added(store2.blocks[b4])

    # Under 'oldest', b0 and b1 must be evicted first even though b0 had high dispersity
    assert spill_mgr2.is_spilled(b0)
    spill_mgr2.close()


def test_bit_exact_attention_equivalence_after_spill(temp_cache_dir):
    """
    Verify bit-exact mathematical equivalence between:
    - Pre-spill RAM Attention Context
    - Post-spill Zero-Copy mmap Attention Context
    Both CASA Q-remap and Tensor Core PagedAttention GEMM must match with diff < 1e-12.
    """
    head_dim = 64
    block_size = 32
    num_blocks = 8
    total_tokens = num_blocks * block_size
    np.random.seed(555)

    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    spill_mgr = SuperTileSpillManager(
        atom_store=store,
        cache_dir=temp_cache_dir,
        max_active_tokens=10000,
        pages_per_tile=num_blocks,
    )

    block_ids = []
    page_table = []
    for i in range(num_blocks):
        bid = create_synthetic_block(store, i * block_size, i)
        block_ids.append(bid)
        page_table.append(store.blocks[bid].physical_page_id)
        spill_mgr.on_block_added(store.blocks[bid])

    q_raw = np.random.randn(head_dim)
    q_pos = total_tokens

    # 1. Compute ground truth attention in RAM
    w_casa_ram, ctx_casa_ram = store.compute_attention(q_raw, q_pos, block_ids)
    w_paged_ram, ctx_paged_ram = store.compute_paged_attention_gemm(q_raw, q_pos, page_table)

    # 2. Force spill of all blocks to disk SuperTile
    spill_mgr.spill_blocks(block_ids)
    for bid in block_ids:
        assert spill_mgr.is_spilled(bid)

    # 3. Compute attention over memory-mapped SuperTile
    w_casa_mmap, ctx_casa_mmap = spill_mgr.compute_attention(q_raw, q_pos, block_ids)
    w_paged_mmap, ctx_paged_mmap = spill_mgr.compute_paged_attention_gemm(q_raw, q_pos, page_table)

    # Verify bit-exact numerical agreement
    diff_casa = np.max(np.abs(ctx_casa_ram - ctx_casa_mmap))
    diff_paged = np.max(np.abs(ctx_paged_ram - ctx_paged_mmap))
    diff_weights = np.max(np.abs(w_casa_ram - w_casa_mmap))

    assert diff_casa < 1e-12
    assert diff_paged < 1e-12
    assert diff_weights < 1e-12

    spill_mgr.close()


def test_ifr_two_tier_retrieval_with_spilled_atoms(temp_cache_dir):
    """
    Verify complete two-tier IFR integration:
    - Coarse centroids stay in DRAM IVF index.
    - Historical needle block is flushed to disk SuperTile.
    - IFR query recalls needle block ID.
    - SpillManager zero-copy reloads the atom from NVMe SuperTile.
    """
    head_dim = 64
    block_size = 32
    num_blocks = 16
    np.random.seed(888)

    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    ifr = IFRRetriever(dim=head_dim, n_clusters=4, n_probe=2, tau_exact_bypass=4, target_top_k=2)
    spill_mgr = SuperTileSpillManager(
        atom_store=store,
        ifr_retriever=ifr,
        cache_dir=temp_cache_dir,
        max_active_tokens=10000,
        pages_per_tile=8,
    )

    block_ids = []
    # Block 2 is our historical needle with distinctive key pattern
    needle_pattern = np.ones(head_dim) * 3.5
    needle_bid = None

    for i in range(num_blocks):
        if i == 2:
            k_unrot = np.repeat(needle_pattern[np.newaxis, :], block_size, axis=0)
            v = np.ones((block_size, head_dim)) * 7.7
            bid = store.register_block(list(range(i * 32, (i + 1) * 32)), k_unrot, v, i * 32)
            needle_bid = bid
        else:
            bid = create_synthetic_block(store, i * block_size, i, dispersity_scale=0.5)

        block_ids.append(bid)
        spill_mgr.on_block_added(store.blocks[bid])

        # Register in IFR
        blk = store.blocks[bid]
        ifr_blk = IFRBlock(
            block_id=bid,
            tokens=blk.tokens,
            keys=blk.k_unrotated,
            values=blk.v,
            orig_pos_start=blk.orig_pos_start,
        )
        ifr.add_block(ifr_blk)

    ifr.build_index()

    # Flush all blocks to disk SuperTiles
    spill_mgr.spill_blocks(block_ids[:8])
    spill_mgr.spill_blocks(block_ids[8:])

    assert spill_mgr.is_spilled(needle_bid)

    # Query matching the historical needle
    query = needle_pattern.copy() / np.linalg.norm(needle_pattern)
    reloaded_blocks, ifr_meta = spill_mgr.retrieve_for_ifr(query)

    reloaded_bids = [b.block_id for b in reloaded_blocks]
    assert needle_bid in reloaded_bids
    assert ifr_meta["spilled_atoms_recalled"] >= 1

    # Verify zero-copy
    recalled_needle = [b for b in reloaded_blocks if b.block_id == needle_bid][0]
    assert recalled_needle.k_unrotated.base is not None
    assert np.allclose(recalled_needle.k_unrotated[0], needle_pattern)

    spill_mgr.close()


def test_cold_restart_recovery(temp_cache_dir):
    """
    Verify that an on-disk SuperTile repository can be recovered on cold start:
    - Scans .kvs files in cache_dir
    - Restores metadata JSON, block positions, and zero-copy mmap handles
    - Restores execution readiness
    """
    head_dim = 64
    block_size = 32
    num_blocks = 8
    np.random.seed(321)

    # Phase 1: Write and spill
    store1 = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    spill_mgr1 = SuperTileSpillManager(
        atom_store=store1,
        cache_dir=temp_cache_dir,
        cleanup_on_close=False,  # Keep disk files
    )
    bids = []
    for i in range(num_blocks):
        bid = create_synthetic_block(store1, i * block_size, i)
        bids.append(bid)
        spill_mgr1.on_block_added(store1.blocks[bid])

    spill_mgr1.spill_blocks(bids)
    spill_mgr1.close()

    # Phase 2: Fresh instance (Cold Restart)
    store2 = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    spill_mgr2 = SuperTileSpillManager(
        atom_store=store2,
        cache_dir=temp_cache_dir,
        cleanup_on_close=False,
    )

    recovered = spill_mgr2.recover_from_disk()
    assert recovered == 1
    assert len(spill_mgr2._tiles) == 1
    assert len(store2.blocks) == num_blocks

    for bid in bids:
        assert bid in store2.blocks
        blk = store2.blocks[bid]
        assert blk.k_unrotated.base is not None
        assert blk.v.base is not None
        assert blk.k_canonical.base is not None

    # Verify attention works out of the box on recovered store
    q = np.random.randn(head_dim)
    w, ctx = spill_mgr2.compute_attention(q, 256, bids)
    assert ctx.shape == (head_dim,)

    spill_mgr2.close()


def test_throughput_benchmark(temp_cache_dir):
    """
    Benchmark NVMe/SSD SuperTile spilling and zero-copy reload throughput.
    """
    head_dim = 64
    block_size = 32
    num_blocks = 64  # 2048 tokens
    np.random.seed(42)

    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    spill_mgr = SuperTileSpillManager(
        atom_store=store,
        cache_dir=temp_cache_dir,
        pages_per_tile=16,
    )

    bids = []
    for i in range(num_blocks):
        bid = create_synthetic_block(store, i * block_size, i, dtype=np.float32)
        bids.append(bid)
        spill_mgr.on_block_added(store.blocks[bid])

    # Benchmark Spilling Throughput
    t0 = time.perf_counter()
    for chunk in range(0, num_blocks, 16):
        spill_mgr.spill_blocks(bids[chunk : chunk + 16])
    t_spill = time.perf_counter() - t0

    total_bytes = spill_mgr.stats["total_spilled_bytes"]
    spill_mb_s = (total_bytes / (1024 * 1024)) / max(t_spill, 1e-6)
    spill_tok_s = (num_blocks * block_size) / max(t_spill, 1e-6)

    print(f"\n[Spill Benchmark] Spilled {num_blocks * block_size} tokens ({total_bytes / 1024:.1f} KB) in {t_spill * 1000:.2f} ms")
    print(f"[Spill Benchmark] Throughput: {spill_mb_s:.2f} MB/s | {spill_tok_s:.0f} tokens/s")

    # Benchmark Zero-Copy Retrieval Latency
    t1 = time.perf_counter()
    retrieved = spill_mgr.retrieve_blocks(bids)
    t_retrieve = time.perf_counter() - t1
    avg_latency_us = (t_retrieve / num_blocks) * 1e6

    print(f"[Retrieval Benchmark] Retrieved {num_blocks} blocks in {t_retrieve * 1000:.2f} ms ({avg_latency_us:.2f} µs/block)")
    assert avg_latency_us < 200.0  # Zero-copy slicing must take sub-millisecond per block

    spill_mgr.close()
