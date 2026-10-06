"""
Unit and integration tests for CASA CanonicalAtomStore:
1. Deduplication and ref-counting.
2. Full context Attention vs CASA Q-remap Attention equivalence (bit-exact / precision check).
3. Mixed-tier Softmax output preservation with tier-bias.
"""

import numpy as np
import pytest
from kvmem_fusion.core import CanonicalAtomStore, apply_rope


def test_casa_store_deduplication():
    store = CanonicalAtomStore(block_size=32, head_dim=64)
    tokens_1 = list(range(32))
    k1 = np.random.randn(32, 64)
    v1 = np.random.randn(32, 64)

    id1 = store.register_block(tokens_1, k1, v1, orig_pos_start=0)
    # Register same tokens
    id2 = store.register_block(tokens_1, k1 + 0.1, v1 + 0.1, orig_pos_start=100)

    assert id1 == id2
    assert store.blocks[id1].ref_count == 2
    assert len(store.blocks) == 1


def test_casa_attention_exact_equivalence():
    """Verify CASA Q-remap attention produces the exact same output as standard full-context attention."""
    head_dim = 64
    block_size = 32
    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    np.random.seed(42)

    # Register 4 blocks (128 tokens total)
    n_blocks = 4
    block_ids = []
    raw_keys_all = []
    raw_values_all = []

    for b in range(n_blocks):
        tokens = list(range(b * block_size, (b + 1) * block_size))
        k = np.random.randn(block_size, head_dim)
        v = np.random.randn(block_size, head_dim)
        raw_keys_all.append(k)
        raw_values_all.append(v)
        b_id = store.register_block(tokens, k, v, orig_pos_start=b * block_size)
        block_ids.append(b_id)

    raw_keys_all = np.concatenate(raw_keys_all, axis=0)
    raw_values_all = np.concatenate(raw_values_all, axis=0)

    # Query at position 500
    q_pos = 500
    q_raw = np.random.randn(head_dim)

    # 1. Standard Full Attention Ground Truth
    q_std = apply_rope(q_raw, pos=q_pos)
    std_scores = np.zeros(n_blocks * block_size)
    for i in range(n_blocks * block_size):
        k_std = apply_rope(raw_keys_all[i], pos=i)
        std_scores[i] = np.dot(q_std, k_std) / np.sqrt(head_dim)

    exp_std = np.exp(std_scores - np.max(std_scores))
    weights_std = exp_std / np.sum(exp_std)
    context_std = np.dot(weights_std, raw_values_all)

    # 2. CASA Q-remap Attention
    weights_casa, context_casa = store.compute_attention(
        q_raw=q_raw,
        query_logical_pos=q_pos,
        block_ids=block_ids
    )

    # Assert exact equivalence
    max_weight_diff = np.max(np.abs(weights_std - weights_casa))
    max_context_diff = np.max(np.abs(context_std - context_casa))

    print(f"\nMax Weight Diff: {max_weight_diff:.2e}, Max Context Diff: {max_context_diff:.2e}")
    assert max_weight_diff < 1e-12
    assert max_context_diff < 1e-12
