"""
Core PyTorch/NumPy reference implementation of:
1. CanonicalAtomStore (CASA Content-Addressed KV Storage with K-Freeze)
2. Q-Remap FlashAttention Score Emulator
3. Mean-K Exact & Dispersity-Check Indexing
4. Mixed-Precision Tier-Bias Softmax
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib
import numpy as np


def apply_rope(x: np.ndarray, pos: int, base: float = 10000.0) -> np.ndarray:
    """Standard 2D RoPE rotation for (..., dim) vector."""
    dim = x.shape[-1]
    assert dim % 2 == 0
    idx = np.arange(0, dim // 2)
    theta = pos * (base ** (-2.0 * idx / dim))
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    out = np.zeros_like(x)
    out[..., 0::2] = x_even * cos_t - x_odd * sin_t
    out[..., 1::2] = x_even * sin_t + x_odd * cos_t
    return out


@dataclass
class KVBlock:
    """32-token logical atom block stored in canonical store."""
    block_id: str
    tokens: List[int]
    k_unrotated: np.ndarray  # [32, head_dim] un-rotated raw key vectors
    v: np.ndarray            # [32, head_dim] raw value vectors
    orig_pos_start: int      # starting logical position in origin sequence
    ref_count: int = 1
    tier: str = "FP8"        # FP8, INT4, INT2, MERGED
    dispersity: float = 0.0  # ||k_i - k_mean|| for bursty needle detection
    mean_k: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self):
        if len(self.mean_k) == 0:
            self.mean_k = np.mean(self.k_unrotated, axis=0)
            self.dispersity = float(np.mean(np.linalg.norm(self.k_unrotated - self.mean_k, axis=-1)))


class CanonicalAtomStore:
    """CASA Store: Content-addressed, immutable K-Freeze store.
    K is never modified or re-rotated once registered.
    """
    def __init__(self, block_size: int = 32, head_dim: int = 64):
        self.block_size = block_size
        self.head_dim = head_dim
        self.blocks: Dict[str, KVBlock] = {}
        self.content_index: Dict[str, str] = {}  # sha256(tokens) -> block_id

    def _hash_tokens(self, tokens: List[int]) -> str:
        h = hashlib.sha256()
        h.update(np.array(tokens, dtype=np.int64).tobytes())
        return h.hexdigest()

    def register_block(self, tokens: List[int], k_unrotated: np.ndarray, v: np.ndarray, orig_pos_start: int) -> str:
        """Register or dedup block (L0 HiRadix/Content dedup)."""
        content_hash = self._hash_tokens(tokens)
        if content_hash in self.content_index:
            existing_id = self.content_index[content_hash]
            self.blocks[existing_id].ref_count += 1
            return existing_id

        block_id = f"atom_{len(self.blocks):06d}"
        block = KVBlock(
            block_id=block_id,
            tokens=tokens,
            k_unrotated=k_unrotated,
            v=v,
            orig_pos_start=orig_pos_start
        )
        self.blocks[block_id] = block
        self.content_index[content_hash] = block_id
        return block_id

    def compute_attention(
        self,
        q_raw: np.ndarray,
        query_logical_pos: int,
        block_ids: List[str],
        tier_biases: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute exact attention with CASA Q-remap.
        Instead of re-rotating K to new compact slots:
        q is transiently rotated to (query_logical_pos - key_orig_pos).
        """
        all_scores = []
        all_values = []

        for b_id in block_ids:
            blk = self.blocks[b_id]
            # Key logical positions in sequence: orig_pos_start + offset
            scores_block = np.zeros(self.block_size, dtype=np.float64)
            for i in range(self.block_size):
                key_pos = blk.orig_pos_start + i
                # CASA Q-remap: q' = R_{query_pos - key_pos} q_raw
                q_remapped = apply_rope(q_raw, pos=query_logical_pos - key_pos)
                score = np.dot(q_remapped, blk.k_unrotated[i]) / np.sqrt(self.head_dim)
                
                # Apply tier-bias if mixed precision is used
                if tier_biases and blk.tier in tier_biases:
                    score += tier_biases[blk.tier]
                scores_block[i] = score

            all_scores.append(scores_block)
            all_values.append(blk.v)

        concat_scores = np.concatenate(all_scores, axis=0)
        concat_values = np.concatenate(all_values, axis=0)

        # Numerically stable Softmax
        exp_scores = np.exp(concat_scores - np.max(concat_scores))
        attn_weights = exp_scores / np.sum(exp_scores)

        # Attended context output
        context = np.dot(attn_weights, concat_values)
        return attn_weights, context
