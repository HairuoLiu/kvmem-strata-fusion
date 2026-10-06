"""
Core PyTorch/NumPy reference implementation of:
1. CanonicalAtomStore (CASA Content-Addressed KV Storage with K-Freeze)
2. Prefix Hash Chain (HiRadix / Merkle DAG Causal Invariant Deduplication)
3. Tensor Core-Compatible PagedAttention Execution Engine
4. Big-Tile Coalescing for GPUDirect Storage (GDS / cuFile)
5. Mixed-Precision Tier-Bias Softmax
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
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


def compute_prefix_hash(
    tokens: List[int],
    parent_hash: Optional[str] = None,
    model_id: str = "meta-llama/Llama-3-70b"
) -> str:
    """Compute causal prefix hash chain (RadixTree / Merkle DAG invariant).
    PrefixHash_i = SHA256(PrefixHash_{i-1} || Tokens_i || ModelID).
    Ensures that identical token blocks under divergent prefixes NEVER collide.
    """
    h = hashlib.sha256()
    if parent_hash is not None:
        h.update(parent_hash.encode("utf-8"))
    else:
        # Root node / system seed
        h.update(f"ROOT:{model_id}".encode("utf-8"))
    h.update(np.array(tokens, dtype=np.int64).tobytes())
    return h.hexdigest()


@dataclass
class KVBlock:
    """Logical atom block stored in canonical store."""
    block_id: str
    tokens: List[int]
    k_unrotated: np.ndarray  # [block_size, head_dim] un-rotated raw key vectors
    v: np.ndarray            # [block_size, head_dim] raw value vectors
    orig_pos_start: int      # starting logical position in origin sequence
    ref_count: int = 1
    tier: str = "FP8"        # FP8, INT4, INT2, MERGED
    dispersity: float = 0.0  # ||k_i - k_mean|| for bursty needle detection
    mean_k: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Prefix Hash Chain fields
    prefix_hash: Optional[str] = None
    parent_hash: Optional[str] = None
    k_canonical: Optional[np.ndarray] = None  # Pre-rotated at canonical pos for Tensor Core GEMM
    physical_page_id: int = -1

    def __post_init__(self):
        if len(self.mean_k) == 0:
            self.mean_k = np.mean(self.k_unrotated, axis=0)
            self.dispersity = float(np.mean(np.linalg.norm(self.k_unrotated - self.mean_k, axis=-1)))
        if self.k_canonical is None:
            # Pre-compute canonical rotated K at logical sequence positions
            # This enables standard batched GEMM / Tensor Core execution without per-token Q-remap
            k_rot = np.zeros_like(self.k_unrotated)
            for i in range(len(self.tokens)):
                k_rot[i] = apply_rope(self.k_unrotated[i], pos=self.orig_pos_start + i)
            self.k_canonical = k_rot


@dataclass
class StorageSuperTile:
    """Big-Tile (Macro-Chunk) coalesced for GPUDirect Storage (cuFile / NVMe DMA).
    Coalesces multiple logical atom pages (e.g. 16 pages x 32 tokens = 512 tokens, ~64KB+)
    to saturate PCIe Gen5 / NVMe RAID bandwidth.
    """
    tile_id: str
    block_ids: List[str]
    total_tokens: int
    size_bytes: int
    nvme_offset_bytes: int = 0


class CanonicalAtomStore:
    """CASA Store: Content-addressed, immutable K-Freeze store.
    Features:
    1. Prefix Hash Chain causal deduplication (replaces flawed flat token hashing)
    2. Tensor Core PagedAttention page-table based GEMM execution
    3. Big-tile coalescing for GPUDirect Storage / cuFile DMA saturation
    4. K-Freeze immutability: Key tensors written once, never re-rotated or moved.
    """
    def __init__(self, block_size: int = 32, head_dim: int = 64, model_id: str = "meta-llama/Llama-3-70b"):
        self.block_size = block_size
        self.head_dim = head_dim
        self.model_id = model_id
        self.blocks: Dict[str, KVBlock] = {}
        # Legacy flat token hash index (kept for backward compatibility)
        self.content_index: Dict[str, str] = {}
        # Prefix Hash Chain index: prefix_hash -> block_id
        self.prefix_chain_index: Dict[str, str] = {}
        # Physical page allocation pool (simulating GPU HBM PagedAttention physical pool)
        self.physical_k_pool: List[np.ndarray] = []
        self.physical_v_pool: List[np.ndarray] = []
        self.page_to_block_id: Dict[int, str] = {}

    def _hash_tokens(self, tokens: List[int]) -> str:
        """Legacy flat hash (only hashes token array). Note: Causally unsafe without prefix!"""
        h = hashlib.sha256()
        h.update(np.array(tokens, dtype=np.int64).tobytes())
        return h.hexdigest()

    def register_block(
        self,
        tokens: List[int],
        k_unrotated: np.ndarray,
        v: np.ndarray,
        orig_pos_start: int,
        tier: str = "FP8"
    ) -> str:
        """Legacy registration via flat hash (backward compatible with Phase 0 tests)."""
        content_hash = self._hash_tokens(tokens)
        if content_hash in self.content_index:
            existing_id = self.content_index[content_hash]
            self.blocks[existing_id].ref_count += 1
            return existing_id

        block_id = f"atom_{len(self.blocks):06d}"
        page_id = len(self.physical_k_pool)
        block = KVBlock(
            block_id=block_id,
            tokens=tokens,
            k_unrotated=k_unrotated,
            v=v,
            orig_pos_start=orig_pos_start,
            tier=tier,
            physical_page_id=page_id
        )
        self.blocks[block_id] = block
        self.content_index[content_hash] = block_id
        self.physical_k_pool.append(block.k_canonical)
        self.physical_v_pool.append(block.v)
        self.page_to_block_id[page_id] = block_id
        return block_id

    def register_prefix_block(
        self,
        tokens: List[int],
        k_unrotated: np.ndarray,
        v: np.ndarray,
        orig_pos_start: int,
        parent_hash: Optional[str] = None,
        tier: str = "FP8"
    ) -> Tuple[str, str]:
        """Causal prefix hash chain registration.
        Returns: (block_id, current_prefix_hash).
        Guarantees:
        - If prefix_hash matches, entire causal history matches, bit-exact KV is reused.
        - If previous tokens diverge, prefix_hash diverges, preventing semantic contamination.
        """
        curr_hash = compute_prefix_hash(tokens, parent_hash=parent_hash, model_id=self.model_id)
        if curr_hash in self.prefix_chain_index:
            existing_id = self.prefix_chain_index[curr_hash]
            self.blocks[existing_id].ref_count += 1
            return existing_id, curr_hash

        block_id = f"atom_{len(self.blocks):06d}"
        page_id = len(self.physical_k_pool)
        block = KVBlock(
            block_id=block_id,
            tokens=tokens,
            k_unrotated=k_unrotated,
            v=v,
            orig_pos_start=orig_pos_start,
            prefix_hash=curr_hash,
            parent_hash=parent_hash,
            tier=tier,
            physical_page_id=page_id
        )
        self.blocks[block_id] = block
        self.prefix_chain_index[curr_hash] = block_id
        self.physical_k_pool.append(block.k_canonical)
        self.physical_v_pool.append(block.v)
        self.page_to_block_id[page_id] = block_id
        return block_id, curr_hash

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
            scores_block = np.zeros(self.block_size, dtype=np.float64)
            for i in range(self.block_size):
                key_pos = blk.orig_pos_start + i
                # CASA Q-remap: q' = R_{query_pos - key_pos} q_raw
                q_remapped = apply_rope(q_raw, pos=query_logical_pos - key_pos)
                score = np.dot(q_remapped, blk.k_unrotated[i]) / np.sqrt(self.head_dim)
                
                if tier_biases:
                    if blk.tier in tier_biases:
                        score += tier_biases[blk.tier]
                    elif b_id in tier_biases:
                        score += tier_biases[b_id]
                scores_block[i] = score

            all_scores.append(scores_block)
            all_values.append(blk.v)

        concat_scores = np.concatenate(all_scores, axis=0)
        concat_values = np.concatenate(all_values, axis=0)

        exp_scores = np.exp(concat_scores - np.max(concat_scores))
        attn_weights = exp_scores / np.sum(exp_scores)
        context = np.dot(attn_weights, concat_values)
        return attn_weights, context

    def compute_paged_attention_gemm(
        self,
        q_raw: np.ndarray,
        query_logical_pos: int,
        page_table: List[int],
        tier_biases: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PagedAttention via Tensor Core-compatible batched GEMM tiles.
        
        Hardware Alignment Theorem:
        - Query q is rotated ONCE at sequence generation pos: q_rot = R_{query_pos} q_raw.
        - Canonical Keys K_canonical are stored in physical pages pre-rotated by their
          canonical sequence positions: K_canonical = R_{key_pos} k_raw.
        - The attention logit is computed via pure 2D GEMM tile contraction:
            logits_tile = q_rot @ K_page.T
          WITHOUT any per-token RoPE branching inside the kernel!
        - The Page Table provides logical-to-physical address translation.
        - Mathematical Equivalence:
            (R_m q)^T (R_n k) == q^T R_{m-n} k == CASA Q-remap == Ground Truth Attention.
        - Dynamic Tier-Bias Softmax:
            Applies tier_biases (from UBBA dynamic budget allocation, supporting either
            tier-level or block_id-level maps) per physical page.
        """
        q_rot = apply_rope(q_raw, pos=query_logical_pos)
        all_scores = []
        all_values = []

        for page_idx in page_table:
            k_page = self.physical_k_pool[page_idx]  # [block_size, head_dim]
            v_page = self.physical_v_pool[page_idx]  # [block_size, head_dim]

            # Pure Tensor Core GEMM matrix-vector multiplication tile
            # (block_size, head_dim) . (head_dim,) -> (block_size,)
            scores_tile = np.dot(k_page, q_rot) / np.sqrt(self.head_dim)
            if tier_biases and page_idx in self.page_to_block_id:
                blk = self.blocks[self.page_to_block_id[page_idx]]
                if blk.tier in tier_biases:
                    scores_tile = scores_tile + tier_biases[blk.tier]
                elif blk.block_id in tier_biases:
                    scores_tile = scores_tile + tier_biases[blk.block_id]
            all_scores.append(scores_tile)
            all_values.append(v_page)

        concat_scores = np.concatenate(all_scores, axis=0)
        concat_values = np.concatenate(all_values, axis=0)

        exp_scores = np.exp(concat_scores - np.max(concat_scores))
        attn_weights = exp_scores / np.sum(exp_scores)
        context = np.dot(attn_weights, concat_values)
        return attn_weights, context

    def coalesce_big_tiles(
        self,
        block_ids: List[str],
        pages_per_tile: int = 16
    ) -> List[StorageSuperTile]:
        """Coalesce fine-grained logical atom blocks into Big-Tile Macro-Chunks for GDS/cuFile.
        E.g. 16 blocks x (32 tokens * 64 dim * 2 bytes FP16 * 2 KV) = 131,072 bytes (128 KB),
        comfortably exceeding cuFile / PCIe Gen5 64 KB saturation threshold.
        """
        tiles = []
        bytes_per_token = self.head_dim * 2 * 2  # FP16 (2B) for both K and V
        bytes_per_block = self.block_size * bytes_per_token

        for i in range(0, len(block_ids), pages_per_tile):
            chunk_ids = block_ids[i:i + pages_per_tile]
            total_tokens = len(chunk_ids) * self.block_size
            total_bytes = len(chunk_ids) * bytes_per_block
            tile = StorageSuperTile(
                tile_id=f"super_tile_{len(tiles):04d}",
                block_ids=chunk_ids,
                total_tokens=total_tokens,
                size_bytes=total_bytes,
                nvme_offset_bytes=len(tiles) * pages_per_tile * bytes_per_block
            )
            tiles.append(tile)
        return tiles
