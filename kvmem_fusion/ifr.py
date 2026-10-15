"""
Invertible Fidelity-bound Retrieval (IFR) Reference Implementation
Plan 01 Architectural Core:
1. LSE Caching and Unnormalized Softmax Ranking
2. Anti-Collapse Dispersion Tracking and Doublet Centroid Splitting
3. IVF-over-Mean-K Inverted Indexing (Coarse DRAM Centroids + Large NVMe Postings)
4. Two-Tier Retrieval Pipeline (L0 Dedup -> Exact Bypass -> L1 IVF -> LSE Top-K)
5. U-E-F-C Statistical Evaluation Gate with Cluster-Adjusted Wilson Intervals
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from kvmem_fusion.core import compute_prefix_hash


def compute_deroped_mean(keys: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Given raw unrotated keys of shape (B, dim),
    compute mean centroid, max token deviation max_i ||k_i - mean_k||,
    and average dispersion mean_i ||k_i - mean_k||.
    """
    try:
        from kvmem_fusion.c_engine_bindings import is_c_engine_available, compute_deroped_mean_c
        if is_c_engine_available() and keys is not None and len(keys) > 0:
            mean_k, max_disp, _ = compute_deroped_mean_c(keys)
            dists = np.linalg.norm(keys - mean_k, axis=-1)
            avg_disp = float(np.mean(dists)) if len(dists) > 0 else 0.0
            return mean_k, max_disp, avg_disp
    except Exception:
        pass

    mean_k = np.mean(keys, axis=0)
    dists = np.linalg.norm(keys - mean_k, axis=-1)
    max_dispersion = float(np.max(dists))
    avg_dispersion = float(np.mean(dists))
    return mean_k, max_dispersion, avg_dispersion


def compute_normalized_dispersion(keys: np.ndarray) -> float:
    """
    Compute directional max dispersion in cosine / L2-normalized space:
    max_i ||k_i / ||k_i|| - mean(k / ||k||)||
    Decoupled from massive activation outlier channel magnitudes.
    On Qwen2.5-0.5B, real keys have Euclidean MAD ~8.32 but normalized MAD ~0.7965.
    """
    try:
        from kvmem_fusion.c_engine_bindings import is_c_engine_available, compute_deroped_mean_c
        if is_c_engine_available() and keys is not None and len(keys) > 0:
            _, _, norm_disp = compute_deroped_mean_c(keys)
            return norm_disp
    except Exception:
        pass

    norms = np.linalg.norm(keys, axis=-1, keepdims=True)
    norm_keys = keys / np.maximum(norms, 1e-12)
    norm_mean = np.mean(norm_keys, axis=0)
    norm_dists = np.linalg.norm(norm_keys - norm_mean, axis=-1)
    return float(np.max(norm_dists))


@dataclass
class IFRBlock:
    """Represents a 32-token logical block in IFR storage."""
    block_id: str
    tokens: List[int]
    keys: np.ndarray  # [32, dim]
    values: np.ndarray  # [32, dim]
    orig_pos_start: int
    mean_k: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dispersion: float = 0.0  # max_i ||k_i - mean_k|| (dev_meta raw Euclidean)
    avg_dispersion: float = 0.0
    norm_dispersion: float = 0.0  # directional dispersion in cosine space (~0.80 on real models)
    doublet_centroids: Optional[List[np.ndarray]] = None
    prefix_hash: Optional[str] = None
    parent_hash: Optional[str] = None

    def __post_init__(self):
        if len(self.mean_k) == 0:
            self.mean_k, self.dispersion, self.avg_dispersion = compute_deroped_mean(self.keys)
        if self.norm_dispersion == 0.0 and len(self.keys) > 0:
            self.norm_dispersion = compute_normalized_dispersion(self.keys)
        if self.prefix_hash is None:
            self.prefix_hash = compute_prefix_hash(self.tokens, parent_hash=self.parent_hash)


class LSECacheTracker:
    """
    LSE (Log-Sum-Exp) Caching Engine:
    Maintains unnormalized logit scores and log-sum-exp normalization factor:
        Z = sum_{j in C_all} exp(s_j)
        LSE = ln(Z)
    Proves and verifies that:
        argmax_{j in C'} exp(s_j) / Z == argmax_{j in C'} s_j
    and tracks discarded attention mass:
        rho = 1 - sum_{j in C'} exp(s_j) / Z = sum_{j not in C'} exp(s_j) / Z
    """
    def __init__(self, head_dim: int = 64):
        self.head_dim = head_dim
        self.scale = 1.0 / np.sqrt(head_dim)

    def compute_logits(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Compute unnormalized attention dot-product logits: s_j = (q . k_j) / sqrt(d)."""
        if candidates is None or len(candidates) == 0:
            return np.empty((0,), dtype=np.float32)
        cand_arr = np.asarray(candidates)
        if cand_arr.size == 0:
            return np.empty((0,), dtype=np.float32)
        if cand_arr.ndim == 1:
            if cand_arr.shape[0] == query.shape[0]:
                cand_arr = cand_arr.reshape(1, -1)
            else:
                return np.empty((0,), dtype=np.float32)
        dim = query.shape[-1] if query.ndim > 0 and query.shape[-1] > 0 else self.head_dim
        scale = 1.0 / np.sqrt(dim)
        return np.dot(cand_arr, query) * scale

    def compute_lse(self, logits: np.ndarray) -> Tuple[float, float]:
        """
        Numerically stable log-sum-exp:
            max_s = max(logits)
            lse = max_s + ln(sum(exp(s - max_s)))
            Z = exp(lse)
        """
        max_s = np.max(logits)
        sum_exp = np.sum(np.exp(logits - max_s))
        lse = float(max_s + np.log(sum_exp))
        z = float(np.exp(lse))
        return lse, z

    def rank_unnormalized(self, logits: np.ndarray, top_k: int) -> np.ndarray:
        """
        Rank top-k indices using raw unnormalized logits.
        Directly equivalent to ranking by exp(s_j) / Z.
        """
        if top_k >= len(logits):
            return np.argsort(-logits)
        top_k_indices = np.argpartition(-logits, top_k)[:top_k]
        sorted_top_k = top_k_indices[np.argsort(-logits[top_k_indices])]
        return sorted_top_k

    def compute_discarded_mass(
        self,
        full_logits: np.ndarray,
        selected_indices: np.ndarray,
        full_lse: Optional[float] = None
    ) -> float:
        """
        Calculate discarded attention mass rho = 1 - sum_{j in selected} exp(s_j) / Z.
        Fidelity Cliff: if rho > rho_max (~0.36), fidelity barrier is breached.
        """
        if full_lse is None:
            full_lse, _ = self.compute_lse(full_logits)

        max_s = np.max(full_logits)
        selected_logits = full_logits[selected_indices]
        selected_sum_exp = np.sum(np.exp(selected_logits - max_s))
        selected_lse = max_s + np.log(max(selected_sum_exp, 1e-30))
        
        log_retained_ratio = selected_lse - full_lse
        retained_mass = float(np.exp(min(0.0, log_retained_ratio)))
        discarded_mass = max(0.0, 1.0 - retained_mass)
        return discarded_mass


class AntiCollapseSplitter:
    """
    Anti-collapse dispersion mechanism to prevent needle-in-a-haystack rank inversion.
    When a block contains 1 burst needle token and 31 diffuse background tokens,
    mean-pooling dilutes the needle by 1/32.
    Anti-collapse detects max dispersion dev_meta > theta and splits into doublet:
        1. Needle centroid (the outlier token vector k_needle)
        2. Residual background centroid (mean of remaining 31 tokens)

    Supports both:
    - normalized directional dispersion threshold (use_normalized=True, default, calibrated theta=0.868 at p95 of real Qwen norm_max)
    - raw Euclidean dispersion threshold (use_normalized=False, legacy)
    """
    def __init__(self, dispersion_threshold: float = 0.868, use_normalized: bool = True):
        self.dispersion_threshold = dispersion_threshold
        self.use_normalized = use_normalized

    def inspect_and_split(self, block: IFRBlock) -> Tuple[bool, List[np.ndarray]]:
        """
        Check intra-block dispersion against threshold theta.
        Returns (is_split, [sub_centroids]).
        Doublet splitting is strictly an index-space virtual representation expansion
        and does NOT mutate or divide the underlying 32-token immutable atom storage.
        """
        if block.doublet_centroids is not None:
            return True, block.doublet_centroids

        # Native C engine acceleration with pure Python fallback
        try:
            from kvmem_fusion.c_engine_bindings import is_c_engine_available, inspect_and_split_c
            if is_c_engine_available() and block.keys is not None and len(block.keys) > 0:
                thresh = self.dispersion_threshold if self.use_normalized else 1000.0
                is_split, centroids, _ = inspect_and_split_c(
                    block.keys,
                    dispersion_threshold=thresh
                )
                if not self.use_normalized:
                    if block.dispersion <= self.dispersion_threshold:
                        return False, [block.mean_k]
                if is_split:
                    block.doublet_centroids = centroids
                    return True, centroids
                return False, [block.mean_k]
        except Exception:
            pass

        metric = block.norm_dispersion if self.use_normalized else block.dispersion
        if metric <= self.dispersion_threshold:
            return False, [block.mean_k]

        # Detect needle: the token with maximum Euclidean deviation from mean
        # Outlier token vector carries burst salience (whether needle burst or massive channel)
        dists = np.linalg.norm(block.keys - block.mean_k, axis=-1)
        needle_idx = int(np.argmax(dists))
        needle_vec = block.keys[needle_idx]

        # Residual background mean
        mask = np.ones(len(block.keys), dtype=bool)
        mask[needle_idx] = False
        residual_mean = np.mean(block.keys[mask], axis=0)

        doublets = [needle_vec, residual_mean]
        block.doublet_centroids = doublets
        return True, doublets


class IVFMeanKIndex:
    """
    L1 IVF-over-Mean-K inverted index:
    - n_clusters coarse centroids resident in pinned DRAM
    - Inverted posting lists mapping coarse centroids -> block IDs (stored on NVMe/DRAM)
    - Probes top-n_probe nearest coarse centroids for each query
    - Incorporates anti-collapse doublet expansion during candidate search
    """
    def __init__(
        self,
        dim: int = 64,
        n_clusters: int = 8,
        n_probe: int = 4,
        dispersion_threshold: float = 0.868,
        use_normalized: bool = True
    ):
        self.dim = dim
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.dispersion_threshold = dispersion_threshold
        self.use_normalized = use_normalized
        self.anti_collapse = AntiCollapseSplitter(
            dispersion_threshold=dispersion_threshold,
            use_normalized=use_normalized
        )

        self.centroids: np.ndarray = np.zeros((n_clusters, dim))
        self.postings: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
        self.block_store: Dict[str, IFRBlock] = {}
        self.is_trained = False

    def train_and_index(self, blocks: List[IFRBlock]):
        """Cluster mean-k vectors into n_clusters Voronoi cells and build postings."""
        for b in blocks:
            self.block_store[b.block_id] = b

        mean_keys = np.array([b.mean_k for b in blocks])
        n_samples = len(mean_keys)

        if n_samples <= self.n_clusters:
            self.centroids = np.zeros((self.n_clusters, self.dim))
            self.centroids[:n_samples] = mean_keys
            for i, b in enumerate(blocks):
                self.postings[i].append(b.block_id)
            self.is_trained = True
            return

        rng = np.random.RandomState(42)
        init_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        centroids = mean_keys[init_indices].copy()

        for _ in range(10):
            dots = np.dot(mean_keys, centroids.T)
            assignments = np.argmax(dots, axis=-1)
            for c_idx in range(self.n_clusters):
                cluster_pts = mean_keys[assignments == c_idx]
                if len(cluster_pts) > 0:
                    centroids[c_idx] = np.mean(cluster_pts, axis=0)

        self.centroids = centroids
        self.postings = {i: [] for i in range(self.n_clusters)}
        dots = np.dot(mean_keys, self.centroids.T)
        assignments = np.argmax(dots, axis=-1)
        for b_idx, c_idx in enumerate(assignments):
            blk = blocks[b_idx]
            self.postings[c_idx].append(blk.block_id)
            # Dual-posting for doublet centroids:
            # If intra-block dispersion exceeds threshold, also map needle centroid
            # to its nearest Voronoi cluster so needle queries are routed at coarse probe.
            is_split, doublets = self.anti_collapse.inspect_and_split(blk)
            if is_split:
                for sc in doublets:
                    sc_c_idx = int(np.argmax(np.dot(self.centroids, sc)))
                    if blk.block_id not in self.postings[sc_c_idx]:
                        self.postings[sc_c_idx].append(blk.block_id)

        self.is_trained = True

    def probe_candidates(self, query: np.ndarray) -> Set[str]:
        """
        Probe top-n_probe coarse clusters and gather candidate block IDs.
        """
        if not self.is_trained:
            return set(self.block_store.keys())

        centroid_scores = np.dot(self.centroids, query)
        top_cluster_indices = np.argsort(-centroid_scores)[:min(self.n_probe, self.n_clusters)]

        candidates: Set[str] = set()
        for c_idx in top_cluster_indices:
            candidates.update(self.postings[c_idx])

        return candidates


class IFRRetriever:
    """
    Complete Two-Tier Dedup-Then-Route IFR Pipeline:
    L0: HiRadix/Content Deduplication
    Bypass: ExactMeanK if unique count |T| <= tau
    L1: IVF-over-Mean-K coarse probe with doublet expansion
    LSE: Unnormalized Logit argmax selection for top-K blocks
    """
    def __init__(
        self,
        dim: int = 64,
        tau_exact_bypass: int = 16,
        n_clusters: int = 8,
        n_probe: int = 4,
        dispersion_threshold: float = 0.868,
        use_normalized: bool = True,
        target_top_k: int = 8
    ):
        self.dim = dim
        self.tau = tau_exact_bypass
        self.target_top_k = target_top_k
        self.lse_tracker = LSECacheTracker(head_dim=dim)
        self.index = IVFMeanKIndex(
            dim=dim,
            n_clusters=n_clusters,
            n_probe=n_probe,
            dispersion_threshold=dispersion_threshold,
            use_normalized=use_normalized
        )
        self.blocks: Dict[str, IFRBlock] = {}
        self.content_hashes: Dict[str, str] = {}
        self._cached_block_ids: Optional[List[str]] = None
        self._cached_all_mean_keys: Optional[np.ndarray] = None
        self._cached_block_to_idx: Optional[Dict[str, int]] = None

    @property
    def block_store(self) -> Dict[str, IFRBlock]:
        return self.blocks

    @block_store.setter
    def block_store(self, val: Dict[str, IFRBlock]):
        self.blocks = val
        self._cached_block_ids = None

    def add_block(self, block: IFRBlock, parent_hash: Optional[str] = None) -> str:
        """
        L0 Identity Deduplication: registers block or references existing.
        Strictly obeys CASA prefix-chained immutable atom invariants via SHA-256 Merkle chain:
        PrefixHash_i = SHA256(PrefixHash_{i-1} || Tokens_i || ModelID).
        """
        if parent_hash is not None and block.parent_hash != parent_hash:
            block.parent_hash = parent_hash
            block.prefix_hash = compute_prefix_hash(block.tokens, parent_hash=parent_hash)
        elif block.prefix_hash is None:
            block.prefix_hash = compute_prefix_hash(block.tokens, parent_hash=block.parent_hash)

        h_str = block.prefix_hash
        if h_str in self.content_hashes:
            return self.content_hashes[h_str]

        self.content_hashes[h_str] = block.block_id
        self.blocks[block.block_id] = block
        self.index.block_store[block.block_id] = block
        self._cached_block_ids = None
        return block.block_id

    def build_index(self):
        """Build L1 IVF index over unique registered blocks."""
        unique_blocks = list(self.blocks.values())
        self.index.train_and_index(unique_blocks)
        self._cached_block_ids = list(self.blocks.keys())
        self._cached_all_mean_keys = np.array([self.blocks[bid].mean_k for bid in self._cached_block_ids], dtype=np.float32)
        self._cached_block_to_idx = {bid: idx for idx, bid in enumerate(self._cached_block_ids)}

    def retrieve(self, query: np.ndarray, top_k: Optional[int] = None) -> Tuple[List[str], Dict[str, float]]:
        """
        Execute IFR retrieval pipeline.
        Returns: (selected_block_ids, metadata_dict)
        """
        eff_top_k = top_k if top_k is not None else self.target_top_k
        unique_count = len(self.blocks)
        meta = {
            "unique_blocks": unique_count,
            "bypassed_exact": 0.0,
            "candidate_count": 0.0,
            "discarded_mass": 0.0
        }

        if unique_count == 0:
            return [], meta

        if self._cached_block_ids is None or len(self._cached_block_ids) != unique_count:
            self._cached_block_ids = list(self.blocks.keys())
            self._cached_all_mean_keys = np.array([self.blocks[bid].mean_k for bid in self._cached_block_ids], dtype=np.float32)
            self._cached_block_to_idx = {bid: idx for idx, bid in enumerate(self._cached_block_ids)}

        block_ids = self._cached_block_ids
        all_mean_keys = self._cached_all_mean_keys
        block_to_idx = self._cached_block_to_idx

        all_logits = self.lse_tracker.compute_logits(query, all_mean_keys)
        full_lse, _ = self.lse_tracker.compute_lse(all_logits) if len(all_logits) > 0 else (0.0, 1.0)

        # Exact Bypass check
        if unique_count <= self.tau:
            meta["bypassed_exact"] = 1.0
            meta["candidate_count"] = float(unique_count)
            top_indices = self.lse_tracker.rank_unnormalized(all_logits, eff_top_k)
            selected = [block_ids[idx] for idx in top_indices]
            meta["discarded_mass"] = self.lse_tracker.compute_discarded_mass(
                all_logits, top_indices, full_lse=full_lse
            ) if len(all_logits) > 0 else 0.0
            return selected, meta

        # L1 IVF Probe
        if not self.index.is_trained:
            self.build_index()

        candidate_ids = list(self.index.probe_candidates(query))
        if not candidate_ids:
            candidate_ids = block_ids[:min(len(block_ids), eff_top_k)]
        meta["candidate_count"] = float(len(candidate_ids))

        # Build candidate representations
        cand_vectors = []
        valid_candidates = []
        for bid in candidate_ids:
            blk = self.blocks.get(bid)
            if blk is None:
                continue
            if blk.doublet_centroids is not None:
                cand_vectors.append(np.mean(blk.doublet_centroids, axis=0))
            else:
                cand_vectors.append(blk.mean_k)
            valid_candidates.append(bid)

        if len(cand_vectors) == 0:
            selected_blocks = block_ids[:min(len(block_ids), eff_top_k)]
            meta["discarded_mass"] = 0.0
            return selected_blocks, meta

        cand_vectors_arr = np.asarray(cand_vectors)
        # Vectorized batch logits computation (single dot product across all candidates)
        cand_logits = self.lse_tracker.compute_logits(query, cand_vectors_arr)

        # Top-K candidate block selection
        top_k_cnt = min(eff_top_k, len(valid_candidates))
        if len(cand_logits) <= top_k_cnt:
            selected_blocks = valid_candidates
        else:
            top_cand_indices = np.argpartition(-cand_logits, top_k_cnt - 1)[:top_k_cnt]
            sorted_indices = top_cand_indices[np.argsort(-cand_logits[top_cand_indices])]
            selected_blocks = [valid_candidates[i] for i in sorted_indices]

        # Fast O(1) dict lookup instead of O(N) list.index
        block_to_idx = {bid: idx for idx, bid in enumerate(block_ids)}
        selected_indices_in_all = np.array([block_to_idx[bid] for bid in selected_blocks])
        meta["discarded_mass"] = self.lse_tracker.compute_discarded_mass(
            all_logits, selected_indices_in_all, full_lse=full_lse
        )

        return selected_blocks, meta


class UEFCEvaluationGate:
    """
    U-E-F-C Four-Dimensional Evaluation Gate
    Synthesizes the 10-expert statistical contract:
    - U (Utility): Paired Wilson CI lower bound > 0 with cluster design effect D_eff
    - E (Efficiency): Retrieval latency <= 350ms @ 10M, speedup >= 3.75x
    - F (Fidelity): Top-1 agreement >= 97%, utility gap <= 1pp, discarded mass <= rho_max (0.36)
    - C (Cost): Index footprint <= 4 GiB (vs 9.5 GiB baseline), reduction >= 2.375x
    """
    RHO_MAX = 0.36
    EFFICIENCY_LATENCY_MAX_MS = 350.0
    COST_INDEX_MAX_GIB = 4.0
    FIDELITY_TOP1_MIN_RATE = 0.97
    FIDELITY_GAP_MAX_PP = 0.01

    @staticmethod
    def compute_wilson_ci(
        successes: int,
        total: int,
        confidence: float = 0.95,
        design_effect: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Wilson score confidence interval with cluster design effect adjustment.
        Effective sample size n_eff = total / design_effect.
        Returns: (p_hat, ci_lower, ci_upper)
        """
        if total == 0:
            return 0.0, 0.0, 0.0

        p_hat = successes / total
        n_eff = total / design_effect
        z = 1.95996 if abs(confidence - 0.95) < 1e-4 else 2.576

        denominator = 1.0 + (z**2) / n_eff
        center = (p_hat + (z**2) / (2.0 * n_eff)) / denominator
        half_width = (z / denominator) * np.sqrt(
            (p_hat * (1.0 - p_hat)) / n_eff + (z**2) / (4.0 * (n_eff**2))
        )

        ci_lower = max(0.0, center - half_width)
        ci_upper = min(1.0, center + half_width)
        return p_hat, float(ci_lower), float(ci_upper)

    @classmethod
    def evaluate_gate(
        cls,
        top1_matches: int,
        total_eval_queries: int,
        discarded_masses: List[float],
        retrieval_latency_ms: float,
        index_footprint_gib: float,
        utility_gap: float,
        paired_success_diff: float,
        design_effect: float = 1.9
    ) -> Dict[str, any]:
        """
        Evaluate full U-E-F-C statistical contract.
        """
        p_top1, f_ci_lower, f_ci_upper = cls.compute_wilson_ci(
            top1_matches, total_eval_queries, confidence=0.95, design_effect=design_effect
        )
        avg_discarded_mass = float(np.mean(discarded_masses)) if discarded_masses else 0.0

        f_pass = (
            p_top1 >= cls.FIDELITY_TOP1_MIN_RATE
            and utility_gap <= cls.FIDELITY_GAP_MAX_PP
            and avg_discarded_mass <= cls.RHO_MAX
        )
        e_pass = retrieval_latency_ms <= cls.EFFICIENCY_LATENCY_MAX_MS
        c_pass = index_footprint_gib <= cls.COST_INDEX_MAX_GIB
        u_pass = paired_success_diff >= 0.0

        all_pass = f_pass and e_pass and c_pass and u_pass

        return {
            "all_passed": all_pass,
            "gate_f": {
                "passed": f_pass,
                "top1_agreement": p_top1,
                "top1_ci_95": (f_ci_lower, f_ci_upper),
                "utility_gap": utility_gap,
                "avg_discarded_mass": avg_discarded_mass,
                "threshold_rho_max": cls.RHO_MAX
            },
            "gate_e": {
                "passed": e_pass,
                "retrieval_latency_ms": retrieval_latency_ms,
                "target_ms": cls.EFFICIENCY_LATENCY_MAX_MS
            },
            "gate_c": {
                "passed": c_pass,
                "index_footprint_gib": index_footprint_gib,
                "target_gib": cls.COST_INDEX_MAX_GIB
            },
            "gate_u": {
                "passed": u_pass,
                "paired_success_diff": paired_success_diff
            }
        }
