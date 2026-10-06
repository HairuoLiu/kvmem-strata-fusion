"""
Unit and Mechanism Verification Tests for Plan 01 (IFR: Invertible Fidelity-bound Retrieval)
Covers:
1. LSE Caching and Unnormalized Softmax Ranking Equivalence (Eq. 10).
2. L1 IVF-over-Mean-K coarse Voronoi clustering and candidate retrieval.
3. Anti-collapse intra-block dispersion check and doublet centroid splitting (resolving rank inversion).
4. Full two-tier Dedup-Then-Route pipeline with Exact bypass (|T| <= tau).
5. U-E-F-C Statistical Evaluation Gate with cluster-robust Wilson CI (D_eff = 1.9).
"""

import numpy as np
import pytest
from kvmem_fusion.ifr import (
    IFRBlock,
    LSECacheTracker,
    AntiCollapseSplitter,
    IVFMeanKIndex,
    IFRRetriever,
    UEFCEvaluationGate,
    compute_deroped_mean,
)


def test_lse_caching_unnormalized_ranking_equivalence():
    """
    Verify Eq. (10) derivation:
    argmax_{j in C} exp(s_j) / Z == argmax_{j in C} s_j, where Z = sum_{j in C_all} exp(s_j).
    Sorting by raw unnormalized logits is mathematically and rank-order identical
    to sorting by normalized softmax probabilities.
    """
    dim = 64
    n_candidates = 500
    top_k = 16
    np.random.seed(42)

    tracker = LSECacheTracker(head_dim=dim)
    query = np.random.randn(dim)
    query /= np.linalg.norm(query)

    candidates = np.random.randn(n_candidates, dim)
    candidates /= np.linalg.norm(candidates, axis=-1, keepdims=True)

    # 1. Unnormalized logits
    logits = tracker.compute_logits(query, candidates)

    # 2. Normalized Softmax Probabilities
    lse, z = tracker.compute_lse(logits)
    softmax_probs = np.exp(logits - lse)

    # 3. Rank via unnormalized logits
    ranked_unnormalized = tracker.rank_unnormalized(logits, top_k)

    # 4. Rank via normalized softmax
    ranked_softmax = np.argsort(-softmax_probs)[:top_k]

    # Verify identical ranking order across all top_k positions
    np.testing.assert_array_equal(
        ranked_unnormalized,
        ranked_softmax,
        err_msg="Unnormalized logit ranking diverged from true softmax probability ranking!"
    )

    # Verify log-sum-exp fidelity and discarded mass calculation
    discarded_mass = tracker.compute_discarded_mass(logits, ranked_unnormalized, full_lse=lse)
    retained_mass = np.sum(softmax_probs[ranked_unnormalized])

    assert abs((1.0 - retained_mass) - discarded_mass) < 1e-6
    print(f"\n✓ Eq.(10) LSE Equivalence confirmed: Top-{top_k} retained mass={retained_mass:.4f}, discarded={discarded_mass:.4f}")


def test_anti_collapse_dispersion_and_doublet_split():
    """
    Verify Meta FAIR / R3 Needle-in-a-Haystack Rank Inversion Pathology:
    Block A: 1 needle token (cos=1.0) + 31 background tokens clustered around background centroid (cos=0)
             -> mean projection = 1/32 = 0.031.
    Block B: 32 diffuse weak tokens, each having projection cos=0.20 -> mean = 0.20.

    1. Without anti-collapse, Mean-K suffers rank inversion: Block B (0.20) > Block A (0.031),
       dropping the needle block despite its true attention logit peak.
    2. With AntiCollapseSplitter (dispersion > theta), Block A splits into doublet centroids:
       [needle_vec, residual_mean], restoring the needle vector to highest rank.
    """
    dim = 64
    np.random.seed(101)

    query = np.zeros(dim)
    query[0] = 1.0  # Unit direction along axis 0

    # Block A: 1 needle token aligned with query, 31 background tokens clustered near orthogonal background
    bg_center = 0.05 * np.random.randn(dim)
    bg_center[0] = 0.0  # background has 0 projection along query
    keys_a = np.zeros((32, dim))
    keys_a[0, 0] = 1.0  # Needle token: cos(q, k_0) = 1.0
    for i in range(1, 32):
        keys_a[i] = bg_center + 0.01 * np.random.randn(dim)
        keys_a[i, 0] = 0.0  # background orthogonal to query

    block_a = IFRBlock(
        block_id="block_needle_a",
        tokens=list(range(32)),
        keys=keys_a,
        values=np.random.randn(32, dim),
        orig_pos_start=0
    )

    # Block B: 32 diffuse weak tokens, each having projection 0.20 along axis 0
    keys_b = np.random.randn(32, dim) * 0.02
    keys_b[:, 0] = 0.20  # cos(q, k_i) = 0.20 for all 32 tokens

    block_b = IFRBlock(
        block_id="block_diffuse_b",
        tokens=list(range(32, 64)),
        keys=keys_b,
        values=np.random.randn(32, dim),
        orig_pos_start=32
    )

    # 1. Check Naive Mean-K scores:
    score_mean_a = np.dot(block_a.mean_k, query)
    score_mean_b = np.dot(block_b.mean_k, query)
    print(f"\nNaive Mean-K score Block A: {score_mean_a:.4f} (diluted by 1/32)")
    print(f"Naive Mean-K score Block B: {score_mean_b:.4f}")

    # Rank inversion occurs: B ranks ahead of A in naive Mean-K!
    assert score_mean_b > score_mean_a, "Expected rank inversion under naive Mean-K"
    assert abs(score_mean_a - 1.0 / 32.0) < 1e-4

    # 2. Check Intra-block dispersion (dev_meta = max_i ||k_i - mean_k||):
    # Block A has high dispersion due to outlier needle (~ 0.97)
    # Block B has low dispersion (~ 0.05)
    assert block_a.dispersion > 0.85
    assert block_b.dispersion < 0.85

    # 3. Apply Anti-collapse doublet splitting
    splitter = AntiCollapseSplitter(dispersion_threshold=0.85)
    is_split_a, doublets_a = splitter.inspect_and_split(block_a)
    is_split_b, doublets_b = splitter.inspect_and_split(block_b)

    assert is_split_a is True, "Block A must trigger anti-collapse doublet split"
    assert is_split_b is False, "Block B should remain single centroid"
    assert len(doublets_a) == 2, "Doublet must yield 2 sub-centroids (needle + residual)"

    # Sub-centroid 0 is the needle vector
    needle_score = np.dot(doublets_a[0], query)
    print(f"Doublet Needle score: {needle_score:.4f} vs Block B: {score_mean_b:.4f}")

    # Rank inversion is cured: Needle sub-centroid dominates Block B
    assert needle_score > score_mean_b
    assert abs(needle_score - 1.0) < 1e-4
    print("✓ Anti-collapse dispersion check and doublet splitting CONFIRMED.")


def test_ivf_clustering_and_probe_recall():
    """
    Verify L1 IVF-over-Mean-K coarse clustering:
    - Verifies building Voronoi cells over de-RoPEd mean-K keys.
    - Probing top clusters retrieves candidate blocks with high recall.
    """
    dim = 64
    n_blocks = 200
    n_clusters = 10
    n_probe = 4
    np.random.seed(202)

    blocks = []
    for b in range(n_blocks):
        # Generate block with cluster bias
        cluster_center = np.random.randn(dim)
        cluster_center /= np.linalg.norm(cluster_center)
        keys = cluster_center + 0.1 * np.random.randn(32, dim)
        blk = IFRBlock(
            block_id=f"blk_{b:04d}",
            tokens=list(range(b * 32, (b + 1) * 32)),
            keys=keys,
            values=np.random.randn(32, dim),
            orig_pos_start=b * 32
        )
        blocks.append(blk)

    index = IVFMeanKIndex(dim=dim, n_clusters=n_clusters, n_probe=n_probe)
    index.train_and_index(blocks)

    assert index.is_trained
    assert len(index.centroids) == n_clusters

    # Query towards block 0's mean-k
    query = blocks[0].mean_k.copy()
    candidates = index.probe_candidates(query)

    # Block 0 must be in probed candidate set
    assert blocks[0].block_id in candidates
    # Number of candidates is a fraction of total blocks (sub-linear search)
    assert len(candidates) < n_blocks
    print(f"\n✓ IVF Index Probing: Probed {len(candidates)}/{n_blocks} candidates with target block hit.")


def test_ifr_two_tier_retriever_pipeline():
    """
    Verify full IFR pipeline:
    - L0 deduplication collapses duplicate blocks.
    - Exact bypass activates when unique blocks |T| <= tau.
    - IVF probe activates when unique blocks |T| > tau.
    - Fidelity cliff bound (rho <= 0.36) is satisfied when relevant blocks are retrieved.
    """
    dim = 64
    tau_exact = 10
    retriever = IFRRetriever(
        dim=dim,
        tau_exact_bypass=tau_exact,
        n_clusters=8,
        n_probe=4,
        target_top_k=4
    )
    np.random.seed(303)

    query = np.zeros(dim)
    query[0] = 1.0  # Query along axis 0

    # Register 8 blocks (should trigger exact bypass because 8 <= tau_exact)
    for i in range(8):
        keys = 0.05 * np.random.randn(32, dim)
        blk = IFRBlock(
            block_id=f"exact_blk_{i}",
            tokens=list(range(i * 32, (i + 1) * 32)),
            keys=keys,
            values=np.random.randn(32, dim),
            orig_pos_start=i * 32
        )
        retriever.add_block(blk)

    selected_exact, meta_exact = retriever.retrieve(query)
    assert meta_exact["bypassed_exact"] == 1.0
    assert len(selected_exact) == 4

    # Add 4 target relevant blocks (high attention logit, matching KVMem A4 top-8 mass distribution)
    for i in range(8, 12):
        keys = 0.05 * np.random.randn(32, dim)
        keys[:, 0] = 28.0  # (q . k) / sqrt(d) = 3.5, exp(3.5) captures ~75% attention mass
        blk = IFRBlock(
            block_id=f"target_blk_{i}",
            tokens=list(range(i * 32, (i + 1) * 32)),
            keys=keys,
            values=np.random.randn(32, dim),
            orig_pos_start=i * 32
        )
        retriever.add_block(blk)

    # Add background distractors (orthogonal to query)
    for i in range(12, 40):
        keys = 0.05 * np.random.randn(32, dim)
        keys[:, 0] = 0.0  # Irrelevant
        blk = IFRBlock(
            block_id=f"distractor_blk_{i}",
            tokens=list(range(i * 32, (i + 1) * 32)),
            keys=keys,
            values=np.random.randn(32, dim),
            orig_pos_start=i * 32
        )
        retriever.add_block(blk)

    retriever.build_index()
    selected_ivf, meta_ivf = retriever.retrieve(query)
    assert meta_ivf["bypassed_exact"] == 0.0
    assert meta_ivf["candidate_count"] > 4

    # The 4 target blocks should be retrieved at the top
    for i in range(8, 12):
        assert f"target_blk_{i}" in selected_ivf

    # Retained mass is high, discarded mass is below the fidelity cliff (rho_max = 0.36)
    assert meta_ivf["discarded_mass"] <= UEFCEvaluationGate.RHO_MAX
    print(f"\n✓ IFR Two-Tier Pipeline confirmed: Exact Bypass and IVF routing both functional, discarded mass={meta_ivf['discarded_mass']:.4f}")


def test_uefc_evaluation_gate_and_wilson_ci():
    """
    Verify U-E-F-C statistical contract:
    - Fidelity (F): Top-1 agreement >= 97%, discarded mass <= 0.36, utility gap <= 1pp.
    - Efficiency (E): retrieval <= 350ms.
    - Cost (C): index <= 4 GiB.
    - Utility (U): paired Wilson CI lower bound.
    - Tests Wilson CI with cluster design effect D_eff = 1.9.
    """
    # 1. Passing Scenario: 98 out of 100 top-1 matches, 310ms latency, 3.8 GiB index
    gate_result_pass = UEFCEvaluationGate.evaluate_gate(
        top1_matches=98,
        total_eval_queries=100,
        discarded_masses=[0.12, 0.18, 0.09, 0.15],
        retrieval_latency_ms=310.0,
        index_footprint_gib=3.8,
        utility_gap=0.005,
        paired_success_diff=0.03,
        design_effect=1.9
    )
    assert gate_result_pass["all_passed"] is True
    assert gate_result_pass["gate_f"]["passed"] is True
    assert gate_result_pass["gate_e"]["passed"] is True
    assert gate_result_pass["gate_c"]["passed"] is True
    assert gate_result_pass["gate_u"]["passed"] is True

    # Check Wilson CI bounds
    ci_low, ci_high = gate_result_pass["gate_f"]["top1_ci_95"]
    assert gate_result_pass["gate_f"]["top1_agreement"] == 0.98
    assert ci_low > 0.85
    assert ci_high <= 1.0

    # 2. Failing Scenario: Top-1 agreement drops to 92% (violates Fidelity gate)
    gate_result_fail = UEFCEvaluationGate.evaluate_gate(
        top1_matches=92,
        total_eval_queries=100,
        discarded_masses=[0.42, 0.38],  # exceeds rho_max 0.36
        retrieval_latency_ms=450.0,     # exceeds 350ms
        index_footprint_gib=5.2,        # exceeds 4 GiB
        utility_gap=0.025,              # exceeds 1pp
        paired_success_diff=-0.01,
        design_effect=1.9
    )
    assert gate_result_fail["all_passed"] is False
    assert gate_result_fail["gate_f"]["passed"] is False
    assert gate_result_fail["gate_e"]["passed"] is False
    assert gate_result_fail["gate_c"]["passed"] is False
    assert gate_result_fail["gate_u"]["passed"] is False

    print(f"✓ U-E-F-C Evaluation Gate & Cluster-Robust Wilson CI CONFIRMED.")
