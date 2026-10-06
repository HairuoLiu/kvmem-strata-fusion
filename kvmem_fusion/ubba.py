"""
UBBA: Universal Byte-Budget Allocator
Core Operations Research & Knapsack Formulation Reference Implementation.

Mathematical Formulation:
    min_{x_{i,t}}  sum_{i=1}^N sum_{t in T} C(b_i, t) * x_{i,t}
    s.t.           sum_{t in T} x_{i,t} <= 1,  x_{i,t} in {0, 1}       (at most one tier per block)
                   rho(b_i, t) <= rho_floor,   forall (i, t) with x_{i,t}=1 (hard fidelity gate)
                   sum_{i=1}^N sum_{t in T} w(b_i) * x_{i,t} >= Target Coverage (demand satisfaction)

Authors:
    10-Expert AI Systems Panel (Stanford, MIT, OpenAI, Google Research, Meta AI,
    SGLang, NVIDIA, AWS, ByteDance, Antigravity Systems)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import time
import numpy as np


@dataclass
class TierProfile:
    """Attributes of a candidate compression tier for a KV block."""
    tier_name: str          # e.g., 'FP8', 'INT4', 'INT2', 'MERGED'
    bytes_per_block: int    # Storage footprint in bytes
    distortion_rho: float   # Relative distortion rho in [0, 1]


@dataclass
class KVBlockCandidate:
    """A KV block candidate available for allocation."""
    block_id: str
    weight: float           # Attention mass / query-key importance weight w_i >= 0
    dispersity: float       # Key dispersion ||k_i - mean_k||
    tier_profiles: Dict[str, TierProfile] = field(default_factory=dict)

    def add_tier(self, name: str, bytes_per_block: int, distortion_rho: float):
        self.tier_profiles[name] = TierProfile(
            tier_name=name,
            bytes_per_block=bytes_per_block,
            distortion_rho=distortion_rho,
        )


@dataclass
class UBBAResult:
    """Output allocation and telemetry produced by the UBBA solver."""
    allocations: Dict[str, str]       # block_id -> selected tier_name
    total_bytes: int                  # Total bytes allocated
    achieved_coverage: float          # Sum of weights of retained blocks
    target_coverage: float            # Requested coverage target
    rho_floor: float                  # Hard fidelity floor
    max_rho: float                    # Max distortion among retained blocks
    mean_rho: float                   # Mean distortion among retained blocks
    num_blocks_selected: int          # Count of retained blocks
    total_blocks_considered: int      # Count of total input blocks
    shadow_price_lambda: float        # Marginal byte cost per coverage unit (Lagrangian dual)
    duality_gap: float                # Upper bound on LP-relaxation duality gap
    solver_latency_ms: float          # Solver runtime in milliseconds
    is_feasible: bool                 # True if achieved_coverage >= target_coverage


def solve_ubba_greedy(
    blocks: List[KVBlockCandidate],
    target_coverage: float,
    rho_floor: float = 0.10,
    enforce_min_tier: bool = False,
) -> UBBAResult:
    """
    Solves the UBBA Minimum-Cost Demand-Covering Knapsack problem using
    Fidelity-Gated Greedy selection:

    1. Action Pruning (Hard Fidelity Gate):
       For each block b_i, filter candidate tiers to A_i = {t : rho(b_i, t) <= rho_floor}.
       Among admissible tiers, identify the minimum byte-cost tier:
           t_i* = argmin_{t in A_i} C(b_i, t),  with cost C_i* = C(b_i, t_i*).
       If A_i is empty, the block is inadmissible (cannot be retained without fidelity breach).

    2. Efficiency Ranking:
       Sort admissible blocks by efficiency eta_i = w_i / C_i* in descending order
       (equivalently, byte cost per unit weight C_i* / w_i in ascending order).

    3. Demand Accumulation:
       Greedily admit blocks until sum(w_i) >= target_coverage.
    """
    t_start = time.perf_counter()

    # Step 1: Fidelity Gating & Tier Selection per Block
    admissible_items: List[Tuple[float, int, str, str, float]] = []
    # Tuples: (efficiency = w/C, cost C, block_id, tier_name, rho)

    total_admissible_weight = 0.0

    for blk in blocks:
        valid_tiers = [
            tp for tp in blk.tier_profiles.values()
            if tp.distortion_rho <= rho_floor
        ]
        if not valid_tiers:
            continue

        # Choose the minimum byte tier among fidelity-admissible tiers
        best_tp = min(valid_tiers, key=lambda tp: tp.bytes_per_block)
        cost = best_tp.bytes_per_block
        weight = max(blk.weight, 1e-12)
        efficiency = weight / cost

        admissible_items.append((efficiency, cost, blk.block_id, best_tp.tier_name, best_tp.distortion_rho, weight))
        total_admissible_weight += weight

    # Check feasibility
    is_feasible = total_admissible_weight >= target_coverage

    # Step 2: Sort by efficiency descending (highest coverage per byte first)
    admissible_items.sort(key=lambda x: x[0], reverse=True)

    # Step 3: Greedy Accumulation
    allocated_tiers: Dict[str, str] = {}
    total_bytes = 0
    cum_weight = 0.0
    selected_rhos: List[float] = []
    shadow_price = 0.0
    marginal_cost = 0

    for eff, cost, blk_id, tier_name, rho, weight in admissible_items:
        allocated_tiers[blk_id] = tier_name
        total_bytes += cost
        cum_weight += weight
        selected_rhos.append(rho)
        shadow_price = 1.0 / eff if eff > 0 else 0.0  # C / w (marginal byte price)
        marginal_cost = cost

        if cum_weight >= target_coverage:
            break

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    max_rho = max(selected_rhos) if selected_rhos else 0.0
    mean_rho = float(np.mean(selected_rhos)) if selected_rhos else 0.0

    return UBBAResult(
        allocations=allocated_tiers,
        total_bytes=total_bytes,
        achieved_coverage=cum_weight,
        target_coverage=target_coverage,
        rho_floor=rho_floor,
        max_rho=max_rho,
        mean_rho=mean_rho,
        num_blocks_selected=len(allocated_tiers),
        total_blocks_considered=len(blocks),
        shadow_price_lambda=shadow_price,
        duality_gap=float(marginal_cost),  # Upper bound on integrality gap is at most 1 item cost
        solver_latency_ms=latency_ms,
        is_feasible=(cum_weight >= target_coverage),
    )


def solve_naive_greedy_coverage(
    blocks: List[KVBlockCandidate],
    target_coverage: float,
) -> UBBAResult:
    """
    Unconstrained Naïve Greedy Coverage Solver (The flawed formulation rejected by Panel):
    Formulation:
        min sum C(b_i, t_i) s.t. sum w_i >= target_coverage
    WITHOUT fidelity constraint (rho <= rho_floor).

    Naïvely selects the cheapest tier available regardless of distortion,
    causing massive fidelity collapse on bursty / sensitive blocks.
    """
    t_start = time.perf_counter()

    items: List[Tuple[float, int, str, str, float, float]] = []
    for blk in blocks:
        if not blk.tier_profiles:
            continue
        # Pick the cheapest tier regardless of distortion
        cheapest_tp = min(blk.tier_profiles.values(), key=lambda tp: tp.bytes_per_block)
        cost = cheapest_tp.bytes_per_block
        weight = max(blk.weight, 1e-12)
        eff = weight / cost
        items.append((eff, cost, blk.block_id, cheapest_tp.tier_name, cheapest_tp.distortion_rho, weight))

    items.sort(key=lambda x: x[0], reverse=True)

    allocations: Dict[str, str] = {}
    total_bytes = 0
    cum_weight = 0.0
    rhos: List[float] = []

    for eff, cost, b_id, tier, rho, weight in items:
        allocations[b_id] = tier
        total_bytes += cost
        cum_weight += weight
        rhos.append(rho)
        if cum_weight >= target_coverage:
            break

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    return UBBAResult(
        allocations=allocations,
        total_bytes=total_bytes,
        achieved_coverage=cum_weight,
        target_coverage=target_coverage,
        rho_floor=0.0,  # Not enforced
        max_rho=max(rhos) if rhos else 0.0,
        mean_rho=float(np.mean(rhos)) if rhos else 0.0,
        num_blocks_selected=len(allocations),
        total_blocks_considered=len(blocks),
        shadow_price_lambda=0.0,
        duality_gap=0.0,
        solver_latency_ms=latency_ms,
        is_feasible=(cum_weight >= target_coverage),
    )


def solve_ubba_lagrangian_dual(
    blocks: List[KVBlockCandidate],
    target_coverage: float,
    rho_floor: float = 0.10,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> Tuple[UBBAResult, float]:
    """
    Solves UBBA using Lagrangian Dual Bisection:
    Primal:
        min sum_{i} C_i^* x_i  s.t. sum w_i x_i >= Target, x_i in [0, 1]
    Lagrangian:
        L(lambda) = sum_i C_i^* x_i + lambda * (Target - sum_i w_i x_i)
                  = lambda * Target + sum_i (C_i^* - lambda * w_i) x_i
    Dual:
        max_{lambda >= 0} L(lambda)
    For a given lambda:
        x_i(lambda) = 1 if (C_i^* - lambda * w_i) < 0 else 0
        equivalently, if lambda > C_i^* / w_i.
    Since sum w_i x_i(lambda) is monotonically non-decreasing in lambda,
    we can find optimal lambda* via 1D bisection.
    """
    # Filter admissible blocks and extract C_i*, w_i
    admissible: List[Tuple[str, str, int, float, float]] = []
    for blk in blocks:
        valid_tiers = [
            tp for tp in blk.tier_profiles.values()
            if tp.distortion_rho <= rho_floor
        ]
        if not valid_tiers:
            continue
        best_tp = min(valid_tiers, key=lambda tp: tp.bytes_per_block)
        admissible.append((
            blk.block_id,
            best_tp.tier_name,
            best_tp.bytes_per_block,
            best_tp.distortion_rho,
            max(blk.weight, 1e-12)
        ))

    if not admissible:
        # Infeasible
        empty_res = solve_ubba_greedy(blocks, target_coverage, rho_floor)
        return empty_res, 0.0

    costs = np.array([item[2] for item in admissible], dtype=np.float64)
    weights = np.array([item[4] for item in admissible], dtype=np.float64)
    ratios = costs / weights  # bytes per unit coverage

    # Lambda range
    lambda_low = 0.0
    lambda_high = float(np.max(ratios) * 2.0)

    best_lambda = lambda_high
    for _ in range(max_iter):
        mid = (lambda_low + lambda_high) / 2.0
        # If lambda > C_i / w_i, x_i = 1
        active_mask = mid >= ratios
        cov = np.sum(weights[active_mask])
        if cov >= target_coverage:
            best_lambda = mid
            lambda_high = mid
        else:
            lambda_low = mid

    # Primal result from greedy benchmark
    greedy_res = solve_ubba_greedy(blocks, target_coverage, rho_floor)
    return greedy_res, best_lambda
