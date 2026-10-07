"""
Unit tests for R10 Sequential Stopping Rule Tool (kvmem_fusion/stopping.py)
Verifies:
- Wilson score interval coverage
- McNemar sample size formula (target N=471 pairs at discordant rate=0.15, delta=0.05)
- Lan-DeMets OBF alpha spending and z-score boundaries
- Cluster design effect Deff = 1 + (m - 1) * ICC
"""

import numpy as np
import pytest
from kvmem_fusion.stopping import (
    compute_wilson_score_interval,
    compute_mcnemar_sample_size,
    compute_lan_demets_obf_boundary,
    compute_cluster_design_effect,
    SequentialAuditHarness,
)


def test_wilson_score_interval_bounds():
    p, low, high = compute_wilson_score_interval(successes=95, trials=100)
    assert p == 0.95
    assert 0.88 <= low <= 0.92
    assert 0.97 <= high <= 0.99


def test_mcnemar_sample_size_matches_r10():
    # R10 contract: discordant_rate=0.15, delta=0.05, alpha=0.05, power=0.80 -> N approx 471
    n_req = compute_mcnemar_sample_size(discordant_rate=0.15, delta_mde=0.05, alpha=0.05, power=0.80)
    assert 460 <= n_req <= 480, f"Expected N approx 471, got {n_req}"


def test_lan_demets_obf_boundary_monotonicity():
    # As information fraction t increases, z-score boundary drops toward 1.96
    t_vals = [0.25, 0.50, 0.75, 1.00]
    z_bounds = [compute_lan_demets_obf_boundary(t)[1] for t in t_vals]
    
    assert z_bounds[0] > z_bounds[1] > z_bounds[2] >= z_bounds[3]
    assert np.isclose(z_bounds[-1], 1.96, atol=0.02)


def test_cluster_design_effect():
    # 4 seeds per task, ICC = 0.3 -> Deff = 1 + 3 * 0.3 = 1.90
    deff = compute_cluster_design_effect(num_seeds_per_task=4, icc=0.3)
    assert np.isclose(deff, 1.90)


def test_sequential_audit_harness_early_stop():
    harness = SequentialAuditHarness(target_pairs=471, num_looks=4, icc=0.3)
    
    # Overwhelming difference at interim look 1 (e.g. 95% vs 40%)
    res = harness.evaluate_interim_look(look_index=0, successes_a=40, successes_b=95, total_tested=118)
    assert bool(res["should_stop_early"]) is True
    assert res["z_statistic"] > res["z_boundary"]
