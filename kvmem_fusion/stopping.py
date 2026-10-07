"""
R10 Statistical Contract and Sequential Stopping Rule Tool
Directly implements E1b from docs/experiment-manual.md:
- Sequential Lan-DeMets O'Brien-Fleming (OBF) monitoring boundaries
- McNemar paired sample size calculator (target N=471)
- Intra-cluster correlation (ICC) Design Effect correction: Deff = 1 + (m - 1) * ICC
- Wilson score interval for proportions
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def compute_wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute Wilson score interval for a binomial proportion.
    
    Args:
        successes: Number of success events
        trials: Total number of trials
        confidence: Confidence level (default 0.95)
    Returns:
        (center, lower_bound, upper_bound)
    """
    if trials == 0:
        return 0.0, 0.0, 0.0
    p = successes / trials
    z = stats.norm.ppf(1 - (1 - confidence) / 2.0)
    denominator = 1.0 + z**2 / trials
    centre_adjusted_probability = (p + z**2 / (2 * trials)) / denominator
    adjusted_std_dev = np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    lower = max(0.0, centre_adjusted_probability - z * adjusted_std_dev)
    upper = min(1.0, centre_adjusted_probability + z * adjusted_std_dev)
    return p, lower, upper


def compute_mcnemar_sample_size(discordant_rate: float = 0.15, delta_mde: float = 0.05, alpha: float = 0.05, power: float = 0.80) -> int:
    """Compute required sample size N of paired (task, seed) units under McNemar's test.
    
    Formula: N = (z_{alpha/2} + z_{beta})^2 * pi_d / delta^2
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    n_req = ((z_alpha + z_beta)**2) * discordant_rate / (delta_mde**2)
    return int(np.ceil(n_req))


def compute_lan_demets_obf_boundary(information_fraction: float) -> Tuple[float, float]:
    """Compute cumulative alpha and z-score boundary at information fraction t in (0, 1].
    
    Lan-DeMets O'Brien-Fleming alpha spending function:
    alpha(t) = 2 * (1 - Phi(z_{alpha/2} / sqrt(t)))
    """
    assert 0.0 < information_fraction <= 1.0
    z_half = 1.95996  # for alpha = 0.05
    cum_alpha = 2.0 * (1.0 - stats.norm.cdf(z_half / np.sqrt(information_fraction)))
    if cum_alpha <= 0.0:
        z_boundary = 8.0
    else:
        z_boundary = float(stats.norm.ppf(1.0 - cum_alpha / 2.0))
    return cum_alpha, z_boundary


def compute_cluster_design_effect(num_seeds_per_task: int = 4, icc: float = 0.3) -> float:
    """Compute cluster design effect: Deff = 1 + (m - 1) * ICC."""
    return 1.0 + (num_seeds_per_task - 1) * icc


class SequentialAuditHarness:
    """Audits sequential test stream against OBF stopping rules."""
    def __init__(self, target_pairs: int = 471, num_looks: int = 4, icc: float = 0.3):
        self.target_pairs = target_pairs
        self.num_looks = num_looks
        self.icc = icc
        self.look_fractions = [float(k) / num_looks for k in range(1, num_looks + 1)]
        self.look_samples = [int(np.ceil(t * target_pairs)) for t in self.look_fractions]
        self.design_effect = compute_cluster_design_effect(4, icc)

    def evaluate_interim_look(self, look_index: int, successes_a: int, successes_b: int, total_tested: int) -> Dict[str, float]:
        t = self.look_fractions[look_index]
        cum_alpha, z_bound = compute_lan_demets_obf_boundary(t)
        
        # Paired difference
        p_a = successes_a / total_tested
        p_b = successes_b / total_tested
        diff = p_b - p_a
        
        # Pooled SE adjusted by design effect
        p_pool = (successes_a + successes_b) / (2 * total_tested)
        se_raw = np.sqrt(p_pool * (1 - p_pool) * 2 / total_tested)
        se_adj = se_raw * np.sqrt(self.design_effect)
        z_stat = diff / (se_adj + 1e-12)
        
        stopped = abs(z_stat) >= z_bound
        return {
            "look_index": look_index + 1,
            "information_fraction": t,
            "total_tested": total_tested,
            "z_boundary": z_bound,
            "z_statistic": z_stat,
            "diff_pp": diff * 100.0,
            "se_adjusted": se_adj,
            "should_stop_early": stopped
        }
