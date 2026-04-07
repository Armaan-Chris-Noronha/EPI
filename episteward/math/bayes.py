"""
Bayesian resistance probability estimator.

Uses ward-level antibiogram data as the prior, then updates on each
culture result via Bayes' rule:

    P(resistant | result) ∝ P(result | resistant) * P(resistant)

Likelihoods:
    P("resistant" | truly resistant)  = sensitivity = 0.95
    P("resistant" | truly sensitive)  = 1 - specificity = 0.05
    P("sensitive" | truly resistant)  = 1 - sensitivity = 0.05
    P("sensitive" | truly sensitive)  = specificity = 0.95

Returns posterior mean and a 95% credible interval (Beta distribution).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import beta as beta_dist


# Culture test characteristics (fixed across pathogens for simplicity)
_SENSITIVITY = 0.95   # P(positive test | truly resistant)
_SPECIFICITY = 0.95   # P(negative test | truly sensitive)


def _beta_from_prior(prior_prob: float) -> Tuple[float, float]:
    """
    Convert a scalar prior probability to Beta(alpha, beta) parameters.

    Uses a weakly informative prior equivalent to 10 pseudo-observations.
    """
    n_pseudo = 10.0
    alpha = prior_prob * n_pseudo
    b = (1.0 - prior_prob) * n_pseudo
    return float(alpha), float(b)


def prior_resistance_prob(
    pathogen: str,
    antibiotic: str,
    ward_id: str,
    resistance_profiles: Dict[str, Any],
) -> float:
    """
    Return the antibiogram prior P(resistant) for a drug-bug-ward triplet.

    Falls back to organism-level, then global default of 0.2.
    """
    ward_data = resistance_profiles.get(ward_id, {})
    bug_data = ward_data.get(pathogen, resistance_profiles.get(pathogen, {}))
    return float(bug_data.get(antibiotic, bug_data.get("default", 0.2)))


def update_posterior(
    prior_prob: float,
    culture_result: Optional[str],  # "resistant", "sensitive", or None (pending)
) -> Tuple[float, float, float]:
    """
    Bayesian update given a culture result.

    Parameters
    ----------
    prior_prob    : P(resistant) before culture result
    culture_result: "resistant", "sensitive", or None (no update)

    Returns
    -------
    posterior_mean : float   updated P(resistant)
    ci_lower       : float   95% credible interval lower bound
    ci_upper       : float   95% credible interval upper bound
    """
    alpha, b = _beta_from_prior(prior_prob)

    if culture_result == "resistant":
        # Positive test: upweight alpha (resistant pseudo-counts)
        likelihood_ratio = _SENSITIVITY / (1 - _SPECIFICITY)
        alpha *= likelihood_ratio
    elif culture_result == "sensitive":
        # Negative test: upweight beta (sensitive pseudo-counts)
        likelihood_ratio = (1 - _SENSITIVITY) / _SPECIFICITY
        alpha *= likelihood_ratio
        b /= likelihood_ratio  # symmetric update
    # else: no update

    # Normalize
    total = alpha + b
    alpha = float(np.clip(alpha, 0.01, total - 0.01))
    b = total - alpha

    dist = beta_dist(alpha, b)
    mean = dist.mean()
    ci_lower, ci_upper = dist.ppf(0.025), dist.ppf(0.975)

    return float(mean), float(ci_lower), float(ci_upper)


def estimate_resistance(
    pathogen: str,
    antibiotic: str,
    ward_id: str,
    resistance_profiles: Dict[str, Any],
    culture_result: Optional[str] = None,
) -> Dict[str, float]:
    """
    Full pipeline: prior → Bayesian update → return summary dict.

    Returns
    -------
    dict with keys: prior, posterior, ci_lower, ci_upper
    """
    prior = prior_resistance_prob(pathogen, antibiotic, ward_id, resistance_profiles)
    posterior, ci_lower, ci_upper = update_posterior(prior, culture_result)
    return {
        "prior": prior,
        "posterior": posterior,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
