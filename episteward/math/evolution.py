"""
Wright-Fisher model for resistance allele evolution under antibiotic pressure.

Each timestep advances the allele frequency by one Wright-Fisher generation:
    p_new ~ Binomial(2N, p_fitness) / 2N
    p_fitness = p * w_R / (p * w_R + (1-p) * w_S)
    w_R = 1.0            (resistant fitness — unaffected by drug)
    w_S = 1.0 - s        (sensitive fitness — reduced under drug pressure)

Resistance is declared emergent when:
    allele_frequency > 0.5  AND  drug pressure sustained > 48 h
"""

from __future__ import annotations

import numpy as np

# Effective population size (work in frequency space: p ∈ [0,1])
_N = int(1e6)  # reduced from 1e8 for runtime; keeps dynamics correct


def selection_coefficient(dose_mg: float, standard_dose_mg: float) -> float:
    """
    Derive selection coefficient *s* from relative drug pressure.

    s = 0.0 → no pressure (w_S = 1.0)
    s = 0.5 → half maximum pressure
    s = 0.9 → near-maximum pressure (high dose)
    """
    ratio = dose_mg / max(standard_dose_mg, 1.0)
    # Sigmoid-like mapping capped at 0.9
    return float(np.clip(0.9 * (1 - np.exp(-ratio)), 0.0, 0.9))


def wright_fisher_step(
    allele_freq: float,
    s: float,
    rng: np.random.Generator,
) -> float:
    """
    Advance allele frequency by one Wright-Fisher generation.

    Parameters
    ----------
    allele_freq : float  in [0, 1] — current resistant allele frequency
    s           : float  selection coefficient against sensitive strain
    rng         : numpy Generator (seeded per-episode)

    Returns
    -------
    float  new allele frequency in [0, 1]
    """
    w_R = 1.0
    w_S = 1.0 - s

    p = float(np.clip(allele_freq, 0.0, 1.0))
    mean_fitness = p * w_R + (1 - p) * w_S
    if mean_fitness <= 0:
        return 1.0  # fully resistant

    p_next = p * w_R / mean_fitness
    # Binomial sampling
    draws = rng.binomial(2 * _N, p_next)
    return draws / (2 * _N)


def evolve_resistance(
    allele_freq: float,
    treatment_hours: float,
    dose_mg: float,
    standard_dose_mg: float,
    rng: np.random.Generator,
    timestep_hours: float = 24.0,
) -> tuple[float, bool]:
    """
    Simulate resistance evolution over *timestep_hours* under a given dose.

    Returns
    -------
    new_freq : float   updated allele frequency
    emerged  : bool    True if resistance is now clinically emergent
    """
    s = selection_coefficient(dose_mg, standard_dose_mg)
    new_freq = wright_fisher_step(allele_freq, s, rng)
    emerged = (new_freq > 0.5) and (treatment_hours >= 48.0)
    return new_freq, emerged
