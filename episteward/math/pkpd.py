"""
One-compartment pharmacokinetic / pharmacodynamic (PK/PD) model.

PK — concentration over time:
    C(t) = (F * D / Vd) * exp(-ke * t)
    ke   = CL / Vd

PD — drug effect via Hill equation:
    effect(C) = Emax * C^n / (EC50^n + C^n)

Therapeutic window (beta-lactams):
    C_min >= MIC * 4   (efficacy threshold)
    C_max <= MIC * 64  (toxicity ceiling)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp


def concentration_profile(
    dose_mg: float,
    pk_params: Dict[str, float],
    t_span: Tuple[float, float] = (0.0, 24.0),
    n_points: int = 241,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the one-compartment PK ODE for a single dose.

    Parameters
    ----------
    dose_mg:
        Administered dose in milligrams.
    pk_params:
        Dict with keys F, Vd_L_kg, CL_L_h_kg, ke (patient weight assumed 70 kg).
    t_span:
        Tuple (t_start, t_end) in hours.
    n_points:
        Number of evaluation points.

    Returns
    -------
    t : np.ndarray   shape (n_points,)
    C : np.ndarray   shape (n_points,)  concentration in mg/L
    """
    weight_kg = 70.0
    F = pk_params["F"]
    Vd = pk_params["Vd_L_kg"] * weight_kg
    CL = pk_params["CL_L_h_kg"] * weight_kg
    ke = CL / Vd

    C0 = F * dose_mg / Vd

    def ode(t: float, y: list[float]) -> list[float]:
        return [-ke * y[0]]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(ode, t_span, [C0], t_eval=t_eval, dense_output=False)
    return sol.t, sol.y[0]


def hill_effect(
    concentration: float,
    emax: float,
    ec50: float,
    hill_n: float = 1.0,
) -> float:
    """
    Hill equation: effect = Emax * C^n / (EC50^n + C^n).

    Returns a value in [0, Emax].
    """
    if concentration <= 0:
        return 0.0
    cn = concentration**hill_n
    return emax * cn / (ec50**hill_n + cn)


def is_in_therapeutic_window(
    dose_mg: float,
    pk_params: Dict[str, float],
    mic: float,
    frequency_hours: float,
) -> bool:
    """
    Return True if trough concentration >= MIC*4 and peak <= MIC*64.

    Evaluates at [0, frequency_hours] since next dose resets the curve.
    """
    _, C = concentration_profile(dose_mg, pk_params, t_span=(0.0, frequency_hours))
    c_peak = C.max()
    c_trough = C.min()
    return (c_trough >= mic * 4) and (c_peak <= mic * 64)


def therapeutic_score(
    dose_mg: float,
    pk_params: Dict[str, float],
    mic: float,
    frequency_hours: float,
) -> float:
    """
    Continuous [0.0, 1.0] score for dose appropriateness.

    1.0 = perfectly in window; degrades linearly outside.
    """
    _, C = concentration_profile(dose_mg, pk_params, t_span=(0.0, frequency_hours))
    c_peak = C.max()
    c_trough = C.min()

    low_target = mic * 4
    high_target = mic * 64

    if c_trough < low_target:
        # Under-dosed: scale by how close trough is to target
        return float(np.clip(c_trough / low_target, 0.0, 1.0))
    if c_peak > high_target:
        # Over-dosed: penalise proportionally
        overshoot = (c_peak - high_target) / high_target
        return float(np.clip(1.0 - overshoot, 0.0, 1.0))
    return 1.0
