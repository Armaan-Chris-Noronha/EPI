"""
OutbreakGrader — scores Task 3 (NetworkOutbreakResponse).

Reward formula (per CLAUDE.md):
    R = α·lives_saved_ratio - β·colistin_overspend - γ·resistance_amplification_events
    α = 0.6, β = 0.25, γ = 0.15

Normalized to [0.0, 1.0] via:
    penalized = max(0.0, base - penalties)
    reward = min(1.0, penalized)

Partial credit at every step:
  Each step graded on marginal lives saved + colistin usage efficiency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from episteward.models import EpiAction
from episteward.state import HospitalState

logger = logging.getLogger(__name__)

_ALPHA = 0.6
_BETA = 0.25
_GAMMA = 0.15


class OutbreakGrader:
    """Grader for NetworkOutbreakResponse task."""

    def grade(
        self,
        action: EpiAction,
        state: HospitalState,
        ground_truth: Dict[str, Any],
        step_number: int,
    ) -> Dict[str, Any]:
        """
        Grade one outbreak management step.

        Uses cumulative ground truth values; provides incremental signal
        proportional to marginal progress this step.
        """
        lives_saved_ratio = ground_truth.get("lives_saved_ratio", 0.0)
        colistin_overspend = ground_truth.get("colistin_overspend", 0)
        resistance_events = ground_truth.get("resistance_amplification_events", 0)
        max_steps = 30

        # Marginal progress signal: scale full reward by step fraction
        step_fraction = step_number / max_steps

        # Lives saved component — scale to current progress
        lives_component = _ALPHA * lives_saved_ratio * step_fraction

        # Colistin penalty — normalise overspend to [0,1] (budget=10)
        colistin_penalty = _BETA * float(min(colistin_overspend, 10) / 10)

        # Resistance amplification penalty — normalise (max reasonable = 20 events)
        resistance_penalty = _GAMMA * float(min(resistance_events, 20) / 20)

        # Isolation bonus: reward containment orders on confirmed CRK hospitals
        containment_bonus = 0.0
        if action.isolation_order:
            crk_hospitals = {
                p.ward_id for p in state.patients.values()
                if p.resistance_frequency > 0.5
            }
            current_ward = None
            for pid, p in state.patients.items():
                if p.antibiotic_history and p.antibiotic_history[-1].get("isolation_order"):
                    current_ward = p.ward_id
                    break
            if current_ward in crk_hospitals:
                containment_bonus = 0.05

        base = lives_component + containment_bonus
        total_penalties = colistin_penalty + resistance_penalty
        reward = float(min(1.0, max(0.0, base - total_penalties)))

        done = step_number >= 30
        return {
            "reward": reward,
            "components": {
                "lives_saved": lives_component,
                "containment_bonus": containment_bonus,
                "colistin_penalty": -colistin_penalty,
                "resistance_penalty": -resistance_penalty,
            },
            "done": done,
            "info": {
                "step": step_number,
                "lives_saved_ratio": lives_saved_ratio,
                "colistin_overspend": colistin_overspend,
                "resistance_events": resistance_events,
                "alpha": _ALPHA,
                "beta": _BETA,
                "gamma": _GAMMA,
            },
        }
