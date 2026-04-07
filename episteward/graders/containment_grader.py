"""
ContainmentGrader — scores Task 2 (ResistanceContainment).

Score breakdown (sums to 1.0 max):
  source_score       : 0.0–0.25  index patient correctly isolated first
  isolation_score    : 0.0–0.25  isolation completeness across cluster
  prescribing_score  : 0.0–0.35  appropriate therapy per patient
  culture_score      : 0.0–0.15  cultures requested on exposed patients

Per-step penalties (applied before normalization):
  -0.05 per new resistance case emerging that step
  -0.03 per unnecessary broad-spectrum prescription

Bonus:
  +0.10 if index patient correctly isolated within first 3 steps
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from episteward.models import EpiAction
from episteward.state import HospitalState

logger = logging.getLogger(__name__)

_BROAD_SPECTRUM = {"carbapenem", "beta_lactam_beta_lactamase_inhibitor", "polymyxin"}
_DRUG_CLASS_MAP: Dict[str, str] = {
    "meropenem": "carbapenem",
    "ertapenem": "carbapenem",
    "piperacillin-tazobactam": "beta_lactam_beta_lactamase_inhibitor",
    "ceftriaxone": "third_gen_cephalosporin",
    "cefazolin": "first_gen_cephalosporin",
    "ciprofloxacin": "fluoroquinolone",
    "nitrofurantoin": "nitrofurantoin",
    "trimethoprim-sulfamethoxazole": "sulfonamide",
    "colistin": "polymyxin",
}


class ContainmentGrader:
    """Grader for ResistanceContainment task."""

    def grade(
        self,
        action: EpiAction,
        state: HospitalState,
        ground_truth: Dict[str, Any],
        step_number: int,
        prev_new_cases: int = 0,
    ) -> Dict[str, Any]:
        """
        Grade one containment step.

        Parameters
        ----------
        action          : agent's EpiAction
        state           : current HospitalState
        ground_truth    : from task.ground_truth
        step_number     : current step (1-indexed)
        prev_new_cases  : new cases from PREVIOUS step (for penalty calc)
        """
        index_pid = ground_truth["index_patient_id"]
        exposed = ground_truth["exposed_patients"]
        isolation_bonus = ground_truth["isolation_bonus_awarded"]

        # --- Source identification score (0.0–0.25) ---
        # Proxy: did agent order isolation on the current patient?
        # If current patient is the index patient and agent isolates → credit
        current_pid = None
        for pid, patient in state.patients.items():
            if patient.antibiotic_history and patient.antibiotic_history[-1] == action.model_dump():
                current_pid = pid
                break

        source_score = 0.0
        if current_pid == index_pid and action.isolation_order:
            source_score = 0.25
        elif action.isolation_order:
            source_score = 0.05  # isolated someone, but not index

        # --- Isolation completeness (0.0–0.25) ---
        isolated_count = sum(1 for p in state.patients.values() if p.is_isolated)
        total = len(state.patients)
        isolation_score = float(isolated_count / total) * 0.25

        # --- Prescribing appropriateness (0.0–0.35) ---
        drug_class = _DRUG_CLASS_MAP.get(action.antibiotic.lower(), "unknown")
        # ESBL E. coli: carbapenems or pip-tazo are correct; fluoroquinolone often resistant
        if drug_class in ("carbapenem", "beta_lactam_beta_lactamase_inhibitor"):
            prescribing_score = 0.35
        elif drug_class in ("third_gen_cephalosporin", "fluoroquinolone"):
            prescribing_score = 0.05  # typically resistant in ESBL
        elif drug_class != "unknown":
            prescribing_score = 0.15
        else:
            prescribing_score = 0.0

        # --- Culture strategy (0.0–0.15) ---
        cultures_requested = sum(1 for p in state.patients.values() if p.culture_pending)
        culture_score = float(min(cultures_requested, len(exposed)) / max(len(exposed), 1)) * 0.15

        # --- Penalties ---
        new_case_penalty = prev_new_cases * 0.05
        broad_penalty = 0.03 if drug_class in _BROAD_SPECTRUM and current_pid not in [index_pid] else 0.0

        # --- Isolation bonus ---
        bonus = 0.10 if isolation_bonus and step_number <= 3 else 0.0

        base = source_score + isolation_score + prescribing_score + culture_score + bonus
        total_penalties = new_case_penalty + broad_penalty
        reward = float(min(1.0, max(0.0, base - total_penalties)))

        done = step_number >= 15
        return {
            "reward": reward,
            "components": {
                "source": source_score,
                "isolation": isolation_score,
                "prescribing": prescribing_score,
                "culture": culture_score,
                "bonus": bonus,
                "new_case_penalty": -new_case_penalty,
                "broad_penalty": -broad_penalty,
            },
            "done": done,
            "info": {
                "step": step_number,
                "isolated_count": isolated_count,
                "cultures_pending": cultures_requested,
            },
        }
