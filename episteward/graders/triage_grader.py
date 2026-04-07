"""
TriageGrader — scores Task 1 (PrescriptionTriage).

Score breakdown (sums to 1.0 max):
  drug_class_score    : 0.0–0.4  correct drug class for pathogen
  pkpd_score          : 0.0–0.3  dose within PK/PD therapeutic window
  stewardship_score   : 0.0–0.3  narrow-spectrum preference when broad unnecessary

Partial credit is provided at every step.
De-escalation bonus: reward increases if agent de-escalates after sensitivities return.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from episteward.models import EpiAction
from episteward.state import HospitalState

logger = logging.getLogger(__name__)

# Drug class → spectrum mapping (broad=1, narrow=0)
_DRUG_SPECTRUM: Dict[str, str] = {
    "carbapenem": "broad",
    "beta_lactam_beta_lactamase_inhibitor": "broad",
    "fluoroquinolone": "narrow",
    "nitrofurantoin": "narrow",
    "first_gen_cephalosporin": "narrow",
    "anti_staphylococcal_penicillin": "narrow",
    "glycopeptide": "narrow",
    "oxazolidinone": "narrow",
    "macrolide": "narrow",
    "sulfonamide": "narrow",
}

# Antibiotic name → drug class (partial lookup)
_DRUG_CLASS_MAP: Dict[str, str] = {
    "meropenem": "carbapenem",
    "ertapenem": "carbapenem",
    "piperacillin-tazobactam": "beta_lactam_beta_lactamase_inhibitor",
    "piperacillin_tazobactam": "beta_lactam_beta_lactamase_inhibitor",
    "ceftriaxone": "third_gen_cephalosporin",
    "cefazolin": "first_gen_cephalosporin",
    "ampicillin": "penicillin",
    "vancomycin": "glycopeptide",
    "linezolid": "oxazolidinone",
    "azithromycin": "macrolide",
    "ciprofloxacin": "fluoroquinolone",
    "nitrofurantoin": "nitrofurantoin",
    "trimethoprim-sulfamethoxazole": "sulfonamide",
    "trimethoprim_sulfamethoxazole": "sulfonamide",
    "colistin": "polymyxin",
}


class TriageGrader:
    """Grader for PrescriptionTriage task."""

    def grade(
        self,
        action: EpiAction,
        state: HospitalState,
        ground_truth: Dict[str, Any],
        step_number: int,
    ) -> Dict[str, Any]:
        """
        Grade a single action against ground truth.

        Returns dict with keys: reward (float [0,1]), components, done, info.
        """
        drug_class = _DRUG_CLASS_MAP.get(action.antibiotic.lower(), "unknown")
        correct_class = ground_truth["correct_drug_class"]
        alt_class = ground_truth.get("alt_drug_class", "")
        needs_broad = ground_truth["needs_broad"]

        # --- Drug class score (0.0–0.4) ---
        if drug_class == correct_class or drug_class == alt_class:
            drug_class_score = 0.4
        elif drug_class != "unknown":
            # Partial credit if at least covering gram direction
            drug_class_score = 0.15
        else:
            drug_class_score = 0.0

        # --- PK/PD therapeutic window score (0.0–0.3) ---
        pkpd_score = self._pkpd_score(action, ground_truth)

        # --- Stewardship / narrow-spectrum score (0.0–0.3) ---
        spectrum = _DRUG_SPECTRUM.get(drug_class, "broad")
        if not needs_broad and spectrum == "broad":
            stewardship_score = 0.0  # unnecessary broad-spectrum
        elif needs_broad and spectrum == "broad":
            stewardship_score = 0.3  # correct choice
        elif not needs_broad and spectrum == "narrow":
            stewardship_score = 0.3  # ideal narrow choice
        else:
            stewardship_score = 0.1  # narrow when broad needed — suboptimal

        # De-escalation bonus: step 4+ with full sensitivities and now narrow
        de_escalation_bonus = 0.0
        if step_number >= 4 and spectrum == "narrow" and not needs_broad:
            # Check if previous steps had broad
            history = []
            for pid, patient in state.patients.items():
                history = patient.antibiotic_history
            if len(history) > 1:
                prev_drug = history[-2].get("antibiotic", "")
                prev_class = _DRUG_CLASS_MAP.get(prev_drug.lower(), "unknown")
                prev_spectrum = _DRUG_SPECTRUM.get(prev_class, "narrow")
                if prev_spectrum == "broad":
                    de_escalation_bonus = 0.1  # correct de-escalation

        base_score = drug_class_score + pkpd_score + stewardship_score + de_escalation_bonus
        reward = float(min(1.0, max(0.0, base_score)))

        done = step_number >= 5
        return {
            "reward": reward,
            "components": {
                "drug_class": drug_class_score,
                "pkpd": pkpd_score,
                "stewardship": stewardship_score,
                "de_escalation_bonus": de_escalation_bonus,
            },
            "done": done,
            "info": {
                "drug_class_matched": drug_class,
                "expected_class": correct_class,
                "needs_broad": needs_broad,
                "step": step_number,
            },
        }

    def _pkpd_score(
        self, action: EpiAction, ground_truth: Dict[str, Any]
    ) -> float:
        """Score dose appropriateness vs ground-truth optimal dose."""
        optimal = ground_truth.get("optimal_dose_mg", 500.0)
        ratio = action.dose_mg / max(optimal, 1.0)
        # Score peaks at 1.0 ratio, degrades ±50%
        deviation = abs(ratio - 1.0)
        if deviation <= 0.1:
            return 0.3
        elif deviation <= 0.3:
            return 0.2
        elif deviation <= 0.5:
            return 0.1
        return 0.0
