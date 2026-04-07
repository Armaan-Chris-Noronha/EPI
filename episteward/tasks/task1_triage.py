"""
Task 1 — PrescriptionTriage (Easy).

Single patient, single ward, complete culture data.
Episode length: 5 steps.

The agent prescribes an antibiotic for one patient whose culture sensitivities
arrive incrementally.  Reward improves if the agent de-escalates correctly
once sensitivities return.

Ground truth:
  - correct_drug_class: the drug class that covers the pathogen
  - optimal_dose_mg:    dose in the therapeutic window
  - needs_broad:        bool — whether broad-spectrum is actually required
  - index_pathogen:     the pathogen name
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from episteward.models import EpiAction, EpiObservation
from episteward.state import HospitalState, PatientRecord
from episteward.tasks.base import BaseTask

logger = logging.getLogger(__name__)

# Scenario pool — sampled by seed
_SCENARIOS = [
    {
        "patient_id": "P001",
        "ward_id": "general_ward",
        "pathogen": "E_coli",
        "infection_site": "urinary_tract",
        "symptoms": ["dysuria", "fever_38_5", "flank_pain"],
        "vitals": {"temp_c": 38.5, "hr_bpm": 96, "wbc_k_ul": 14.2, "crp_mg_l": 48.0, "procalcitonin_ng_ml": 0.6},
        "correct_drug_class": "fluoroquinolone",
        "alt_drug_class": "nitrofurantoin",
        "needs_broad": False,
        "optimal_dose_mg": 500.0,
    },
    {
        "patient_id": "P002",
        "ward_id": "icu",
        "pathogen": "Klebsiella_pneumoniae",
        "infection_site": "bloodstream",
        "symptoms": ["fever_39_2", "hypotension", "tachycardia"],
        "vitals": {"temp_c": 39.2, "hr_bpm": 118, "wbc_k_ul": 18.7, "crp_mg_l": 120.0, "procalcitonin_ng_ml": 4.2},
        "correct_drug_class": "carbapenem",
        "alt_drug_class": "beta_lactam_beta_lactamase_inhibitor",
        "needs_broad": True,
        "optimal_dose_mg": 1000.0,
    },
    {
        "patient_id": "P003",
        "ward_id": "surgical_ward",
        "pathogen": "S_aureus",
        "infection_site": "wound",
        "symptoms": ["erythema", "purulent_discharge", "fever_38_1"],
        "vitals": {"temp_c": 38.1, "hr_bpm": 88, "wbc_k_ul": 12.0, "crp_mg_l": 32.0, "procalcitonin_ng_ml": 0.3},
        "correct_drug_class": "anti_staphylococcal_penicillin",
        "alt_drug_class": "first_gen_cephalosporin",
        "needs_broad": False,
        "optimal_dose_mg": 500.0,
    },
]

# Culture reveal schedule: step → what sensitivity info becomes available
_CULTURE_STEPS = {
    1: {"status": "pending", "organism": None},
    2: {"status": "gram_stain", "organism": None},
    3: {"status": "organism_id", "sensitivities": {}},
    4: {"status": "full_sensitivities"},
    5: {"status": "full_sensitivities"},
}


class PrescriptionTriage(BaseTask):
    """Single-patient antibiotic prescribing task with de-escalation signal."""

    max_steps = 5
    name = "task1_triage"

    def __init__(self) -> None:
        super().__init__()
        self._scenario: Dict[str, Any] = {}
        self._culture_history: List[Dict[str, Any]] = []

    def reset(self, seed: int = 0) -> EpiObservation:
        """Initialise episode, return first observation with pending culture."""
        self.state = HospitalState(task_id=self.name, episode_seed=seed)
        self.state.seed(seed)

        # Pick scenario deterministically from seed
        idx = seed % len(_SCENARIOS)
        self._scenario = dict(_SCENARIOS[idx])
        self._culture_history = []

        patient = PatientRecord(
            patient_id=self._scenario["patient_id"],
            ward_id=self._scenario["ward_id"],
            pathogen=self._scenario["pathogen"],
        )
        self.state.patients[patient.patient_id] = patient
        self.state.step_number = 1

        return self._make_observation()

    def step(self, action: EpiAction) -> Tuple[EpiObservation, bool]:
        """Apply prescription action, advance culture reveal."""
        self._assert_ready()
        assert self.state is not None

        pid = self._scenario["patient_id"]
        patient = self.state.patients[pid]

        # Record treatment
        patient.antibiotic_history.append(action.model_dump())
        patient.is_treated = True
        if action.isolation_order:
            patient.is_isolated = True
        if action.culture_requested:
            patient.culture_pending = True

        self.state.step_number += 1
        done = self.state.step_number > self.max_steps
        if done:
            self.state.is_done = True

        return self._make_observation(), done

    def _make_observation(self) -> EpiObservation:
        """Build observation with appropriate culture reveal for current step."""
        assert self.state is not None
        step = self.state.step_number
        pid = self._scenario["patient_id"]
        patient = self.state.patients[pid]

        culture = self._culture_at_step(step)

        return EpiObservation(
            patient_id=pid,
            ward_id=self._scenario["ward_id"],
            infection_site=self._scenario["infection_site"],
            symptoms=list(self._scenario["symptoms"]),
            vitals=dict(self._scenario["vitals"]),
            culture_results=culture,
            resistance_flags=[],
            transfer_history=list(patient.transfer_history),
            antibiotic_history=list(patient.antibiotic_history),
            network_alert=None,
            step_number=step,
        )

    def _culture_at_step(self, step: int) -> Dict[str, Any]:
        """Return culture information appropriate for the current step."""
        pathogen = self._scenario["pathogen"]
        if step == 1:
            return {"status": "pending"}
        elif step == 2:
            return {"status": "gram_stain_available", "gram_stain": "negative"}
        elif step == 3:
            return {"status": "organism_identified", "organism": pathogen, "sensitivities": {}}
        else:
            return {
                "status": "full_sensitivities",
                "organism": pathogen,
                "sensitivities": {
                    "correct_class": "susceptible",
                    "broad_class": "susceptible",
                },
            }

    @property
    def ground_truth(self) -> Dict[str, Any]:
        """Ground truth for triage grader."""
        return {
            "correct_drug_class": self._scenario["correct_drug_class"],
            "alt_drug_class": self._scenario["alt_drug_class"],
            "optimal_dose_mg": self._scenario["optimal_dose_mg"],
            "needs_broad": self._scenario["needs_broad"],
            "index_pathogen": self._scenario["pathogen"],
        }
