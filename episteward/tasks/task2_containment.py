"""
Task 2 — ResistanceContainment (Medium).

6-patient ESBL E. coli cluster, 3-day transfer logs, partial culture data.
Episode length: 15 steps.

The agent must:
  1. Identify the index patient (source of transmission)
  2. Issue correct isolation orders
  3. Adjust empiric therapy for exposed patients
  4. Request appropriate cultures

Simulation:
  - Each step = 8 hours of simulated time
  - New resistance cases generated via network.py if isolation is incomplete
  - Wright-Fisher evolution runs per patient per step

Ground truth:
  - index_patient_id: the source patient
  - exposed_patients: list of patient IDs who need adjusted therapy
  - transmission_chain: ordered list (source → ... → current)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from episteward.models import EpiAction, EpiObservation
from episteward.state import HospitalState, PatientRecord
from episteward.tasks.base import BaseTask

logger = logging.getLogger(__name__)

_WARD_ID = "ward_B"
_INDEX_PATIENT = "P_IDX"
_PATIENTS = [
    {"patient_id": _INDEX_PATIENT, "days_in_ward": 5, "has_culture": True},
    {"patient_id": "P_A1",         "days_in_ward": 3, "has_culture": False},
    {"patient_id": "P_A2",         "days_in_ward": 3, "has_culture": False},
    {"patient_id": "P_B1",         "days_in_ward": 2, "has_culture": False},
    {"patient_id": "P_B2",         "days_in_ward": 2, "has_culture": False},
    {"patient_id": "P_C1",         "days_in_ward": 1, "has_culture": False},
]


class ResistanceContainment(BaseTask):
    """ESBL E. coli cluster containment task."""

    max_steps = 15
    name = "task2_containment"

    def __init__(self) -> None:
        super().__init__()
        self._new_cases_this_episode: int = 0
        self._isolation_bonus_awarded: bool = False
        self._current_patient_idx: int = 0  # which patient the action applies to

    def reset(self, seed: int = 0) -> EpiObservation:
        """Initialise 6-patient ESBL cluster."""
        self.state = HospitalState(task_id=self.name, episode_seed=seed)
        self.state.seed(seed)
        self._new_cases_this_episode = 0
        self._isolation_bonus_awarded = False
        self._current_patient_idx = 0

        for spec in _PATIENTS:
            p = PatientRecord(
                patient_id=spec["patient_id"],
                ward_id=_WARD_ID,
                pathogen="E_coli_ESBL",
                resistance_frequency=1.0 if spec["patient_id"] == _INDEX_PATIENT else 0.05,
            )
            # Build plausible transfer history
            if spec["days_in_ward"] >= 3:
                p.transfer_history = ["admit_ward_A", _WARD_ID]
            else:
                p.transfer_history = [_WARD_ID]
            self.state.patients[spec["patient_id"]] = p

        self.state.ward_infection_counts[_WARD_ID] = 1  # index patient only
        self.state.step_number = 1
        return self._make_observation()

    def step(self, action: EpiAction) -> Tuple[EpiObservation, bool]:
        """Apply action to current patient, simulate spread, advance state."""
        self._assert_ready()
        assert self.state is not None

        pid = _PATIENTS[self._current_patient_idx]["patient_id"]
        patient = self.state.patients[pid]

        patient.antibiotic_history.append(action.model_dump())
        if action.isolation_order:
            patient.is_isolated = True
        if action.culture_requested:
            patient.culture_pending = True

        # Check isolation bonus (within first 3 steps)
        if self.state.step_number <= 3 and not self._isolation_bonus_awarded:
            if patient.is_isolated and pid == _INDEX_PATIENT:
                self._isolation_bonus_awarded = True

        # Simulate spread for non-isolated infected patients
        self._simulate_spread_step()

        # Cycle through patients
        self._current_patient_idx = (self._current_patient_idx + 1) % len(_PATIENTS)
        self.state.step_number += 1
        done = self.state.step_number > self.max_steps
        if done:
            self.state.is_done = True

        return self._make_observation(), done

    def _simulate_spread_step(self) -> None:
        """Simplified transmission: non-isolated infected patients may spread."""
        assert self.state is not None
        rng = self.state.rng
        infected = [
            pid for pid, p in self.state.patients.items()
            if p.resistance_frequency > 0.5 and not p.is_isolated
        ]
        for pid in infected:
            # Each infected patient has small chance of spreading to neighbours
            for target_pid, tp in self.state.patients.items():
                if target_pid == pid or tp.resistance_frequency > 0.5:
                    continue
                if rng.random() < 0.05:  # simplified β
                    tp.resistance_frequency = min(1.0, tp.resistance_frequency + 0.3)
                    self._new_cases_this_episode += 1

    def _make_observation(self) -> EpiObservation:
        assert self.state is not None
        pid = _PATIENTS[self._current_patient_idx]["patient_id"]
        patient = self.state.patients[pid]
        spec = _PATIENTS[self._current_patient_idx]

        culture: Dict[str, Any] = {}
        if spec["has_culture"] or patient.culture_pending:
            culture = {"status": "positive", "organism": "E_coli_ESBL"}
        else:
            culture = {"status": "pending"}

        return EpiObservation(
            patient_id=pid,
            ward_id=_WARD_ID,
            infection_site="urinary_tract",
            symptoms=["fever", "dysuria"],
            vitals={"temp_c": 38.3, "hr_bpm": 92, "wbc_k_ul": 13.5, "crp_mg_l": 45.0, "procalcitonin_ng_ml": 0.5},
            culture_results=culture,
            resistance_flags=["ESBL"] if patient.resistance_frequency > 0.5 else [],
            transfer_history=list(patient.transfer_history),
            antibiotic_history=list(patient.antibiotic_history),
            network_alert=f"ESBL cluster detected in {_WARD_ID}" if self._new_cases_this_episode > 0 else None,
            step_number=self.state.step_number,
        )

    @property
    def ground_truth(self) -> Dict[str, Any]:
        return {
            "index_patient_id": _INDEX_PATIENT,
            "exposed_patients": [p["patient_id"] for p in _PATIENTS if p["patient_id"] != _INDEX_PATIENT],
            "transmission_chain": [_INDEX_PATIENT, "P_A1", "P_A2"],
            "isolation_bonus_awarded": self._isolation_bonus_awarded,
            "new_cases_total": self._new_cases_this_episode,
        }
