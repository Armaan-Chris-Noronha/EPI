"""
Task 3 — NetworkOutbreakResponse (Hard).

10-hospital network, CRK spreading, finite colistin budget.
Episode length: 30 steps.

The agent must:
  1. Trace phylogenetic spread from transfer logs and resistance typing
  2. Issue hospital-level containment orders (economic penalty per order)
  3. Allocate colistin budget (fixed 10 courses total)
  4. Maintain treatment for non-CRK patients simultaneously

Reward formula:
    R = α·lives_saved_ratio - β·colistin_overspend - γ·resistance_amplification_events
    α=0.6, β=0.25, γ=0.15

Ground truth:
  - source_hospital: where CRK originated
  - transmission_tree: graph of spread events
  - total_crk_patients: final infected count
  - lives_at_risk: patients who need colistin
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

from episteward.models import EpiAction, EpiObservation
from episteward.state import HospitalState, PatientRecord
from episteward.tasks.base import BaseTask

logger = logging.getLogger(__name__)

_ALPHA = 0.6
_BETA = 0.25
_GAMMA = 0.15
_COLISTIN_BUDGET = 10

# 10-hospital network (simplified; full topology in hospital_network.json)
_HOSPITALS = [f"H{i}" for i in range(1, 11)]
_SOURCE_HOSPITAL = "H3"

_INITIAL_PATIENTS: List[Dict[str, Any]] = [
    {"patient_id": f"H{h}_P{p}", "hospital": f"H{h}", "crk": h == 3 and p == 1}
    for h in range(1, 11)
    for p in range(1, 4)  # 3 patients per hospital = 30 total
]


class NetworkOutbreakResponse(BaseTask):
    """10-hospital CRK network outbreak response task."""

    max_steps = 30
    name = "task3_outbreak"

    def __init__(self) -> None:
        super().__init__()
        self._containment_orders: Set[str] = set()
        self._resistance_events: int = 0
        self._lives_saved: int = 0
        self._total_at_risk: int = 0

    def reset(self, seed: int = 0) -> EpiObservation:
        """Initialise 10-hospital network with CRK seeded in H3."""
        self.state = HospitalState(task_id=self.name, episode_seed=seed)
        self.state.seed(seed)
        self.state.colistin_budget = _COLISTIN_BUDGET
        self.state.colistin_used = 0
        self._containment_orders = set()
        self._resistance_events = 0
        self._lives_saved = 0

        for spec in _INITIAL_PATIENTS:
            p = PatientRecord(
                patient_id=spec["patient_id"],
                ward_id=spec["hospital"],
                pathogen="K_pneumoniae_CRK" if spec["crk"] else "K_pneumoniae",
                resistance_frequency=1.0 if spec["crk"] else 0.01,
            )
            self.state.patients[p.patient_id] = p

        self._total_at_risk = sum(
            1 for p in self.state.patients.values()
            if p.resistance_frequency > 0.5
        )

        self.state.ward_infection_counts[_SOURCE_HOSPITAL] = 1
        self.state.step_number = 1
        return self._make_observation()

    def step(self, action: EpiAction) -> Tuple[EpiObservation, bool]:
        """Apply action (may target a hospital-level patient), spread CRK."""
        self._assert_ready()
        assert self.state is not None

        # Determine current focus patient (cycle by step)
        pid = _INITIAL_PATIENTS[(self.state.step_number - 1) % len(_INITIAL_PATIENTS)]["patient_id"]
        patient = self.state.patients[pid]

        patient.antibiotic_history.append(action.model_dump())

        if action.isolation_order:
            self._containment_orders.add(patient.ward_id)
            patient.is_isolated = True

        # Colistin allocation
        if action.antibiotic.lower() == "colistin":
            if self.state.colistin_used < self.state.colistin_budget:
                self.state.colistin_used += 1
                if patient.resistance_frequency > 0.5:
                    patient.alive = True
                    self._lives_saved += 1
            else:
                logger.warning("Colistin budget exhausted")

        # Simulate CRK spread across network
        self._simulate_network_spread()

        self.state.step_number += 1
        done = self.state.step_number > self.max_steps
        if done:
            self.state.is_done = True

        return self._make_observation(), done

    def _simulate_network_spread(self) -> None:
        """Simplified network spread step."""
        assert self.state is not None
        rng = self.state.rng

        crk_hospitals = {
            p.ward_id for p in self.state.patients.values()
            if p.resistance_frequency > 0.5 and p.ward_id not in self._containment_orders
        }

        for h in crk_hospitals:
            # Each CRK hospital can spread to a random neighbour
            idx = int(rng.integers(0, len(_HOSPITALS)))
            target_h = _HOSPITALS[idx]
            if target_h in self._containment_orders:
                continue
            if rng.random() < 0.08:  # β for CRK
                # Infect first non-CRK patient in target hospital
                for pid, p in self.state.patients.items():
                    if p.ward_id == target_h and p.resistance_frequency < 0.5:
                        p.resistance_frequency = 0.9
                        self._resistance_events += 1
                        break

    def _make_observation(self) -> EpiObservation:
        assert self.state is not None
        step = self.state.step_number
        pid = _INITIAL_PATIENTS[(step - 1) % len(_INITIAL_PATIENTS)]["patient_id"]
        patient = self.state.patients[pid]

        crk_count = sum(
            1 for p in self.state.patients.values() if p.resistance_frequency > 0.5
        )
        budget_remaining = self.state.colistin_budget - self.state.colistin_used

        return EpiObservation(
            patient_id=pid,
            ward_id=patient.ward_id,
            infection_site="bloodstream",
            symptoms=["fever", "hypotension", "tachycardia"],
            vitals={"temp_c": 39.1, "hr_bpm": 115, "wbc_k_ul": 19.0, "crp_mg_l": 140.0, "procalcitonin_ng_ml": 5.5},
            culture_results={"status": "positive", "organism": "K_pneumoniae_CRK"} if patient.resistance_frequency > 0.5 else {"status": "pending"},
            resistance_flags=["CRK"] if patient.resistance_frequency > 0.5 else [],
            transfer_history=list(patient.transfer_history),
            antibiotic_history=list(patient.antibiotic_history),
            network_alert=(
                f"CRK outbreak: {crk_count} cases across network. "
                f"Colistin budget: {budget_remaining}/{self.state.colistin_budget}"
            ),
            step_number=step,
        )

    @property
    def ground_truth(self) -> Dict[str, Any]:
        assert self.state is not None
        crk_patients = [
            pid for pid, p in self.state.patients.items()
            if p.resistance_frequency > 0.5
        ]
        colistin_overspend = max(0, self.state.colistin_used - self.state.colistin_budget)
        return {
            "source_hospital": _SOURCE_HOSPITAL,
            "total_crk_patients": len(crk_patients),
            "lives_at_risk": self._total_at_risk,
            "lives_saved": self._lives_saved,
            "colistin_overspend": colistin_overspend,
            "resistance_amplification_events": self._resistance_events,
            "containment_orders": list(self._containment_orders),
            "lives_saved_ratio": self._lives_saved / max(1, self._total_at_risk),
        }
