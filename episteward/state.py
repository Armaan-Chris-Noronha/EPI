"""
HospitalState — the mutable episode state shared across tasks and graders.

This dataclass holds all simulation state for one episode. It is seeded in
reset() so episodes are reproducible. Tasks mutate it; /state serializes it
read-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class PatientRecord:
    """Per-patient mutable state within an episode."""

    patient_id: str
    ward_id: str
    pathogen: Optional[str] = None
    resistance_frequency: float = 0.0  # Wright-Fisher allele frequency
    is_isolated: bool = False
    is_treated: bool = False
    culture_pending: bool = False
    treatment_hours_elapsed: float = 0.0
    transfer_history: List[str] = field(default_factory=list)
    antibiotic_history: List[Dict[str, Any]] = field(default_factory=list)
    alive: bool = True


@dataclass
class HospitalState:
    """Full mutable state for one EpiSteward episode."""

    task_id: str
    episode_seed: int
    step_number: int = 0
    is_done: bool = False

    patients: Dict[str, PatientRecord] = field(default_factory=dict)
    ward_infection_counts: Dict[str, int] = field(default_factory=dict)
    colistin_budget: int = 10  # Task 3 only — total courses available
    colistin_used: int = 0

    # Loaded static data (shared across episodes, not re-seeded)
    _antibiotics: Dict[str, Any] = field(default_factory=dict)
    _pathogens: Dict[str, Any] = field(default_factory=dict)
    _resistance_profiles: Dict[str, Any] = field(default_factory=dict)
    _hospital_network: Dict[str, Any] = field(default_factory=dict)

    # Per-episode RNG — seeded in reset() for reproducibility
    rng: Any = field(default_factory=lambda: np.random.default_rng())

    def __post_init__(self) -> None:
        """Load static data files once on construction."""
        self._antibiotics = json.loads((DATA_DIR / "antibiotics.json").read_text())
        self._pathogens = json.loads((DATA_DIR / "pathogens.json").read_text())
        self._resistance_profiles = json.loads(
            (DATA_DIR / "resistance_profiles.json").read_text()
        )
        self._hospital_network = json.loads(
            (DATA_DIR / "hospital_network.json").read_text()
        )

    def seed(self, seed: int) -> None:
        """Re-initialize the episode RNG for reproducibility."""
        self.rng = np.random.default_rng(seed)
        self.episode_seed = seed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict for /state response (read-only snapshot)."""
        return {
            "task_id": self.task_id,
            "episode_seed": self.episode_seed,
            "step_number": self.step_number,
            "is_done": self.is_done,
            "colistin_budget": self.colistin_budget,
            "colistin_used": self.colistin_used,
            "patients": {
                pid: {
                    "ward_id": p.ward_id,
                    "pathogen": p.pathogen,
                    "resistance_frequency": p.resistance_frequency,
                    "is_isolated": p.is_isolated,
                    "alive": p.alive,
                }
                for pid, p in self.patients.items()
            },
            "ward_infection_counts": dict(self.ward_infection_counts),
        }
