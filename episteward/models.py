"""
Pydantic v2 models for EpiSteward.

All API boundary types live here:
  - EpiObservation  — what the agent sees each step
  - EpiAction       — what the agent submits each step
  - EpiReward       — reward breakdown returned by graders
  - StepResult      — envelope returned by /step and /reset
  - StateResult     — envelope returned by /state (read-only)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EpiObservation(BaseModel):
    """Full observation delivered to the agent at each step."""

    patient_id: str
    ward_id: str
    infection_site: str
    symptoms: List[str]
    vitals: Dict[str, float]  # temp_c, hr_bpm, wbc_k_ul, crp_mg_l, procalcitonin_ng_ml
    culture_results: Dict[str, Any]  # may have missing fields for realism
    resistance_flags: List[str]  # e.g. ["ESBL", "MRSA", "CRE"]
    transfer_history: List[str]  # ward IDs visited, chronological
    antibiotic_history: List[Dict[str, Any]]
    network_alert: Optional[str] = None
    step_number: int


class EpiAction(BaseModel):
    """Action submitted by the agent."""

    antibiotic: str  # must match a key in antibiotics.json
    dose_mg: float = Field(gt=0)
    frequency_hours: float = Field(gt=0)
    duration_days: int = Field(gt=0)
    route: str  # "IV", "PO", or "IM"
    isolation_order: bool = False
    culture_requested: bool = False
    specialist_consult: bool = False
    reasoning: Optional[str] = None  # logged but not graded


class EpiReward(BaseModel):
    """Reward breakdown returned by a grader."""

    value: float  # 0.0–1.0 after clamping
    components: Dict[str, float]  # e.g. {"pkpd": 0.3, "stewardship": 0.2, ...}
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Envelope returned by /reset and /step."""

    observation: EpiObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    """Read-only snapshot returned by /state. Must not advance episode."""

    task_id: str
    step_number: int
    episode_seed: int
    hospital_state: Dict[str, Any]  # serialized HospitalState
    is_done: bool


class ResetRequest(BaseModel):
    """Optional body for POST /reset. All fields optional so {} is valid."""

    task_id: Optional[str] = "task1_triage"
    seed: Optional[int] = None
