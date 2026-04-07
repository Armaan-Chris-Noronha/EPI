"""Tests for Pydantic model round-trip serialization and validation."""

import pytest
from episteward.models import EpiAction, EpiObservation, EpiReward, StepResult


def test_epi_action_round_trip():
    action = EpiAction(
        antibiotic="meropenem",
        dose_mg=1000.0,
        frequency_hours=8.0,
        duration_days=7,
        route="IV",
        isolation_order=False,
        culture_requested=True,
    )
    data = action.model_dump()
    restored = EpiAction.model_validate(data)
    assert restored.antibiotic == "meropenem"
    assert restored.dose_mg == 1000.0


def test_epi_observation_round_trip():
    obs = EpiObservation(
        patient_id="P001",
        ward_id="icu",
        infection_site="bloodstream",
        symptoms=["fever", "hypotension"],
        vitals={"temp_c": 39.0, "hr_bpm": 110, "wbc_k_ul": 18.0, "crp_mg_l": 100.0, "procalcitonin_ng_ml": 3.0},
        culture_results={"status": "pending"},
        resistance_flags=["ESBL"],
        transfer_history=["ward_A", "icu"],
        antibiotic_history=[],
        step_number=1,
    )
    data = obs.model_dump()
    restored = EpiObservation.model_validate(data)
    assert restored.patient_id == "P001"
    assert "ESBL" in restored.resistance_flags


def test_step_result_defaults():
    obs = EpiObservation(
        patient_id="P001", ward_id="icu", infection_site="lung",
        symptoms=[], vitals={}, culture_results={},
        resistance_flags=[], transfer_history=[], antibiotic_history=[],
        step_number=1,
    )
    result = StepResult(observation=obs)
    assert result.reward == 0.0
    assert result.done is False


def test_action_dose_positive():
    with pytest.raises(Exception):
        EpiAction(
            antibiotic="meropenem", dose_mg=-100.0,
            frequency_hours=8.0, duration_days=7, route="IV",
        )
