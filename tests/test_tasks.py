"""Tests for task reset/step and observation validity."""

import pytest
from episteward.models import EpiAction
from episteward.tasks import PrescriptionTriage, ResistanceContainment, NetworkOutbreakResponse


_BASIC_ACTION = EpiAction(
    antibiotic="meropenem",
    dose_mg=1000.0,
    frequency_hours=8.0,
    duration_days=7,
    route="IV",
    isolation_order=False,
    culture_requested=True,
)


def test_triage_reset_returns_observation():
    task = PrescriptionTriage()
    obs = task.reset(seed=0)
    assert obs.step_number == 1
    assert obs.patient_id != ""


def test_triage_step_advances():
    task = PrescriptionTriage()
    task.reset(seed=0)
    obs, done = task.step(_BASIC_ACTION)
    assert obs.step_number == 2


def test_triage_terminates_at_max_steps():
    task = PrescriptionTriage()
    task.reset(seed=0)
    done = False
    for _ in range(10):
        _, done = task.step(_BASIC_ACTION)
        if done:
            break
    assert done


def test_containment_reset():
    task = ResistanceContainment()
    obs = task.reset(seed=1)
    assert obs.ward_id == "ward_B"


def test_outbreak_reset():
    task = NetworkOutbreakResponse()
    obs = task.reset(seed=2)
    assert obs.network_alert is not None


def test_ground_truth_not_in_observation():
    task = PrescriptionTriage()
    obs = task.reset(seed=0)
    # Ground truth fields must not bleed into observation
    obs_dict = obs.model_dump()
    assert "correct_drug_class" not in obs_dict
    assert "optimal_dose_mg" not in obs_dict
