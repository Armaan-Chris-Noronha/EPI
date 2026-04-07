"""Tests for grader correctness: [0,1] range, determinism, partial credit."""

import pytest
from episteward.models import EpiAction
from episteward.graders import TriageGrader, ContainmentGrader, OutbreakGrader
from episteward.tasks import PrescriptionTriage, ResistanceContainment, NetworkOutbreakResponse


_ACTION = EpiAction(
    antibiotic="meropenem",
    dose_mg=1000.0,
    frequency_hours=8.0,
    duration_days=7,
    route="IV",
    isolation_order=True,
    culture_requested=True,
)

_BAD_ACTION = EpiAction(
    antibiotic="azithromycin",
    dose_mg=50.0,
    frequency_hours=24.0,
    duration_days=1,
    route="PO",
    isolation_order=False,
    culture_requested=False,
)


def _setup_triage():
    task = PrescriptionTriage()
    task.reset(seed=0)
    task.step(_ACTION)  # put some history in
    return task, TriageGrader()


def test_triage_reward_in_range():
    task, grader = _setup_triage()
    result = grader.grade(_ACTION, task.state, task.ground_truth, step_number=2)
    assert 0.0 <= result["reward"] <= 1.0


def test_triage_partial_credit_bad_action():
    task, grader = _setup_triage()
    result = grader.grade(_BAD_ACTION, task.state, task.ground_truth, step_number=2)
    assert result["reward"] >= 0.0  # never negative
    # Bad action should score less than good action
    good = grader.grade(_ACTION, task.state, task.ground_truth, step_number=2)
    assert good["reward"] >= result["reward"]


def test_triage_deterministic():
    task, grader = _setup_triage()
    r1 = grader.grade(_ACTION, task.state, task.ground_truth, step_number=3)
    r2 = grader.grade(_ACTION, task.state, task.ground_truth, step_number=3)
    assert r1["reward"] == r2["reward"]


def test_containment_reward_in_range():
    task = ResistanceContainment()
    task.reset(seed=0)
    task.step(_ACTION)
    grader = ContainmentGrader()
    result = grader.grade(_ACTION, task.state, task.ground_truth, step_number=2)
    assert 0.0 <= result["reward"] <= 1.0


def test_outbreak_reward_in_range():
    task = NetworkOutbreakResponse()
    task.reset(seed=0)
    task.step(_ACTION)
    grader = OutbreakGrader()
    result = grader.grade(_ACTION, task.state, task.ground_truth, step_number=2)
    assert 0.0 <= result["reward"] <= 1.0


def test_all_graders_have_components():
    """Each grader must return a non-empty components dict."""
    for task_cls, grader in [
        (PrescriptionTriage, TriageGrader()),
        (ResistanceContainment, ContainmentGrader()),
        (NetworkOutbreakResponse, OutbreakGrader()),
    ]:
        task = task_cls()
        task.reset(seed=0)
        task.step(_ACTION)
        result = grader.grade(_ACTION, task.state, task.ground_truth, step_number=2)
        assert isinstance(result["components"], dict)
        assert len(result["components"]) > 0
