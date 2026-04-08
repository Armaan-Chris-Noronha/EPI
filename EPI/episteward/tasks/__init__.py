"""
Tasks sub-package — the three EpiSteward episodes.

  task1_triage      — PrescriptionTriage        (Easy,   5 steps)
  task2_containment — ResistanceContainment      (Medium, 15 steps)
  task3_outbreak    — NetworkOutbreakResponse    (Hard,   30 steps)

Each task implements BaseTask and is responsible for:
  - Generating the initial observation (reset)
  - Advancing simulation state (step)
  - Providing ground-truth data for its paired grader
"""

from episteward.tasks.base import BaseTask
from episteward.tasks.task1_triage import PrescriptionTriage
from episteward.tasks.task2_containment import ResistanceContainment
from episteward.tasks.task3_outbreak import NetworkOutbreakResponse

TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "task1_triage": PrescriptionTriage,
    "task2_containment": ResistanceContainment,
    "task3_outbreak": NetworkOutbreakResponse,
}

__all__ = [
    "BaseTask",
    "PrescriptionTriage",
    "ResistanceContainment",
    "NetworkOutbreakResponse",
    "TASK_REGISTRY",
]
