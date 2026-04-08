"""
BaseTask — abstract base class for all EpiSteward tasks.

Contract:
  - reset(seed)  → EpiObservation   initialise state, return first observation
  - step(action) → (EpiObservation, bool)   advance state, return (obs, done)
  - ground_truth → dict             ONLY accessed by paired grader, never agent

Subclasses must implement all three.  The done flag must be True on the final
step (max_steps reached or terminal condition met).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from episteward.models import EpiAction, EpiObservation
from episteward.state import HospitalState


class BaseTask(ABC):
    """Abstract base for EpiSteward episode tasks."""

    #: Maximum number of steps before forced episode termination.
    max_steps: int = 5

    #: Human-readable task name.
    name: str = "base"

    def __init__(self) -> None:
        self.state: HospitalState | None = None

    @abstractmethod
    def reset(self, seed: int = 0) -> EpiObservation:
        """
        Initialise a new episode with the given random seed.

        Must create a fresh HospitalState, seed its RNG, populate patients,
        and return the first observation.
        """
        ...

    @abstractmethod
    def step(self, action: EpiAction) -> Tuple[EpiObservation, bool]:
        """
        Apply *action*, advance simulation by one step.

        Returns
        -------
        observation : EpiObservation  — new observation after action
        done        : bool            — True if episode is finished
        """
        ...

    @property
    @abstractmethod
    def ground_truth(self) -> Dict[str, Any]:
        """
        Ground-truth data for grading.

        This property is intentionally not part of the observation — it is
        only accessed by the grader, never by the agent.
        """
        ...

    def _assert_ready(self) -> None:
        """Raise RuntimeError if reset() has not been called yet."""
        if self.state is None:
            raise RuntimeError(
                f"Task '{self.name}' has not been reset. Call reset() first."
            )
