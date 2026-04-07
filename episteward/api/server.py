"""
FastAPI application — EpiSteward OpenEnv HTTP interface.

Endpoints:
  GET  /           → {"status": "ok", "env": "episteward"}
  GET  /health     → {"status": "ok"}
  POST /reset      → StepResult  (accepts optional body or empty {})
  POST /step       → StepResult
  GET  /state      → StateResult
  GET  /tasks      → list of task ids

Run with:
  uvicorn episteward.api.server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query

from episteward.graders.containment_grader import ContainmentGrader
from episteward.graders.outbreak_grader import OutbreakGrader
from episteward.graders.triage_grader import TriageGrader
from episteward.models import (
    EpiAction,
    ResetRequest,
    StateResult,
    StepResult,
)
from episteward.tasks import TASK_REGISTRY, BaseTask

logger = logging.getLogger(__name__)

app = FastAPI(title="EpiSteward", version="1.0.0")

# ---------------------------------------------------------------------------
# In-memory episode state (single-process, single-session)
# ---------------------------------------------------------------------------

_current_task: Optional[BaseTask] = None
_current_task_id: str = "task1_triage"
_current_seed: int = 0
_step_rewards: List[float] = []
_prev_new_cases: int = 0  # for containment grader

_GRADERS: Dict[str, Any] = {
    "task1_triage": TriageGrader(),
    "task2_containment": ContainmentGrader(),
    "task3_outbreak": OutbreakGrader(),
}


def _get_grader_result(action: EpiAction, task: BaseTask, task_id: str) -> Dict[str, Any]:
    """Dispatch to correct grader and return result dict."""
    assert task.state is not None
    gt = task.ground_truth
    step = task.state.step_number

    if task_id == "task1_triage":
        return _GRADERS["task1_triage"].grade(action, task.state, gt, step)
    elif task_id == "task2_containment":
        global _prev_new_cases
        result = _GRADERS["task2_containment"].grade(
            action, task.state, gt, step, _prev_new_cases
        )
        _prev_new_cases = gt.get("new_cases_total", 0)
        return result
    elif task_id == "task3_outbreak":
        return _GRADERS["task3_outbreak"].grade(action, task.state, gt, step)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_id}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> Dict[str, str]:
    """Health check and env identity."""
    return {"status": "ok", "env": "episteward"}


@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness probe for container startup polling."""
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks() -> List[str]:
    """List all available task IDs."""
    return list(TASK_REGISTRY.keys())


@app.post("/reset")
async def reset_endpoint(
    request: Optional[ResetRequest] = None,
    task: Optional[str] = Query(default=None),
) -> StepResult:
    """
    Reset the environment to a new episode.

    Accepts:
      - Empty body {}
      - Body with optional task_id and seed fields
      - ?task= query parameter
    """
    global _current_task, _current_task_id, _current_seed, _step_rewards, _prev_new_cases

    # Resolve task_id: query param > body > default
    task_id = task or (request.task_id if request else None) or "task1_triage"
    seed = (request.seed if request else None) or 0

    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_id}")

    _current_task_id = task_id
    _current_seed = seed
    _step_rewards = []
    _prev_new_cases = 0

    task_cls = TASK_REGISTRY[task_id]
    _current_task = task_cls()
    observation = _current_task.reset(seed=seed)

    return StepResult(observation=observation, reward=0.0, done=False, info={})


@app.post("/step")
async def step_endpoint(action: EpiAction) -> StepResult:
    """Advance the episode by one step with the given action."""
    global _step_rewards

    if _current_task is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first.",
        )

    observation, done = _current_task.step(action)
    grade = _get_grader_result(action, _current_task, _current_task_id)
    reward = float(grade["reward"])
    _step_rewards.append(reward)

    return StepResult(
        observation=observation,
        reward=reward,
        done=done,
        info=grade.get("info", {}),
    )


@app.get("/state")
async def state_endpoint() -> StateResult:
    """Return a read-only snapshot of the current episode state."""
    if _current_task is None or _current_task.state is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first.",
        )

    state = _current_task.state
    return StateResult(
        task_id=_current_task_id,
        step_number=state.step_number,
        episode_seed=state.episode_seed,
        hospital_state=state.to_dict(),
        is_done=state.is_done,
    )
