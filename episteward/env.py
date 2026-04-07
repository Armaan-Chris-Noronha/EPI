"""
EpiStewardEnv — the top-level OpenEnv environment class.

Supports two usage modes:
  1. In-process (default): tasks run directly in the same Python process.
     Used for testing and when inference.py runs inside the container.
  2. Docker client mode: from_docker_image() launches a container and
     proxies all calls over HTTP. Used by inference.py outside the container.

Both modes expose the same async interface:
    env = await EpiStewardEnv.from_docker_image(image_name)
    result = await env.reset(task_id="task1_triage")
    result = await env.step(action)
    await env.close()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

from episteward.models import EpiAction, ResetRequest, StepResult, StateResult

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 7860
_HEALTH_TIMEOUT = 60  # seconds to wait for container /health


class EpiStewardEnv:
    """
    OpenEnv-compliant environment for antibiotic stewardship RL.

    Use ``from_docker_image`` to get an HTTP-backed instance, or
    instantiate directly for in-process use.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        _in_process: bool = False,
    ) -> None:
        self._base_url = base_url
        self._in_process = _in_process
        self._client: Optional[httpx.AsyncClient] = None
        self._container_id: Optional[str] = None

        # In-process backend (lazy import to avoid circular at module level)
        self._backend: Any = None

    # ------------------------------------------------------------------
    # Factory — Docker client mode
    # ------------------------------------------------------------------

    @classmethod
    async def from_docker_image(cls, image_name: str) -> "EpiStewardEnv":
        """
        Launch a Docker container from *image_name*, wait for /health,
        and return a configured HTTP client wrapping the container.

        Raises RuntimeError if the container does not become healthy within
        _HEALTH_TIMEOUT seconds.
        """
        import subprocess

        port = _DEFAULT_PORT
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "-p",
                f"{port}:{port}",
                image_name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        container_id = result.stdout.strip()
        base_url = f"http://localhost:{port}"

        env = cls(base_url=base_url)
        env._container_id = container_id
        env._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

        # Poll /health until ready
        deadline = time.monotonic() + _HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            try:
                r = await env._client.get("/health")
                if r.status_code == 200:
                    logger.info("Container %s healthy at %s", container_id[:12], base_url)
                    return env
            except httpx.ConnectError:
                pass
            await asyncio.sleep(1)

        await env.close()
        raise RuntimeError(
            f"Container {container_id[:12]} did not become healthy within "
            f"{_HEALTH_TIMEOUT}s"
        )

    # ------------------------------------------------------------------
    # Factory — in-process mode (for tests / running inside container)
    # ------------------------------------------------------------------

    @classmethod
    def in_process(cls) -> "EpiStewardEnv":
        """Return an in-process environment (no Docker, no HTTP)."""
        env = cls(_in_process=True)
        from episteward.api import server as _server_module

        env._backend = _server_module
        return env

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    async def reset(self, task_id: str = "task1_triage") -> StepResult:
        """Reset the environment to a new episode for *task_id*."""
        payload = ResetRequest(task_id=task_id)
        if self._in_process:
            return await self._backend.reset_endpoint(payload)
        assert self._client is not None
        r = await self._client.post(
            "/reset",
            json=payload.model_dump(),
            params={"task": task_id},
        )
        r.raise_for_status()
        return StepResult.model_validate(r.json())

    async def step(self, action: EpiAction) -> StepResult:
        """Submit *action* and advance the episode by one step."""
        if self._in_process:
            return await self._backend.step_endpoint(action)
        assert self._client is not None
        r = await self._client.post("/step", json=action.model_dump())
        r.raise_for_status()
        return StepResult.model_validate(r.json())

    async def state(self) -> StateResult:
        """Return a read-only snapshot of the current episode state."""
        if self._in_process:
            return await self._backend.state_endpoint()
        assert self._client is not None
        r = await self._client.get("/state")
        r.raise_for_status()
        return StateResult.model_validate(r.json())

    async def close(self) -> None:
        """Stop and remove the Docker container (no-op in in-process mode)."""
        if self._client:
            await self._client.aclose()
        if self._container_id:
            import subprocess

            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
            )
            logger.info("Stopped container %s", self._container_id[:12])
