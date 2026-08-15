"""Async job scheduler + animation runner.

The scheduler is a lightweight cooperative scheduler with three capabilities:

* **One-shot jobs** with an optional delay.
* **Periodic jobs** with a fixed interval.
* **Cancelable tasks** that can be stopped by handle.

It is the only place that owns the per-job ``anyio`` task group, so cancelling
the whole scheduler is a single operation.
"""

from __future__ import annotations

import contextlib
import inspect
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import anyio

from robot.errors import AnimationError
from robot.logging import get_logger
from robot.utils.clock import Clock, SystemClock

_log = get_logger("animation.scheduler")


Callback = Callable[[float], Any]
AsyncCallback = Callable[[float], Awaitable[Any]]


async def _invoke_callback(callback: Callback | AsyncCallback, value: float) -> None:
    result = callback(value)
    if inspect.isawaitable(result):
        await result


@dataclass(slots=True)
class ScheduledTask:
    """Handle to a running job; pass to :meth:`AnimationScheduler.cancel`."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "task"
    _scope: anyio.CancelScope | None = None

    def cancel(self) -> None:
        if self._scope is not None:
            self._scope.cancel()


class AnimationScheduler:
    """Cooperative scheduler with periodic + one-shot jobs."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock or SystemClock()
        self._task_group: anyio.abc.TaskGroup | None = None
        self._tasks: dict[str, ScheduledTask] = {}
        self._closed = False

    # ------------------------------------------------------------------ start/stop
    async def start(self) -> None:
        if self._task_group is not None:
            return
        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()

    async def stop(self) -> None:
        if self._task_group is None:
            return
        try:
            self._task_group.cancel_scope.cancel()
        finally:
            with contextlib.suppress(Exception):
                await self._task_group.__aexit__(None, None, None)
            self._task_group = None
            self._tasks.clear()
            self._closed = True

    # ------------------------------------------------------------------ jobs
    def schedule(
        self,
        callback: Callback | AsyncCallback,
        *,
        delay_s: float = 0.0,
        interval_s: float | None = None,
        name: str = "job",
    ) -> ScheduledTask:
        """Schedule a one-shot or periodic job."""
        if self._task_group is None:
            raise AnimationError("scheduler not started")
        if interval_s is not None and interval_s <= 0:
            raise AnimationError("interval_s must be > 0")

        task = ScheduledTask(name=name)
        task._scope = anyio.CancelScope()
        self._tasks[task.id] = task
        scope = task._scope

        async def _runner() -> None:
            with scope:
                try:
                    if delay_s > 0:
                        await self._clock.sleep(delay_s)
                    if interval_s is None:
                        await _invoke_callback(callback, 0.0)
                        return
                    while True:
                        await _invoke_callback(callback, 0.0)
                        await self._clock.sleep(interval_s)
                except anyio.get_cancelled_exc_class():
                    raise
                except Exception:
                    _log.exception("scheduler.job_failed", name=name)
                    raise

        self._task_group.start_soon(_runner)
        return task

    def cancel(self, task: ScheduledTask) -> None:
        task.cancel()
        self._tasks.pop(task.id, None)

    def cancel_all(self) -> None:
        for task in list(self._tasks.values()):
            self.cancel(task)

    @contextlib.asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        await self.start()
        try:
            yield
        finally:
            await self.stop()


__all__ = ["AnimationScheduler", "ScheduledTask"]
