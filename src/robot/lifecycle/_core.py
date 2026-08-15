"""Async-safe application lifecycle manager.

The lifecycle is the only place that owns the root task group and the
event-bus subscription. Components subscribe to events in ``startup`` and
unsubscribe in ``shutdown``; the manager guarantees both are called exactly
once and that all errors are surfaced.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import anyio

from robot.errors import LifecycleError
from robot.interfaces.event_bus import EventBus
from robot.logging import get_logger

_log = get_logger("lifecycle")

StartupHook: TypeAlias = Callable[[], Awaitable[None]]
ShutdownHook: TypeAlias = Callable[[], Awaitable[None]]


class LifecycleState(str, Enum):
    """High-level state of the application lifecycle."""

    NEW = "new"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(slots=True, frozen=True)
class LifecycleHooks:
    """Container for startup and shutdown callbacks."""

    on_startup: tuple[StartupHook, ...] = ()
    on_shutdown: tuple[ShutdownHook, ...] = ()


class Lifecycle:
    """Cooperative startup/shutdown controller.

    Example
    -------
    >>> lifecycle = Lifecycle(bus=event_bus)
    >>> lifecycle.add_startup(my_component.start)
    >>> async with lifecycle.running() as task_group:
    ...     await anyio.sleep_forever()
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._state: LifecycleState = LifecycleState.NEW
        self._startup_hooks: list[StartupHook] = []
        self._shutdown_hooks: list[ShutdownHook] = []
        self._task_group: anyio.abc.TaskGroup | None = None

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state is LifecycleState.RUNNING

    # ------------------------------------------------------------------ hooks
    def add_startup(self, hook: StartupHook) -> None:
        if self._state is not LifecycleState.NEW:
            raise LifecycleError("cannot add startup hooks after startup")
        self._startup_hooks.append(hook)

    def add_shutdown(self, hook: ShutdownHook) -> None:
        if self._state is not LifecycleState.STOPPED:
            self._shutdown_hooks.append(hook)

    # ------------------------------------------------------------------ run
    @contextlib.asynccontextmanager
    async def running(self) -> AsyncIterator[anyio.abc.TaskGroup]:
        """Run the application inside an ``anyio`` task group.

        Yields the task group so callers can spawn long-running services
        (HTTP server, audio loop, etc.) alongside the lifecycle.
        """
        if self._state is not LifecycleState.NEW:
            raise LifecycleError(f"cannot start from state {self._state}")

        async with anyio.create_task_group() as tg:
            self._task_group = tg
            self._state = LifecycleState.STARTING
            await self._run_hooks(self._startup_hooks, "startup")
            self._state = LifecycleState.RUNNING
            _log.info("lifecycle.running")
            try:
                yield tg
            finally:
                self._state = LifecycleState.STOPPING
                await self._run_hooks(self._shutdown_hooks, "shutdown")
                self._state = LifecycleState.STOPPED
                self._task_group = None
                _log.info("lifecycle.stopped")

    # ------------------------------------------------------------------ misc
    async def _run_hooks(self, hooks: list[Callable[[], Awaitable[None]]], phase: str) -> None:
        for hook in hooks:
            try:
                await hook()
            except Exception:
                _log.exception("lifecycle.hook_failed", phase=phase, hook=hook)
                raise


__all__ = ["Lifecycle", "LifecycleHooks", "LifecycleState", "ShutdownHook", "StartupHook"]
