"""Tests for the lifecycle manager."""

from __future__ import annotations

import anyio
import pytest

from robot.errors import LifecycleError
from robot.events.bus import InMemoryEventBus
from robot.lifecycle import Lifecycle


async def test_lifecycle_runs_hooks() -> None:
    bus = InMemoryEventBus()
    lifecycle = Lifecycle(bus=bus)
    started = False
    stopped = False

    async def startup() -> None:
        nonlocal started
        started = True

    async def shutdown() -> None:
        nonlocal stopped
        stopped = True

    lifecycle.add_startup(startup)
    lifecycle.add_shutdown(shutdown)
    async with lifecycle.running():
        assert started
        assert lifecycle.is_running
    assert stopped
    assert lifecycle.state.value == "stopped"


async def test_lifecycle_rejects_double_start() -> None:
    bus = InMemoryEventBus()
    lifecycle = Lifecycle(bus=bus)
    async with lifecycle.running():
        with pytest.raises(LifecycleError):
            async with lifecycle.running():
                pass


async def test_lifecycle_rejects_startup_after_start() -> None:
    bus = InMemoryEventBus()
    lifecycle = Lifecycle(bus=bus)
    async with lifecycle.running():
        with pytest.raises(LifecycleError):
            lifecycle.add_startup(lambda: anyio.sleep(0))
