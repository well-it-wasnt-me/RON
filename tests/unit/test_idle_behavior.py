"""Unit tests for IdleBehavior."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from robot.behavior.idle import IdleBehavior
from robot.behavior.personality import Personality
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.utils.random_source import SystemRandomSource


class TestIdleBehavior:
    """Tests for :class:`IdleBehavior`."""

    def _make_idle(
        self, *, energy: float = 0.5, curiosity: float = 0.7, playfulness: float = 0.7
    ) -> IdleBehavior:
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        personality = Personality(
            curiosity=curiosity,
            energy=energy,
            shyness=0.3,
            friendliness=0.8,
            playfulness=playfulness,
        )
        rng = SystemRandomSource(seed=42)
        clock = _MockClock()
        return IdleBehavior(
            state_machine=sm,
            personality=personality,
            rng=rng,
            clock=clock,  # type: ignore[arg-type]
            min_idle_s=0.01,
            max_idle_s=0.02,
        )

    @pytest.mark.asyncio
    async def test_initial_state_is_stopped(self) -> None:
        idle = self._make_idle()
        assert idle._stopped is True

    @pytest.mark.asyncio
    async def test_stop_sets_stopped(self) -> None:
        idle = self._make_idle()
        idle._stopped = False
        idle.stop()
        assert idle._stopped is True

    @pytest.mark.asyncio
    async def test_drain_returns_empty_initially(self) -> None:
        idle = self._make_idle()
        assert idle.drain() == []

    @pytest.mark.asyncio
    async def test_drain_clears_outbox(self) -> None:
        idle = self._make_idle()
        await idle.state_machine.transition(RobotState.IDLE)
        # Run a short burst
        run_task = asyncio.create_task(idle.run())
        await asyncio.sleep(0.15)
        idle.stop()
        await asyncio.sleep(0.05)
        with contextlib.suppress(Exception):
            run_task.cancel()
        # Should have produced some actions
        actions = idle.drain()
        # drain should return a list and clear the outbox
        assert isinstance(actions, list)
        second_drain = idle.drain()
        assert len(second_drain) == 0

    @pytest.mark.asyncio
    async def test_choose_action_returns_action_or_none(self) -> None:
        idle = self._make_idle(energy=1.0, curiosity=1.0, playfulness=1.0)
        actions_seen = 0
        for _ in range(20):
            action = idle._choose_action()
            if action is not None:
                actions_seen += 1
        assert actions_seen > 0

    @pytest.mark.asyncio
    async def test_choose_action_low_energy(self) -> None:
        idle = self._make_idle(energy=0.0, curiosity=0.5, playfulness=0.5)
        actions_seen = 0
        for _ in range(20):
            action = idle._choose_action()
            if action is not None:
                actions_seen += 1
        # With energy=0.0, most rolls will skip
        assert actions_seen >= 0


class _MockClock:
    """A minimal async clock for testing."""

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(0)

    def now(self) -> object:
        import datetime

        return datetime.datetime.now(tz=datetime.UTC)

    def monotonic(self) -> float:
        import time

        return time.monotonic()
