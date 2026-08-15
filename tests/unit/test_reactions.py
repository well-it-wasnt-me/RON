"""Unit tests for ReactionEngine."""

from __future__ import annotations

import pytest

from robot.behavior.actions import (
    CelebrateAction,
    LookAroundAction,
    RequestBlinkAction,
)
from robot.behavior.reactions import ReactionEngine
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    FaceDetected,
    SpeechRecognized,
    StateChanged,
    WakeWordDetected,
)


class TestReactionEngine:
    """Tests for :class:`ReactionEngine`."""

    def _make_engine(self) -> ReactionEngine:
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        return ReactionEngine(bus=bus, state_machine=sm)

    def test_attach_subscribes_to_events(self) -> None:
        engine = self._make_engine()
        engine.attach()
        engine.detach()

    def test_detach_unsubscribes(self) -> None:
        engine = self._make_engine()
        engine.attach()
        engine.detach()

    def test_drain_returns_empty_initially(self) -> None:
        engine = self._make_engine()
        assert engine.drain() == []

    def test_drain_clears_outbox(self) -> None:
        engine = self._make_engine()
        engine._outbox.append(RequestBlinkAction(speed=1.0))
        result = engine.drain()
        assert len(result) == 1
        assert engine.drain() == []

    @pytest.mark.asyncio
    async def test_wake_word_produces_actions(self) -> None:
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        engine = ReactionEngine(bus=bus, state_machine=sm)
        engine.attach()

        await bus.publish(WakeWordDetected(phrase="hey deskbot"))
        actions = engine.drain()

        assert len(actions) >= 1
        action_types = [type(a) for a in actions]
        assert RequestBlinkAction in action_types

        engine.detach()

    @pytest.mark.asyncio
    async def test_face_detected_produces_look_action(self) -> None:
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        engine = ReactionEngine(bus=bus, state_machine=sm)
        engine.attach()

        await bus.publish(FaceDetected(x=0.5, y=0.5, confidence=0.9))
        actions = engine.drain()

        assert len(actions) >= 1
        action_types = [type(a) for a in actions]
        assert LookAroundAction in action_types

        engine.detach()

    @pytest.mark.asyncio
    async def test_speech_recognized_produces_blink(self) -> None:
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        engine = ReactionEngine(bus=bus, state_machine=sm)
        engine.attach()

        await bus.publish(SpeechRecognized(text="hello"))
        actions = engine.drain()

        assert len(actions) >= 1
        action_types = [type(a) for a in actions]
        assert RequestBlinkAction in action_types

        engine.detach()

    @pytest.mark.asyncio
    async def test_state_change_to_speaking_produces_celebrate(self) -> None:
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        await sm.transition(RobotState.IDLE)
        engine = ReactionEngine(bus=bus, state_machine=sm)
        engine.attach()

        await bus.publish(StateChanged(previous=RobotState.IDLE, current=RobotState.SPEAKING))
        actions = engine.drain()

        assert len(actions) >= 1
        action_types = [type(a) for a in actions]
        assert CelebrateAction in action_types

        engine.detach()

    @pytest.mark.asyncio
    async def test_state_change_from_speaking_to_speaking_no_celebrate(self) -> None:
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        engine = ReactionEngine(bus=bus, state_machine=sm)
        engine.attach()

        await bus.publish(StateChanged(previous=RobotState.SPEAKING, current=RobotState.SPEAKING))
        actions = engine.drain()

        celebrate_count = sum(1 for a in actions if isinstance(a, CelebrateAction))
        assert celebrate_count == 0

        engine.detach()
