"""Tests for the event bus and event types."""

from __future__ import annotations

import pytest

from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    BlinkRequested,
    EmotionChanged,
    EmotionName,
    FaceDetected,
    RobotStarted,
    SpeechRecognized,
)


async def test_bus_publish_invokes_handlers() -> None:
    bus = InMemoryEventBus()
    received: list[object] = []

    async def handler(event: object) -> None:
        received.append(event)

    bus.subscribe(RobotStarted, handler)
    await bus.publish(RobotStarted())
    await bus.publish(BlinkRequested())  # different type - should be ignored
    await bus.close()
    assert len(received) == 1
    assert isinstance(received[0], RobotStarted)


async def test_bus_handler_exceptions_do_not_break_other_handlers() -> None:
    bus = InMemoryEventBus()
    received: list[object] = []

    def bad_handler(_: object) -> None:
        raise RuntimeError("boom")

    def good_handler(event: object) -> None:
        received.append(event)

    bus.subscribe(RobotStarted, bad_handler)
    bus.subscribe(RobotStarted, good_handler)
    await bus.publish(RobotStarted())
    await bus.close()
    assert received and isinstance(received[0], RobotStarted)


async def test_bus_unsubscribe() -> None:
    bus = InMemoryEventBus()
    received: list[object] = []

    def handler(event: object) -> None:
        received.append(event)

    bus.subscribe(RobotStarted, handler)
    bus.unsubscribe(RobotStarted, handler)
    await bus.publish(RobotStarted())
    await bus.close()
    assert received == []


async def test_event_immutability() -> None:
    event = EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY)
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        event.current = EmotionName.SAD  # type: ignore[misc]


def test_event_payload_helpers() -> None:
    e = FaceDetected(x=0.5, y=0.5, confidence=0.9)
    assert e.confidence == 0.9
    s = SpeechRecognized(text="hi", language="en")
    assert s.text == "hi"
    assert s.language == "en"
