"""Tests for the FaceAnimator's bus event handlers."""

from __future__ import annotations

import pytest
from tests.fakes.clock import FakeClock

from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    BlinkRequested,
    EmotionChanged,
    EmotionName,
    LookRequested,
)
from robot.face.animator import FaceAnimator
from robot.face.components import MouthShape
from robot.face.emotions import EmotionEngine
from robot.face.renderer import FaceRenderer
from robot.face.themes.minimal import MinimalTheme
from robot.hardware.displays.mock_display import MockDisplay


def _make_animator(bus: InMemoryEventBus | None = None) -> FaceAnimator:
    renderer = FaceRenderer(width=32, height=32)
    return FaceAnimator(
        renderer=renderer,
        display=MockDisplay(width=32, height=32),
        clock=FakeClock(),
        emotions=EmotionEngine(width=32, height=32),
        theme=MinimalTheme(),
        fps=20,
        bus=bus,
        width=32,
        height=32,
    )


async def test_emotion_handler_accepts_emotionname_enum() -> None:
    bus = InMemoryEventBus()
    a = _make_animator(bus=bus)
    # The app publishes EmotionChanged with an EmotionName enum value.
    await bus.publish(
        EmotionChanged(
            previous=EmotionName.NEUTRAL,
            current=EmotionName.HAPPY,
            intensity=1.0,
        )
    )
    for _ in range(15):
        a.step()
    assert a.current.mouth.shape is MouthShape.SMILE


async def test_emotion_handler_accepts_string() -> None:
    bus = InMemoryEventBus()
    a = _make_animator(bus=bus)
    # Defensive: a plain string payload must also work.
    await bus.publish(
        EmotionChanged(
            previous="neutral",  # type: ignore[arg-type]
            current="happy",  # type: ignore[arg-type]
            intensity=1.0,
        )
    )
    for _ in range(15):
        a.step()
    assert a.current.mouth.shape is MouthShape.SMILE


async def test_emotion_handler_ignores_unknown_names() -> None:
    """Unknown names should be ignored, not raise."""
    bus = InMemoryEventBus()
    _make_animator(bus=bus)
    # Build an event with an unknown current - should be a no-op.
    event = EmotionChanged(
        previous=EmotionName.NEUTRAL,
        current="mystery",  # type: ignore[arg-type]
        intensity=1.0,
    )
    # Should NOT raise.
    await bus.publish(event)


async def test_blink_handler_triggers_blink() -> None:
    bus = InMemoryEventBus()
    a = _make_animator(bus=bus)
    await bus.publish(BlinkRequested(left=True, right=True))
    a.step()
    # The blink animation reduces openness on the next step.
    assert a.current.left_eye.openness < 1.0


async def test_look_handler_moves_gaze() -> None:
    bus = InMemoryEventBus()
    a = _make_animator(bus=bus)
    await bus.publish(LookRequested(x=0.5, y=0.0, duration_s=0.0))
    a.step()
    assert a.current.left_eye.gaze.x == pytest.approx(0.5, abs=1e-6)
