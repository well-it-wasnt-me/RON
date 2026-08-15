"""Tests for the :class:`FaceAnimator`."""

from __future__ import annotations

import pytest
from tests.fakes.clock import FakeClock

from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    BlinkRequested,
    EmotionChanged,
    LookRequested,
)
from robot.face.animator import FaceAnimator
from robot.face.emotions import EmotionEngine
from robot.face.renderer import FaceRenderer
from robot.face.themes.minimal import MinimalTheme
from robot.hardware.displays.mock_display import MockDisplay


def _make(bus: InMemoryEventBus | None = None) -> FaceAnimator:
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


def test_constructor_validates_fps() -> None:
    renderer = FaceRenderer(width=32, height=32)
    with pytest.raises(ValueError):
        FaceAnimator(
            renderer=renderer,
            display=MockDisplay(width=32, height=32),
            clock=FakeClock(),
            emotions=EmotionEngine(width=32, height=32),
            theme=MinimalTheme(),
            fps=0,
        )


def test_current_starts_at_neutral() -> None:
    a = _make()
    assert a.current.left_eye.openness == 1.0
    assert a.current.eyelids.top == 0.0


def test_set_emotion_changes_model() -> None:
    from robot.face.components import MouthShape

    a = _make()
    a.set_emotion("happy")
    # Manually drive the timeline to t=1.0 (the timeline is 0.3s but the
    # interpolation snaps mouth/eyebrow at t=0.5).
    assert a._timeline is not None
    for anim in a._timeline._items:
        if hasattr(anim, "on_update"):
            anim.on_update(0.6)  # past the 0.5 snap threshold
    assert a.current.mouth.shape is MouthShape.SMILE


def test_blink_drops_openness_via_eye_animator() -> None:
    a = _make()
    a.blink()
    a.step()
    # The eye animator reduces openness
    assert a.current.left_eye.openness < 1.0


async def test_bus_event_emotion_changed() -> None:
    from robot.face.components import MouthShape

    bus = InMemoryEventBus()
    a = _make(bus=bus)
    await bus.publish(
        EmotionChanged(previous=None, current="excited", intensity=1.0)  # type: ignore[arg-type]
    )  # Step many times to complete the 0.3s emotion timeline (fps=20 => 6 steps)
    for _ in range(15):
        a.step()
    assert a.current.mouth.shape is MouthShape.SMILE_OPEN


async def test_bus_event_blink() -> None:
    bus = InMemoryEventBus()
    a = _make(bus=bus)
    await bus.publish(BlinkRequested(left=True, right=True))
    a.step()
    assert a.current.left_eye.openness < 1.0


async def test_bus_event_look() -> None:
    bus = InMemoryEventBus()
    a = _make(bus=bus)
    await bus.publish(LookRequested(x=0.5, y=0.0, duration_s=0.0))
    a.step()
    assert a.current.left_eye.gaze.x == pytest.approx(0.5, abs=1e-6)


def test_step_returns_a_frame() -> None:
    a = _make()
    frame = a.step()
    assert frame.width == 32
    assert frame.height == 32
    assert len(frame.pixels) == 32 * 32 * 3


def test_bounce_animation_runs() -> None:
    a = _make()
    a.bounce()
    # The bounce should set up a timeline. Verify the timeline is set
    # and the items contain a tween with our on_update callback.
    assert a._timeline is not None
    assert len(a._timeline._items) == 1
    from robot.animation.timelines import Tween

    assert isinstance(a._timeline._items[0], Tween)
    # Manually trigger the on_update (step()'s integration with the
    # animation framework is tested in the broader integration suite).
    a._timeline._items[0].on_update(0.1)
    assert a.current.bounce == pytest.approx(0.5, abs=1e-3)


def test_reset_returns_to_neutral() -> None:
    a = _make()
    a.set_emotion("happy")
    a.step()
    a.reset()
    from robot.face.components import MouthShape

    assert a.current.mouth.shape is MouthShape.NEUTRAL
