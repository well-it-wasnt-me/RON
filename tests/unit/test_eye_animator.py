"""Tests for the single-display :class:`EyeDisplayAnimator` orchestrator.

Covers the bus event handlers, the run loop, and the back-compat
``left``/``right`` aliases.
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from tests.fakes.clock import FakeClock
from tests.fakes.display import FakeDisplay

from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    BlinkRequested,
    EmotionChanged,
    EmotionName,
    LookRequested,
)
from robot.eye_engine.animator import EyeDisplayAnimator
from robot.eye_engine.renderer import EyeRenderer
from robot.hardware.displays.mock_display import MockDisplay


def _make_animator(bus: InMemoryEventBus | None = None) -> EyeDisplayAnimator:
    renderer = EyeRenderer(width=32, height=32)
    return EyeDisplayAnimator(
        renderer=renderer,
        display=FakeDisplay(width=32, height=32),
        clock=FakeClock(),
        bus=bus,
        fps=20,
        width=32,
        height=32,
    )


def test_constructor_creates_eye_animator() -> None:
    a = _make_animator()
    assert a.eye is not None
    # back-compat aliases both return the same single animator
    assert a.left is a.eye
    assert a.right is a.eye


def test_constructor_validates_fps() -> None:
    renderer = EyeRenderer(width=32, height=32)
    with pytest.raises(ValueError):
        EyeDisplayAnimator(
            renderer=renderer,
            display=FakeDisplay(width=32, height=32),
            clock=FakeClock(),
            fps=0,
        )


def test_constructor_validates_dimensions() -> None:
    renderer = EyeRenderer(width=32, height=32)
    with pytest.raises(ValueError):
        EyeDisplayAnimator(
            renderer=renderer,
            display=FakeDisplay(width=32, height=32),
            clock=FakeClock(),
            fps=30,
            width=0,
            height=0,
        )


async def test_step_pushes_to_display() -> None:
    a = _make_animator()
    frame = await a._step_async(drift=False)
    assert cast("MockDisplay", a.display).frames[-1] is frame
    assert a.frame_count == 1


async def test_run_forever_can_be_stopped() -> None:
    a = _make_animator()
    task = asyncio.create_task(a.run_forever())
    await asyncio.sleep(0.05)
    a.stop()
    await asyncio.wait_for(task, timeout=1.0)
    assert not a.is_running
    assert a.frame_count >= 1


# ---------------------------------------------------------------------------
# Bus event handlers
# ---------------------------------------------------------------------------
async def test_emotion_changed_updates_eye() -> None:
    bus = InMemoryEventBus()
    a = _make_animator(bus=bus)
    await bus.publish(EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY))
    assert a.eye.emotion is EmotionName.HAPPY


async def test_blink_requested_blinks_eye() -> None:
    bus = InMemoryEventBus()
    a = _make_animator(bus=bus)
    await bus.publish(BlinkRequested(left=True, right=True))
    await a._step_async(drift=False)
    assert a.eye.openness < 1.0


async def test_blink_requested_with_neither_flag_is_noop() -> None:
    bus = InMemoryEventBus()
    a = _make_animator(bus=bus)
    openness_before = a.eye.openness
    await bus.publish(BlinkRequested(left=False, right=False))
    await a._step_async(drift=False)
    assert a.eye.openness == pytest.approx(openness_before, abs=1e-6)


async def test_look_requested_moves_gaze() -> None:
    bus = InMemoryEventBus()
    a = _make_animator(bus=bus)
    await bus.publish(LookRequested(x=0.5, y=0.0, duration_s=0.0))
    await a._step_async(drift=False)
    assert a.eye.render_state().gaze_x == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Direct API
# ---------------------------------------------------------------------------
def test_set_emotion() -> None:
    a = _make_animator()
    a.set_emotion(EmotionName.SLEEPY)
    assert a.eye.emotion is EmotionName.SLEEPY


def test_blink_double_blink_wink() -> None:
    a = _make_animator()
    a.blink()
    a.double_blink()
    a.eye.wink()


def test_look_helpers() -> None:
    a = _make_animator()
    a.look_left()
    a.look_right()
    a.look_up()
    a.look_down()
    a.look_center()


def test_reset() -> None:
    a = _make_animator()
    a.set_emotion(EmotionName.ANGRY)
    a.reset()
    assert a.eye.emotion is EmotionName.NEUTRAL


def test_enable_sync_and_independent_are_noop() -> None:
    """Single-display: both modes look the same. Just verify the API works."""
    a = _make_animator()
    a.enable_independent()
    assert a.sync is False
    a.enable_sync()
    assert a.sync is True


# ---------------------------------------------------------------------------
# Back-compat
# ---------------------------------------------------------------------------
def test_current_eye_state_returns_legacy_view() -> None:
    a = _make_animator()
    a.set_emotion(EmotionName.SLEEPY)
    state = a.current_eye_state()
    assert state.emotion is EmotionName.SLEEPY
    assert state.openness == pytest.approx(0.20, abs=1e-3)


def test_render_state_delegates_to_eye() -> None:
    a = _make_animator()
    state = a.render_state()
    assert state.cx == pytest.approx(16.0)  # width 32 / 2
    assert state.cy == pytest.approx(16.0)
