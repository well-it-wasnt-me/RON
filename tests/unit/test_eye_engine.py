"""Tests for the eye engine."""

from __future__ import annotations

import pytest
from tests.fakes.clock import FakeClock
from tests.fakes.display import FakeDisplay

from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    EmotionName,
)
from robot.eye_engine.blink import BlinkController
from robot.eye_engine.emotions import EmotionLibrary
from robot.eye_engine.eye_state import EyeState, GazeVector
from robot.eye_engine.renderer import EyeRenderer


def test_gaze_vector_clamps() -> None:
    g = GazeVector(x=2.0, y=-2.0).clamped()
    assert g.x == 1.0
    assert g.y == -1.0


def test_eye_state_helpers() -> None:
    state = EyeState().with_gaze(GazeVector(0.5, 0.5)).with_openness(2.0)
    assert state.openness == 1.0
    assert state.gaze.x == 0.5


def test_renderer_produces_correct_size() -> None:
    from robot.eye_engine.render_state import EyeRenderState

    r = EyeRenderer(width=64, height=64)
    state = EyeRenderState(cx=32, cy=32, eye_radius=24)
    frame = r.render(state)
    assert frame.width == 64
    assert frame.height == 64
    assert len(frame.pixels) == 64 * 64 * 3


def test_renderer_paints_non_black_pixels() -> None:
    from robot.eye_engine.render_state import EyeRenderState

    r = EyeRenderer(width=64, height=64)
    state = EyeRenderState(cx=32, cy=32, eye_radius=24, openness=1.0)
    frame = r.render(state)
    assert any(b != 0 for b in frame.pixels)


def test_blink_progress_starts_open() -> None:
    ctrl = BlinkController()
    assert ctrl.blink_progress(0.0) <= 1.0
    assert ctrl.blink_progress(0.5) <= 1.0
    assert ctrl.blink_progress(1.0) == pytest.approx(1.0, abs=1e-3)


def test_blink_progress_closes_at_some_point() -> None:
    ctrl = BlinkController()
    samples = [ctrl.blink_progress(t / 10) for t in range(11)]
    assert min(samples) < 0.5


def test_emotion_library_resolves_all() -> None:
    lib = EmotionLibrary()
    for name in EmotionName:
        state = lib.get(name).to_eye_state()
        assert state.emotion == name


def test_emotion_library_unknown() -> None:
    lib = EmotionLibrary()
    with pytest.raises(ValueError):
        lib.get("nope")  # type: ignore[arg-type]


async def test_animator_reacts_to_events() -> None:
    """The dual-eye animator reacts to bus events and pushes frames."""
    bus = InMemoryEventBus()
    display = FakeDisplay(width=32, height=32)
    renderer = EyeRenderer(width=32, height=32)
    clock = FakeClock()

    from robot.eye_engine.animator import EyeDisplayAnimator

    animator = EyeDisplayAnimator(
        renderer=renderer,
        display=display,
        clock=clock,
        bus=bus,
        fps=20,
        width=32,
        height=32,
    )
    # Push 3 frames via the async _step_async helper
    for _ in range(3):
        await animator._step_async(drift=False)
    assert display.frames, "animator did not push any frame to the left display"
    # second display removed

    # Use the direct API (which works regardless of the clock) to verify
    # the wiring: set an emotion, look right, and check the render state.
    # Apply via the direct API; the look should be accepted (right or wrong target).
    animator.set_emotion(EmotionName.HAPPY)
    animator.look(0.5, 0.0, duration_s=0.0)
    # Step several frames and verify the left eye (which the dual animator
    # actually drives) is looking right.
    for _ in range(3):
        await animator._step_async(drift=False)
    state = animator.eye.render_state()
    assert state.gaze_x > 0.0, f"left eye should be looking right, got {state.gaze_x}"
