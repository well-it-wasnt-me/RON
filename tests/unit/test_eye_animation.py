"""Tests for the per-eye :class:`EyeAnimator` state machine."""

from __future__ import annotations

import pytest

from robot.events.events import EmotionName
from robot.eye_engine.animation import EyeAnimator, EyeSide
from robot.eye_engine.render_state import EyeRenderState


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_animator_defaults() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    assert a.side is EyeSide.LEFT
    assert a.width == 240
    assert a.height == 240
    assert a.emotion is EmotionName.NEUTRAL
    assert a.openness == 1.0
    assert a.frame_count == 0


def test_animator_validates_fps() -> None:
    with pytest.raises(ValueError):
        EyeAnimator(side=EyeSide.LEFT, fps=0)


def test_animator_step_increments_frame_count() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.step()
    a.step()
    assert a.frame_count == 2


def test_render_state_returns_correct_centre() -> None:
    a = EyeAnimator(side=EyeSide.LEFT, width=128, height=64)
    s = a.render_state()
    assert s.cx == 64.0
    assert s.cy == 32.0


# ---------------------------------------------------------------------------
# Blink
# ---------------------------------------------------------------------------
def test_blink_starts_with_openness_drop() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.blink()
    a.step()  # closing
    assert a.openness < 1.0


def test_blink_reaches_closed_state() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.blink()
    # Blink takes ~0.18s total; at 30fps that's ~5 frames. Run 10 frames.
    min_open = 1.0
    for _ in range(15):
        a.step()
        min_open = min(min_open, a.openness)
    assert min_open < 0.1, f"blink never closed, min_openness={min_open}"


def test_blink_returns_to_open() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.blink()
    for _ in range(20):
        a.step()
    assert a.openness == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Double blink
# ---------------------------------------------------------------------------
def test_double_blink_closes_twice() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.double_blink()
    min_open = 1.0
    closes = 0
    was_open = True
    for _ in range(30):
        a.step()
        if a.openness < 0.3 and was_open:
            closes += 1
            was_open = False
        if a.openness > 0.9:
            was_open = True
        min_open = min(min_open, a.openness)
    assert closes >= 2
    assert min_open < 0.1


# ---------------------------------------------------------------------------
# Look
# ---------------------------------------------------------------------------
def test_look_left_moves_gaze() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.look_left(duration_s=0.0)
    a.step()
    assert a.render_state().gaze_x == pytest.approx(-1.0, abs=1e-6)


def test_look_right_moves_gaze() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.look_right(duration_s=0.0)
    a.step()
    assert a.render_state().gaze_x == pytest.approx(1.0, abs=1e-6)


def test_look_up_moves_gaze() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.look_up(duration_s=0.0)
    a.step()
    assert a.render_state().gaze_y == pytest.approx(-1.0, abs=1e-6)


def test_look_down_moves_gaze() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.look_down(duration_s=0.0)
    a.step()
    assert a.render_state().gaze_y == pytest.approx(1.0, abs=1e-6)


def test_look_center() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.look(0.5, 0.5, duration_s=0.0)
    a.step()
    a.look_center(duration_s=0.0)
    a.step()
    assert a.render_state().gaze_x == pytest.approx(0.0, abs=1e-6)
    assert a.render_state().gaze_y == pytest.approx(0.0, abs=1e-6)


def test_look_clamps_out_of_range() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.look(2.0, -2.0, duration_s=0.0)
    a.step()
    s = a.render_state()
    assert s.gaze_x == 1.0
    assert s.gaze_y == -1.0


def test_look_smoothstep_interpolation() -> None:
    """A look with non-zero duration should ease over time."""
    a = EyeAnimator(side=EyeSide.LEFT, fps=30)
    a.look(1.0, 0.0, duration_s=0.5)
    a.step()  # first frame: eased to a small value
    first = a.render_state().gaze_x
    # Run enough frames to reach the peak of the look (15 frames at 30fps = 0.5s).
    for _ in range(15):
        a.step()
    peak = a.render_state().gaze_x
    assert first < peak <= 1.0
    assert peak > 0.9, f"expected look to reach >0.9, got {peak}"


# ---------------------------------------------------------------------------
# Emotions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("emotion", "expected_openness"),
    [
        (EmotionName.HAPPY, 0.7),
        (EmotionName.SAD, 0.7),
        (EmotionName.ANGRY, 0.85),
        (EmotionName.SURPRISED, 1.0),
        (EmotionName.SLEEPY, 0.20),
        (EmotionName.THINKING, 0.7),
        (EmotionName.CURIOUS, 0.95),
    ],
)
def test_emotion_sets_openness(emotion: EmotionName, expected_openness: float) -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.set_emotion(emotion)
    assert a.openness == pytest.approx(expected_openness, abs=1e-3)


def test_set_emotion_returns_to_neutral() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.set_emotion(EmotionName.SLEEPY)
    a.set_emotion(EmotionName.NEUTRAL)
    assert a.openness == pytest.approx(1.0, abs=1e-3)


def test_set_emotion_clamps_intensity() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.set_emotion(EmotionName.HAPPY, intensity=2.0)
    a.set_emotion(EmotionName.HAPPY, intensity=-1.0)
    # Should not raise


# ---------------------------------------------------------------------------
# Wink
# ---------------------------------------------------------------------------
def test_wink_drops_top_lid() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.wink()
    a.step()
    assert a.render_state().lid_top > 0.0


def test_wink_releases_after_duration() -> None:
    a = EyeAnimator(side=EyeSide.LEFT, fps=60)
    a.wink()
    for _ in range(60):
        a.step()
    assert a.render_state().lid_top == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
def test_drift_oscillates_gaze() -> None:
    a = EyeAnimator(side=EyeSide.LEFT, fps=30)
    # Look up-right (away from the emotion default) and let it finish.
    a.look(0.5, 0.5, duration_s=0.0)
    a.step()  # this step completes the look and sets _just_completed
    samples: list[float] = []
    # Each iteration: step (which eases the gaze back) then drift (which
    # oscillates around the target). After enough frames, the ease-back
    # has converged to the target and the drift noise dominates.
    for _ in range(30):
        a.step()
        a.drift(amplitude=0.5, speed=1.0)
        samples.append(a.render_state().gaze_x)
    # The samples should oscillate with a non-trivial range.
    assert max(samples) - min(samples) > 0.2, (
        f"drift did not oscillate: {min(samples)} .. {max(samples)}"
    )


def test_drift_does_not_change_target() -> None:
    """Drift adds noise around the last look target, not around (0, 0)."""
    a = EyeAnimator(side=EyeSide.LEFT)
    a.look(0.5, 0.0, duration_s=0.0)
    a.step()
    # Sample drift
    seen = []
    for _ in range(60):
        a.drift(amplitude=0.1, speed=1.0)
        seen.append(a.render_state().gaze_x)
    # Average should be around 0.5
    avg = sum(seen) / len(seen)
    assert 0.3 < avg < 0.7


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
def test_reset_returns_to_neutral() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    a.set_emotion(EmotionName.ANGRY)
    a.look_left(duration_s=0.0)
    a.wink()
    a.reset()
    assert a.emotion is EmotionName.NEUTRAL
    assert a.render_state().gaze_x == pytest.approx(0.0, abs=1e-6)
    assert a.render_state().gaze_y == pytest.approx(0.0, abs=1e-6)
    assert a.render_state().openness == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Sides
# ---------------------------------------------------------------------------
def test_left_and_right_animators_are_independent() -> None:
    a = EyeAnimator(side=EyeSide.LEFT)
    b = EyeAnimator(side=EyeSide.RIGHT)
    a.look_left(duration_s=0.0)
    a.step()
    b.look_right(duration_s=0.0)
    b.step()
    assert a.render_state().gaze_x == pytest.approx(-1.0, abs=1e-6)
    assert b.render_state().gaze_x == pytest.approx(1.0, abs=1e-6)


def test_eye_side_enum_values() -> None:
    assert EyeSide.LEFT.value == "left"
    assert EyeSide.RIGHT.value == "right"
    assert EyeSide.BOTH.value == "both"


# ---------------------------------------------------------------------------
# EyeRenderState properties
# ---------------------------------------------------------------------------
def test_eye_render_state_iris_radius() -> None:
    s = EyeRenderState(cx=10, cy=10, eye_radius=20, iris_radius_ratio=0.5)
    assert s.iris_radius == pytest.approx(10.0)


def test_eye_render_state_pupil_radius_scales_with_dilation() -> None:
    base = EyeRenderState(cx=10, cy=10, eye_radius=20, pupil_dilation=0.0)
    big = EyeRenderState(cx=10, cy=10, eye_radius=20, pupil_dilation=1.0)
    assert big.pupil_radius > base.pupil_radius
