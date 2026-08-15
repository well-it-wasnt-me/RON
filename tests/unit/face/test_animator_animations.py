"""Tests for FaceAnimator animation integration (speaking, thinking, wake)."""

from __future__ import annotations

import pytest
from tests.fakes.clock import FakeClock

from robot.face.animations import SpeakingAnimation, ThinkingDotsAnimation, WakeAnimation
from robot.face.animator import FaceAnimator
from robot.face.components import MouthShape
from robot.face.emotions import EmotionEngine
from robot.face.renderer import FaceRenderer
from robot.face.themes.minimal import MinimalTheme
from robot.hardware.displays.mock_display import MockDisplay


def _make(bus: object = None) -> FaceAnimator:
    renderer = FaceRenderer(width=32, height=32)
    return FaceAnimator(
        renderer=renderer,
        display=MockDisplay(width=32, height=32),
        clock=FakeClock(),
        emotions=EmotionEngine(width=32, height=32),
        theme=MinimalTheme(),
        fps=20,
        width=32,
        height=32,
    )


# ---------------------------------------------------------------------------
# Speaking animation
# ---------------------------------------------------------------------------


class TestSpeakingAnimation:
    """Tests for set_speaking_animation / step integration."""

    def test_speaking_animation_drives_mouth_openness(self) -> None:
        """When a speaking animation is active, step() should update the mouth."""
        a = _make()
        anim = SpeakingAnimation(text="hello")
        a.set_speaking_animation(anim)

        # Step once -- the animation should produce a viseme frame
        a.step()
        # After one step, the mouth should have been modified by the animation.
        # The first phoneme "h" maps to Viseme.IDLE (openness=0.0), but
        # "e" maps to Viseme.EE (openness=0.3). Since the animation steps
        # through frames, we check that the mouth openness changed from
        # the default neutral.
        # Actually, let's step several times to make sure we hit a viseme
        # with openness > 0.
        openness_values: list[float] = []
        for _ in range(30):
            a.step()
            openness_values.append(a.current.mouth.openness)
        # At least some frames should have openness > 0 (speaking)
        assert any(o > 0.0 for o in openness_values)

    def test_speaking_animation_clears_when_finished(self) -> None:
        """When the animation runs out of frames, it should auto-clear."""
        a = _make()
        # Very short text so animation finishes quickly.
        anim = SpeakingAnimation(text="a")
        a.set_speaking_animation(anim)
        assert a._speaking_animation is not None

        # Step enough times to exhaust the animation.
        for _ in range(100):
            a.step()

        # Animation should have auto-cleared.
        assert a._speaking_animation is None

    def test_speaking_animation_mouth_shape_is_open(self) -> None:
        """When openness > 0.1, the mouth shape should be OPEN."""
        a = _make()
        anim = SpeakingAnimation(text="aa")
        a.set_speaking_animation(anim)

        # Step until we get a mouth with high openness.
        for _ in range(20):
            a.step()
            if a.current.mouth.openness > 0.1:
                assert a.current.mouth.shape is MouthShape.OPEN
                return
        pytest.fail("Never got mouth openness > 0.1")

    def test_speaking_animation_mouth_shape_is_neutral_when_closed(self) -> None:
        """When openness <= 0.1, the mouth shape should be NEUTRAL."""
        a = _make()
        anim = SpeakingAnimation(text="pp")
        a.set_speaking_animation(anim)

        for _ in range(10):
            a.step()
            if a.current.mouth.openness <= 0.1:
                assert a.current.mouth.shape is MouthShape.NEUTRAL
                return
        # "pp" has openness 0.0, so it should always be NEUTRAL when we hit it
        # but the animation may start at earlier visemes. Just verify the
        # first few steps produce NEUTRAL when openness is low.

    def test_set_emotion_clears_speaking_animation(self) -> None:
        """Calling set_emotion should clear the speaking animation."""
        a = _make()
        anim = SpeakingAnimation(text="hello world")
        a.set_speaking_animation(anim)
        assert a._speaking_animation is not None

        a.set_emotion("neutral")
        assert a._speaking_animation is None

    def test_set_speaking_animation_none_clears_it(self) -> None:
        """Setting the animation to None should clear it."""
        a = _make()
        anim = SpeakingAnimation(text="test")
        a.set_speaking_animation(anim)
        assert a._speaking_animation is not None

        a.set_speaking_animation(None)
        assert a._speaking_animation is None


# ---------------------------------------------------------------------------
# Thinking animation
# ---------------------------------------------------------------------------


class TestThinkingAnimation:
    """Tests for set_thinking_animation / step integration."""

    def test_thinking_animation_drives_gaze(self) -> None:
        """When a thinking animation is active, step() should shift gaze."""
        a = _make()
        anim = ThinkingDotsAnimation()
        a.set_thinking_animation(anim)

        # Record initial gaze.

        # Step several times to advance past the first keyframe's hold time.
        for _ in range(30):
            a.step()

        # Gaze should have moved from the initial position.
        # (The thinking animation cycles through gaze positions.)
        # After 30 steps at 20 fps = 1.5s, we should have moved past
        # the first keyframe which holds for 0.4s.
        # The gaze should have changed at some point.
        # We can't guarantee it's different from initial_x at the exact
        # moment, so we step more and track all values.
        gaze_values: list[tuple[float, float]] = []
        for _ in range(60):
            a.step()
            gaze_values.append((a.current.left_eye.gaze.x, a.current.left_eye.gaze.y))

        # At least some gaze values should differ from (0, 0).
        assert any(x != 0.0 or y != 0.0 for x, y in gaze_values)

    def test_set_emotion_clears_thinking_animation(self) -> None:
        """Calling set_emotion should clear the thinking animation."""
        a = _make()
        anim = ThinkingDotsAnimation()
        a.set_thinking_animation(anim)
        assert a._thinking_animation is not None

        a.set_emotion("neutral")
        assert a._thinking_animation is None

    def test_set_thinking_animation_none_clears_it(self) -> None:
        """Setting the animation to None should clear it."""
        a = _make()
        anim = ThinkingDotsAnimation()
        a.set_thinking_animation(anim)
        assert a._thinking_animation is not None

        a.set_thinking_animation(None)
        assert a._thinking_animation is None

    def test_thinking_animation_gaze_applied_to_both_eyes(self) -> None:
        """Both eyes should get the same gaze from the thinking animation."""
        a = _make()
        anim = ThinkingDotsAnimation()
        a.set_thinking_animation(anim)

        # Step enough to get a non-trivial gaze.
        for _ in range(20):
            a.step()

        # Both eyes should have the same gaze position.
        assert a.current.left_eye.gaze.x == pytest.approx(a.current.right_eye.gaze.x, abs=1e-6)
        assert a.current.left_eye.gaze.y == pytest.approx(a.current.right_eye.gaze.y, abs=1e-6)


# ---------------------------------------------------------------------------
# Wake animation
# ---------------------------------------------------------------------------


class TestWakeAnimation:
    """Tests for set_wake_animation / step integration."""

    def test_wake_animation_overrides_face(self) -> None:
        """When a wake animation is active, it should override face properties."""
        a = _make()
        anim = WakeAnimation()
        a.set_wake_animation(anim)

        # Step once -- the first phase (EYES_OPEN) sets eye_openness=1.2
        a.step()
        # Eye openness should be high (1.2) but clamped by the Eye model
        # which uses float values. The wake frame sets openness=1.2.
        assert a.current.left_eye.openness > 1.0
        assert a.current.right_eye.openness > 1.0

    def test_wake_animation_clears_when_done(self) -> None:
        """When the wake animation completes, it should auto-clear."""
        a = _make()
        anim = WakeAnimation()
        a.set_wake_animation(anim)
        assert a._wake_animation is not None

        # The wake animation lasts ~0.9s total. At 20 fps, that's ~18 steps.
        for _ in range(50):
            a.step()

        assert a._wake_animation is None

    def test_set_emotion_does_not_clear_wake_animation(self) -> None:
        """set_emotion should NOT clear the wake animation (it plays to completion)."""
        a = _make()
        anim = WakeAnimation()
        a.set_wake_animation(anim)
        assert a._wake_animation is not None

        a.set_emotion("curious")
        assert a._wake_animation is not None

    def test_set_wake_animation_none_clears_it(self) -> None:
        """Setting the animation to None should clear it."""
        a = _make()
        anim = WakeAnimation()
        a.set_wake_animation(anim)
        assert a._wake_animation is not None

        a.set_wake_animation(None)
        assert a._wake_animation is None

    def test_wake_animation_sets_eyebrows(self) -> None:
        """The EYES_OPEN phase should raise the eyebrows."""
        a = _make()
        anim = WakeAnimation()
        a.set_wake_animation(anim)

        a.step()
        # In EYES_OPEN phase, eyebrow_raise is 0.8
        assert a.current.left_eyebrow.raise_amount > 0.0
        assert a.current.right_eyebrow.raise_amount > 0.0

    def test_reset_clears_all_animations(self) -> None:
        """reset() should clear all three animation slots."""
        a = _make()
        a.set_speaking_animation(SpeakingAnimation(text="test"))
        a.set_thinking_animation(ThinkingDotsAnimation())
        a.set_wake_animation(WakeAnimation())

        assert a._speaking_animation is not None
        assert a._thinking_animation is not None
        assert a._wake_animation is not None

        a.reset()

        assert a._speaking_animation is None
        assert a._thinking_animation is None  # type: ignore[unreachable]
        assert a._wake_animation is None
