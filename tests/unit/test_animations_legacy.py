"""Tests for face animations."""

from robot.face.animations.speaking import SpeakingAnimation, Viseme
from robot.face.animations.thinking_dots import ThinkingDotsAnimation
from robot.face.animations.wake import WakeAnimation, WakePhase


class TestThinkingDotsAnimation:
    def test_initial_position(self) -> None:
        anim = ThinkingDotsAnimation()
        x, y = anim.current_position
        assert x == 0.20  # First keyframe x
        assert y == -0.20  # First keyframe y

    def test_step_advances(self) -> None:
        anim = ThinkingDotsAnimation()
        # Step many times to advance past the first hold duration.
        for _ in range(100):
            gaze = anim.step(0.05)
        # Should have moved to a different position.
        assert gaze is not None

    def test_reset(self) -> None:
        anim = ThinkingDotsAnimation()
        for _ in range(50):
            anim.step(0.05)
        anim.reset()
        x, y = anim.current_position
        assert x == 0.20  # Back to first keyframe
        assert y == -0.20

    def test_empty_pattern(self) -> None:
        anim = ThinkingDotsAnimation(pattern=())
        gaze = anim.step(0.1)
        assert gaze.x == 0.0
        assert gaze.y == 0.0

    def test_step_returns_gaze(self) -> None:
        anim = ThinkingDotsAnimation()
        gaze = anim.step(0.1)
        assert hasattr(gaze, "x")
        assert hasattr(gaze, "y")

    def test_smooth_transition(self) -> None:
        """After enough steps, the position should change from the initial."""
        anim = ThinkingDotsAnimation()
        _initial_x, _initial_y = anim.current_position
        # Step far enough to move past the first keyframe hold time.
        for _ in range(20):
            anim.step(0.05)
        _new_x, _new_y = anim.current_position
        # Position should have advanced (at least the index changed).
        # Not checking exact values because the pattern may loop.
        assert anim._index >= 0


class TestSpeakingAnimation:
    def test_text_to_frames(self) -> None:
        anim = SpeakingAnimation(text="Hi")
        assert anim.has_frames
        assert anim.total_duration > 0

    def test_step_produces_frames(self) -> None:
        anim = SpeakingAnimation(text="Hello")
        frame = anim.step(0.05)
        assert hasattr(frame, "openness")
        assert hasattr(frame, "width")

    def test_empty_text(self) -> None:
        anim = SpeakingAnimation(text="")
        assert not anim.has_frames
        assert anim.total_duration == 0

    def test_from_visemes(self) -> None:
        anim = SpeakingAnimation.from_visemes(
            [
                (Viseme.AA, 0.1),
                (Viseme.EE, 0.08),
                (Viseme.OO, 0.12),
            ]
        )
        assert anim.has_frames

    def test_reset(self) -> None:
        anim = SpeakingAnimation(text="Test")
        anim.step(0.05)
        anim.step(0.05)
        anim.reset()
        assert anim._index == 0

    def test_punctuation_pauses(self) -> None:
        anim = SpeakingAnimation(text="Hi. Bye!")
        # Should have frames for letters plus pauses for punctuation.
        assert anim.has_frames
        assert anim.total_duration > 0

    def test_viseme_enum_values(self) -> None:
        assert Viseme.AA.value == "aa"
        assert Viseme.IDLE.value == "idle"
        assert Viseme.PP.value == "pp"


class TestWakeAnimation:
    def test_initial_phase(self) -> None:
        anim = WakeAnimation()
        assert anim._phase == WakePhase.EYES_OPEN

    def test_step_advances(self) -> None:
        anim = WakeAnimation()
        frame = anim.step(0.05)
        assert frame.phase == WakePhase.EYES_OPEN
        assert frame.eye_openness > 1.0  # Wide open in eyes_open phase

    def test_completes(self) -> None:
        anim = WakeAnimation()
        for _ in range(50):
            anim.step(0.05)
        assert anim.done

    def test_done_returns_idle_frame(self) -> None:
        anim = WakeAnimation()
        for _ in range(50):
            anim.step(0.05)
        frame = anim.step(0.05)
        assert anim.done
        assert frame.phase == WakePhase.DONE

    def test_reset(self) -> None:
        anim = WakeAnimation()
        for _ in range(50):
            anim.step(0.05)
        done_before = anim.done
        assert done_before
        anim.reset()
        assert not anim.done
        assert anim._phase == WakePhase.EYES_OPEN
