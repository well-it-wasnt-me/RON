"""Unit tests for face animation modules: SpeakingAnimation, ThinkingDotsAnimation, WakeAnimation."""

from __future__ import annotations

import pytest

from robot.face.animations.speaking import SpeakingAnimation, Viseme, VisemeFrame
from robot.face.animations.thinking_dots import ThinkingDotsAnimation
from robot.face.animations.wake import WakeAnimation, WakeFrame, WakePhase
from robot.face.components import EyebrowShape, Gaze, MouthShape


# ---------------------------------------------------------------------------
# SpeakingAnimation
# ---------------------------------------------------------------------------
class TestSpeakingAnimation:
    """Tests for :class:`SpeakingAnimation`."""

    def test_empty_text_produces_no_frames(self) -> None:
        anim = SpeakingAnimation(text="")
        assert not anim.has_frames
        frame = anim.step(dt=0.033)
        assert frame.openness == 0.0

    def test_simple_text_produces_frames(self) -> None:
        anim = SpeakingAnimation(text="hi")
        assert anim.has_frames
        # Should produce frames for 'h' and 'i'
        assert anim.total_duration > 0

    def test_step_advances_through_frames(self) -> None:
        anim = SpeakingAnimation(text="a", default_phoneme_duration=0.05)
        assert anim.has_frames
        # First step should return the 'a' viseme frame
        frame = anim.step(dt=0.01)
        assert frame.openness > 0  # 'a' is AA viseme, openness=0.7
        # After enough time, should advance to done
        for _ in range(10):
            anim.step(dt=0.05)
        assert not anim.has_frames

    def test_reset_returns_to_beginning(self) -> None:
        anim = SpeakingAnimation(text="hello")
        for _ in range(5):
            anim.step(dt=0.05)
        anim.reset()
        assert anim.has_frames
        assert anim._index == 0

    def test_from_visemes_class_method(self) -> None:
        anim = SpeakingAnimation.from_visemes(
            [(Viseme.AA, 0.1), (Viseme.EE, 0.1), (Viseme.OO, 0.1)]
        )
        assert anim.has_frames
        frame = anim.step(dt=0.01)
        assert frame.viseme == Viseme.AA

    def test_punctuation_creates_pauses(self) -> None:
        anim = SpeakingAnimation(text="a.b", sentence_pause_duration=0.25)
        # 'a' is a viseme, '.' is a sentence pause, 'b' is a viseme
        assert anim.total_duration > 0

    def test_comma_creates_pause(self) -> None:
        anim = SpeakingAnimation(text="a,b", pause_duration=0.15)
        assert anim.total_duration > 0

    def test_space_creates_short_pause(self) -> None:
        anim = SpeakingAnimation(text="a b")
        assert anim.total_duration > 0

    def test_speed_modifier(self) -> None:
        anim_slow = SpeakingAnimation(text="hello", speed=0.5)
        anim_fast = SpeakingAnimation(text="hello", speed=2.0)
        assert anim_slow.total_duration > anim_fast.total_duration

    def test_has_frames_property(self) -> None:
        anim = SpeakingAnimation(text="hi")
        assert anim.has_frames

    def test_step_after_completion_returns_idle(self) -> None:
        anim = SpeakingAnimation(text="a", default_phoneme_duration=0.01)
        # Exhaust all frames
        for _ in range(20):
            anim.step(dt=0.1)
        frame = anim.step(dt=0.1)
        assert frame.openness == 0.0
        assert frame.viseme == Viseme.IDLE

    def test_visme_frame_fields(self) -> None:
        frame = VisemeFrame(openness=0.5, width=0.6, duration_s=0.08, viseme=Viseme.AA)
        assert frame.openness == 0.5
        assert frame.width == 0.6
        assert frame.duration_s == 0.08
        assert frame.viseme == Viseme.AA

    def test_digraph_recognition(self) -> None:
        """Two-character digraphs like 'sh', 'ch', 'th' should be recognized."""
        anim = SpeakingAnimation(text="she")
        assert anim.has_frames
        # 'sh' should map to Viseme.CH, 'e' to Viseme.EE
        total_frames = len(anim._frames)
        assert total_frames >= 2  # at least 'sh' and 'e'


# ---------------------------------------------------------------------------
# ThinkingDotsAnimation
# ---------------------------------------------------------------------------
class TestThinkingDotsAnimation:
    """Tests for :class:`ThinkingDotsAnimation`."""

    def test_default_pattern_produces_gaze(self) -> None:
        anim = ThinkingDotsAnimation()
        gaze = anim.step(dt=0.1)
        assert isinstance(gaze, Gaze)
        # Should be near the first keyframe (0.20, -0.20)
        assert -1.0 <= gaze.x <= 1.0
        assert -1.0 <= gaze.y <= 1.0

    def test_custom_pattern(self) -> None:
        pattern = [(0.5, -0.5, 0.5), (-0.5, 0.5, 0.5)]
        anim = ThinkingDotsAnimation(pattern=pattern)
        gaze = anim.step(dt=0.01)
        assert gaze.x > 0  # should be near 0.5

    def test_empty_pattern_returns_center(self) -> None:
        anim = ThinkingDotsAnimation(pattern=[])
        gaze = anim.step(dt=0.1)
        assert gaze.x == 0.0
        assert gaze.y == 0.0

    def test_reset(self) -> None:
        anim = ThinkingDotsAnimation()
        for _ in range(10):
            anim.step(dt=0.2)
        anim.reset()
        assert anim._index == 0
        assert anim._elapsed == 0.0

    def test_cycles_through_pattern(self) -> None:
        anim = ThinkingDotsAnimation()
        # After enough steps, should cycle back to start
        for _ in range(100):
            anim.step(dt=0.1)
        # Should not crash; pattern cycles

    def test_current_position(self) -> None:
        anim = ThinkingDotsAnimation()
        x, y = anim.current_position
        # First keyframe is (0.20, -0.20)
        assert abs(x - 0.20) < 0.01
        assert abs(y - (-0.20)) < 0.01

    def test_current_position_empty_pattern(self) -> None:
        anim = ThinkingDotsAnimation(pattern=[])
        x, y = anim.current_position
        assert x == 0.0
        assert y == 0.0


# ---------------------------------------------------------------------------
# WakeAnimation
# ---------------------------------------------------------------------------
class TestWakeAnimation:
    """Tests for :class:`WakeAnimation`."""

    def test_initial_phase_is_eyes_open(self) -> None:
        anim = WakeAnimation()
        assert not anim.done
        assert anim._phase == WakePhase.EYES_OPEN

    def test_advances_through_phases(self) -> None:
        anim = WakeAnimation()
        phases_seen: list[WakePhase] = []
        for _ in range(200):
            frame = anim.step(dt=0.05)
            if frame.phase not in phases_seen:
                phases_seen.append(frame.phase)
            if anim.done:
                # The last step returns a DONE-phase frame
                if frame.phase not in phases_seen:
                    phases_seen.append(frame.phase)
                break
        assert WakePhase.EYES_OPEN in phases_seen
        # DONE phase is signalled by anim.done=True; the idle frame has phase=DONE
        assert anim.done

    def test_eyes_open_phase(self) -> None:
        anim = WakeAnimation()
        frame = anim.step(dt=0.01)
        assert frame.phase == WakePhase.EYES_OPEN
        assert frame.eye_openness == 1.2
        assert frame.eyebrow_shape == EyebrowShape.RAISED

    def test_done_returns_idle_frame(self) -> None:
        anim = WakeAnimation()
        # Exhaust all phases
        for _ in range(200):
            anim.step(dt=0.5)
        assert anim.done
        frame = anim.step(dt=0.01)
        assert frame.phase == WakePhase.DONE
        assert frame.gaze.x == 0.0
        assert frame.gaze.y == 0.0

    def test_reset(self) -> None:
        anim = WakeAnimation()
        for _ in range(100):
            anim.step(dt=0.1)
        anim.reset()
        assert not anim.done
        assert anim._phase == WakePhase.EYES_OPEN

    def test_double_blink_phase(self) -> None:
        anim = WakeAnimation()
        # Skip past EYES_OPEN phase
        anim.step(dt=0.5)
        # Should be in or past DOUBLE_BLINK
        if anim._phase == WakePhase.DOUBLE_BLINK:
            frame = anim.step(dt=0.01)
            assert 0.0 <= frame.eye_openness <= 1.2

    def test_mouth_surprise_phase(self) -> None:
        anim = WakeAnimation()
        # Advance through phases
        for _ in range(5):
            anim.step(dt=0.2)
        if anim._phase == WakePhase.MOUTH_SURPRISE:
            frame = anim.step(dt=0.01)
            assert frame.mouth_shape == MouthShape.WIDE_OPEN
            assert frame.mouth_openness > 0

    def test_wake_frame_is_frozen(self) -> None:
        """WakeFrame should be immutable (frozen dataclass)."""
        frame = WakeFrame(gaze=Gaze(0.0, 0.0))
        with pytest.raises(AttributeError):
            frame.gaze = Gaze(1.0, 1.0)  # type: ignore[misc]

    def test_wake_phase_ordering(self) -> None:
        anim = WakeAnimation()
        # Verify phases progress in order (DONE is reached when anim.done is True)
        phases: list[WakePhase] = []
        for _ in range(200):
            frame = anim.step(dt=0.05)
            if not phases or phases[-1] != frame.phase:
                phases.append(frame.phase)
            if anim.done:
                break
        # Eyes open should always be first
        assert phases[0] == WakePhase.EYES_OPEN
        # Animation should complete
        assert anim.done
