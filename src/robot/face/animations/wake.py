"""Wake animation - attention-getting face animation on wake word detection.

When the robot detects a wake word, it should immediately signal to the
user that it's listening. The :class:`WakeAnimation` produces a sequence
of face model deltas (gaze shift, blink, and emotion change) that
create a bright "I'm awake!" effect:

1.  Eyes snap to center and open wide (0.1s)
2.  Quick double-blink (0.3s)
3.  Mouth opens to a surprised ``O`` shape (0.2s)
4.  Transition to ``curious`` emotion (0.3s)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from robot.face.components import (
    EyebrowShape,
    Gaze,
    MouthShape,
)


class WakePhase(str, Enum):
    """Phases of the wake animation."""

    IDLE = "idle"
    EYES_OPEN = "eyes_open"
    DOUBLE_BLINK = "double_blink"
    MOUTH_SURPRISE = "mouth_surprise"
    TRANSITION = "transition"
    DONE = "done"


# Duration for each phase in seconds.
_PHASE_DURATIONS: dict[WakePhase, float] = {
    WakePhase.IDLE: 0.0,
    WakePhase.EYES_OPEN: 0.10,
    WakePhase.DOUBLE_BLINK: 0.30,
    WakePhase.MOUTH_SURPRISE: 0.20,
    WakePhase.TRANSITION: 0.30,
    WakePhase.DONE: 0.0,
}


@dataclass(slots=True, frozen=True)
class WakeFrame:
    """A single frame of the wake animation.

    Provides target values for the face model at a given animation
    time. The caller interpolates between the current state and
    this target.
    """

    gaze: Gaze
    eye_openness: float = 1.0
    eyelid_top: float = 0.0
    mouth_shape: MouthShape = MouthShape.NEUTRAL
    mouth_openness: float = 0.0
    mouth_width: float = 0.5
    eyebrow_shape: EyebrowShape = EyebrowShape.NEUTRAL
    eyebrow_raise: float = 0.0
    phase: WakePhase = WakePhase.IDLE


@dataclass(slots=True)
class WakeAnimation:
    """Produces a sequence of :class:`WakeFrame` objects for the wake animation.

    Usage::

        anim = WakeAnimation()
        while not anim.done:
            frame = anim.step(dt=0.033)
            # Apply frame to face model...
    """

    _phase: WakePhase = WakePhase.EYES_OPEN
    _elapsed: float = 0.0

    def step(self, dt: float) -> WakeFrame:
        """Advance the animation by ``dt`` seconds and return the current frame."""
        if self._phase == WakePhase.DONE:
            return self._idle_frame()

        self._elapsed += dt
        phase_duration = _PHASE_DURATIONS.get(self._phase, 0.3)

        frame = self._frame_for_phase(self._phase)

        # Transition to next phase when duration expires.
        if self._elapsed >= phase_duration:
            self._elapsed -= phase_duration
            self._phase = self._next_phase(self._phase)

        return frame

    @property
    def done(self) -> bool:
        """Whether the animation has completed."""
        return self._phase == WakePhase.DONE

    def reset(self) -> None:
        """Reset the animation to the beginning."""
        self._phase = WakePhase.EYES_OPEN
        self._elapsed = 0.0

    def _next_phase(self, phase: WakePhase) -> WakePhase:
        order = [
            WakePhase.EYES_OPEN,
            WakePhase.DOUBLE_BLINK,
            WakePhase.MOUTH_SURPRISE,
            WakePhase.TRANSITION,
            WakePhase.DONE,
        ]
        idx = order.index(phase) if phase in order else 0
        next_idx = idx + 1
        if next_idx >= len(order):
            return WakePhase.DONE
        return order[next_idx]

    def _frame_for_phase(self, phase: WakePhase) -> WakeFrame:
        if phase == WakePhase.EYES_OPEN:
            return WakeFrame(
                gaze=Gaze(0.0, 0.0),
                eye_openness=1.2,
                eyelid_top=0.0,
                eyebrow_shape=EyebrowShape.RAISED,
                eyebrow_raise=0.8,
                phase=phase,
            )
        if phase == WakePhase.DOUBLE_BLINK:
            # The blink cycle happens within the 0.3s window.
            # First 0.15s: eyes closed. Next 0.15s: eyes open, then closed again.
            blink_t = self._elapsed / max(0.001, _PHASE_DURATIONS[WakePhase.DOUBLE_BLINK])
            if blink_t < 0.15:
                openness = max(0.1, 1.0 - blink_t * 6.0)
            elif blink_t < 0.30:
                openness = 0.1 + (blink_t - 0.15) * 6.0
            elif blink_t < 0.50:
                openness = max(0.1, 1.0 - (blink_t - 0.30) * 5.0)
            else:
                openness = 0.1 + (blink_t - 0.50) * 1.8
            openness = max(0.1, min(1.2, openness))
            return WakeFrame(
                gaze=Gaze(0.0, 0.05),
                eye_openness=openness,
                eyelid_top=1.0 - openness,
                mouth_shape=MouthShape.NEUTRAL,
                mouth_openness=0.1,
                phase=phase,
            )
        if phase == WakePhase.MOUTH_SURPRISE:
            return WakeFrame(
                gaze=Gaze(0.0, 0.05),
                eye_openness=1.1,
                eyelid_top=0.05,
                mouth_shape=MouthShape.WIDE_OPEN,
                mouth_openness=0.7,
                mouth_width=0.6,
                eyebrow_shape=EyebrowShape.RAISED,
                eyebrow_raise=0.6,
                phase=phase,
            )
        if phase == WakePhase.TRANSITION:
            # Smooth transition to "curious" look.
            return WakeFrame(
                gaze=Gaze(0.1, 0.1),
                eye_openness=1.0,
                eyelid_top=0.0,
                mouth_shape=MouthShape.OPEN,
                mouth_openness=0.2,
                mouth_width=0.4,
                eyebrow_shape=EyebrowShape.RAISED,
                eyebrow_raise=0.3,
                phase=phase,
            )
        return self._idle_frame()

    @staticmethod
    def _idle_frame() -> WakeFrame:
        return WakeFrame(gaze=Gaze(0.0, 0.0), phase=WakePhase.DONE)


__all__ = ["WakeAnimation", "WakeFrame", "WakePhase"]
