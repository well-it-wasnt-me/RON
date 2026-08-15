"""Thinking dots animation.

Produces periodic gaze-shift events that give the face a "thinking"
look while the LLM is generating a response. The animation cycles
through a series of subtle gaze positions, creating a gentle
left-right-up pattern that conveys thoughtful processing.

The animation is driven by the :class:`FaceOrchestrator` subscribing
to :class:`LLMTokenReceived` events, but this module provides the
pure animation logic that can be used independently of the event bus.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from robot.face.components import Gaze

# ---------------------------------------------------------------------------
# Gaze pattern keyframes
# ---------------------------------------------------------------------------
# Each keyframe is (x, y, hold_seconds). The animation cycles through
# these positions, holding each for ``hold_seconds`` before moving to
# the next. The pattern mimics a person thinking: look up-right,
# glance left, look up-center, return to center.
_THINKING_PATTERN: Sequence[tuple[float, float, float]] = (
    (0.20, -0.20, 0.40),  # up-right (classic "thinking" pose)
    (0.05, -0.15, 0.30),  # slightly left
    (-0.10, -0.10, 0.35),  # look left
    (0.00, -0.05, 0.25),  # center-up
    (-0.05, 0.05, 0.30),  # slightly down-left
    (0.15, -0.25, 0.40),  # up-right again
    (0.00, 0.00, 0.50),  # return to center
)


@dataclass(slots=True)
class ThinkingDotsAnimation:
    """Produces a sequence of gaze positions for the "thinking dots" effect.

    The animation advances through keyframes each time :meth:`step`
    is called, returning the current gaze position. The caller is
    responsible for applying the gaze to the face model and publishing
    any events.
    """

    pattern: Sequence[tuple[float, float, float]] = _THINKING_PATTERN
    _index: int = 0
    _elapsed: float = 0.0

    def step(self, dt: float) -> Gaze:
        """Advance the animation by ``dt`` seconds and return the current gaze.

        Parameters
        ----------
        dt:
            Time elapsed since the last call, in seconds.
        """
        if not self.pattern:
            return Gaze(0.0, 0.0)

        self._elapsed += dt
        _, _, hold = self.pattern[self._index]

        # Advance to the next keyframe when the hold time expires.
        while self._elapsed >= hold:
            self._elapsed -= hold
            self._index = (self._index + 1) % len(self.pattern)
            _, _, hold = self.pattern[self._index]

        # Lerp between the current and next keyframe for smooth transitions.
        x, y, _ = self.pattern[self._index]
        next_idx = (self._index + 1) % len(self.pattern)
        nx, ny, _ = self.pattern[next_idx]

        # Transition ratio: how far we are into the hold period.
        transition_start = hold * 0.7  # start transitioning at 70% of hold
        if self._elapsed >= transition_start:
            t = (self._elapsed - transition_start) / max(0.001, hold - transition_start)
            t = max(0.0, min(1.0, t))
            # Smooth the transition.
            t = t * t * (3.0 - 2.0 * t)  # smoothstep
            x = x + (nx - x) * t
            y = y + (ny - y) * t

        return Gaze(x=x, y=y)

    def reset(self) -> None:
        """Reset the animation to the beginning."""
        self._index = 0
        self._elapsed = 0.0

    @property
    def current_position(self) -> tuple[float, float]:
        """Return the current gaze (x, y) without advancing."""
        if not self.pattern:
            return 0.0, 0.0
        x, y, _ = self.pattern[self._index]
        return x, y


__all__ = ["ThinkingDotsAnimation"]
