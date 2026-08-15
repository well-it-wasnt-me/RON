"""Blink controller - schedules natural-looking blinks."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from robot.utils.clock import Clock, SystemClock
from robot.utils.random_source import RandomSource, SystemRandomSource


class BlinkPhase(str, enum.Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    OPENING = "opening"


@dataclass(slots=True)
class BlinkController:
    """Produces blink events at naturalistic intervals.

    The controller is stateless from the outside; call :meth:`next_blink_in`
    whenever you need the next blink delay, and :meth:`blink_progress` to
    convert a progress value ``t in [0, 1]`` to an openness factor.
    """

    base_interval_s: float = 4.0
    interval_jitter_s: float = 2.5
    closing_s: float = 0.06
    closed_s: float = 0.02
    opening_s: float = 0.10
    _clock: Clock = field(default_factory=SystemClock, repr=False)
    _rng: RandomSource = field(default_factory=SystemRandomSource, repr=False)

    def configure(self, clock: Clock, rng: RandomSource) -> None:
        self._clock = clock
        self._rng = rng

    def next_blink_in(self) -> float:
        """Return the seconds until the next natural blink."""
        jitter = self._rng.uniform(-self.interval_jitter_s, self.interval_jitter_s)
        return max(0.5, self.base_interval_s + jitter)

    def total_duration_s(self) -> float:
        return self.closing_s + self.closed_s + self.opening_s

    def blink_progress(self, t: float) -> float:
        """Map a normalised progress value to an openness factor (1 = open)."""
        if t < 0.0 or t > 1.0:
            raise ValueError("t must be in [0, 1]")
        phase_share_close = self.closing_s / self.total_duration_s()
        phase_share_closed = self.closed_s / self.total_duration_s()
        if t < phase_share_close:
            # ease out: from 1.0 to 0.0
            p = t / phase_share_close
            return 1.0 - p * p
        if t < phase_share_close + phase_share_closed:
            return 0.0
        p = (t - phase_share_close - phase_share_closed) / (
            1.0 - phase_share_close - phase_share_closed
        )
        return p * (2.0 - p)  # ease out


__all__ = ["BlinkController", "BlinkPhase"]
