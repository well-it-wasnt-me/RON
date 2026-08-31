"""Post-hoc human feedback ledger: transition_id -> feedback reward delta.

Human feedback is *post-hoc*: the human reacts to an action the robot already
took. The :class:`FeedbackLedger` stores the attributed feedback keyed by the
transition it applies to, so the reward used at training time
(:meth:`robot.learning.learning_service.LearningService.reward_for_transition`)
can amend the original immediate reward recorded with the transition.

Semantics
---------
* **Last-wins.** If the human gives feedback for the same transition twice, the
  most recent entry replaces the earlier one. The ledger never *sums* repeated
  feedback — it reflects the human's final word.
* **Never invents.** A transition absent from the ledger yields ``0.0``; the
  reward is only amended when feedback was actually attributed to that
  transition.
* **Thread-safe.** A teaching controller on the event loop and a training cycle
  on a worker may both read/write, so access is guarded by a lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class FeedbackEntry:
    """A single attributed human-feedback record.

    Attributes
    ----------
    transition_id:
        The transition the feedback was attributed to.
    polarity:
        ``+1`` (positive) or ``-1`` (negative).
    magnitude:
        Strength of the feedback.
    source:
        Origin (``"speech"``, ``"api"``, ``"cli"`` …).
    interaction_id:
        The teaching interaction the feedback belongs to, if known.
    text:
        The raw utterance that produced the feedback, for auditing.
    timestamp:
        When the feedback was recorded (UTC).
    """

    transition_id: str
    polarity: int
    magnitude: float
    source: str
    interaction_id: str | None = None
    text: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @property
    def reward_delta(self) -> float:
        """The signed reward delta this entry contributes: ``polarity*magnitude``."""
        return float(self.polarity) * float(self.magnitude)


@dataclass(slots=True)
class FeedbackLedger:
    """Maps a transition id to its most-recent feedback (last-wins).

    The ledger only *stores* feedback that the
    :class:`~robot.learning.feedback_service.FeedbackService` has already
    attributed to a real transition. It never mints or invents entries.
    """

    _entries: dict[str, FeedbackEntry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, entry: FeedbackEntry) -> FeedbackEntry:
        """Record feedback for a transition (last-wins).

        Returns the entry now stored for that transition.
        """
        with self._lock:
            self._entries[entry.transition_id] = entry
        return entry

    def feedback_for_transition(self, transition_id: str) -> float:
        """Return ``polarity*magnitude`` for the transition, or ``0.0`` if absent.

        Never invents a reward — ``0.0`` is the explicit "no feedback" signal.
        """
        with self._lock:
            entry = self._entries.get(transition_id)
        if entry is None:
            return 0.0
        return entry.reward_delta

    def get(self, transition_id: str) -> FeedbackEntry | None:
        """Return the stored entry for a transition, or ``None``."""
        with self._lock:
            return self._entries.get(transition_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Drop all recorded feedback (e.g. on reset)."""
        with self._lock:
            self._entries.clear()


__all__ = ["FeedbackEntry", "FeedbackLedger"]
