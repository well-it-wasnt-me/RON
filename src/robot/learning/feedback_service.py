"""Attribution of explicit human feedback to a recent real transition.

Human feedback ("good" / "no") is a *post-hoc* signal: the human reacts to an
action the robot already took. The :class:`FeedbackService` attributes a
feedback event to the most-recent **eligible** transition in the recorder's
working memory, records it in the :class:`FeedbackLedger`, and publishes a
:class:`~robot.events.events.HumanFeedback` event.

It **never invents** a target. If no recent transition is eligible (none within
the feedback window, or no transitions at all), the feedback is dropped with a
log line and ``None`` is returned — no transition is fabricated, no reward is
invented, no counter is bumped.

Eligibility
-----------
From ``recorder.working_memory.recent(20)`` the service considers experiences
whose recorded timestamp is within ``feedback_window_s`` of now. When the
caller passes an ``interaction_id``, transitions from that same interaction are
preferred (the human is reacting to the teaching interaction they're in); if
none of those are eligible, the most-recent eligible transition overall is
used. The most-recent (last-executed) eligible transition wins.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from robot.events.events import HumanFeedback
from robot.learning.experience import Experience
from robot.learning.feedback_ledger import FeedbackEntry, FeedbackLedger
from robot.logging import get_logger

if TYPE_CHECKING:
    from robot.learning.recorder import ExperienceRecorder

_log = get_logger("learning.feedback_service")

#: How many recent experiences to scan for an attribution target.
_RECENT_SCAN = 20


def _age_s(timestamp: datetime, now: datetime) -> float:
    """Age of a timestamp in seconds relative to ``now`` (never negative)."""
    delta = now - timestamp
    return max(0.0, delta.total_seconds())


class FeedbackService:
    """Attribute human feedback to the most-recent eligible real transition.

    Parameters
    ----------
    recorder:
        The experience recorder whose ``working_memory`` is scanned for an
        attribution target and whose ``bus`` publishes the
        :class:`HumanFeedback` event.
    ledger:
        The :class:`FeedbackLedger` the attributed feedback is recorded in.
    feedback_window_s:
        Maximum age (seconds) of a transition to be eligible for feedback
        attribution. Feedback that arrives after this window is dropped.
    staleness_s:
        Broader staleness bound used downstream by
        :meth:`~robot.learning.learning_service.LearningService.reward_for_transition`
        to decide whether a recorded feedback is still applied. Stored here so
        the teaching configuration has a single owner.
    """

    def __init__(
        self,
        recorder: ExperienceRecorder,
        ledger: FeedbackLedger,
        feedback_window_s: float = 5.0,
        staleness_s: float = 30.0,
    ) -> None:
        self._recorder = recorder
        self._ledger = ledger
        self.feedback_window_s = float(feedback_window_s)
        self.staleness_s = float(staleness_s)

    async def handle_feedback(
        self,
        polarity: int,
        magnitude: float = 1.0,
        source: str = "speech",
        interaction_id: str | None = None,
        text: str = "",
    ) -> FeedbackEntry | None:
        """Attribute feedback to the most-recent eligible transition.

        Returns the recorded :class:`FeedbackEntry`, or ``None`` if no recent
        transition was eligible (feedback dropped — never invented).
        """
        target = self._select_eligible(interaction_id)
        if target is None:
            _log.info(
                "feedback.no_eligible_transition",
                polarity=polarity,
                interaction_id=interaction_id,
                source=source,
            )
            return None

        transition_id = str(target.metadata.get("transition_id", ""))
        if not transition_id:
            # Defensive: a transition without a transition_id cannot be keyed.
            _log.warning("feedback.target_missing_transition_id", source=source)
            return None

        entry = FeedbackEntry(
            transition_id=transition_id,
            polarity=int(polarity),
            magnitude=float(magnitude),
            source=source,
            interaction_id=interaction_id,
            text=text,
        )
        self._ledger.record(entry)
        await self._recorder.bus.publish(
            HumanFeedback(
                polarity=entry.polarity,
                magnitude=entry.magnitude,
                source=entry.source,
                interaction_id=entry.interaction_id,
                transition_id=entry.transition_id,
                text=entry.text,
            )
        )
        _log.info(
            "feedback.attributed",
            transition_id=transition_id,
            polarity=entry.polarity,
            magnitude=entry.magnitude,
            source=entry.source,
            interaction_id=interaction_id,
        )
        return entry

    def _select_eligible(self, interaction_id: str | None) -> Experience | None:
        """Pick the most-recent eligible experience, preferring a matching interaction.

        Returns the chosen :class:`Experience` or ``None``.
        """
        now = datetime.now(tz=UTC)
        recent = self._recorder.working_memory.recent(_RECENT_SCAN)
        # most-recent first
        ordered = list(reversed(recent))

        def is_eligible(exp: object) -> bool:
            ts = getattr(exp, "timestamp", None)
            if ts is None:
                return False
            return _age_s(ts, now) <= self.feedback_window_s

        eligible = [e for e in ordered if is_eligible(e)]
        if not eligible:
            return None

        if interaction_id is not None:
            same_interaction = [
                e
                for e in eligible
                if str(e.metadata.get("interaction_id", "")) == interaction_id
            ]
            if same_interaction:
                return same_interaction[0]
        return eligible[0]


__all__ = ["FeedbackService"]
