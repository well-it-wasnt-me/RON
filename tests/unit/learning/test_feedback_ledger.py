"""Phase 5: FeedbackLedger maps transition_id -> feedback (last-wins)."""

from __future__ import annotations

from robot.learning.feedback_ledger import FeedbackEntry, FeedbackLedger


def _entry(
    transition_id: str, polarity: int, magnitude: float = 1.0
) -> FeedbackEntry:
    return FeedbackEntry(
        transition_id=transition_id,
        polarity=polarity,
        magnitude=magnitude,
        source="speech",
    )


class TestFeedbackForTransition:
    def test_absent_returns_zero(self) -> None:
        """No feedback ever invented — absent transition yields 0.0."""
        ledger = FeedbackLedger()
        assert ledger.feedback_for_transition("t1") == 0.0

    def test_positive_delta(self) -> None:
        ledger = FeedbackLedger()
        ledger.record(_entry("t1", polarity=+1, magnitude=1.0))
        assert ledger.feedback_for_transition("t1") == 1.0

    def test_negative_delta(self) -> None:
        ledger = FeedbackLedger()
        ledger.record(_entry("t1", polarity=-1, magnitude=1.0))
        assert ledger.feedback_for_transition("t1") == -1.0

    def test_magnitude_scales(self) -> None:
        ledger = FeedbackLedger()
        ledger.record(_entry("t1", polarity=+1, magnitude=0.5))
        assert ledger.feedback_for_transition("t1") == 0.5
        ledger.record(_entry("t2", polarity=-1, magnitude=2.0))
        assert ledger.feedback_for_transition("t2") == -2.0


class TestLastWins:
    def test_repeat_feedback_replaces(self) -> None:
        """Repeated feedback for the same transition: last wins, not summed."""
        ledger = FeedbackLedger()
        ledger.record(_entry("t1", polarity=+1, magnitude=1.0))
        ledger.record(_entry("t1", polarity=-1, magnitude=1.0))
        assert ledger.feedback_for_transition("t1") == -1.0

    def test_independent_transitions(self) -> None:
        ledger = FeedbackLedger()
        ledger.record(_entry("t1", polarity=+1))
        ledger.record(_entry("t2", polarity=-1))
        assert ledger.feedback_for_transition("t1") == 1.0
        assert ledger.feedback_for_transition("t2") == -1.0


class TestGetAndClear:
    def test_get_returns_stored_entry(self) -> None:
        ledger = FeedbackLedger()
        e = _entry("t1", polarity=+1, magnitude=0.7)
        ledger.record(e)
        got = ledger.get("t1")
        assert got is e
        assert got is not None
        assert got.reward_delta == 0.7

    def test_get_absent_is_none(self) -> None:
        assert FeedbackLedger().get("nope") is None

    def test_clear_drops_all(self) -> None:
        ledger = FeedbackLedger()
        ledger.record(_entry("t1", polarity=+1))
        ledger.record(_entry("t2", polarity=-1))
        assert len(ledger) == 2
        ledger.clear()
        assert len(ledger) == 0
        assert ledger.feedback_for_transition("t1") == 0.0
