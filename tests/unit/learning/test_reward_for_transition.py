"""Phase 6: ``LearningService.reward_for_transition`` amends recorded reward.

The amended reward is the transition's recorded reward plus any post-hoc human
feedback the :class:`FeedbackLedger` attributes to it, clamped to ``[-2, 2]``.
The transition is looked up by ``metadata["transition_id"]`` in
``working_memory.recent(256)``. Unknown transitions yield ``0.0``; with no
ledger wired the recorded reward is used unchanged. The lookup is exercised
end-to-end through the canonical transition lifecycle.
"""

from __future__ import annotations

import pytest

from robot.events.bus import InMemoryEventBus
from robot.learning.action_learning import deskbot_action_space
from robot.learning.experience import WorkingMemory
from robot.learning.feedback_ledger import FeedbackEntry, FeedbackLedger
from robot.learning.learning_service import LearningService


def _make_service(*, ledger: FeedbackLedger | None = None) -> LearningService:
    svc = LearningService(bus=InMemoryEventBus(), action_space=deskbot_action_space())
    svc.feedback_ledger = ledger
    return svc


def _record_wave(svc: LearningService, reward: float, transition_id: str) -> None:
    """Record a real wave transition through the canonical lifecycle."""
    assert svc.recorder is not None
    pending = svc.recorder.begin_transition(action_index=13, policy_version="test")
    svc.recorder.complete_transition(
        pending,
        reward=reward,
        done=False,
        metadata={"transition_id": transition_id, "behavior_action_name": "wave"},
    )


def test_unknown_transition_is_zero() -> None:
    svc = _make_service()
    assert svc.reward_for_transition("does_not_exist") == 0.0


def test_no_ledger_uses_recorded_reward() -> None:
    svc = _make_service(ledger=None)
    _record_wave(svc, reward=0.5, transition_id="t1")
    assert svc.reward_for_transition("t1") == pytest.approx(0.5)


def test_ledger_with_no_feedback_uses_recorded_reward() -> None:
    """A wired ledger that has no feedback for the transition adds nothing."""
    svc = _make_service(ledger=FeedbackLedger())
    _record_wave(svc, reward=0.4, transition_id="t2")
    assert svc.reward_for_transition("t2") == pytest.approx(0.4)


def test_positive_feedback_amends_reward() -> None:
    ledger = FeedbackLedger()
    svc = _make_service(ledger=ledger)
    _record_wave(svc, reward=0.0, transition_id="t3")
    ledger.record(FeedbackEntry(transition_id="t3", polarity=1, magnitude=1.0, source="speech"))
    assert svc.reward_for_transition("t3") == pytest.approx(1.0)


def test_negative_feedback_amends_reward() -> None:
    ledger = FeedbackLedger()
    svc = _make_service(ledger=ledger)
    _record_wave(svc, reward=0.5, transition_id="t4")
    ledger.record(FeedbackEntry(transition_id="t4", polarity=-1, magnitude=1.0, source="speech"))
    assert svc.reward_for_transition("t4") == pytest.approx(-0.5)


def test_amended_reward_clamped_high() -> None:
    ledger = FeedbackLedger()
    svc = _make_service(ledger=ledger)
    _record_wave(svc, reward=1.5, transition_id="t5")
    ledger.record(FeedbackEntry(transition_id="t5", polarity=1, magnitude=1.0, source="speech"))
    # 1.5 + 1.0 = 2.5 -> clamped to 2.0
    assert svc.reward_for_transition("t5") == pytest.approx(2.0)


def test_amended_reward_clamped_low() -> None:
    ledger = FeedbackLedger()
    svc = _make_service(ledger=ledger)
    _record_wave(svc, reward=-1.5, transition_id="t6")
    ledger.record(FeedbackEntry(transition_id="t6", polarity=-1, magnitude=1.0, source="speech"))
    # -1.5 - 1.0 = -2.5 -> clamped to -2.0
    assert svc.reward_for_transition("t6") == pytest.approx(-2.0)


def test_uses_recent_256_not_just_default() -> None:
    """The lookup scans recent(256), so a transition beyond default recent(10)
    is still found."""
    ledger = FeedbackLedger()
    svc = _make_service(ledger=ledger)
    # Record 15 transitions; the one we amend is the oldest (index 0), which
    # default recent(10) would miss but recent(256) catches.
    for i in range(15):
        _record_wave(svc, reward=0.0, transition_id=f"old-{i}")
    ledger.record(FeedbackEntry(transition_id="old-0", polarity=1, magnitude=1.0, source="speech"))
    assert svc.reward_for_transition("old-0") == pytest.approx(1.0)


def test_capacity_keeps_recent_transition_lookup() -> None:
    """Even with a small working memory, the most-recent transition is found."""
    # Custom service with a small working memory.
    svc = LearningService(
        bus=InMemoryEventBus(),
        action_space=deskbot_action_space(),
    )
    svc.feedback_ledger = FeedbackLedger()
    # Replace working_memory with a small-capacity one and re-wire the recorder.
    svc.working_memory = WorkingMemory(capacity=4)
    assert svc.recorder is not None
    svc.recorder.working_memory = svc.working_memory
    _record_wave(svc, reward=0.0, transition_id="fresh")
    assert svc.reward_for_transition("fresh") == pytest.approx(0.0)
