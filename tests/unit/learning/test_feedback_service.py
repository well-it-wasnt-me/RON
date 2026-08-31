"""Phase 5: FeedbackService attributes feedback to a recent real transition.

The service scans ``recorder.working_memory.recent(20)`` for the most-recent
eligible transition (within ``feedback_window_s``), preferring one whose
``interaction_id`` matches the feedback. It records into the ledger and
publishes :class:`HumanFeedback`. It never invents a target: if nothing is
eligible, the feedback is dropped with a log and ``None`` returned.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from robot.events.bus import InMemoryEventBus
from robot.events.events import HumanFeedback
from robot.learning.experience import Experience, WorkingMemory
from robot.learning.feedback_ledger import FeedbackLedger
from robot.learning.feedback_service import FeedbackService


def _exp(
    *,
    transition_id: str,
    interaction_id: str | None = None,
    age_s: float = 0.0,
) -> Experience:
    ts = datetime.now(tz=UTC) - timedelta(seconds=age_s)
    metadata: dict[str, object] = {"transition_id": transition_id}
    if interaction_id is not None:
        metadata["interaction_id"] = interaction_id
    return Experience(
        timestamp=ts,
        state=[0.0],
        action=[0.0],
        reward=0.0,
        next_state=[0.0],
        metadata=metadata,
    )


def _make_service(
    *,
    feedback_window_s: float = 5.0,
    staleness_s: float = 30.0,
) -> tuple[FeedbackService, WorkingMemory, InMemoryEventBus, FeedbackLedger]:
    bus = InMemoryEventBus()
    working = WorkingMemory(capacity=256)
    recorder = SimpleNamespace(bus=bus, working_memory=working)
    ledger = FeedbackLedger()
    service = FeedbackService(
        recorder,  # type: ignore[arg-type]
        ledger,
        feedback_window_s=feedback_window_s,
        staleness_s=staleness_s,
    )
    return service, working, bus, ledger


async def test_positive_feedback_attributed_to_most_recent() -> None:
    """The most-recent eligible transition receives the feedback."""
    service, working, _bus, ledger = _make_service()
    working.add(_exp(transition_id="t_old", age_s=2.0))
    working.add(_exp(transition_id="t_recent", age_s=0.5))

    entry = await service.handle_feedback(polarity=+1, source="speech", text="good")
    assert entry is not None
    assert entry.transition_id == "t_recent"
    assert entry.polarity == 1
    assert ledger.feedback_for_transition("t_recent") == 1.0
    assert ledger.feedback_for_transition("t_old") == 0.0


async def test_no_eligible_transition_drops_feedback() -> None:
    """Feedback with no eligible (recent) target is dropped, never invented."""
    service, working, _bus, ledger = _make_service()
    # Only a stale transition exists, outside the 5s window.
    working.add(_exp(transition_id="t_stale", age_s=20.0))

    entry = await service.handle_feedback(polarity=+1, source="speech", text="good")
    assert entry is None
    assert len(ledger) == 0
    assert ledger.feedback_for_transition("t_stale") == 0.0


async def test_no_transitions_at_all_drops_feedback() -> None:
    service, _working, _bus, ledger = _make_service()
    entry = await service.handle_feedback(polarity=-1, source="speech", text="no")
    assert entry is None
    assert len(ledger) == 0


async def test_staleness_window_excludes_old() -> None:
    """A transition older than feedback_window_s is not eligible."""
    service, working, _bus, ledger = _make_service(feedback_window_s=3.0)
    working.add(_exp(transition_id="t1", age_s=5.0))  # older than 3s window
    entry = await service.handle_feedback(polarity=+1, source="speech", text="good")
    assert entry is None
    assert len(ledger) == 0


async def test_interaction_id_preference() -> None:
    """Feedback with an interaction_id prefers a transition from that interaction."""
    service, working, _bus, ledger = _make_service()
    # A more recent transition from a DIFFERENT interaction...
    working.add(_exp(transition_id="t_other", interaction_id="ix_other", age_s=0.2))
    # ...and a slightly older one from the SAME interaction.
    working.add(_exp(transition_id="t_same", interaction_id="ix_same", age_s=1.0))

    entry = await service.handle_feedback(
        polarity=+1, interaction_id="ix_same", source="speech", text="good"
    )
    assert entry is not None
    assert entry.transition_id == "t_same"
    assert ledger.feedback_for_transition("t_same") == 1.0
    assert ledger.feedback_for_transition("t_other") == 0.0


async def test_interaction_id_falls_back_when_no_match() -> None:
    """If no transition matches the interaction_id, the most-recent eligible wins."""
    service, working, _bus, _ledger = _make_service()
    working.add(_exp(transition_id="t_a", interaction_id="ix_a", age_s=0.5))
    entry = await service.handle_feedback(
        polarity=+1, interaction_id="ix_unknown", source="speech", text="good"
    )
    assert entry is not None
    assert entry.transition_id == "t_a"


async def test_negative_feedback_records_negative_delta() -> None:
    service, working, _bus, ledger = _make_service()
    working.add(_exp(transition_id="t1", age_s=0.5))
    entry = await service.handle_feedback(
        polarity=-1, magnitude=2.0, source="speech", text="no"
    )
    assert entry is not None
    assert entry.polarity == -1
    assert ledger.feedback_for_transition("t1") == -2.0


async def test_publishes_human_feedback_event() -> None:
    """Attribution publishes a HumanFeedback event with the transition_id set."""
    service, working, bus, _ledger = _make_service()
    working.add(_exp(transition_id="t1", age_s=0.5))
    # Subscribe a capture handler.
    captured: list[HumanFeedback] = []

    async def _on_feedback(event: HumanFeedback) -> None:
        captured.append(event)

    bus.subscribe(HumanFeedback, _on_feedback)

    await service.handle_feedback(polarity=+1, source="speech", text="good")
    assert len(captured) == 1
    assert captured[0].transition_id == "t1"
    assert captured[0].polarity == 1


async def test_last_wins_on_repeat() -> None:
    """Repeated feedback for the same transition: ledger last-wins."""
    service, working, _bus, ledger = _make_service()
    working.add(_exp(transition_id="t1", age_s=0.2))
    await service.handle_feedback(polarity=+1, source="speech", text="good")
    await service.handle_feedback(polarity=-1, source="speech", text="no")
    assert ledger.feedback_for_transition("t1") == -1.0
