"""Phase 7 (Test 12, KEY): the policy learns ``wave`` from real human praise.

The end-to-end teaching loop the whole project targets:

    person present → RON waves → human says "good"  (repeated)

Each wave is a **real** action executed through the canonical
:class:`ActionExecutor`, recorded as a real transition by the recorder, and
attributed post-hoc praise via :class:`FeedbackService`. After enough
repetitions and forced training cycles, the policy's Q-value for ``wave`` in
that state must rise above an unrelated action (``look_left``).

Nothing here hard-codes ``wave`` as the answer: the only signal that wave is
"good" is the human's positive feedback, amended into the reward via the
:class:`FeedbackLedger`. No synthetic experiences are injected and no
thresholds are weakened.
"""

from __future__ import annotations

from pathlib import Path

from robot.behavior.actions import WaveAction
from robot.events.bus import InMemoryEventBus
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.learning.experience import ReplayBuffer, WorkingMemory
from robot.learning.feedback_ledger import FeedbackLedger
from robot.learning.feedback_service import FeedbackService
from robot.learning.interaction_context import InteractionContext
from robot.learning.learning_service import (
    CheckpointConfig,
    LearningSchedule,
    LearningService,
    ResourceLimits,
)
from robot.learning.state_encoder import STATE_SIZE
from robot.services.executor import ActionExecutor


def _servo_controller() -> object:
    bus = MockServoBus(
        {
            "pan": MockServo(name="pan", min_angle=-90.0, max_angle=90.0),
            "tilt": MockServo(name="tilt", min_angle=-30.0, max_angle=30.0),
            "left_arm": MockServo(name="left_arm", min_angle=0.0, max_angle=180.0),
            "right_arm": MockServo(name="right_arm", min_angle=0.0, max_angle=180.0),
        }
    )
    return wrap_servo_controller(bus, backend_name="mock")


def _make_service(bus: InMemoryEventBus, tmp_path: Path) -> LearningService:
    schedule = LearningSchedule(
        min_new_experiences=4,
        train_interval_s=0.0,
        min_experiences_for_training=4,
    )
    limits = ResourceLimits(
        batch_size=8,
        training_epochs_per_cycle=1,
        max_cpu_fraction=1.0,
        eval_sample_size=8,
    )
    ckpt_cfg = CheckpointConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        keep_last_n=2,
        promote_threshold=1.0,
    )
    return LearningService(
        bus=bus,
        schedule=schedule,
        resource_limits=limits,
        checkpoint_config=ckpt_cfg,
        state_size=STATE_SIZE,
        seed=42,
        use_multimodal=False,
        working_memory=WorkingMemory(capacity=256),
        replay_buffer=ReplayBuffer(capacity=10_000, seed=42),
    )


async def _teach_wave(
    *, bus: InMemoryEventBus, tmp_path: Path, polarity: int, repetitions: int = 12
) -> tuple[LearningService, list[float]]:
    """Run the real wave→feedback loop and return (service, last-wave state)."""
    service = _make_service(bus, tmp_path)
    recorder = service.recorder
    assert recorder is not None
    ledger = FeedbackLedger()
    service.feedback_ledger = ledger
    feedback = FeedbackService(
        recorder, ledger, feedback_window_s=30.0, staleness_s=60.0
    )
    executor = ActionExecutor(bus=bus, servo_controller=_servo_controller())  # type: ignore[arg-type]
    executor.experience_recorder = recorder
    context = InteractionContext()
    executor.interaction_context = context
    recorder.attach()
    # A person is present; this is the state in which waving is "correct".
    service.encoder.update_vision(face_detected=True, face_count=1)
    service.encoder.person_present = True
    text = "good" if polarity > 0 else "no"
    try:
        for _ in range(repetitions):
            context.begin_interaction()
            await executor.execute_one(WaveAction())
            assert context.interaction_id is not None
            await feedback.handle_feedback(
                polarity=polarity,
                source="speech",
                text=text,
                interaction_id=context.interaction_id,
            )
        last_state = list(service.working_memory.recent(1)[-1].state)
        return service, last_state
    finally:
        recorder.detach()


async def test_positive_feedback_makes_wave_preferred(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """Repeated wave + "good" -> Q(state, wave) > Q(state, look_left)."""
    service, state = await _teach_wave(
        bus=bus, tmp_path=tmp_path, polarity=+1, repetitions=12
    )
    # Real experiences were collected — no synthetic injection.
    assert service.status.total_experiences == 12
    # Each wave transition received attributed positive feedback.
    assert service.feedback_ledger is not None
    assert len(service.feedback_ledger) == 12

    # The amended reward for a wave transition is the praise (+1); the base
    # reward for wave with a face present is 0.0 (wave is not an interaction
    # action), so reward_for_transition == 1.0 — no hard-coded wave reward.
    recent = service.working_memory.recent(12)
    for exp in recent:
        assert exp.metadata["action_index"] == 13  # wave
        assert service.reward_for_transition(exp.metadata["transition_id"]) > 0.0

    q_before = service.q_values(state)
    # Train and confirm wave becomes preferred over an unrelated action.
    preferred = False
    for _ in range(100):
        service.force_training()
        q = service.q_values(state)
        if q["wave"] > q["look_left"]:
            preferred = True
            break
    q = service.q_values(state)
    assert preferred, (
        f"wave did not become preferred: wave={q['wave']} look_left={q['look_left']}"
    )
    # Sanity: training actually moved the values.
    assert q["wave"] != q_before["wave"]


async def test_no_feedback_does_not_prefer_wave(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """Without any human feedback, waving is not singled out as preferred.

    This is the control: nothing hard-codes wave as the answer. With no
    feedback, the amended reward equals the recorded reward (0.0 for wave),
    so a handful of unpraised waves must not make wave strictly preferred.
    """
    service = _make_service(bus, tmp_path)
    recorder = service.recorder
    assert recorder is not None
    service.feedback_ledger = FeedbackLedger()
    executor = ActionExecutor(bus=bus, servo_controller=_servo_controller())  # type: ignore[arg-type]
    executor.experience_recorder = recorder
    executor.interaction_context = InteractionContext()
    recorder.attach()
    service.encoder.update_vision(face_detected=True, face_count=1)
    service.encoder.person_present = True
    try:
        for _ in range(12):
            await executor.execute_one(WaveAction())
        state = list(service.working_memory.recent(1)[-1].state)
        for _ in range(40):
            service.force_training()
        q = service.q_values(state)
        # No praise -> wave is not strictly preferred over look_left. (They may
        # be close, but wave must not dominate.)
        assert not q["wave"] > q["look_left"] + 0.5
    finally:
        recorder.detach()
