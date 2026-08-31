"""Phase 7 (Test 13): negative feedback weakens ``wave`` below an unrelated action.

The complement of the positive-feedback KEY test: when the human *corrects*
the wave (``"no"``), the policy's Q-value for ``wave`` must fall below an
unrelated action (``look_left``). The only signal that wave is "wrong" is the
human's negative feedback, amended into the reward via the
:class:`FeedbackLedger` — nothing hard-codes wave as wrong.
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


async def test_negative_feedback_makes_wave_dispreferred(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """Repeated wave + "no" -> Q(state, wave) < Q(state, look_left)."""
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
    service.encoder.update_vision(face_detected=True, face_count=1)
    service.encoder.person_present = True
    try:
        for _ in range(12):
            context.begin_interaction()
            await executor.execute_one(WaveAction())
            assert context.interaction_id is not None
            await feedback.handle_feedback(
                polarity=-1,
                source="speech",
                text="no",
                interaction_id=context.interaction_id,
            )

        assert service.status.total_experiences == 12
        assert len(ledger) == 12
        # The amended reward for a corrected wave is negative (praise delta -1).
        for exp in service.working_memory.recent(12):
            assert exp.metadata["action_index"] == 13  # wave
            assert service.reward_for_transition(exp.metadata["transition_id"]) < 0.0

        state = list(service.working_memory.recent(1)[-1].state)
        # Train and confirm wave becomes dispreferred below look_left.
        dispreferred = False
        for _ in range(100):
            service.force_training()
            q = service.q_values(state)
            if q["wave"] < q["look_left"]:
                dispreferred = True
                break
        q = service.q_values(state)
        assert dispreferred, (
            f"wave was not weakened: wave={q['wave']} look_left={q['look_left']}"
        )
    finally:
        recorder.detach()
