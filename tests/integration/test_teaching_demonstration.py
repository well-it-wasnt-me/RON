"""Phase 8: a gesture triggers a *demonstration* through the executor.

The end-to-end teaching path in demonstrate mode:

1. ``arm_from_instruction("RON, when I wave, wave back")`` arms a session
   via the constrained parser (no LLM).
2. A ``GestureDetected``-style call to ``on_gesture_detected("wave", state)``
   opens an interaction and executes the desired ``WaveAction`` through the
   canonical :class:`ActionExecutor` — so it is recorded as a **real**
   transition, tagged with the teaching-session + interaction id.
3. A non-matching gesture is a no-op (no transition, no interaction).

Nothing here is synthetic: the experience comes from the real action
execution path, exactly as Phase 3 requires.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot.events.bus import InMemoryEventBus
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.learning.experience import ReplayBuffer, WorkingMemory
from robot.learning.interaction_context import InteractionContext
from robot.learning.learning_service import (
    CheckpointConfig,
    LearningSchedule,
    LearningService,
    ResourceLimits,
)
from robot.learning.safety_gate import SafetyGate
from robot.learning.state_encoder import STATE_SIZE
from robot.learning.teaching_controller import TeachingController
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
        min_new_experiences=8,
        train_interval_s=0.1,
        min_experiences_for_training=8,
    )
    limits = ResourceLimits(
        batch_size=8,
        training_epochs_per_cycle=1,
        max_cpu_fraction=1.0,
        eval_sample_size=16,
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


def _make_controller(service: LearningService) -> tuple[TeachingController, ActionExecutor]:
    assert service.recorder is not None
    assert service.action_learner is not None
    executor = ActionExecutor(bus=service.bus, servo_controller=_servo_controller())  # type: ignore[arg-type]
    executor.experience_recorder = service.recorder
    executor.interaction_context = InteractionContext()
    safety_gate = SafetyGate(
        action_space=service.action_space,
        servo_limits={
            "pan": (-90.0, 90.0),
            "tilt": (-30.0, 30.0),
            "left_arm": (0.0, 180.0),
            "right_arm": (0.0, 180.0),
        },
        cooldown_s=0.0,
    )
    controller = TeachingController(
        action_learner=service.action_learner,
        safety_gate=safety_gate,
        action_space=service.action_space,
        interaction_context=executor.interaction_context,
        executor=executor,
        min_experiences_for_practice=64,
    )
    return controller, executor


@pytest.fixture
def bus() -> InMemoryEventBus:
    return InMemoryEventBus()


async def test_gesture_triggers_real_demonstration(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """A matching gesture executes wave and records one real, tagged experience."""
    service = _make_service(bus, tmp_path)
    controller, _executor = _make_controller(service)
    recorder = service.recorder
    assert recorder is not None

    session_id = controller.arm_from_instruction("ron when i wave, wave back")
    assert session_id is not None
    assert controller.in_teaching_mode is True

    recorder.attach()
    try:
        state = service.encoder.encode()
        executed = await controller.on_gesture_detected("wave", state)
        assert executed == 13  # wave

        assert service.status.total_experiences == 1
        exp = service.working_memory.recent(1)[-1]
        assert exp.metadata["action_index"] == 13
        assert exp.metadata["teaching_session_id"] == session_id
        assert "interaction_id" in exp.metadata
    finally:
        recorder.detach()


async def test_non_matching_gesture_is_noop(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """A gesture that does not match the trigger records nothing."""
    service = _make_service(bus, tmp_path)
    controller, _executor = _make_controller(service)
    recorder = service.recorder
    assert recorder is not None

    controller.arm_from_instruction("when i wave, wave back")

    recorder.attach()
    try:
        state = service.encoder.encode()
        executed = await controller.on_gesture_detected("point", state)
        assert executed is None
        assert service.status.total_experiences == 0
    finally:
        recorder.detach()


async def test_end_session_disarms(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """After end_session, a gesture no longer triggers."""
    service = _make_service(bus, tmp_path)
    controller, _executor = _make_controller(service)
    recorder = service.recorder
    assert recorder is not None

    controller.arm_from_instruction("when i wave, wave back")
    controller.end_session()
    assert controller.in_teaching_mode is False

    recorder.attach()
    try:
        state = service.encoder.encode()
        executed = await controller.on_gesture_detected("wave", state)
        assert executed is None
        assert service.status.total_experiences == 0
    finally:
        recorder.detach()


async def test_arm_from_instruction_non_teaching_returns_none(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """A non-teaching utterance does not arm a session."""
    service = _make_service(bus, tmp_path)
    controller, _executor = _make_controller(service)
    assert controller.arm_from_instruction("what is the weather") is None
    assert controller.in_teaching_mode is False
