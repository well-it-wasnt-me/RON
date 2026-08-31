"""Phase 3: the new behaviour actions flow through the canonical executor.

Each of the new learnable actions added in Phase 2 (``wave``, ``speak``,
``change_emotion``, ``set_state``, ``move_arm``) must execute through the
:class:`ActionExecutor` and produce exactly one stored experience — with the
correct ``behavior_action_name`` in the transition metadata and
``execution_success=True``. This proves the canonical execution layer is the
single point that records real actions, including the new ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot.behavior.actions import (
    ChangeEmotionAction,
    MoveArmAction,
    SetStateAction,
    SpeakAction,
    WaveAction,
)
from robot.events.bus import InMemoryEventBus
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.learning.experience import ReplayBuffer, WorkingMemory
from robot.learning.learning_service import (
    CheckpointConfig,
    LearningSchedule,
    LearningService,
    ResourceLimits,
)
from robot.learning.state_encoder import STATE_SIZE
from robot.services.executor import ActionExecutor


def _servo_controller() -> object:
    """Mock servo controller with gaze + both arm servos."""
    bus = MockServoBus(
        {
            "pan": MockServo(name="pan", min_angle=-90.0, max_angle=90.0),
            "tilt": MockServo(name="tilt", min_angle=-30.0, max_angle=30.0),
            "left_arm": MockServo(name="left_arm", min_angle=0.0, max_angle=180.0),
            "right_arm": MockServo(name="right_arm", min_angle=0.0, max_angle=180.0),
        }
    )
    return wrap_servo_controller(bus, backend_name="mock")


def _make_service(*, bus: InMemoryEventBus, tmp_path: Path) -> LearningService:
    schedule = LearningSchedule(
        min_new_experiences=8,
        train_interval_s=0.1,
        min_experiences_for_training=8,
    )
    limits = ResourceLimits(
        batch_size=8,
        training_epochs_per_cycle=3,
        max_cpu_fraction=1.0,
        eval_sample_size=16,
    )
    ckpt_cfg = CheckpointConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        keep_last_n=3,
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
        multimodal_history_length=5,
        working_memory=WorkingMemory(capacity=256),
        replay_buffer=ReplayBuffer(capacity=10_000, seed=42),
    )


@pytest.fixture
def bus() -> InMemoryEventBus:
    return InMemoryEventBus()


@pytest.fixture
def executor(bus: InMemoryEventBus) -> ActionExecutor:
    return ActionExecutor(bus=bus, servo_controller=_servo_controller())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("action", "expected_name", "expected_index"),
    [
        (WaveAction(), "wave", 13),
        (SpeakAction(text="hello there"), "speak", 10),
        (ChangeEmotionAction(emotion="happy", intensity=1.0), "change_emotion", 11),
        (SetStateAction(state="curious"), "set_state", 12),
        (MoveArmAction(servo="left_arm", angle=90.0), "move_arm", 14),
        (MoveArmAction(servo="right_arm", angle=120.0), "move_arm", 15),
    ],
)
async def test_new_action_creates_one_experience(
    bus: InMemoryEventBus,
    executor: ActionExecutor,
    tmp_path: Path,
    action: object,
    expected_name: str,
    expected_index: int,
) -> None:
    service = _make_service(bus=bus, tmp_path=tmp_path)
    recorder = service.recorder
    assert recorder is not None
    executor.experience_recorder = recorder
    recorder.attach()
    try:
        assert service.status.total_experiences == 0
        await executor.execute_one(action)  # type: ignore[arg-type]
        # Exactly one experience stored.
        assert service.status.total_experiences == 1
        exp = service.working_memory.recent(1)[-1]
        # ``behavior_action_name`` is the BehaviourAction class name; the
        # precise action-space identity is the resolved ``action_index``.
        assert exp.metadata["behavior_action_name"] == expected_name
        assert exp.metadata["action_index"] == expected_index
        assert exp.metadata["execution_success"] is True
        assert exp.metadata["execution_failure_reason"] == ""
    finally:
        recorder.detach()


async def test_failed_new_action_recorded_as_failure(
    bus: InMemoryEventBus,
    executor: ActionExecutor,
    tmp_path: Path,
) -> None:
    """An out-of-range arm move is recorded with execution_success=False."""
    service = _make_service(bus=bus, tmp_path=tmp_path)
    recorder = service.recorder
    assert recorder is not None
    executor.experience_recorder = recorder
    recorder.attach()
    try:
        # right_arm range is [0, 180]; 999 is out of range -> ServoError.
        from robot.errors import ServoError

        with pytest.raises(ServoError):
            await executor.execute_one(MoveArmAction(servo="right_arm", angle=999.0))
        assert service.status.total_experiences == 1
        exp = service.working_memory.recent(1)[-1]
        assert exp.metadata["behavior_action_name"] == "move_arm"
        assert exp.metadata["action_index"] == 15
        assert exp.metadata["execution_success"] is False
        assert "ServoError" in exp.metadata["execution_failure_reason"]
    finally:
        recorder.detach()

