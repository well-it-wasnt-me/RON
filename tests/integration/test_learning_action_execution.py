"""Integration tests: real action execution reaches the learning transition lifecycle.

These tests exercise the same runtime path the robot uses —
``ActionExecutor.execute_one`` → ``_execute_one`` (bus events + servo commands) —
with a real :class:`ExperienceRecorder` / :class:`LearningService` wired in, a real
:class:`InMemoryEventBus`, and the production mock servo bus.  They prove:

* a real action creates exactly one stored experience (Test A),
* pure perception events create none (Test B),
* the stored transition's ``next_state`` is the *post*-action state (Test C),
* a failed hardware action is recorded with ``execution_success=False`` and the
  robot keeps running (Test D),
* experiences reach the :class:`LearningService` counters + replay buffer (Test E),
* enough real experiences let the existing training schedule trigger a cycle
  (Test F),
* multimodal (570-dim) encoding remains functional through the real path.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest

from robot.behavior.actions import (
    CelebrateAction,
    RequestBlinkAction,
    RequestLookAction,
    RequestServoMoveAction,
)
from robot.behavior.state_machine import RobotState
from robot.errors import ServoError
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    EmotionName,
    FaceDetected,
    ServoMoved,
    SpeechRecognized,
    StateChanged,
)
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.learning.action_learning import deskbot_action_space
from robot.learning.experience import ReplayBuffer, WorkingMemory
from robot.learning.learning_service import (
    CheckpointConfig,
    LearningSchedule,
    LearningService,
    ResourceLimits,
)
from robot.learning.state_encoder import STATE_SIZE
from robot.services.executor import ActionExecutor

# Index of the HAPPY emotion in the state vector (EmotionName.HAPPY is the 2nd
# member of the enum, and StateEncoder encodes emotions at offset 0).
_HAPPY_INDEX = list(EmotionName).index(EmotionName.HAPPY)


def _servo_controller() -> object:
    """Build a mock servo controller with pan/tilt gaze servos, like the app."""
    bus = MockServoBus(
        {
            "pan": MockServo(name="pan", min_angle=-90.0, max_angle=90.0),
            "tilt": MockServo(name="tilt", min_angle=-30.0, max_angle=30.0),
            "left_arm": MockServo(name="left_arm", min_angle=0.0, max_angle=180.0),
        }
    )
    return wrap_servo_controller(bus, backend_name="mock")


def _make_service(
    *,
    bus: InMemoryEventBus,
    tmp_path: Path,
    use_multimodal: bool = False,
    min_experiences_for_training: int = 8,
) -> LearningService:
    schedule = LearningSchedule(
        min_new_experiences=8,
        train_interval_s=0.1,
        min_experiences_for_training=min_experiences_for_training,
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
        use_multimodal=use_multimodal,
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


class TestRealActionCreatesExperience:
    """Test A + E: a real action through the runtime path stores an experience."""

    async def test_real_action_creates_one_experience(
        self, bus: InMemoryEventBus, executor: ActionExecutor, tmp_path: Path
    ) -> None:
        service = _make_service(bus=bus, tmp_path=tmp_path)
        recorder = service.recorder
        assert recorder is not None
        executor.experience_recorder = recorder
        recorder.attach()
        try:
            assert service.status.total_experiences == 0
            await executor.execute_one(CelebrateAction(intensity=1.0))
            # Test A: exactly one experience stored.
            assert service.status.total_experiences == 1
            assert len(service.working_memory) == 1
            # Test E: it reached the replay buffer and the new-since-train counter.
            assert len(service.replay_buffer) == 1
            assert service.status.new_experiences_since_train == 1
        finally:
            recorder.detach()


class TestObservationAloneCreatesNoExperience:
    """Test B: pure perception events never create experiences."""

    async def test_observations_do_not_increment_experiences(
        self, bus: InMemoryEventBus, executor: ActionExecutor, tmp_path: Path
    ) -> None:
        service = _make_service(bus=bus, tmp_path=tmp_path)
        recorder = service.recorder
        assert recorder is not None
        executor.experience_recorder = recorder
        recorder.attach()
        try:
            await bus.publish(FaceDetected(x=0.4, y=0.3, confidence=0.9))
            await bus.publish(SpeechRecognized(text="hello there"))
            await bus.publish(
                StateChanged(previous=RobotState.IDLE, current=RobotState.LISTENING)
            )
            await bus.publish(ServoMoved(name="pan", angle=10.0))
            # Yield so any async handlers settle.
            await asyncio.sleep(0.01)

            assert service.status.total_experiences == 0
            assert len(service.working_memory) == 0
            assert len(service.replay_buffer) == 0
        finally:
            recorder.detach()


class TestNextStateIsPostActionState:
    """Test C: state=before, next_state=after — not two identical snapshots."""

    async def test_transition_captures_post_action_state(
        self, bus: InMemoryEventBus, executor: ActionExecutor, tmp_path: Path
    ) -> None:
        service = _make_service(bus=bus, tmp_path=tmp_path)
        recorder = service.recorder
        assert recorder is not None
        executor.experience_recorder = recorder
        recorder.attach()
        try:
            # A CelebrateAction publishes EmotionChanged(HAPPY); the recorder's
            # observation handler updates the encoder, so the post-action state
            # must reflect the new emotion.
            await executor.execute_one(CelebrateAction(intensity=1.0))

            exp = service.working_memory.recent(1)[-1]
            assert exp.state != exp.next_state, "state and next_state must differ"
            # Before the action the robot was not happy; after it is.
            assert exp.state[_HAPPY_INDEX] == 0.0
            assert exp.next_state[_HAPPY_INDEX] > 0.0
        finally:
            recorder.detach()


class TestFailedHardwareAction:
    """Test D: a failed action is recorded safely and the robot keeps running."""

    async def test_failed_action_recorded_and_robot_continues(
        self, bus: InMemoryEventBus, executor: ActionExecutor, tmp_path: Path
    ) -> None:
        service = _make_service(bus=bus, tmp_path=tmp_path)
        recorder = service.recorder
        assert recorder is not None
        executor.experience_recorder = recorder
        recorder.attach()
        try:
            # pan servo range is [-90, 90]; 200 is out of range -> ServoError.
            with pytest.raises(ServoError):
                await executor.execute_one(RequestServoMoveAction(servo="pan", angle=200.0))

            # The failure was still recorded as a completed transition.
            assert service.status.total_experiences == 1
            exp = service.working_memory.recent(1)[-1]
            assert exp.metadata["execution_success"] is False
            assert exp.metadata["execution_failure_reason"] != ""
            assert "ServoError" in exp.metadata["execution_failure_reason"]

            # The robot continues: a subsequent valid action still works and
            # records a second, successful experience.
            await executor.execute_one(CelebrateAction(intensity=0.5))
            assert service.status.total_experiences == 2
            ok_exp = service.working_memory.recent(1)[-1]
            assert ok_exp.metadata["execution_success"] is True
        finally:
            recorder.detach()


class TestTrainingThresholdFromRealActions:
    """Test F: enough real actions let the existing schedule trigger training."""

    async def test_force_training_triggers_from_real_actions(
        self, bus: InMemoryEventBus, executor: ActionExecutor, tmp_path: Path
    ) -> None:
        service = _make_service(bus=bus, tmp_path=tmp_path, min_experiences_for_training=8)
        recorder = service.recorder
        assert recorder is not None
        executor.experience_recorder = recorder
        recorder.attach()
        try:
            # Drive a variety of real actions through the runtime path.
            actions = [
                CelebrateAction(intensity=0.8),
                RequestBlinkAction(left=True, right=True, speed=1.0),
                RequestBlinkAction(left=True, right=False, speed=1.5),  # wink
                RequestLookAction(x=-0.6, y=0.0, duration_s=0.4),  # look_left
                RequestLookAction(x=0.6, y=0.0, duration_s=0.4),  # look_right
                RequestLookAction(x=0.0, y=0.0, duration_s=0.4),  # look_center
                RequestServoMoveAction(servo="pan", angle=0.0),  # look_left
                RequestServoMoveAction(servo="pan", angle=90.0),  # look_center
                CelebrateAction(intensity=0.4),
                RequestBlinkAction(speed=1.0),
                RequestLookAction(x=0.0, y=-0.6, duration_s=0.4),  # look_up
                RequestLookAction(x=0.0, y=0.6, duration_s=0.4),  # look_down
                CelebrateAction(intensity=1.0),
                RequestBlinkAction(left=True, right=False, speed=1.5),
                RequestServoMoveAction(servo="pan", angle=90.0),  # look_center
                CelebrateAction(intensity=0.6),
            ]
            for action in actions:
                await executor.execute_one(action)

            assert service.status.total_experiences == len(actions)
            assert len(service.replay_buffer) == len(actions)

            # The existing training schedule can now trigger a cycle from these
            # real experiences — no manual injection.
            triggered = service.force_training()
            assert triggered is True
            assert service.status.training_cycles_completed >= 1
            assert math.isfinite(service.status.current_model_loss)
        finally:
            recorder.detach()


class TestMultimodalStaysFunctional:
    """Task 6: use_multimodal=true keeps producing 570-dim experiences."""

    async def test_multimodal_state_size_570(
        self, bus: InMemoryEventBus, executor: ActionExecutor, tmp_path: Path
    ) -> None:
        service = _make_service(bus=bus, tmp_path=tmp_path, use_multimodal=True)
        recorder = service.recorder
        assert recorder is not None
        executor.experience_recorder = recorder
        recorder.attach()
        try:
            assert service.status.multimodal_state_size == 570
            assert service.status.use_multimodal is True

            await executor.execute_one(CelebrateAction(intensity=1.0))

            assert service.status.total_experiences == 1
            exp = service.working_memory.recent(1)[-1]
            assert len(exp.state) == 570
            assert len(exp.next_state) == 570
            assert exp.state != exp.next_state
        finally:
            recorder.detach()


class TestActionSpaceIdentity:
    """Sanity: the mapping resolves the configured ActionSpace identity."""

    def test_celebrate_maps_to_celebrate(self) -> None:
        from robot.learning.action_mapping import behavior_action_to_index

        space = deskbot_action_space()
        idx = behavior_action_to_index(CelebrateAction(intensity=1.0), space)
        assert idx is not None
        assert space.get(idx).name == "celebrate"

    def test_arm_servo_move_is_not_mappable(self) -> None:
        from robot.learning.action_mapping import behavior_action_to_index

        space = deskbot_action_space()
        # Non-gaze servo moves have no action-space identity -> not recorded.
        assert (
            behavior_action_to_index(
                RequestServoMoveAction(servo="left_arm", angle=45.0), space
            )
            is None
        )
