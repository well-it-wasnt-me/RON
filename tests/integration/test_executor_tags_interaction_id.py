"""Phase 4: the executor tags transitions with the active interaction context.

When an :class:`InteractionContext` is wired onto the :class:`ActionExecutor`,
each stored experience's metadata must carry the active
``interaction_id``/``teaching_session_id``/``episode_id``. With no context wired
(or an empty context) the metadata is untagged — ambient actions stay
unattributed to any teaching session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot.behavior.actions import CelebrateAction
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


async def test_action_tagged_with_full_context(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """An action during an active teaching session is tagged with all ids."""
    service = _make_service(bus=bus, tmp_path=tmp_path)
    recorder = service.recorder
    assert recorder is not None

    ctx = InteractionContext()
    interaction = ctx.begin_interaction()
    session = ctx.begin_teaching_session()
    episode = ctx.begin_episode()

    executor = ActionExecutor(bus=bus, servo_controller=_servo_controller())  # type: ignore[arg-type]
    executor.experience_recorder = recorder
    executor.interaction_context = ctx

    recorder.attach()
    try:
        await executor.execute_one(CelebrateAction(intensity=1.0))
        assert service.status.total_experiences == 1
        exp = service.working_memory.recent(1)[-1]
        assert exp.metadata["interaction_id"] == interaction
        assert exp.metadata["teaching_session_id"] == session
        assert exp.metadata["episode_id"] == episode
    finally:
        recorder.detach()


async def test_action_untagged_when_context_empty(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """A wired but empty context tags nothing (ambient action)."""
    service = _make_service(bus=bus, tmp_path=tmp_path)
    recorder = service.recorder
    assert recorder is not None

    ctx = InteractionContext()  # nothing active

    executor = ActionExecutor(bus=bus, servo_controller=_servo_controller())  # type: ignore[arg-type]
    executor.experience_recorder = recorder
    executor.interaction_context = ctx

    recorder.attach()
    try:
        await executor.execute_one(CelebrateAction(intensity=1.0))
        assert service.status.total_experiences == 1
        exp = service.working_memory.recent(1)[-1]
        assert "interaction_id" not in exp.metadata
        assert "teaching_session_id" not in exp.metadata
        assert "episode_id" not in exp.metadata
    finally:
        recorder.detach()


async def test_action_untagged_when_context_not_wired(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """No interaction_context wired -> untagged (default, ambient) experience."""
    service = _make_service(bus=bus, tmp_path=tmp_path)
    recorder = service.recorder
    assert recorder is not None

    executor = ActionExecutor(bus=bus, servo_controller=_servo_controller())  # type: ignore[arg-type]
    executor.experience_recorder = recorder
    # interaction_context left None

    recorder.attach()
    try:
        await executor.execute_one(CelebrateAction(intensity=1.0))
        assert service.status.total_experiences == 1
        exp = service.working_memory.recent(1)[-1]
        assert "interaction_id" not in exp.metadata
        assert "teaching_session_id" not in exp.metadata
        assert "episode_id" not in exp.metadata
    finally:
        recorder.detach()


async def test_tags_track_lifecycle_changes(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """Ending a span stops tagging subsequent actions with that id."""
    service = _make_service(bus=bus, tmp_path=tmp_path)
    recorder = service.recorder
    assert recorder is not None

    ctx = InteractionContext()
    executor = ActionExecutor(bus=bus, servo_controller=_servo_controller())  # type: ignore[arg-type]
    executor.experience_recorder = recorder
    executor.interaction_context = ctx

    recorder.attach()
    try:
        # First action: interaction active.
        interaction = ctx.begin_interaction()
        await executor.execute_one(CelebrateAction(intensity=0.5))
        exp1 = service.working_memory.recent(1)[-1]
        assert exp1.metadata["interaction_id"] == interaction

        # End the interaction; next action is untagged for interaction_id.
        ctx.end_interaction()
        await executor.execute_one(CelebrateAction(intensity=0.5))
        exp2 = service.working_memory.recent(1)[-1]
        assert "interaction_id" not in exp2.metadata
    finally:
        recorder.detach()
