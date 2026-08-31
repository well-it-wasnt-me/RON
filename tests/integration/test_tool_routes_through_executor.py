"""Phase 3, Test 9: a learnable builtin tool routes through the ActionExecutor.

When a ``ToolExecutor`` has its ``action_executor`` wired, a learnable builtin
tool call (here ``change_emotion``) must route through the canonical executor
and produce a learning transition — i.e. exactly one stored experience tagged
with the resolved action name. The direct ``_handle_*`` path is bypassed, so
the call is recorded once, in one place.

When ``action_executor`` is ``None`` the direct path runs and no transition is
recorded (preserves the pre-teaching-loop behaviour and the existing tool tests).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot.ai.tools.executor import ToolExecutor
from robot.ai.tools.registry import BUILTIN_TOOLS as _BUILTIN_TOOLS, ToolRegistry
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


async def _noop_tool_handler(**_kwargs: object) -> dict[str, str]:
    """Placeholder handler matching the app's registration shape."""
    return {"status": "ok"}


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for definition in _BUILTIN_TOOLS.values():
        registry.add(definition, handler=_noop_tool_handler)
    return registry


@pytest.fixture
def bus() -> InMemoryEventBus:
    return InMemoryEventBus()


async def test_change_emotion_tool_creates_transition(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """A learnable tool routed through the executor creates one experience."""
    service = _make_service(bus=bus, tmp_path=tmp_path)
    recorder = service.recorder
    assert recorder is not None

    executor = ActionExecutor(bus=bus, servo_controller=_servo_controller())  # type: ignore[arg-type]
    executor.experience_recorder = recorder
    tool_executor = ToolExecutor(
        registry=_make_registry(),
        bus=bus,
        servo_controller=None,
        tts=None,
        audio=None,
        action_executor=executor,
    )

    recorder.attach()
    try:
        assert service.status.total_experiences == 0
        result = await tool_executor.execute_tool_call(
            "change_emotion", {"emotion": "happy", "intensity": 0.8}
        )
        # The routed handler returns the same shape as the direct handler.
        assert result["status"] == "ok"
        assert result["emotion"] == "happy"
        # Exactly one transition was recorded, tagged with the action name.
        assert service.status.total_experiences == 1
        exp = service.working_memory.recent(1)[-1]
        assert exp.metadata["behavior_action_name"] == "change_emotion"
        assert exp.metadata["execution_success"] is True
    finally:
        recorder.detach()


async def test_direct_path_records_no_transition(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """Without a wired executor, the direct handler runs and records nothing."""
    service = _make_service(bus=bus, tmp_path=tmp_path)
    recorder = service.recorder
    assert recorder is not None

    # No action_executor wired -> direct _handle_* path.
    tool_executor = ToolExecutor(
        registry=_make_registry(),
        bus=bus,
        servo_controller=None,
        tts=None,
        audio=None,
        action_executor=None,
    )
    # Wire the recorder onto a throwaway executor just so we can attach it to
    # the bus; the direct path does not go through any executor.
    recorder.attach()
    try:
        result = await tool_executor.execute_tool_call(
            "change_emotion", {"emotion": "sad", "intensity": 0.5}
        )
        assert result["status"] == "ok"
        assert service.status.total_experiences == 0
    finally:
        recorder.detach()

