"""Phase 3, Test 10: a non-learnable tool does not create a transition.

``play_sound`` is not registered in the :class:`ActionSpace`, so even when an
:class:`ActionExecutor` is wired it stays a direct bus publish and is logged as
``action not learnable``. No learning transition is opened and no experience is
stored — observation events (``SoundEffectPlayed``) update state only, they
never create transitions.
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


async def test_play_sound_creates_no_transition(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """play_sound is not in the ActionSpace -> no transition, just a bus publish."""
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
        result = await tool_executor.execute_tool_call("play_sound", {"name": "greet"})
        # The direct handler still runs and publishes the sound effect.
        assert result["status"] == "ok"
        assert result["sound"] == "greet"
        # But no learning transition was recorded — play_sound is not learnable.
        assert service.status.total_experiences == 0
        assert len(service.working_memory) == 0
    finally:
        recorder.detach()


async def test_play_sound_not_learnable_log_emitted(
    bus: InMemoryEventBus, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """The non-learnable path logs ``action_not_learnable`` for play_sound.

    structlog renders to stdout (not the stdlib logging handlers), so the log
    is asserted via ``capfd`` (file-descriptor capture) rather than ``caplog``
    — structlog may hold a reference to the original ``sys.stdout``, which only
    fd-level capture reliably intercepts.
    """
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
        await tool_executor.execute_tool_call("play_sound", {"name": "talk"})
        out = capfd.readouterr().out
        assert "action_not_learnable" in out, (
            "expected action_not_learnable log for play_sound"
        )
        assert service.status.total_experiences == 0
    finally:
        recorder.detach()

