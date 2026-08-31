"""Phase 8: practice mode is safety-gated and falls back to demonstration.

Practice mode lets the trained policy propose an action on each gesture,
gated by the :class:`SafetyGate`. Three invariants are exercised:

1. **Fallback**: below the experience floor the policy has not learned enough,
   so practice falls back to demonstrating the desired action (a real
   transition is still recorded).
2. **Accepted proposal**: above the floor, an in-range proposal is executed
   through the executor — the *policy's* action, not the demonstration — and
   recorded as a real transition.
3. **Rejected proposal is a no-op**: a proposal that the full mutating safety
   gate rejects (e.g. on cooldown) becomes a no-op: no execution, no
   transition, and never a ``ServoError``.
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
from robot.learning.teaching_parser import parse_teaching_instruction
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


def _build(
    service: LearningService,
    *,
    cooldown_s: float,
    min_experiences_for_practice: int,
) -> tuple[TeachingController, SafetyGate, ActionExecutor]:
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
        cooldown_s=cooldown_s,
    )
    controller = TeachingController(
        action_learner=service.action_learner,
        safety_gate=safety_gate,
        action_space=service.action_space,
        interaction_context=executor.interaction_context,
        executor=executor,
        min_experiences_for_practice=min_experiences_for_practice,
    )
    return controller, safety_gate, executor


@pytest.fixture
def bus() -> InMemoryEventBus:
    return InMemoryEventBus()


async def test_practice_falls_back_to_demonstration_below_floor(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """With too few experiences, practice demonstrates the desired action."""
    service = _make_service(bus, tmp_path)
    controller, _gate, _executor = _build(
        service, cooldown_s=0.0, min_experiences_for_practice=64
    )
    recorder = service.recorder
    assert recorder is not None
    spec = parse_teaching_instruction("when i wave, wave back", service.action_space)
    assert spec is not None
    controller.start_session(spec, mode="practice")

    recorder.attach()
    try:
        state = service.encoder.encode()
        executed = await controller.on_gesture_detected("wave", state)
        # Falls back to the demonstration (wave), not a policy proposal.
        assert executed == spec.desired_action_index == 13
        assert service.status.total_experiences == 1
        exp = service.working_memory.recent(1)[-1]
        assert exp.metadata["action_index"] == 13
    finally:
        recorder.detach()


async def test_practice_executes_accepted_proposal(
    bus: InMemoryEventBus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Above the floor, an in-range policy proposal is executed and recorded."""
    service = _make_service(bus, tmp_path)
    controller, _gate, _executor = _build(
        service, cooldown_s=0.0, min_experiences_for_practice=0
    )
    recorder = service.recorder
    assert recorder is not None
    learner = service.action_learner
    assert learner is not None
    # Force the policy to propose blink (index 5): a no-servo, always-safe
    # action. This is distinct from the demonstration target (wave=13), proving
    # the executed action is the policy's proposal, not the demonstration.
    # ActionLearner is a slotted dataclass, so patch the class method (the
    # monkeypatch is auto-reverted at test teardown).
    monkeypatch.setattr(
        type(learner), "select_action", lambda self, state, validator=None: 5
    )

    spec = parse_teaching_instruction("when i wave, wave back", service.action_space)
    assert spec is not None
    controller.start_session(spec, mode="practice")

    recorder.attach()
    try:
        state = service.encoder.encode()
        executed = await controller.on_gesture_detected("wave", state)
        assert executed == 5
        assert service.status.total_experiences == 1
        exp = service.working_memory.recent(1)[-1]
        assert exp.metadata["action_index"] == 5
    finally:
        recorder.detach()


async def test_practice_rejected_proposal_is_noop(
    bus: InMemoryEventBus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proposal rejected by the safety gate becomes a no-op (no transition)."""
    service = _make_service(bus, tmp_path)
    controller, gate, _executor = _build(
        service, cooldown_s=10.0, min_experiences_for_practice=0
    )
    recorder = service.recorder
    assert recorder is not None
    learner = service.action_learner
    assert learner is not None
    # Force the policy to propose wave (13). See the accepted-proposal test for
    # why the class method is patched.
    monkeypatch.setattr(
        type(learner), "select_action", lambda self, state, validator=None: 13
    )
    # Pre-arm wave's cooldown so the full mutating gate rejects it on
    # re-validation. The non-mutating validator used during selection still
    # allows wave (static layer only), so the proposal reaches the gate.
    assert gate.validate(13).allowed

    spec = parse_teaching_instruction("when i wave, wave back", service.action_space)
    assert spec is not None
    controller.start_session(spec, mode="practice")

    recorder.attach()
    try:
        state = service.encoder.encode()
        executed = await controller.on_gesture_detected("wave", state)
        assert executed is None
        assert service.status.total_experiences == 0
    finally:
        recorder.detach()


async def test_practice_no_learner_is_noop(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """Without a learner, practice cannot propose and returns None."""
    service = _make_service(bus, tmp_path)
    controller, _gate, _executor = _build(
        service, cooldown_s=0.0, min_experiences_for_practice=0
    )
    recorder = service.recorder
    assert recorder is not None
    # Remove the learner to simulate the disabled-policy path.
    controller._action_learner = None

    spec = parse_teaching_instruction("when i wave, wave back", service.action_space)
    assert spec is not None
    controller.start_session(spec, mode="practice")

    recorder.attach()
    try:
        state = service.encoder.encode()
        executed = await controller.on_gesture_detected("wave", state)
        assert executed is None
        assert service.status.total_experiences == 0
    finally:
        recorder.detach()
