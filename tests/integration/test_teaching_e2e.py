"""Phase 10 (E2E): the full human teaching loop through the real pipeline.

This is the end-to-end test the whole project targets, driven entirely by
real events on the real event bus — no LLM, no TTS, no synthetic experiences:

1. ``SpeechRecognized("when I wave, wave back")`` is published on the bus;
   a speech handler arms a teaching session through the constrained parser
   (:func:`parse_teaching_instruction` → :meth:`start_session`). The LLM never
   decides what action the robot should learn.
2. Repeatedly: ``FaceDetected`` (a person is present) then
   ``GestureDetected(wave)`` is published. A gesture observation handler sets
   the encoder's gesture slot, then the teaching-controller handler runs
   ``on_gesture_detected`` → the canonical :class:`ActionExecutor` executes
   the real ``WaveAction`` and the recorder stores a **real** transition tagged
   with the teaching-session + interaction id.
3. ``SpeechRecognized("good")`` is published; the speech handler reads it as
   praise and the :class:`FeedbackService` attributes +1 to the most-recent
   eligible real transition via the :class:`FeedbackLedger`.

After enough repetitions and forced training cycles, the policy's Q-value for
``wave`` in the (person + wave + teaching) state must rise above an unrelated
action (``look_left``). Nothing hard-codes ``wave`` as the answer: the only
signal that wave is "good" is the human's positive feedback, amended into the
reward through the ledger.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot.events.bus import InMemoryEventBus
from robot.events.events import FaceDetected, GestureDetected, SpeechRecognized
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
from robot.learning.safety_gate import SafetyGate
from robot.learning.state_encoder import _GESTURE_NAMES, _GESTURE_START, STATE_SIZE
from robot.learning.teaching_controller import TeachingController
from robot.services.executor import ActionExecutor

#: Spoken praise / correction cues (mirrors ConversationService._match_feedback).
_PRAISE = {"good", "great", "yes", "nice", "well done", "good robot"}
_CORRECTION = {"no", "bad", "wrong", "nope", "stop"}

_WAVE_INDEX = 13
_GESTURE_WAVE_SLOT = _GESTURE_START + _GESTURE_NAMES.index("wave")


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


class _Harness:
    """Wires the real teaching pipeline onto a bus, event-driven (no LLM/TTS)."""

    def __init__(
        self, bus: InMemoryEventBus, service: LearningService, tmp_path: Path
    ) -> None:
        self.bus = bus
        self.service = service
        recorder = service.recorder
        assert recorder is not None
        assert service.action_learner is not None

        ledger = FeedbackLedger()
        service.feedback_ledger = ledger
        self.ledger = ledger
        self.feedback = FeedbackService(
            recorder, ledger, feedback_window_s=30.0, staleness_s=60.0
        )

        executor = ActionExecutor(bus=bus, servo_controller=_servo_controller())  # type: ignore[arg-type]
        executor.experience_recorder = recorder
        context = InteractionContext()
        executor.interaction_context = context
        self.context = context
        self.executor = executor

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
        self.controller = TeachingController(
            action_learner=service.action_learner,
            safety_gate=safety_gate,
            action_space=service.action_space,
            interaction_context=context,
            executor=executor,
            min_experiences_for_practice=64,
        )

        # Observation handlers: update the encoder from perception events.
        # Critical so they run before the (non-critical) gesture trigger handler
        # and the encoder state is current when on_gesture_detected encodes.
        bus.subscribe_critical(FaceDetected, self._on_face)
        bus.subscribe_critical(GestureDetected, self._on_gesture_observation)
        # Speech handler mirrors ConversationService: teaching-first, then
        # praise/correction as a side effect.
        bus.subscribe(SpeechRecognized, self._on_speech)
        # Gesture trigger handler: runs the teaching controller.
        bus.subscribe(GestureDetected, self._on_gesture_trigger)

    # -- observation -----------------------------------------------------
    async def _on_face(self, _event: FaceDetected) -> None:
        self.service.encoder.update_vision(face_detected=True, face_count=1)
        self.service.encoder.update_person_present(True)

    async def _on_gesture_observation(self, event: GestureDetected) -> None:
        self.service.encoder.update_gesture(event.gesture)

    # -- speech ----------------------------------------------------------
    async def _on_speech(self, event: SpeechRecognized) -> None:
        sid = self.controller.arm_from_instruction(event.text)
        if sid is not None:
            self.service.encoder.update_teaching_context(True)
            return
        cue = event.text.strip().lower()
        if cue in _PRAISE:
            await self.feedback.handle_feedback(
                polarity=+1, source="speech", text=event.text
            )
        elif cue in _CORRECTION:
            await self.feedback.handle_feedback(
                polarity=-1, source="speech", text=event.text
            )

    # -- gesture trigger -------------------------------------------------
    async def _on_gesture_trigger(self, event: GestureDetected) -> None:
        await self.controller.on_gesture_detected(event.gesture, self.service.encoder.encode())


@pytest.fixture
def bus() -> InMemoryEventBus:
    return InMemoryEventBus()


async def test_e2e_human_teaching_loop_learns_wave(
    bus: InMemoryEventBus, tmp_path: Path
) -> None:
    """The full event-driven loop: speech arm → face+gesture → praise → Q rises."""
    service = _make_service(bus, tmp_path)
    recorder = service.recorder
    assert recorder is not None
    harness = _Harness(bus, service, tmp_path)
    controller = harness.controller

    recorder.attach()
    try:
        # 1. Arm a teaching session from a spoken instruction (no LLM).
        await bus.publish(SpeechRecognized(text="when I wave, wave back"))
        assert controller.in_teaching_mode is True
        assert controller.session_id is not None
        session_id = controller.session_id

        # 2. Repeatedly: person present + wave gesture -> real wave -> praise.
        repetitions = 12
        for _ in range(repetitions):
            await bus.publish(FaceDetected(x=0.5, y=0.5, confidence=0.9))
            await bus.publish(GestureDetected(gesture="wave", confidence=0.9))
            await bus.publish(SpeechRecognized(text="good"))

        # 3. Real experiences were collected through the canonical executor.
        assert service.status.total_experiences == repetitions

        recent = service.working_memory.recent(repetitions)
        for exp in recent:
            assert exp.metadata["action_index"] == _WAVE_INDEX  # wave
            assert exp.metadata["teaching_session_id"] == session_id
            assert exp.metadata.get("interaction_id") is not None
            # The amended reward carries the attributed praise (+1); the base
            # reward for wave with a face present is 0.0 — no hard-coded reward.
            tid = exp.metadata["transition_id"]
            assert service.reward_for_transition(tid) > 0.0

        # The last wave transition's recorded state carries person + wave +
        # teaching context (the state in which waving was "correct").
        last_state = list(recent[-1].state)
        assert last_state[_GESTURE_WAVE_SLOT] == 1.0
        assert last_state[_GESTURE_START] == 0.0  # "none" slot cleared

        # 4. The ledger carries the attributed praise for every transition.
        assert len(harness.ledger) == repetitions
        for exp in recent:
            entry = harness.ledger.get(exp.metadata["transition_id"])
            assert entry is not None
            assert entry.polarity == +1
            assert entry.reward_delta == 1.0

        # 5. Train and confirm wave becomes preferred over look_left in the
        # (person + wave + teaching) state — the real learning signal.
        q_before = service.q_values(last_state)
        preferred = False
        for _ in range(100):
            service.force_training()
            q = service.q_values(last_state)
            if q["wave"] > q["look_left"]:
                preferred = True
                break
        q = service.q_values(last_state)
        assert preferred, (
            f"wave did not become preferred: wave={q['wave']} look_left={q['look_left']}"
        )
        assert q["wave"] != q_before["wave"]
    finally:
        recorder.detach()
