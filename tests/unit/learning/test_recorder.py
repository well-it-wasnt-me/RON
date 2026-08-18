"""Tests for the ExperienceRecorder that bridges events to memory.

Observation events update the encoder only; only the transition
lifecycle (begin/complete with a real action from ActionSpace) produces
a stored experience.
"""

from __future__ import annotations

import asyncio

import pytest

from robot.behavior.state_machine import RobotState
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    EmotionChanged,
    EmotionName,
    FaceDetected,
    IdleTimeout,
    ServoMoved,
    SpeechRecognized,
    StateChanged,
)
from robot.learning.experience import (
    EpisodicMemory,
    Experience,
    InMemoryExperienceStore,
    ReplayBuffer,
    WorkingMemory,
)
from robot.learning.recorder import ExperienceRecorder
from robot.learning.state_encoder import STATE_SIZE
from robot.learning.transition import TransitionError

# ========================================================================
# Observation events update the encoder but do NOT produce transitions
# ========================================================================


class TestObservationEventsOnlyUpdateEncoder:
    """Observation events must not create fake transitions."""

    @pytest.fixture
    def bus(self) -> InMemoryEventBus:
        return InMemoryEventBus()

    @pytest.fixture
    def recorder(self, bus: InMemoryEventBus) -> ExperienceRecorder:
        rec = ExperienceRecorder(
            bus=bus,
            working_memory=WorkingMemory(capacity=100),
            replay_buffer=ReplayBuffer(capacity=100, seed=42),
        )
        rec.attach()
        return rec

    async def test_state_changed_no_experience(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """StateChanged updates encoder but does not produce a transition."""
        await bus.publish(StateChanged(previous=RobotState.BOOT, current=RobotState.IDLE))
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) == 0
        assert recorder.encoder.state == RobotState.IDLE

    async def test_face_detected_no_experience(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """FaceDetected updates encoder but does not produce a transition."""
        await bus.publish(FaceDetected(x=0.5, y=0.3, confidence=0.9))
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) == 0
        assert recorder.encoder.vision.face_detected == 1.0
        assert recorder.encoder.vision.face_x == 0.5

    async def test_speech_recognized_no_experience(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """SpeechRecognized is an observation, not an action."""
        await bus.publish(SpeechRecognized(text="hello", confidence=0.95))
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) == 0

    async def test_emotion_changed_no_experience(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """EmotionChanged updates encoder but does not produce a transition."""
        await bus.publish(
            EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY, intensity=0.8)
        )
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) == 0
        assert recorder.encoder.emotions.get("happy") == 0.8

    async def test_idle_timeout_no_experience(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """IdleTimeout updates encoder but does not produce a transition."""
        await bus.publish(IdleTimeout(seconds_idle=30.0))
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) == 0
        assert recorder.encoder.idle_seconds == 30.0

    async def test_servo_moved_no_experience(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """ServoMoved updates encoder but does not produce a transition."""
        await bus.publish(ServoMoved(name="pan", angle=45.0))
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) == 0
        assert recorder.encoder.servos.get("pan") == 45.0

    async def test_detach(self, bus: InMemoryEventBus, recorder: ExperienceRecorder) -> None:
        """After detaching, events should not update the encoder."""
        recorder.detach()
        await bus.publish(FaceDetected(x=0.5, y=0.3, confidence=0.9))
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) == 0
        assert recorder.encoder.vision.face_detected == 0.0


# ========================================================================
# Transition lifecycle produces experiences
# ========================================================================


class TestTransitionLifecycle:
    """The begin/complete transition lifecycle produces valid experiences."""

    @pytest.fixture
    def bus(self) -> InMemoryEventBus:
        return InMemoryEventBus()

    @pytest.fixture
    def recorder(self, bus: InMemoryEventBus) -> ExperienceRecorder:
        rec = ExperienceRecorder(
            bus=bus,
            working_memory=WorkingMemory(capacity=100),
            replay_buffer=ReplayBuffer(capacity=100, seed=42),
        )
        rec.attach()
        return rec

    def test_begin_and_complete_produces_experience(self, recorder: ExperienceRecorder) -> None:
        """begin + complete produces exactly one experience."""

        # Set an observation
        recorder.encoder.update_state(RobotState.CURIOUS)

        pending = recorder.begin_transition(action_index=2)  # look_center
        assert recorder.transition_store.pending_count == 1
        assert len(recorder.working_memory) == 0

        # Simulate execution outcome
        recorder.encoder.update_state(RobotState.IDLE)
        transition = recorder.complete_transition(pending, reward=0.5)

        assert recorder.transition_store.pending_count == 0
        assert len(recorder.working_memory) == 1
        assert len(recorder.replay_buffer) == 1

        # The experience has the correct action identity
        exp = recorder.working_memory.recent(1)[0]
        assert exp.metadata["action_name"] == "look_center"
        assert exp.metadata["action_index"] == 2
        assert exp.reward == 0.5
        assert len(exp.state) == STATE_SIZE
        assert len(exp.next_state) == STATE_SIZE
        assert transition.action_name == "look_center"

    def test_no_action_means_no_completed_transition(self, recorder: ExperienceRecorder) -> None:
        """Without calling complete, no experience is stored."""
        recorder.begin_transition(action_index=0)
        assert recorder.transition_store.pending_count == 1
        assert len(recorder.working_memory) == 0
        assert len(recorder.replay_buffer) == 0

    def test_complete_captures_next_state_after_execution(
        self, recorder: ExperienceRecorder
    ) -> None:
        """next_state reflects the encoder state after the action."""
        recorder.encoder.update_state(RobotState.IDLE)
        pending = recorder.begin_transition(action_index=0)
        # Simulate the robot acting — encoder state changes
        recorder.encoder.update_emotion(EmotionName.HAPPY, 0.9)
        recorder.encoder.update_state(RobotState.SPEAKING)
        recorder.complete_transition(pending, reward=1.0)

        # The next_state should reflect the post-execution observation
        exp = recorder.working_memory.recent(1)[0]
        # State was IDLE when begin was called
        assert exp.state != exp.next_state
        # next_state should have HAPPY emotion set
        assert exp.next_state[1] == 0.9  # happy is index 1 in emotions

    def test_failed_action_execution_recorded(self, recorder: ExperienceRecorder) -> None:
        """A failed action execution is recorded with success=False."""
        pending = recorder.begin_transition(action_index=0)
        transition = recorder.complete_transition(
            pending,
            reward=-1.0,
            execution_success=False,
            execution_failure_reason="servo timeout",
        )
        assert transition.execution_success is False
        assert transition.execution_failure_reason == "servo timeout"

        exp = recorder.working_memory.recent(1)[0]
        assert exp.metadata["execution_success"] is False
        assert exp.metadata["execution_failure_reason"] == "servo timeout"

    def test_action_id_belongs_to_action_space(self, recorder: ExperienceRecorder) -> None:
        """The action index must be a valid index in the action space."""

        for i in range(recorder.action_space.size):
            pending = recorder.begin_transition(action_index=i)
            transition = recorder.complete_transition(pending, reward=0.0)
            assert transition.action_index == i
            assert transition.action_name == recorder.action_space.get(i).name

    def test_invalid_action_index_rejected(self, recorder: ExperienceRecorder) -> None:
        """An out-of-range action index is rejected at begin."""
        with pytest.raises(TransitionError, match="out of range"):
            recorder.begin_transition(action_index=999)

    def test_invalid_action_index_negative(self, recorder: ExperienceRecorder) -> None:
        """A negative action index is rejected."""
        with pytest.raises(TransitionError, match="out of range"):
            recorder.begin_transition(action_index=-1)

    def test_complete_twice_rejected(self, recorder: ExperienceRecorder) -> None:
        """Completing the same transition twice is rejected."""
        pending = recorder.begin_transition(action_index=0)
        recorder.complete_transition(pending, reward=0.0)
        with pytest.raises(TransitionError, match="already completed"):
            recorder.complete_transition(pending, reward=0.0)

    def test_timestamps_monotonic(self, recorder: ExperienceRecorder) -> None:
        """Start timestamp is before completion timestamp."""
        pending = recorder.begin_transition(action_index=0)
        transition = recorder.complete_transition(pending, reward=0.0)
        assert transition.completion_timestamp_ns >= transition.start_timestamp_ns
        assert transition.latency_ms >= 0.0

    def test_execution_metadata_recorded(self, recorder: ExperienceRecorder) -> None:
        """Execution metadata is present in the stored transition."""
        pending = recorder.begin_transition(
            action_index=5,
            execution_id="exec-001",
            policy_version="v1.2.3",
        )
        transition = recorder.complete_transition(pending, reward=0.3, metadata={"scene": "test"})
        assert transition.execution_id == "exec-001"
        assert transition.policy_version == "v1.2.3"
        assert transition.metadata["scene"] == "test"
        assert transition.transition_id != ""
        assert transition.latency_ms >= 0.0

    def test_default_reward_used(self, recorder: ExperienceRecorder) -> None:
        """When reward is None, default_reward is used."""
        recorder.default_reward = 0.42
        pending = recorder.begin_transition(action_index=0)
        transition = recorder.complete_transition(pending, reward=None)
        assert transition.reward == 0.42

    def test_with_episodic_memory(self) -> None:
        """EpisodicMemory is populated when configured."""
        bus = InMemoryEventBus()
        store = InMemoryExperienceStore()
        em = EpisodicMemory(store=store, capacity=100)
        recorder = ExperienceRecorder(
            bus=bus,
            working_memory=WorkingMemory(capacity=100),
            replay_buffer=ReplayBuffer(capacity=100, seed=42),
            episodic_memory=em,
        )

        pending = recorder.begin_transition(action_index=0)
        recorder.complete_transition(pending, reward=0.5)

        assert len(em) == 1
        assert store.count() == 1

    def test_callback_invoked(self) -> None:
        """The on_experience_recorded callback is invoked once per transition."""
        recorded: list[Experience] = []
        bus = InMemoryEventBus()
        recorder = ExperienceRecorder(
            bus=bus,
            working_memory=WorkingMemory(capacity=100),
            replay_buffer=ReplayBuffer(capacity=100, seed=42),
            on_experience_recorded=recorded.append,
        )
        pending = recorder.begin_transition(action_index=0)
        recorder.complete_transition(pending, reward=0.1)
        assert len(recorded) == 1
        assert recorded[0].metadata["action_name"] is not None


# ========================================================================
# Backward compatibility: manual record() still works
# ========================================================================


class TestManualRecord:
    """The legacy record() method still stores experiences directly."""

    def test_manual_record(self) -> None:
        bus = InMemoryEventBus()
        recorder = ExperienceRecorder(
            bus=bus,
            working_memory=WorkingMemory(capacity=100),
            replay_buffer=ReplayBuffer(capacity=100, seed=42),
        )
        exp = recorder.record(
            state=[1.0, 2.0],
            action=[0.5],
            reward=1.0,
            next_state=[1.1, 2.1],
            metadata={"source": "manual"},
        )
        assert len(recorder.working_memory) == 1
        assert exp.state == [1.0, 2.0]
        assert exp.metadata["source"] == "manual"


# ========================================================================
# Encoder helpers
# ========================================================================


class TestEncoderHelpers:
    """Test update_context and encoder determinism."""

    def test_update_context(self) -> None:
        bus = InMemoryEventBus()
        recorder = ExperienceRecorder(bus=bus)
        recorder.update_context(state=RobotState.IDLE)
        assert recorder.encoder.state == RobotState.IDLE
        recorder.update_context(emotions={"happy": 0.8})
        assert recorder.encoder.emotions.get("happy") == 0.8

    def test_encoder_deterministic(self) -> None:
        bus = InMemoryEventBus()
        recorder = ExperienceRecorder(bus=bus)
        recorder.encoder.update_state(RobotState.IDLE)
        recorder.encoder.update_emotion(EmotionName.HAPPY, 0.7)
        vec1 = recorder.encoder.encode()
        vec2 = recorder.encoder.encode()
        assert vec1 == vec2, "Same inputs should produce identical vectors"

    def test_state_size_correct(self) -> None:
        bus = InMemoryEventBus()
        recorder = ExperienceRecorder(bus=bus)
        pending = recorder.begin_transition(action_index=0)
        recorder.complete_transition(pending, reward=0.0)
        exp = recorder.working_memory.recent(1)[0]
        assert len(exp.state) == STATE_SIZE
        assert len(exp.next_state) == STATE_SIZE
