"""Tests for the ExperienceRecorder that bridges events to memory."""

from __future__ import annotations

import asyncio

import pytest

from robot.behavior.state_machine import RobotState, StateMachine
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
    InMemoryExperienceStore,
    ReplayBuffer,
    WorkingMemory,
)
from robot.learning.recorder import ExperienceRecorder
from robot.learning.state_encoder import STATE_SIZE

# ========================================================================
# ExperienceRecorder
# ========================================================================


class TestExperienceRecorder:
    """Tests for the ExperienceRecorder event bus integration."""

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

    @pytest.fixture
    def state_machine(self, bus: InMemoryEventBus) -> StateMachine:
        return StateMachine(bus=bus)

    async def test_state_changed_produces_experience(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder, state_machine: StateMachine
    ) -> None:
        """State transitions should produce experience tuples."""
        await state_machine.transition(RobotState.IDLE)
        await bus.publish(StateChanged(previous=RobotState.BOOT, current=RobotState.IDLE))
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) > 0

    async def test_emotion_changed_produces_experience(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """Emotion changes should produce experience tuples."""
        await bus.publish(
            EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY, intensity=0.8)
        )
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) > 0
        exp = recorder.working_memory.recent(1)[0]
        assert exp.metadata.get("event_type") == "EmotionChanged"

    async def test_face_detected_produces_experience(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """Face detection events should produce experience tuples."""
        await bus.publish(FaceDetected(x=0.5, y=0.3, confidence=0.9))
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) > 0
        exp = recorder.working_memory.recent(1)[0]
        assert exp.metadata.get("event_type") == "FaceDetected"
        assert exp.reward > 0  # face detection gets a positive reward

    async def test_speech_recognized_produces_experience(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """Speech recognition events should produce experience tuples."""
        await bus.publish(SpeechRecognized(text="hello", confidence=0.95))
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) > 0
        exp = recorder.working_memory.recent(1)[0]
        assert exp.metadata.get("event_type") == "SpeechRecognized"

    async def test_idle_timeout_produces_experience(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """Idle timeout events should produce experience tuples."""
        await bus.publish(IdleTimeout(seconds_idle=30.0))
        await asyncio.sleep(0.01)
        assert len(recorder.working_memory) > 0
        exp = recorder.working_memory.recent(1)[0]
        assert exp.metadata.get("event_type") == "IdleTimeout"
        assert exp.reward < 0  # idling gets a negative reward

    async def test_manual_record(self, bus: InMemoryEventBus, recorder: ExperienceRecorder) -> None:
        """Manual recording should work without events."""
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

    async def test_servo_moved_updates_encoder(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """ServoMoved events should update the encoder's servo state."""
        await bus.publish(ServoMoved(name="pan", angle=45.0))
        await asyncio.sleep(0.01)
        # Check that the encoder updated its servo position
        assert recorder.encoder.servos.get("pan") == 45.0

    async def test_detach(self, bus: InMemoryEventBus, recorder: ExperienceRecorder) -> None:
        """After detaching, no more experiences should be recorded."""
        recorder.detach()
        await bus.publish(FaceDetected(x=0.5, y=0.3, confidence=0.9))
        await asyncio.sleep(0.01)
        initial_count = len(recorder.working_memory)
        assert initial_count == 0

    async def test_with_episodic_memory(self, bus: InMemoryEventBus) -> None:
        """EpisodicMemory should be populated when configured."""
        store = InMemoryExperienceStore()
        em = EpisodicMemory(store=store, capacity=100)
        recorder = ExperienceRecorder(
            bus=bus,
            working_memory=WorkingMemory(capacity=100),
            replay_buffer=ReplayBuffer(capacity=100, seed=42),
            episodic_memory=em,
        )
        recorder.attach()

        await bus.publish(FaceDetected(x=0.5, y=0.5, confidence=0.8))
        await asyncio.sleep(0.01)

        assert len(em) > 0
        assert store.count() > 0

    async def test_experience_has_correct_state_size(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """Recorded experiences should have state vectors of the correct size."""
        await bus.publish(FaceDetected(x=0.5, y=0.3, confidence=0.9))
        await asyncio.sleep(0.01)

        exp = recorder.working_memory.recent(1)[0]
        assert len(exp.state) == STATE_SIZE, (
            f"State should have {STATE_SIZE} elements, got {len(exp.state)}"
        )
        assert len(exp.next_state) == STATE_SIZE, f"next_state should have {STATE_SIZE} elements"

    async def test_record_with_encoder(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """record_with_encoder should use the current encoder state."""
        recorder.encoder.update_state(RobotState.CURIOUS)
        recorder.encoder.update_emotion(EmotionName.HAPPY, 0.8)

        action = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # FaceDetected one-hot
        exp = recorder.record_with_encoder(
            action=action,
            reward=0.5,
            metadata={"source": "test"},
        )
        assert len(exp.state) == STATE_SIZE
        assert len(exp.next_state) == STATE_SIZE
        assert exp.reward == 0.5

    async def test_update_context(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """update_context should update the encoder."""
        recorder.update_context(state=RobotState.IDLE)
        assert recorder.encoder.state == RobotState.IDLE
        recorder.update_context(emotions={"happy": 0.8})
        assert recorder.encoder.emotions.get("happy") == 0.8

    async def test_encoder_deterministic(
        self, bus: InMemoryEventBus, recorder: ExperienceRecorder
    ) -> None:
        """Same encoder state should produce same vector."""
        recorder.encoder.update_state(RobotState.IDLE)
        recorder.encoder.update_emotion(EmotionName.HAPPY, 0.7)
        vec1 = recorder.encoder.encode()
        vec2 = recorder.encoder.encode()
        assert vec1 == vec2, "Same inputs should produce identical vectors"
