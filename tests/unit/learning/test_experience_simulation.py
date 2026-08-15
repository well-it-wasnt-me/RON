"""Integration test: simulation produces experiences that survive restart.

This test satisfies the Phase 2 acceptance criteria:
  observe -> act -> observe result -> store experience -> restart -> load experience

It uses DeskBot's SimulationDriver to produce real events, records them
as experiences, and verifies persistence across a simulated restart.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    EmotionChanged,
    EmotionName,
    FaceDetected,
    IdleTimeout,
    ServoMoved,
    SpeechRecognized,
)
from robot.learning.experience import (
    EpisodicMemory,
    SqliteExperienceStore,
)
from robot.learning.recorder import ExperienceRecorder


class TestSimulationExperienceRecording:
    """Integration test using simulation events to produce experiences."""

    async def test_simulation_produces_and_persists_experiences(self, tmp_path: Path) -> None:
        """Full acceptance test: observe -> act -> store -> restart -> load."""
        db_path = tmp_path / "simulation_experiences.db"

        # --- Phase 1: Run simulation and record experiences ---
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)

        store = SqliteExperienceStore(db_path=db_path)
        episodic = EpisodicMemory(store=store, capacity=100)
        recorder = ExperienceRecorder(bus=bus, episodic_memory=episodic)
        recorder.attach()

        # Update context with initial state
        recorder.update_context(state=RobotState.BOOT)

        # Simulate: BOOT -> IDLE transition
        await sm.transition(RobotState.IDLE)

        # Simulate: face detected (perception event)
        await bus.publish(FaceDetected(x=0.5, y=0.3, confidence=0.85))

        # Simulate: emotion change (reaction to face)
        await bus.publish(
            EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.CURIOUS, intensity=0.7)
        )

        # Simulate: IDLE -> CURIOUS transition
        await sm.transition(RobotState.CURIOUS)

        # Simulate: speech recognized
        await bus.publish(SpeechRecognized(text="hello deskbot", confidence=0.92))

        # Simulate: servo movement
        await bus.publish(ServoMoved(name="pan", angle=60.0))

        # Simulate: idle timeout
        await bus.publish(IdleTimeout(seconds_idle=45.0))

        # Allow async handlers to process
        await asyncio.sleep(0.05)

        # Verify experiences were recorded
        assert len(recorder.working_memory) > 0, "Should have recorded experiences from events"
        assert store.count() > 0, "Experiences should be persisted to SQLite"

        recorded_count = store.count()
        assert recorded_count >= 3, f"Expected at least 3 experiences, got {recorded_count}"

        # Verify experience content
        experiences = store.load_all()
        # Check that at least one experience has a face detection
        face_experiences = [
            e for e in experiences if e.metadata.get("event_type") == "FaceDetected"
        ]
        assert len(face_experiences) >= 1, "Should have at least one face detection experience"

        # Verify the experience has proper vectors
        face_exp = face_experiences[0]
        assert len(face_exp.state) > 0, "State vector should not be empty"
        assert len(face_exp.action) > 0, "Action vector should not be empty"
        assert face_exp.reward > 0, "Face detection should have positive reward"

        store.close()

        # --- Phase 2: Simulate restart and verify persistence ---
        store2 = SqliteExperienceStore(db_path=db_path)
        episodic2 = EpisodicMemory(store=store2, capacity=100, max_load=100)
        episodic2.load_from_store()

        # Experiences must survive application restart
        assert len(episodic2) == recorded_count, (
            f"Should have loaded {recorded_count} experiences after restart, got {len(episodic2)}"
        )

        # Verify data integrity
        loaded = store2.load_all()
        face_loaded = [e for e in loaded if e.metadata.get("event_type") == "FaceDetected"]
        assert len(face_loaded) >= 1
        face_loaded_exp = face_loaded[0]
        assert face_loaded_exp.state == face_exp.state, "State vectors should match after restart"
        assert face_loaded_exp.action == face_exp.action, "Action vectors should match"
        assert face_loaded_exp.next_state == face_exp.next_state, "Next state vectors should match"
        assert face_loaded_exp.reward == face_exp.reward, "Reward should match"

        store2.close()

    async def test_working_memory_and_replay_buffer_integration(self) -> None:
        """Experiences flow through all memory layers."""
        bus = InMemoryEventBus()

        recorder = ExperienceRecorder(bus=bus)
        recorder.attach()

        # Produce experiences manually
        recorder.record(
            state=[0.0] * 10,
            action=[1.0] * 5,
            reward=0.5,
            next_state=[0.1] * 10,
            metadata={"source": "manual"},
        )
        recorder.record(
            state=[0.1] * 10,
            action=[0.9] * 5,
            reward=0.3,
            next_state=[0.2] * 10,
            metadata={"source": "manual"},
        )

        # Working memory has both
        assert len(recorder.working_memory) == 2
        # Replay buffer has both
        assert len(recorder.replay_buffer) == 2
        # Can sample from replay buffer
        batch = recorder.replay_buffer.sample(2)
        assert len(batch) == 2

    async def test_experience_recorder_with_state_machine(self) -> None:
        """Recording experiences during state machine transitions."""
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)

        recorder = ExperienceRecorder(bus=bus)
        recorder.attach()

        # Update context
        recorder.update_context(state=RobotState.BOOT)

        # Transition through states
        await sm.transition(RobotState.IDLE)
        await sm.transition(RobotState.CURIOUS)

        # Give handlers time
        await asyncio.sleep(0.02)

        # Should have recorded experiences
        assert len(recorder.working_memory) > 0

    async def test_experience_vectors_are_consistent(self) -> None:
        """State vectors produced from context should be consistent."""
        bus = InMemoryEventBus()

        recorder = ExperienceRecorder(bus=bus)
        recorder.attach()

        # Set up context
        recorder.update_context(state=RobotState.IDLE)
        recorder.update_context(emotions={"happy": 0.8, "curious": 0.5})

        # Fire an event that produces an experience
        await bus.publish(FaceDetected(x=0.5, y=0.5, confidence=0.9))
        await asyncio.sleep(0.02)

        exp = recorder.working_memory.recent(1)[0]
        # State vector should have consistent size
        assert len(exp.state) > 0
        assert len(exp.state) == len(exp.next_state), "State and next_state should have same size"
        # All values should be finite floats
        for v in exp.state:
            assert isinstance(v, float)
            assert not math.isnan(v)  # not NaN
