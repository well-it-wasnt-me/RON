"""Integration test: simulation produces experiences that survive restart.

Integration test proving the transition lifecycle acceptance criteria:
  observe -> act -> observe result -> store experience -> restart -> load experience

Observation events update the encoder; real actions selected from the
ActionSpace produce transitions through the lifecycle.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

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
)
from robot.learning.experience import (
    EpisodicMemory,
    SqliteExperienceStore,
)
from robot.learning.recorder import ExperienceRecorder
from robot.learning.state_encoder import STATE_SIZE


class TestSimulationExperienceRecording:
    """Integration test using simulation events + transition lifecycle."""

    async def test_transition_lifecycle_persists_experiences(self, tmp_path: Path) -> None:
        """Full acceptance test: observe -> act -> store -> restart -> load."""
        db_path = tmp_path / "simulation_experiences.db"

        # --- Run simulation and record experiences ---
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)

        store = SqliteExperienceStore(db_path=db_path)
        episodic = EpisodicMemory(store=store, capacity=100)
        recorder = ExperienceRecorder(bus=bus, episodic_memory=episodic)
        recorder.attach()

        # Observe: BOOT -> IDLE transition (updates encoder)
        recorder.update_context(state=RobotState.BOOT)
        await sm.transition(RobotState.IDLE)

        # Observe: face detected (updates encoder only — no transition)
        await bus.publish(FaceDetected(x=0.5, y=0.3, confidence=0.85))
        await asyncio.sleep(0.01)

        # --- Act: select action look_center (index 2) from ActionSpace ---
        pending = recorder.begin_transition(action_index=2)

        # Observe: outcome after action — encoder state reflects the new state
        await bus.publish(
            EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY, intensity=0.7)
        )
        await asyncio.sleep(0.01)

        # Complete the transition with the post-execution observation
        transition = recorder.complete_transition(pending, reward=0.5)
        assert transition.action_name == "look_center"
        assert transition.execution_success is True

        # Verify experience was recorded and persisted
        assert len(recorder.working_memory) == 1
        assert store.count() == 1

        experiences = store.load_all()
        assert len(experiences) == 1
        exp = experiences[0]
        assert exp.metadata["action_name"] == "look_center"
        assert exp.metadata["action_index"] == 2
        assert exp.reward == 0.5
        assert len(exp.state) == STATE_SIZE
        assert len(exp.next_state) == STATE_SIZE
        assert exp.metadata["execution_success"] is True
        assert exp.metadata["transition_id"] != ""

        store.close()

        # --- Simulate restart and verify persistence ---
        store2 = SqliteExperienceStore(db_path=db_path)
        episodic2 = EpisodicMemory(store=store2, capacity=100, max_load=100)
        episodic2.load_from_store()

        assert len(episodic2) == 1, "Should have loaded 1 experience after restart"

        loaded = store2.load_all()
        assert len(loaded) == 1
        loaded_exp = loaded[0]
        assert loaded_exp.state == exp.state, "State vectors should match after restart"
        assert loaded_exp.action == exp.action, "Action vectors should match"
        assert loaded_exp.next_state == exp.next_state, "Next state vectors should match"
        assert loaded_exp.reward == exp.reward, "Reward should match"
        assert loaded_exp.metadata["action_name"] == "look_center"

        store2.close()

    async def test_multiple_transitions_persisted(self, tmp_path: Path) -> None:
        """Multiple transitions through the lifecycle are all persisted."""
        db_path = tmp_path / "multi_transitions.db"

        bus = InMemoryEventBus()
        store = SqliteExperienceStore(db_path=db_path)
        episodic = EpisodicMemory(store=store, capacity=100)
        recorder = ExperienceRecorder(bus=bus, episodic_memory=episodic)

        # Record 5 transitions with different actions
        for action_idx in range(5):
            pending = recorder.begin_transition(action_index=action_idx)
            recorder.complete_transition(pending, reward=float(action_idx) * 0.1)

        assert store.count() == 5
        loaded = store.load_all()
        assert len(loaded) == 5

        # Each transition should have the correct action identity
        for i, exp in enumerate(loaded):
            assert exp.metadata["action_index"] == i
            assert exp.reward == pytest.approx(float(i) * 0.1)

        store.close()

    async def test_observation_events_do_not_produce_experiences(self, tmp_path: Path) -> None:
        """Observation events alone produce zero experiences."""
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)

        store = SqliteExperienceStore(db_path=":memory:")
        episodic = EpisodicMemory(store=store, capacity=100)
        recorder = ExperienceRecorder(bus=bus, episodic_memory=episodic)
        recorder.attach()

        recorder.update_context(state=RobotState.BOOT)
        await sm.transition(RobotState.IDLE)
        await bus.publish(FaceDetected(x=0.5, y=0.3, confidence=0.85))
        await bus.publish(SpeechRecognized(text="hello", confidence=0.9))
        await bus.publish(ServoMoved(name="pan", angle=60.0))
        await bus.publish(IdleTimeout(seconds_idle=45.0))
        await asyncio.sleep(0.05)

        # No transitions created — only encoder updated
        assert len(recorder.working_memory) == 0
        assert store.count() == 0
        assert recorder.transition_store.pending_count == 0

        recorder.detach()
        store.close()

    async def test_working_memory_and_replay_buffer_integration(self) -> None:
        """Experiences flow through all memory layers."""
        bus = InMemoryEventBus()
        recorder = ExperienceRecorder(bus=bus)

        # Record 3 transitions
        for i in range(3):
            pending = recorder.begin_transition(action_index=i)
            recorder.complete_transition(pending, reward=float(i) * 0.1)

        assert len(recorder.working_memory) == 3
        assert len(recorder.replay_buffer) == 3
        batch = recorder.replay_buffer.sample(2)
        assert len(batch) == 2

    async def test_experience_vectors_are_consistent(self) -> None:
        """State vectors produced from the lifecycle are consistent."""
        bus = InMemoryEventBus()
        recorder = ExperienceRecorder(bus=bus)

        recorder.update_context(state=RobotState.IDLE)
        recorder.update_context(emotions={"happy": 0.8})

        pending = recorder.begin_transition(action_index=0)
        recorder.complete_transition(pending, reward=0.1)

        exp = recorder.working_memory.recent(1)[0]
        assert len(exp.state) == STATE_SIZE
        assert len(exp.state) == len(exp.next_state)
        for v in exp.state:
            assert isinstance(v, float)
            assert not math.isnan(v)
