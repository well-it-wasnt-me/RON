"""Integration test: StateEncoder with simulation events.

Verifies that the StateEncoder produces correct vectors when fed
real DeskBot events through the ExperienceRecorder, and that the
vectors are suitable for neural-network training.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    EmotionChanged,
    EmotionName,
    FaceDetected,
)
from robot.learning.recorder import ExperienceRecorder
from robot.learning.state_encoder import (
    STATE_SIZE,
    AudioFeatures,
    StateEncoder,
)


class TestStateEncoderSimulation:
    """Integration tests using real DeskBot events."""

    async def test_state_encoder_after_event_sequence(self) -> None:
        """Full event sequence should produce a well-formed state vector."""
        encoder = StateEncoder()

        # Simulate a sequence of events
        encoder.update_state(RobotState.IDLE)
        encoder.update_emotion(EmotionName.NEUTRAL, 1.0)
        vec_idle = encoder.encode()

        # Face detected
        encoder.update_vision(
            face_detected=True, face_x=0.4, face_y=0.6, face_confidence=0.9, face_count=1
        )
        encoder.update_state(RobotState.CURIOUS)
        encoder.update_emotion(EmotionName.CURIOUS, 0.7)
        vec_curious = encoder.encode()

        # Vectors should differ
        assert vec_idle != vec_curious
        # Curious state should have face_detected=1.0
        assert vec_curious[33] == 1.0
        # Emotion CURIOUS should be set
        emotions = list(EmotionName)
        curious_idx = emotions.index(EmotionName.CURIOUS)
        assert vec_curious[curious_idx] == pytest.approx(0.7)

    async def test_state_encoder_with_audio(self) -> None:
        """Audio features should integrate into the state vector."""
        encoder = StateEncoder()

        # Simulate silence
        silence_pcm = struct.pack(f"<{480}h", *([0] * 480))
        audio = AudioFeatures.from_pcm(silence_pcm, sample_rate=16000)
        encoder.update_audio(audio)
        vec = encoder.encode()
        assert vec[39] == 0.0  # rms_energy
        assert vec[40] == 0.0  # peak_amplitude

        # Simulate audio
        samples = [
            int(16000 * (__import__("math").sin(2 * __import__("math").pi * 440 * i / 16000)))
            for i in range(480)
        ]
        audio_loud = AudioFeatures.from_pcm(struct.pack(f"<{480}h", *samples), sample_rate=16000)
        encoder.update_audio(audio_loud)
        vec_loud = encoder.encode()
        assert vec_loud[39] > 0.0  # rms_energy should be > 0
        assert vec_loud[40] > 0.0  # peak_amplitude should be > 0

    async def test_encoder_with_recorder_and_events(self) -> None:
        """StateEncoder should produce consistent vectors through the recorder."""
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)

        encoder = StateEncoder()
        encoder.update_state(RobotState.BOOT)

        recorder = ExperienceRecorder(bus=bus, encoder=encoder)
        recorder.attach()

        # Simulate events
        await sm.transition(RobotState.IDLE)
        await bus.publish(FaceDetected(x=0.5, y=0.3, confidence=0.85))
        await bus.publish(
            EmotionChanged(
                previous=EmotionName.NEUTRAL,
                current=EmotionName.CURIOUS,
                intensity=0.7,
            )
        )

        await asyncio.sleep(0.05)

        # Verify experiences were recorded with correct state size
        assert len(recorder.working_memory) > 0
        for exp in recorder.working_memory:
            assert len(exp.state) == STATE_SIZE, (
                f"State vector should be {STATE_SIZE} elements, got {len(exp.state)}"
            )

    async def test_acceptance_criteria_predictable_representation(self) -> None:
        """Given the same robot state, encoder must produce a predictable representation."""
        # Setup: identical state in two encoders
        enc1 = StateEncoder()
        enc2 = StateEncoder()

        for enc in [enc1, enc2]:
            enc.update_state(RobotState.IDLE)
            enc.update_emotion(EmotionName.HAPPY, 0.8)
            enc.update_emotion(EmotionName.CURIOUS, 0.5)
            enc.update_servo("pan", 45.0)
            enc.update_servo("tilt", 90.0)
            enc.update_vision(
                face_detected=True,
                face_x=0.3,
                face_y=0.4,
                face_confidence=0.75,
                face_size=0.12,
                face_count=1,
            )
            enc.update_idle(5.0)
            enc.push_reward(0.5)

        vec1 = enc1.encode()
        vec2 = enc2.encode()

        # Identical inputs -> identical outputs (deterministic)
        assert vec1 == vec2

        # Verify all values are finite
        import math

        for v in vec1:
            assert math.isfinite(v), f"Non-finite value: {v}"

        # Verify the vector is suitable for neural network input
        tensor = enc1.encode_tensor()
        assert tensor.shape == (STATE_SIZE,)

    async def test_missing_inputs_handled_gracefully(self) -> None:
        """Missing camera and microphone inputs should produce valid defaults."""
        encoder = StateEncoder()
        # No camera, no microphone, no face detection
        vec = encoder.encode()

        # Vision: no face detected
        assert vec[33] == 0.0  # face_detected
        assert vec[34] == 0.5  # face_x defaults to centre
        assert vec[35] == 0.5  # face_y defaults to centre

        # Audio: no input
        assert vec[39] == 0.0  # rms_energy
        assert vec[40] == 0.0  # peak_amplitude
        assert vec[41] == 0.0  # zero_crossing_rate

        # All values should be finite
        import math

        for v in vec:
            assert math.isfinite(v)
