"""Tests for the StateEncoder, VisionFeatures, and AudioFeatures."""

from __future__ import annotations

import math
import struct

import pytest

from robot.behavior.state_machine import RobotState
from robot.events.events import EmotionName
from robot.learning.state_encoder import (
    ENCODER_VERSION,
    STATE_SIZE,
    AudioFeatures,
    StateEncoder,
    VisionFeatures,
    state_layout,
    state_size,
)
from robot.learning.tensor import Tensor

# ========================================================================
# VisionFeatures
# ========================================================================


class TestVisionFeatures:
    """Tests for VisionFeatures dataclass."""

    def test_default_no_face(self) -> None:
        vf = VisionFeatures()
        assert vf.face_detected == 0.0
        assert vf.face_x == 0.5  # centre when unknown
        assert vf.face_y == 0.5
        assert vf.face_confidence == 0.0
        assert vf.face_size == 0.0
        assert vf.face_count == 0.0

    def test_from_face_event(self) -> None:
        vf = VisionFeatures.from_face_event(x=0.3, y=0.6, confidence=0.85, face_count=1)
        assert vf.face_detected == 1.0
        assert vf.face_x == 0.3
        assert vf.face_y == 0.6
        assert vf.face_confidence == 0.85
        assert vf.face_count == pytest.approx(1.0 / 3.0)

    def test_from_face_event_multiple_faces(self) -> None:
        vf = VisionFeatures.from_face_event(x=0.5, y=0.4, confidence=0.9, face_count=3)
        assert vf.face_detected == 1.0
        assert vf.face_count == pytest.approx(1.0)  # 3/3 = 1.0

    def test_from_face_event_overflow(self) -> None:
        """Face count should be capped at max_faces."""
        vf = VisionFeatures.from_face_event(x=0.5, y=0.5, confidence=0.7, face_count=10)
        assert vf.face_count == pytest.approx(1.0)  # capped at 3/3 = 1.0

    def test_no_face(self) -> None:
        vf = VisionFeatures.no_face()
        assert vf.face_detected == 0.0
        assert vf.face_x == 0.5
        assert vf.face_y == 0.5

    def test_to_vector(self) -> None:
        vf = VisionFeatures(
            face_detected=1.0,
            face_x=0.3,
            face_y=0.6,
            face_confidence=0.85,
            face_size=0.2,
            face_count=0.33,
        )
        vec = vf.to_vector()
        assert len(vec) == 6
        assert vec[0] == 1.0
        assert vec[1] == 0.3
        assert vec[2] == 0.6
        assert vec[3] == 0.85
        assert vec[4] == 0.2
        assert vec[5] == pytest.approx(0.33)

    def test_frozen(self) -> None:
        vf = VisionFeatures()
        with pytest.raises(AttributeError):
            vf.face_detected = 1.0  # type: ignore[misc]


# ========================================================================
# AudioFeatures
# ========================================================================


class TestAudioFeatures:
    """Tests for AudioFeatures extraction from PCM data."""

    def test_no_audio(self) -> None:
        af = AudioFeatures.no_audio()
        assert af.rms_energy == 0.0
        assert af.peak_amplitude == 0.0
        assert af.zero_crossing_rate == 0.0

    def test_silence(self) -> None:
        """All-zero PCM should produce near-zero features."""
        n_samples = 480  # 30ms at 16kHz
        pcm = struct.pack(f"<{n_samples}h", *([0] * n_samples))
        af = AudioFeatures.from_pcm(pcm, sample_rate=16000)
        assert af.rms_energy == 0.0
        assert af.peak_amplitude == 0.0
        assert af.zero_crossing_rate == 0.0

    def test_constant_signal(self) -> None:
        """A constant non-zero signal has zero crossings."""
        n_samples = 480
        pcm = struct.pack(f"<{n_samples}h", *([1000] * n_samples))
        af = AudioFeatures.from_pcm(pcm, sample_rate=16000)
        assert af.peak_amplitude == pytest.approx(1000 / 32767.0, rel=1e-3)
        assert af.zero_crossing_rate == 0.0  # no crossings in constant signal

    def test_alternating_signal(self) -> None:
        """An alternating signal should have high zero-crossing rate."""
        n_samples = 480
        samples = [16000 if i % 2 == 0 else -16000 for i in range(n_samples)]
        pcm = struct.pack(f"<{n_samples}h", *samples)
        af = AudioFeatures.from_pcm(pcm, sample_rate=16000)
        assert af.zero_crossing_rate > 0.4  # almost every sample crosses zero
        assert af.rms_energy > 0.0

    def test_sine_wave(self) -> None:
        """A sine wave should have mid-range energy and consistent ZCR."""
        import math

        n_samples = 1600
        samples = [int(20000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n_samples)]
        pcm = struct.pack(f"<{n_samples}h", *samples)
        af = AudioFeatures.from_pcm(pcm, sample_rate=16000)
        assert af.rms_energy > 0.0
        assert af.peak_amplitude > 0.0
        assert af.zero_crossing_rate > 0.0

    def test_empty_pcm(self) -> None:
        af = AudioFeatures.from_pcm(b"", sample_rate=16000)
        assert af.rms_energy == 0.0

    def test_short_pcm(self) -> None:
        """Single-sample PCM should produce valid features."""
        pcm = struct.pack("<1h", 16000)
        af = AudioFeatures.from_pcm(pcm, sample_rate=16000)
        assert af.peak_amplitude > 0.0

    def test_to_vector(self) -> None:
        af = AudioFeatures(rms_energy=0.3, peak_amplitude=0.7, zero_crossing_rate=0.1)
        vec = af.to_vector()
        assert len(vec) == 3
        assert vec == [0.3, 0.7, 0.1]

    def test_clipping(self) -> None:
        """Values should be clipped to [0, 1]."""
        n_samples = 480
        samples = [32767] * n_samples
        pcm = struct.pack(f"<{n_samples}h", *samples)
        af = AudioFeatures.from_pcm(pcm, sample_rate=16000)
        assert af.rms_energy <= 1.0
        assert af.peak_amplitude <= 1.0


# ========================================================================
# StateEncoder
# ========================================================================


class TestStateEncoder:
    """Tests for the unified StateEncoder."""

    def test_default_state_size(self) -> None:
        """The state vector must have a fixed, documented size."""
        enc = StateEncoder()
        vec = enc.encode()
        assert len(vec) == STATE_SIZE
        assert len(vec) == state_size()

    def test_all_finite(self) -> None:
        """No NaN or inf values in the output."""
        enc = StateEncoder()
        vec = enc.encode()
        for i, v in enumerate(vec):
            assert not math.isnan(v), f"NaN at index {i}"
            assert not math.isinf(v), f"inf at index {i}"

    def test_deterministic_encoding(self) -> None:
        """Same inputs must always produce the same output."""
        enc1 = StateEncoder()
        enc1.update_state(RobotState.CURIOUS)
        enc1.update_emotion(EmotionName.HAPPY, 0.8)

        enc2 = StateEncoder()
        enc2.update_state(RobotState.CURIOUS)
        enc2.update_emotion(EmotionName.HAPPY, 0.8)

        assert enc1.encode() == enc2.encode()

    def test_default_encoding(self) -> None:
        """Default encoder should produce valid output."""
        enc = StateEncoder()
        vec = enc.encode()
        assert len(vec) == STATE_SIZE
        # Default state is IDLE (index 1)
        assert vec[11] == 1.0  # IDLE is index 1 in RobotState enum
        # All other states should be 0
        for i in range(8):
            if i != 1:
                assert vec[10 + i] == 0.0
        # Default emotions should be 0
        for i in range(10):
            assert vec[i] == 0.0

    def test_emotion_encoding(self) -> None:
        """Emotion intensities should appear at the correct indices."""
        enc = StateEncoder()
        enc.update_emotion(EmotionName.HAPPY, 0.7)
        enc.update_emotion(EmotionName.CURIOUS, 0.5)
        vec = enc.encode()

        emotions = list(EmotionName)
        happy_idx = emotions.index(EmotionName.HAPPY)
        curious_idx = emotions.index(EmotionName.CURIOUS)

        assert vec[happy_idx] == pytest.approx(0.7)
        assert vec[curious_idx] == pytest.approx(0.5)

    def test_state_encoding(self) -> None:
        """Robot state should be one-hot encoded at the correct position."""
        enc = StateEncoder()
        for state in RobotState:
            enc.update_state(state)
            vec = enc.encode()
            states = list(RobotState)
            idx = states.index(state)
            assert vec[10 + idx] == 1.0
            for j in range(8):
                if j != idx:
                    assert vec[10 + j] == 0.0

    def test_personality_encoding(self) -> None:
        """Personality traits should appear at indices 18-22."""
        enc = StateEncoder()
        enc.personality = {
            "curiosity": 0.9,
            "energy": 0.5,
            "shyness": 0.2,
            "friendliness": 0.8,
            "playfulness": 0.6,
        }
        vec = enc.encode()
        assert vec[18] == pytest.approx(0.9)
        assert vec[19] == pytest.approx(0.5)
        assert vec[20] == pytest.approx(0.2)
        assert vec[21] == pytest.approx(0.8)
        assert vec[22] == pytest.approx(0.6)

    def test_servo_encoding(self) -> None:
        """Servo positions should be normalised to [-1, 1]."""
        enc = StateEncoder()
        enc.update_servo("pan", 0.0)  # -90° normalisation from centre
        enc.update_servo("tilt", 90.0)  # centre
        enc.update_servo("left_arm", 180.0)  # max
        enc.update_servo("right_arm", 45.0)  # -45° from centre

        vec = enc.encode()
        assert vec[23] == pytest.approx(-1.0)
        assert vec[24] == pytest.approx(0.0)
        assert vec[25] == pytest.approx(1.0)
        assert vec[26] == pytest.approx(-0.5)

    def test_vision_encoding(self) -> None:
        """Vision features should appear at indices 33-38."""
        enc = StateEncoder()
        enc.update_vision(
            face_detected=True,
            face_x=0.3,
            face_y=0.6,
            face_confidence=0.85,
            face_size=0.15,
            face_count=1,
        )
        vec = enc.encode()
        assert vec[33] == 1.0  # face_detected
        assert vec[34] == pytest.approx(0.3)  # face_x
        assert vec[35] == pytest.approx(0.6)  # face_y
        assert vec[36] == pytest.approx(0.85)  # confidence
        assert vec[37] == pytest.approx(0.15)  # size
        assert vec[38] == pytest.approx(1.0 / 3.0)  # face_count normalised

    def test_audio_encoding(self) -> None:
        """Audio features should appear at indices 39-41."""
        enc = StateEncoder()
        audio = AudioFeatures(rms_energy=0.3, peak_amplitude=0.7, zero_crossing_rate=0.1)
        enc.update_audio(audio)
        vec = enc.encode()
        assert vec[39] == pytest.approx(0.3)
        assert vec[40] == pytest.approx(0.7)
        assert vec[41] == pytest.approx(0.1)

    def test_flags_encoding(self) -> None:
        """Binary flags should appear at indices 42-45."""
        enc = StateEncoder()
        enc.update_state(RobotState.SPEAKING)
        vec = enc.encode()
        assert vec[42] == 1.0  # speaking
        assert vec[43] == 0.0  # not listening

        enc.update_state(RobotState.LISTENING)
        vec = enc.encode()
        assert vec[42] == 0.0  # not speaking
        assert vec[43] == 1.0  # listening

    def test_interaction_flag(self) -> None:
        """Interaction flag should be 1.0 when face detected or in interactive state."""
        enc = StateEncoder()
        # IDLE with no face
        vec = enc.encode()
        assert vec[44] == 0.0

        # CURIOUS state (interaction flag = 1.0)
        enc.update_state(RobotState.CURIOUS)
        vec = enc.encode()
        assert vec[44] == 1.0

        # IDLE with face detected
        enc.update_state(RobotState.IDLE)
        enc.update_vision(face_detected=True, face_x=0.5, face_y=0.5, face_count=1)
        vec = enc.encode()
        assert vec[44] == 1.0

    def test_idle_seconds_encoding(self) -> None:
        """Idle time should be normalised (60s = 1.0, capped)."""
        enc = StateEncoder()
        enc.update_idle(30.0)
        vec = enc.encode()
        assert vec[45] == pytest.approx(0.5)

        enc.update_idle(120.0)
        vec = enc.encode()
        assert vec[45] == pytest.approx(1.0)  # capped

    def test_reward_history(self) -> None:
        """Recent rewards should appear at indices 46-50."""
        enc = StateEncoder()
        enc.push_reward(0.5)
        enc.push_reward(-0.1)
        enc.push_reward(0.8)
        vec = enc.encode()
        assert vec[46] == pytest.approx(0.5)
        assert vec[47] == pytest.approx(-0.1)
        assert vec[48] == pytest.approx(0.8)
        assert vec[49] == 0.0
        assert vec[50] == 0.0

    def test_reward_history_overflow(self) -> None:
        """Reward history should be capped at 5 entries."""
        enc = StateEncoder()
        for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            enc.push_reward(r)
        vec = enc.encode()
        assert vec[46] == pytest.approx(0.2)  # oldest kept
        assert vec[50] == pytest.approx(0.6)  # newest

    def test_reset(self) -> None:
        """Reset should return encoder to default state."""
        enc = StateEncoder()
        enc.update_state(RobotState.CURIOUS)
        enc.update_emotion(EmotionName.HAPPY, 0.8)
        enc.update_servo("pan", 45.0)
        enc.update_vision(face_detected=True, face_count=1)
        enc.push_reward(0.5)
        enc.update_idle(30.0)

        enc.reset()

        vec = enc.encode()
        assert vec[11] == 1.0  # back to IDLE (index 1 in enum)
        # Emotions reset to 0
        for i in range(10):
            assert vec[i] == 0.0
        # Vision reset
        assert vec[33] == 0.0  # no face
        assert vec[34] == 0.5  # x default to centre

    def test_encode_tensor(self) -> None:
        """encode_tensor should return a Tensor with correct shape."""
        enc = StateEncoder()
        enc.update_state(RobotState.THINKING)
        tensor = enc.encode_tensor()
        assert isinstance(tensor, Tensor)
        assert tensor.shape == (STATE_SIZE,)

    def test_missing_camera(self) -> None:
        """No camera input should produce valid defaults."""
        enc = StateEncoder()
        vec = enc.encode()
        # No face detected
        assert vec[33] == 0.0
        assert vec[34] == 0.5  # default x (centre)

    def test_missing_microphone(self) -> None:
        """No microphone input should produce zero audio features."""
        enc = StateEncoder()
        vec = enc.encode()
        assert vec[39] == 0.0
        assert vec[40] == 0.0
        assert vec[41] == 0.0

    def test_layout_matches_constants(self) -> None:
        """state_layout() should return sections matching the vector."""
        layout = state_layout()
        assert layout["emotions"] == (0, 10)
        assert layout["robot_state"] == (10, 18)
        assert layout["personality"] == (18, 23)
        assert layout["servos"] == (23, 33)
        assert layout["vision"] == (33, 39)
        assert layout["audio"] == (39, 42)
        assert layout["flags"] == (42, 46)
        assert layout["rewards"] == (46, 51)
        # [51..61) is the teaching/gesture/conversation context block.
        assert layout["teaching_context"] == (51, 52)
        assert layout["interaction_active"] == (52, 53)
        assert layout["person_present"] == (53, 54)
        assert layout["gesture"] == (54, 59)
        assert layout["conversation_turn"] == (59, 60)
        assert layout["last_action_index"] == (60, 61)
        # Only [61..91) remains reserved/zero.
        assert layout["reserved"] == (61, 91)

    def test_reserved_section_is_zero(self) -> None:
        """Remaining reserved section [61..91) should be all zeros."""
        enc = StateEncoder()
        vec = enc.encode()
        for i in range(61, STATE_SIZE):
            assert vec[i] == 0.0, f"Reserved index {i} should be 0"

    def test_encoder_version(self) -> None:
        """Encoder version should be a positive integer."""
        assert ENCODER_VERSION >= 1
        assert isinstance(ENCODER_VERSION, int)


# ========================================================================
# ========================================================================


class TestStateEncoderIntegration:
    """Integration tests combining StateEncoder with experience recording."""

    def test_encoder_produces_consistent_vectors_for_training(self) -> None:
        """Multiple encodes of the same state should produce identical vectors."""
        enc = StateEncoder()
        enc.update_state(RobotState.IDLE)
        enc.update_emotion(EmotionName.NEUTRAL, 1.0)
        enc.update_vision(face_detected=True, face_x=0.4, face_y=0.6, face_count=1)

        vec1 = enc.encode()
        vec2 = enc.encode()
        assert vec1 == vec2

    def test_state_changes_produce_different_vectors(self) -> None:
        """Different states should produce different vectors."""
        enc = StateEncoder()

        enc.update_state(RobotState.IDLE)
        idle_vec = enc.encode()

        enc.update_state(RobotState.CURIOUS)
        curious_vec = enc.encode()

        # Vectors should differ
        assert idle_vec != curious_vec

    def test_encoder_output_suitable_for_neural_network(self) -> None:
        """State vectors should be suitable as neural network input."""
        enc = StateEncoder()
        enc.update_state(RobotState.IDLE)
        enc.update_emotion(EmotionName.HAPPY, 0.8)
        enc.update_vision(
            face_detected=True, face_x=0.5, face_y=0.5, face_confidence=0.9, face_count=1
        )

        vec = enc.encode()
        tensor = enc.encode_tensor()

        # Verify dimensions
        assert len(vec) == STATE_SIZE
        assert tensor.shape == (STATE_SIZE,)

        # Verify all values are finite and in reasonable range
        for v in vec:
            assert math.isfinite(v)
            assert -1.0 <= v <= 1.0 or abs(v) <= 2.0, f"Value {v} out of reasonable range"

    def test_audio_from_realistic_pcm(self) -> None:
        """Audio features from realistic PCM data should be well-formed."""
        import math

        # 30ms of 440Hz sine wave at 16kHz, 16-bit
        n_samples = 480
        samples = [int(16000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n_samples)]
        pcm = struct.pack(f"<{n_samples}h", *samples)
        af = AudioFeatures.from_pcm(pcm, sample_rate=16000)

        assert 0.0 < af.rms_energy < 1.0
        assert 0.0 < af.peak_amplitude < 1.0
        assert 0.0 < af.zero_crossing_rate < 1.0
