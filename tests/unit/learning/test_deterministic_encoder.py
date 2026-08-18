"""Tests for the deterministic multimodal encoder.

Phase 3: Make multimodal encoding deterministic.

Tests prove:
- same input => identical output
- no NaN
- no inf
- fixed output dimension
- history ordering
- empty history
- full history
- reset behavior (stateless — no reset needed)
- serialization/version compatibility
- 10,000 encodings of the same fixture are identical
"""

from __future__ import annotations

import math

import pytest

from robot.behavior.state_machine import RobotState
from robot.events.events import EmotionName
from robot.learning.deterministic_encoder import (
    DEFAULT_HISTORY_LENGTH,
    DETERMINISTIC_ENCODER_VERSION,
    DeterministicMultimodalEncoder,
    ObservationContext,
    deterministic_encoding_size,
)
from robot.learning.observation import Observation
from robot.learning.state_encoder import StateEncoder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def encoder() -> DeterministicMultimodalEncoder:
    return DeterministicMultimodalEncoder()


@pytest.fixture
def base_observation() -> Observation:
    """A frozen observation fixture for determinism testing."""
    enc = StateEncoder()
    enc.update_state(RobotState.IDLE)
    enc.update_emotion(EmotionName.HAPPY, 0.7)
    enc.update_vision(face_detected=True, face_x=0.3, face_y=0.7, face_confidence=0.9, face_count=1)
    return Observation.from_encoder(enc)


@pytest.fixture
def context(base_observation: Observation) -> ObservationContext:
    return ObservationContext(current=base_observation)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same input → identical output, every time."""

    def test_same_input_identical_output(
        self, encoder: DeterministicMultimodalEncoder, context: ObservationContext
    ) -> None:
        """Encoding the same context twice produces identical results."""
        v1 = encoder.encode(context)
        v2 = encoder.encode(context)
        assert v1 == v2

    def test_ten_thousand_encodings_identical(
        self, encoder: DeterministicMultimodalEncoder, context: ObservationContext
    ) -> None:
        """Encoding a frozen fixture 10,000 times produces exactly the same result.

        This is the Definition of Done for Phase 3.
        """
        first = encoder.encode(context)
        for _ in range(9999):
            result = encoder.encode(context)
            assert result == first, "Encoding is not deterministic"

    def test_encode_does_not_mutate_encoder(
        self, encoder: DeterministicMultimodalEncoder, context: ObservationContext
    ) -> None:
        """Calling encode() does not change the encoder's internal state."""
        v1 = encoder.encode(context)
        # Encode a different context in between
        other = ObservationContext(current=Observation())
        encoder.encode(other)
        # The original should still produce the same result
        v2 = encoder.encode(context)
        assert v1 == v2

    def test_two_encoders_same_seed_identical(self) -> None:
        """Two encoders with the same config produce identical outputs."""
        """Different history lengths produce the same output size (temporal MLP compresses)."""
        enc1 = DeterministicMultimodalEncoder(history_length=3)
        enc2 = DeterministicMultimodalEncoder(history_length=3)
        obs = Observation()
        ctx = ObservationContext(current=obs)
        assert enc1.encode(ctx) == enc2.encode(ctx)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


class TestOutputValidation:
    """No NaN, no inf, fixed dimension."""

    def test_no_nan(
        self, encoder: DeterministicMultimodalEncoder, context: ObservationContext
    ) -> None:
        """Output contains no NaN values."""
        vec = encoder.encode(context)
        for v in vec:
            assert not math.isnan(v), "NaN found in output"

    def test_no_inf(
        self, encoder: DeterministicMultimodalEncoder, context: ObservationContext
    ) -> None:
        """Output contains no inf values."""
        vec = encoder.encode(context)
        for v in vec:
            assert not math.isinf(v), "Inf found in output"

    def test_fixed_output_dimension(
        self, encoder: DeterministicMultimodalEncoder, context: ObservationContext
    ) -> None:
        """Output has a fixed, correct dimension."""
        vec = encoder.encode(context)
        assert len(vec) == encoder.output_size
        assert len(vec) == deterministic_encoding_size()

    def test_validate_output(
        self, encoder: DeterministicMultimodalEncoder, context: ObservationContext
    ) -> None:
        """validate_output returns True for a well-formed vector."""
        vec = encoder.encode(context)
        assert encoder.validate_output(vec) is True

    def test_validate_output_rejects_nan(self, encoder: DeterministicMultimodalEncoder) -> None:
        """validate_output returns False for a vector with NaN."""
        vec = [1.0] * encoder.output_size
        vec[5] = float("nan")
        assert encoder.validate_output(vec) is False

    def test_validate_output_rejects_wrong_size(
        self, encoder: DeterministicMultimodalEncoder
    ) -> None:
        """validate_output returns False for a wrong-sized vector."""
        vec = [1.0] * 10  # wrong size
        assert encoder.validate_output(vec) is False


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestHistory:
    """History ordering, empty, full."""

    def test_empty_history(
        self, encoder: DeterministicMultimodalEncoder, base_observation: Observation
    ) -> None:
        """Empty history is handled (zero-padded)."""
        ctx = ObservationContext(current=base_observation, history=())
        vec = encoder.encode(ctx)
        assert len(vec) == encoder.output_size
        assert encoder.validate_output(vec)

    def test_full_history(
        self, encoder: DeterministicMultimodalEncoder, base_observation: Observation
    ) -> None:
        """Full history produces a valid output."""
        history = [Observation() for _ in range(DEFAULT_HISTORY_LENGTH)]
        ctx = ObservationContext(current=base_observation, history=tuple(history))
        vec = encoder.encode(ctx)
        assert encoder.validate_output(vec)

    def test_history_longer_than_max(
        self, encoder: DeterministicMultimodalEncoder, base_observation: Observation
    ) -> None:
        """History longer than history_length is truncated (most recent kept)."""
        history = [Observation() for _ in range(DEFAULT_HISTORY_LENGTH + 5)]
        ctx = ObservationContext(current=base_observation, history=tuple(history))
        vec = encoder.encode(ctx)
        assert encoder.validate_output(vec)

    def test_history_ordering_matters(
        self, encoder: DeterministicMultimodalEncoder, base_observation: Observation
    ) -> None:
        """Different history orderings produce different outputs."""
        obs_a = Observation()
        obs_b = Observation()
        # Different robot states to make them distinguishable
        enc_a = StateEncoder()
        enc_a.update_state(RobotState.IDLE)
        obs_a = Observation.from_encoder(enc_a)
        enc_b = StateEncoder()
        enc_b.update_state(RobotState.CURIOUS)
        obs_b = Observation.from_encoder(enc_b)

        ctx1 = ObservationContext(current=base_observation, history=(obs_a, obs_b))
        ctx2 = ObservationContext(current=base_observation, history=(obs_b, obs_a))
        v1 = encoder.encode(ctx1)
        v2 = encoder.encode(ctx2)
        assert v1 != v2, "Different history orderings should produce different outputs"

    def test_same_history_same_output(
        self, encoder: DeterministicMultimodalEncoder, base_observation: Observation
    ) -> None:
        """Same history content in the same order produces the same output."""
        history = tuple(Observation() for _ in range(3))
        ctx1 = ObservationContext(current=base_observation, history=history)
        ctx2 = ObservationContext(current=base_observation, history=history)
        assert encoder.encode(ctx1) == encoder.encode(ctx2)


# ---------------------------------------------------------------------------
# Statelessness / reset
# ---------------------------------------------------------------------------


class TestStatelessness:
    """The encoder is stateless — no reset needed."""

    def test_no_reset_needed(
        self, encoder: DeterministicMultimodalEncoder, context: ObservationContext
    ) -> None:
        """Encoding the same context after other encodings produces the same result."""
        v1 = encoder.encode(context)
        # Do many other encodings
        for _ in range(100):
            other_ctx = ObservationContext(current=Observation())
            encoder.encode(other_ctx)
        v2 = encoder.encode(context)
        assert v1 == v2


# ---------------------------------------------------------------------------
# Version and serialization
# ---------------------------------------------------------------------------


class TestVersion:
    """Version compatibility."""

    def test_version_is_positive(self, encoder: DeterministicMultimodalEncoder) -> None:
        """The encoder has a version number."""
        assert encoder.version > 0
        assert encoder.version == DETERMINISTIC_ENCODER_VERSION

    def test_same_version_same_layout(self) -> None:
        """Encoders with the same version produce the same output size."""
        enc1 = DeterministicMultimodalEncoder(history_length=5)
        enc2 = DeterministicMultimodalEncoder(history_length=5)
        assert enc1.version == enc2.version
        assert enc1.output_size == enc2.output_size

    def test_different_history_length_same_output_size(self) -> None:
        """Different history lengths produce different output sizes."""
        """Different history lengths produce the same output size (temporal MLP compresses)."""
        enc1 = DeterministicMultimodalEncoder(history_length=3)
        enc2 = DeterministicMultimodalEncoder(history_length=5)
        assert enc1.output_size == enc2.output_size  # temporal MLP output is fixed


# ---------------------------------------------------------------------------
# Integration with Observation
# ---------------------------------------------------------------------------


class TestIntegrationWithObservation:
    """The encoder works with Observation objects from the Phase 2 types."""

    def test_encode_from_state_encoder(self, encoder: DeterministicMultimodalEncoder) -> None:
        """Encoding an observation captured from a StateEncoder works."""
        enc = StateEncoder()
        enc.update_state(RobotState.SPEAKING)
        enc.update_emotion(EmotionName.EXCITED, 0.9)
        obs = Observation.from_encoder(enc)
        ctx = ObservationContext(current=obs)
        vec = encoder.encode(ctx)
        assert encoder.validate_output(vec)

    def test_context_with_history_from_observations(
        self, encoder: DeterministicMultimodalEncoder
    ) -> None:
        """A context with history built from observations encodes correctly."""
        enc = StateEncoder()
        observations = []
        for i in range(5):
            enc.update_state(RobotState.IDLE)
            enc.update_emotion(EmotionName.HAPPY, float(i) * 0.2)
            observations.append(Observation.from_encoder(enc))
        ctx = ObservationContext(current=observations[-1], history=tuple(observations[:-1]))
        vec = encoder.encode(ctx)
        assert encoder.validate_output(vec)
