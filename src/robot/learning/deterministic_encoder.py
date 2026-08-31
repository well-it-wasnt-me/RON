"""Deterministic multimodal encoder: same input -> identical output, every time.

The multimodal encoder is a **reproducible representation function**.
The previous :class:`MultimodalEncoder` mutated its internal history
buffer on every ``encode()`` call, which means calling ``encode()`` twice
on the same observation could produce different outputs - fatal for replay,
debugging, and evaluation.

This module provides:

* :class:`ObservationContext` - an immutable bundle of the current
  observation plus a fixed-size history tuple.  History management lives
  **outside** the encoder.
* :class:`DeterministicMultimodalEncoder` - a stateless encoder that
  takes an :class:`ObservationContext` and always produces the same
  vector.  No internal mutation.  No trainable parameters.  Vision and
  audio are deterministic feature normalisation; robot state is the
  existing :class:`StateEncoder` layout; temporal context is a fixed
  MLP over a flattened history window.

::

    Vision features (normalised)
          |
    Audio features (normalised)
          |
    Robot state (deterministic)
          |
          v
    Observation encoder (concat)
          |
          v
    Temporal encoder (MLP over history)
          |
          v
    64-128 dimensional latent state

No transformers.  No pretrained models.  No trainable modality encoders.
Make it correct before making it clever.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from robot.learning.network import MLP
from robot.learning.observation import Observation
from robot.learning.state_encoder import STATE_SIZE
from robot.learning.tensor import Tensor
from robot.logging import get_logger

_log = get_logger("learning.deterministic_encoder")

# ---------------------------------------------------------------------------
# Version and sizes
# ---------------------------------------------------------------------------

DETERMINISTIC_ENCODER_VERSION = 1

# Deterministic sub-encoder output sizes (fixed, no training)
DETERMINISTIC_VISION_OUTPUT = 6  # = VisionFeatures.to_vector() - normalised
DETERMINISTIC_AUDIO_OUTPUT = 3  # = AudioFeatures.to_vector() - normalised

# Temporal encoder: MLP over flattened history, fixed seed for determinism
TEMPORAL_HIDDEN_SIZES: list[int] = [64]
TEMPORAL_OUTPUT_SIZE = 64  # final latent dimension
TEMPORAL_SEED = 42

DEFAULT_HISTORY_LENGTH = 5


def deterministic_encoding_size(history_length: int = DEFAULT_HISTORY_LENGTH) -> int:
    """Return the total size of the deterministic encoding vector.

    Layout: ``[robot_state(91) | vision(6) | audio(3) | temporal(64)]``
    """
    return (
        STATE_SIZE + DETERMINISTIC_VISION_OUTPUT + DETERMINISTIC_AUDIO_OUTPUT + TEMPORAL_OUTPUT_SIZE
    )


# ---------------------------------------------------------------------------
# ObservationContext: immutable current + history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationContext:
    """Immutable bundle of the current observation and its history.

    Attributes
    ----------
    current:
        The most recent observation.
    history:
        A tuple of past observations, oldest first.  When shorter than
        the encoder's ``history_length``, missing entries are
        zero-padded by the encoder.
    """

    current: Observation
    history: tuple[Observation, ...] = ()

    @classmethod
    def from_observations(
        cls,
        current: Observation,
        history: list[Observation] | tuple[Observation, ...] | None = None,
    ) -> ObservationContext:
        """Create a context from a current observation and optional history."""
        return cls(current=current, history=tuple(history) if history else ())

    def with_history(
        self, history: list[Observation] | tuple[Observation, ...]
    ) -> ObservationContext:
        """Return a new context with updated history."""
        return ObservationContext(current=self.current, history=tuple(history))


# ---------------------------------------------------------------------------
# Deterministic multimodal encoder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DeterministicMultimodalEncoder:
    """Stateless, deterministic multimodal encoder.

    Given the same :class:`ObservationContext`, this encoder always
    produces the same output vector.  It never mutates internal state.

    The encoder uses:

    * **Robot state**: :class:`StateEncoder` layout (91 elements) -
      deterministic hand-crafted features.
    * **Vision**: raw :class:`VisionFeatures` vector (6 elements) -
      deterministic normalisation only.
    * **Audio**: raw :class:`AudioFeatures` vector (3 elements) -
      deterministic normalisation only.
    * **Temporal context**: a fixed-seed MLP over the flattened history
      window, producing a 64-element latent state.

    Parameters
    ----------
    history_length:
        Number of recent observations to include in the temporal
        context window.
    """

    history_length: int = DEFAULT_HISTORY_LENGTH
    _temporal_mlp: MLP = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the fixed-seed temporal MLP (not trained - deterministic init)."""
        temporal_input = STATE_SIZE * self.history_length
        self._temporal_mlp = MLP(
            input_size=temporal_input,
            hidden_sizes=TEMPORAL_HIDDEN_SIZES,
            output_size=TEMPORAL_OUTPUT_SIZE,
            activation="relu",
            output_activation="linear",
            weight_init="he",
            seed=TEMPORAL_SEED,
        )

    @property
    def output_size(self) -> int:
        """Total size of the encoding vector."""
        return deterministic_encoding_size(self.history_length)

    @property
    def version(self) -> int:
        """Encoder version (increment when layout changes incompatibly)."""
        return DETERMINISTIC_ENCODER_VERSION

    # ------------------------------------------------------------------ encode
    def encode(self, context: ObservationContext) -> list[float]:
        """Produce the deterministic encoding vector.

        The same ``context`` always produces the same vector.  This
        method does **not** mutate any internal state.

        Parameters
        ----------
        context:
            The observation context (current + history).

        Returns
        -------
        list[float]
            A flat vector of :attr:`output_size` elements.
        """
        # 1. Robot state (deterministic StateEncoder layout)
        robot_state = context.current.to_vector()

        # 2. Vision (deterministic normalisation - raw features)
        vision_vec = context.current.vision.features.to_vector()

        # 3. Audio (deterministic normalisation - raw features)
        audio_vec = context.current.audio.features.to_vector()

        # 4. Temporal context (fixed-seed MLP over flattened history)
        temporal_vec = self._encode_temporal(context)

        # Concatenate
        result = robot_state + vision_vec + audio_vec + temporal_vec

        # Sanity: no NaN or inf
        for i, v in enumerate(result):
            if math.isnan(v) or math.isinf(v):
                result[i] = 0.0

        return result

    def encode_numpy(self, context: ObservationContext) -> np.ndarray:
        """Produce the encoding vector as a numpy array."""
        return np.array(self.encode(context), dtype=np.float64)

    def _encode_temporal(self, context: ObservationContext) -> list[float]:
        """Encode the temporal history through the fixed-seed MLP.

        The history is flattened into a single vector of
        ``STATE_SIZE * history_length`` elements.  Missing entries
        (when the history is shorter than ``history_length``) are
        zero-padded at the beginning (oldest slots).
        """
        window_size = STATE_SIZE * self.history_length
        flattened = [0.0] * window_size

        # Take the most recent history_length observations
        history = list(context.history)
        if len(history) > self.history_length:
            history = history[-self.history_length :]

        # Fill from the end (most recent at the end)
        for i, obs in enumerate(history):
            offset = (self.history_length - len(history) + i) * STATE_SIZE
            vec = obs.to_vector()
            for j, v in enumerate(vec):
                if offset + j < window_size:
                    flattened[offset + j] = v

        # Run through the fixed-seed MLP (deterministic)
        x = np.array(flattened, dtype=np.float64).reshape(1, -1)
        pred = self._temporal_mlp.predict(Tensor(x))
        return [float(v) for v in pred.data.flatten().tolist()]

    # ------------------------------------------------------------------ helpers
    def validate_output(self, vec: list[float]) -> bool:
        """Check that an output vector is well-formed.

        Returns True if the vector has the correct size and contains
        no NaN/inf values.
        """
        if len(vec) != self.output_size:
            return False
        return all(not (math.isnan(v) or math.isinf(v)) for v in vec)


__all__ = [
    "DEFAULT_HISTORY_LENGTH",
    "DETERMINISTIC_AUDIO_OUTPUT",
    "DETERMINISTIC_ENCODER_VERSION",
    "DETERMINISTIC_VISION_OUTPUT",
    "TEMPORAL_OUTPUT_SIZE",
    "DeterministicMultimodalEncoder",
    "ObservationContext",
    "deterministic_encoding_size",
]
