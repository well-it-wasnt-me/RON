"""Multimodal learning: unified representation from vision, audio, and robot state.

.. note::
    This module is **experimental / future-phase** code. It is not wired
    into the production :class:`LearningService` and is only exercised by
    unit tests. Do not rely on it for production behaviour. It is tested in
    isolation but is not yet wired into :class:`LearningService`. The
    production learning path currently uses :class:`StateEncoder` and
    :class:`WorldModel` directly. Integration is planned for a future
    phase once the base world model demonstrates reliable improvement.

This module extends the Phase 3 state encoder with **trainable** sub-encoders
for each sensory channel and a **temporal history window** that provides
recent context.  The result is a single fixed-size vector that combines:

* **Vision** - face detection features processed through a lightweight MLP.
* **Audio** - audio signal features processed through a lightweight MLP.
* **Robot state** - emotions, behaviour state, servo positions, personality,
  and flags (kept as deterministic hand-crafted features).
* **Recent history** - a window of the last N state vectors, flattened
  into the final representation.

No pretrained encoders or external models are used.  All sub-networks
are small MLPs built on the Phase 1 neural network core.

Architecture
------------

::

    camera/vision ──► VisionEncoder ──┐
    audio ──────────► AudioEncoder ────┤
    robot state ──────────────────────►├──► concat ──► history ──► output
    recent rewards ───────────────────►┘

The ``MultimodalEncoder`` produces a flat vector of
``MULTIMODAL_SIZE`` elements.  The ``MultimodalWorldModel`` uses this
representation for prediction, allowing the world model to learn
cross-modal relationships (e.g. "a loud sound + no face -> user is
away, not just sleeping").
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robot.learning.network import MLP
from robot.learning.optimizers import Adam
from robot.learning.state_encoder import (
    STATE_SIZE,
    AudioFeatures,
    StateEncoder,
    VisionFeatures,
)
from robot.learning.tensor import Tensor
from robot.logging import get_logger

_log = get_logger("learning.multimodal")

# ---------------------------------------------------------------------------
# Version and sizes
# ---------------------------------------------------------------------------

MULTIMODAL_VERSION = 2  # Bump when layout changes incompatibly

# Sub-encoder output sizes (hyperparameters, can be tuned)
VISION_ENCODER_INPUT = 6  # matches VisionFeatures.to_vector()
VISION_ENCODER_OUTPUT = 16

AUDIO_ENCODER_INPUT = 3  # matches AudioFeatures.to_vector()
AUDIO_ENCODER_OUTPUT = 8

# History window
DEFAULT_HISTORY_LENGTH = 5  # Number of recent state snapshots to include

# Final multimodal vector:
#   = 91 + 16 + 8 + 455 = 570  (with default history_length=5)
# Without history: 91 + 16 + 8 = 115
MULTIMODAL_BASE_SIZE = STATE_SIZE + VISION_ENCODER_OUTPUT + AUDIO_ENCODER_OUTPUT


def multimodal_size(history_length: int = DEFAULT_HISTORY_LENGTH) -> int:
    """Return the total size of the multimodal representation vector.

    Parameters
    ----------
    history_length:
        Number of recent state snapshots to include in the history.
    """
    return MULTIMODAL_BASE_SIZE + STATE_SIZE * history_length


# ---------------------------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VisionEncoder:
    """Trainable encoder for vision (face detection) features.

    Takes the 6-element vision feature vector from :class:`VisionFeatures`
    and produces a ``VISION_ENCODER_OUTPUT``-dimensional representation.

    Parameters
    ----------
    hidden_sizes:
        Hidden layer sizes for the MLP.
    learning_rate:
        Learning rate for the Adam optimiser.
    seed:
        Random seed for reproducibility.
    """

    hidden_sizes: list[int] = field(default_factory=lambda: [32, 16])
    learning_rate: float = 0.001
    seed: int = 42
    _model: MLP | None = field(default=None, init=False, repr=False)
    _optimizer: Adam | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_model()

    def _build_model(self) -> None:
        self._model = MLP(
            input_size=VISION_ENCODER_INPUT,
            hidden_sizes=self.hidden_sizes,
            output_size=VISION_ENCODER_OUTPUT,
            activation="relu",
            output_activation="linear",
            weight_init="he",
            seed=self.seed,
        )
        self._optimizer = Adam(learning_rate=self.learning_rate)

    @property
    def model(self) -> MLP:
        assert self._model is not None
        return self._model

    def encode(self, vision: VisionFeatures) -> np.ndarray:
        """Encode vision features into a representation vector."""
        x = np.array(vision.to_vector(), dtype=np.float64).reshape(1, -1)
        pred = self.model.predict(Tensor(x))
        return pred.data.flatten()

    def encode_batch(self, vision_batch: list[VisionFeatures]) -> np.ndarray:
        """Encode a batch of vision features.

        Returns an array of shape ``(batch, VISION_ENCODER_OUTPUT)``.
        """
        if not vision_batch:
            return np.empty((0, VISION_ENCODER_OUTPUT), dtype=np.float64)
        x = np.array([v.to_vector() for v in vision_batch], dtype=np.float64)
        pred = self.model.predict(Tensor(x))
        return pred.data

    def train_step(self, vision: np.ndarray, target: np.ndarray) -> float:
        """Run one training step.

        Parameters
        ----------
        vision:
            Input vision features, shape ``(batch, VISION_ENCODER_INPUT)``.
        target:
            Target representation, shape ``(batch, VISION_ENCODER_OUTPUT)``.
        """
        loss, _ = self.model.network.train_step(
            Tensor(vision), Tensor(target), loss_fn="mse", optimizer=self._optimizer
        )
        return loss

    def save(self, path: str | Path) -> None:
        self.model.save(path)

    def load(self, path: str | Path) -> None:
        self._model = MLP.load(path)
        self._optimizer = Adam(learning_rate=self.learning_rate)

    def param_count(self) -> int:
        return self.model.network.param_count()


# ---------------------------------------------------------------------------
# Audio encoder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AudioEncoder:
    """Trainable encoder for audio features.

    Takes the 3-element audio feature vector from :class:`AudioFeatures`
    and produces an ``AUDIO_ENCODER_OUTPUT``-dimensional representation.

    Parameters
    ----------
    hidden_sizes:
        Hidden layer sizes for the MLP.
    learning_rate:
        Learning rate for the Adam optimiser.
    seed:
        Random seed for reproducibility.
    """

    hidden_sizes: list[int] = field(default_factory=lambda: [16, 8])
    learning_rate: float = 0.001
    seed: int = 42
    _model: MLP | None = field(default=None, init=False, repr=False)
    _optimizer: Adam | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_model()

    def _build_model(self) -> None:
        self._model = MLP(
            input_size=AUDIO_ENCODER_INPUT,
            hidden_sizes=self.hidden_sizes,
            output_size=AUDIO_ENCODER_OUTPUT,
            activation="relu",
            output_activation="linear",
            weight_init="he",
            seed=self.seed,
        )
        self._optimizer = Adam(learning_rate=self.learning_rate)

    @property
    def model(self) -> MLP:
        assert self._model is not None
        return self._model

    def encode(self, audio: AudioFeatures) -> np.ndarray:
        """Encode audio features into a representation vector."""
        x = np.array(audio.to_vector(), dtype=np.float64).reshape(1, -1)
        pred = self.model.predict(Tensor(x))
        return pred.data.flatten()

    def encode_batch(self, audio_batch: list[AudioFeatures]) -> np.ndarray:
        """Encode a batch of audio features.

        Returns an array of shape ``(batch, AUDIO_ENCODER_OUTPUT)``.
        """
        if not audio_batch:
            return np.empty((0, AUDIO_ENCODER_OUTPUT), dtype=np.float64)
        x = np.array([a.to_vector() for a in audio_batch], dtype=np.float64)
        pred = self.model.predict(Tensor(x))
        return pred.data

    def train_step(self, audio: np.ndarray, target: np.ndarray) -> float:
        """Run one training step.

        Parameters
        ----------
        audio:
            Input audio features, shape ``(batch, AUDIO_ENCODER_INPUT)``.
        target:
            Target representation, shape ``(batch, AUDIO_ENCODER_OUTPUT)``.
        """
        loss, _ = self.model.network.train_step(
            Tensor(audio), Tensor(target), loss_fn="mse", optimizer=self._optimizer
        )
        return loss

    def save(self, path: str | Path) -> None:
        self.model.save(path)

    def load(self, path: str | Path) -> None:
        self._model = MLP.load(path)
        self._optimizer = Adam(learning_rate=self.learning_rate)

    def param_count(self) -> int:
        return self.model.network.param_count()


# ---------------------------------------------------------------------------
# History buffer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HistoryBuffer:
    """Fixed-size ring buffer of recent state vectors.

    Provides temporal context by storing the last ``history_length``
    state snapshots.  When the buffer is not yet full, missing entries
    are filled with zeros.

    Parameters
    ----------
    state_size:
        Dimension of each state vector.
    history_length:
        Number of recent states to keep.
    """

    state_size: int = STATE_SIZE
    history_length: int = DEFAULT_HISTORY_LENGTH
    _buffer: deque[list[float]] = field(default_factory=deque, init=False, repr=False)

    def push(self, state: list[float]) -> None:
        """Add a state to the history, evicting the oldest if at capacity."""
        assert len(state) == self.state_size, (
            f"state length {len(state)} does not match state_size {self.state_size}"
        )
        self._buffer.append(state)
        while len(self._buffer) > self.history_length:
            self._buffer.popleft()

    def encode(self) -> list[float]:
        """Flatten the history into a single vector.

        Returns a list of ``state_size * history_length`` floats.
        Missing entries (when buffer is not yet full) are filled with
        zeros.
        """
        result = [0.0] * (self.state_size * self.history_length)
        entries = list(self._buffer)
        for i, state in enumerate(entries):
            # Most recent entries go at the end
            offset = (self.history_length - len(entries) + i) * self.state_size
            for j, v in enumerate(state):
                result[offset + j] = v
        return result

    def encode_numpy(self) -> np.ndarray:
        """Flatten the history into a numpy array."""
        return np.array(self.encode(), dtype=np.float64)

    def clear(self) -> None:
        """Clear the history buffer."""
        self._buffer.clear()

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.history_length

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Multimodal encoder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MultimodalEncoder:
    """Unified encoder that combines vision, audio, robot state, and history.

    The encoder produces a fixed-size vector of ``multimodal_size()``
    elements containing:

    1.  **Robot state** (``STATE_SIZE`` elements) - the deterministic
        hand-crafted features from :class:`StateEncoder`.
    2.  **Vision encoded** (``VISION_ENCODER_OUTPUT`` elements) - the
        trainable vision sub-encoder output.
    3.  **Audio encoded** (``AUDIO_ENCODER_OUTPUT`` elements) - the
        trainable audio sub-encoder output.
    4.  **History** (``STATE_SIZE * history_length`` elements) - recent
        state snapshots for temporal context.

    Parameters
    ----------
    vision_encoder:
        Trainable vision sub-encoder. Created with defaults if None.
    audio_encoder:
        Trainable audio sub-encoder. Created with defaults if None.
    state_encoder:
        Deterministic robot state encoder. Created with defaults if None.
    history_length:
        Number of recent state snapshots to include.
    """

    vision_encoder: VisionEncoder = field(default_factory=VisionEncoder)
    audio_encoder: AudioEncoder = field(default_factory=AudioEncoder)
    state_encoder: StateEncoder = field(default_factory=StateEncoder)
    history_length: int = DEFAULT_HISTORY_LENGTH
    _history: HistoryBuffer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._history = HistoryBuffer(
            state_size=STATE_SIZE,
            history_length=self.history_length,
        )

    @property
    def output_size(self) -> int:
        """Total size of the multimodal representation vector."""
        return multimodal_size(self.history_length)

    def encode(self) -> list[float]:
        """Produce the full multimodal representation vector.

        The layout is:

        ``[robot_state | vision_encoded | audio_encoded | history]``

        where ``|`` denotes concatenation.
        """
        # 1. Robot state (deterministic)
        robot_state = self.state_encoder.encode()

        # 2. Vision (trainable)
        vision_vec = self.vision_encoder.encode(self.state_encoder.vision)

        # 3. Audio (trainable)
        audio_vec = self.audio_encoder.encode(self.state_encoder.audio)

        # 4. Push the current state into history and encode it
        self._history.push(robot_state)
        history_vec = self._history.encode()

        # Concatenate all parts
        result = (
            robot_state
            + [float(x) for x in vision_vec.tolist()]
            + [float(x) for x in audio_vec.tolist()]
            + history_vec
        )

        # Sanity: no NaN or inf
        for i, v in enumerate(result):
            if np.isnan(v) or np.isinf(v):
                result[i] = 0.0

        return result

    def encode_numpy(self) -> np.ndarray:
        """Produce the multimodal vector as a numpy array."""
        return np.array(self.encode(), dtype=np.float64)

    def encode_tensor(self) -> Tensor:
        """Produce the multimodal vector as a :class:`Tensor`."""
        return Tensor(self.encode())

    def encode_unimodal_vision(self) -> list[float]:
        """Encode only vision + robot state (no audio, no history).

        Used for ablation studies comparing unimodal vs multimodal.
        """
        robot_state = self.state_encoder.encode()
        vision_vec = self.vision_encoder.encode(self.state_encoder.vision)
        result = robot_state + [float(x) for x in vision_vec.tolist()]
        for i, v in enumerate(result):
            if np.isnan(v) or np.isinf(v):
                result[i] = 0.0
        return result

    def encode_unimodal_audio(self) -> list[float]:
        """Encode only audio + robot state (no vision, no history).

        Used for ablation studies comparing unimodal vs multimodal.
        """
        robot_state = self.state_encoder.encode()
        audio_vec = self.audio_encoder.encode(self.state_encoder.audio)
        result = robot_state + [float(x) for x in audio_vec.tolist()]
        for i, v in enumerate(result):
            if np.isnan(v) or np.isinf(v):
                result[i] = 0.0
        return result

    def encode_no_history(self) -> list[float]:
        """Encode all modalities but without temporal history.

        Used for ablation studies comparing with vs without history.
        """
        robot_state = self.state_encoder.encode()
        vision_vec = self.vision_encoder.encode(self.state_encoder.vision)
        audio_vec = self.audio_encoder.encode(self.state_encoder.audio)
        result = (
            robot_state
            + [float(x) for x in vision_vec.tolist()]
            + [float(x) for x in audio_vec.tolist()]
        )
        for i, v in enumerate(result):
            if np.isnan(v) or np.isinf(v):
                result[i] = 0.0
        return result

    def push_state_to_history(self, state: list[float] | None = None) -> None:
        """Manually push a state vector into the history buffer.

        If ``state`` is None, the current encoder state is used.
        """
        if state is None:
            state = self.state_encoder.encode()
        self._history.push(state)

    def clear_history(self) -> None:
        """Clear the history buffer."""
        self._history.clear()

    def reset(self) -> None:
        """Reset all encoder state and clear history."""
        self.state_encoder.reset()
        self._history.clear()


# ---------------------------------------------------------------------------
# Multimodal simulation environment
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MultimodalEnvironment:
    """A simulation environment where vision, audio, and state all matter.

    The environment simulates a desk scene where:

    - **Vision** (face detection) affects the optimal action: if a face
      is present, the robot should look at it; if not, it should listen
      for audio cues.
    - **Audio** (sound energy) affects the optimal action: loud sounds
      suggest the user is present and speaking; silence suggests the
      user is away.
    - **Robot state** (idle vs. curious) interacts with perception:
      a curious robot explores more; an idle robot conserves energy.
    - **Both together** are more informative than either alone: a face
      AND a loud sound means "interact"; no face AND no sound means
      "sleep".

    The state vector layout matches the :class:`StateEncoder` layout
    so the world model can use it directly.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    noise_std:
        Standard deviation of Gaussian noise added to transitions.
    """

    seed: int = 42
    noise_std: float = 0.01
    _rng: np.random.Generator = field(init=False, repr=False)
    _step_count: int = field(default=0, init=False)
    _face_x: float = field(default=0.5, init=False)
    _face_y: float = field(default=0.5, init=False)
    _face_detected: float = field(default=1.0, init=False)
    _face_confidence: float = field(default=0.9, init=False)
    _audio_energy: float = field(default=0.3, init=False)
    _idle_time: float = field(default=0.0, init=False)
    _interaction: float = field(default=1.0, init=False)

    # Action indices (matching deskbot_action_space)
    _ACTION_LOOK_CENTER = 2
    _ACTION_CELEBRATE = 7
    _ACTION_SLEEP = 8
    _ACTION_LOOK_AROUND = 9

    @property
    def action_names(self) -> list[str]:
        return [
            "look_left",
            "look_right",
            "look_center",
            "look_up",
            "look_down",
            "blink",
            "wink",
            "celebrate",
            "sleep",
            "look_around",
        ]

    @property
    def action_size(self) -> int:
        return 10

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state."""
        self._face_x = 0.5
        self._face_y = 0.5
        self._face_detected = 1.0
        self._face_confidence = 0.9
        self._audio_energy = 0.3
        self._idle_time = 0.0
        self._interaction = 1.0
        self._step_count = 0
        return self.state

    @property
    def state(self) -> np.ndarray:
        """Return the current state as a full STATE_SIZE vector.

        Layout matches the :class:`StateEncoder` layout:
        [emotions(10), state(8), personality(5), servos(10),
         vision(6), audio(3), flags(4), rewards(5), reserved(40)]

        .. warning::
            The index constants below (33, 39, 42, etc.) must mirror the
            offsets in :class:`StateEncoder`. If the encoder layout changes,
            these must be updated. Import the shared constants when available.
        """
        vec = np.zeros(STATE_SIZE, dtype=np.float64)

        # Emotions: neutral=1.0 at index 0
        vec[0] = 1.0  # neutral

        # Robot state: IDLE=0 at index 10 -> one-hot at index 10
        vec[10] = 1.0

        # Personality defaults
        vec[18] = 0.7  # curiosity
        vec[19] = 0.6  # energy
        vec[20] = 0.3  # shyness
        vec[21] = 0.8  # friendliness
        vec[22] = 0.7  # playfulness

        # Servos at centre
        vec[23] = 0.0  # pan
        vec[24] = 0.0  # tilt

        # Vision
        vec[33] = self._face_detected
        vec[34] = self._face_x
        vec[35] = self._face_y
        vec[36] = self._face_confidence
        vec[37] = 0.0  # face size (not simulated)
        vec[38] = self._face_detected  # face count normalised

        # Audio
        vec[39] = self._audio_energy
        vec[40] = self._audio_energy * 0.8  # peak correlated with energy
        vec[41] = 0.3  # zero-crossing rate

        # Flags
        vec[42] = 0.0  # speaking
        vec[43] = 0.0  # listening
        vec[44] = self._interaction
        vec[45] = self._idle_time

        return vec

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool]:  # noqa: PLR0912
        """Take an action and return (next_state, reward, done).

        Reward structure emphasises multimodal information:
        - **celebrate** when face detected AND audio high: +1.5 (best)
        - **celebrate** when face detected only: +1.0
        - **celebrate** when audio high only: +0.5
        - **celebrate** when neither: -0.5 (bad)
        - **look_center** when face detected: +0.5
        - **look_around** when no face: +0.3 (exploration)
        - **sleep** when idle and no stimuli: +0.2 (energy saving)
        - **sleep** when face or audio present: -1.0 (missing interaction)
        """
        noise = self._rng.normal(0, self.noise_std, 4)
        face = self._face_detected > 0.5
        audio = self._audio_energy > 0.3
        reward = 0.0

        action_name = (
            self.action_names[action_index] if action_index < len(self.action_names) else "blink"
        )

        if action_name == "look_left":
            self._face_x = max(0.0, self._face_x - 0.1 + noise[0])
            reward = 0.1 if face else -0.05
        elif action_name == "look_right":
            self._face_x = min(1.0, self._face_x + 0.1 + noise[0])
            reward = 0.1 if face else -0.05
        elif action_name == "look_center":
            self._face_x = 0.5 + noise[0]
            self._face_y = 0.5 + noise[1]
            reward = 0.5 if face else -0.05
        elif action_name == "look_up":
            self._face_y = max(0.0, self._face_y - 0.1 + noise[1])
            reward = 0.05
        elif action_name == "look_down":
            self._face_y = min(1.0, self._face_y + 0.1 + noise[1])
            reward = 0.05
        elif action_name in {"blink", "wink"}:
            reward = 0.0
        elif action_name == "celebrate":
            if face and audio:
                reward = 1.5  # Best: both modalities confirm interaction
            elif face:
                reward = 1.0
            elif audio:
                reward = 0.5
            else:
                reward = -0.5
        elif action_name == "sleep":
            reward = 0.2 if not face and not audio else -1.0  # save energy vs miss interaction
            self._face_detected = 0.0
            self._audio_energy *= 0.5
        elif action_name == "look_around":
            reward = 0.3 if not face else 0.1
            if not face and self._rng.random() < 0.3:
                self._face_detected = 1.0
                self._face_x = 0.5
                self._face_confidence = 0.6

        # Dynamics: face and audio change over time
        # Recompute face/audio from the (possibly mutated) fields so the
        # dynamics reflect the post-action state.
        face = self._face_detected > 0.5
        audio = self._audio_energy > 0.3
        if face:
            # Face is present — reset idle time (user is interacting).
            self._idle_time = 0.0
            # Face might drift slightly
            self._face_x += noise[2] * 0.05
            self._face_x = max(0.0, min(1.0, self._face_x))
            self._face_y += noise[3] * 0.05
            self._face_y = max(0.0, min(1.0, self._face_y))
            # Audio fluctuates
            self._audio_energy = max(0.0, min(1.0, self._audio_energy + noise[0] * 0.1))
            self._interaction = 1.0
        else:
            # No face: audio might increase (someone speaking off-screen)
            self._audio_energy = max(0.0, min(1.0, self._audio_energy + self._rng.normal(0, 0.1)))
            # Face might reappear
            if self._rng.random() < 0.05:
                self._face_detected = 1.0
                self._face_x = 0.5
                self._face_confidence = 0.7

        self._face_detected = max(0.0, min(1.0, self._face_detected))
        self._audio_energy = max(0.0, min(1.0, self._audio_energy))
        self._idle_time = min(1.0, self._idle_time + 0.01)

        self._step_count += 1
        done = self._step_count >= 200

        return self.state.copy(), reward, done

    def action_onehot(self, action_index: int) -> np.ndarray:
        """Convert an action index to a one-hot vector."""
        vec = np.zeros(self.action_size, dtype=np.float64)
        if 0 <= action_index < self.action_size:
            vec[action_index] = 1.0
        return vec

    def scenario_vision_matters(self) -> np.ndarray:
        """Return a state where vision is informative and audio is not."""
        state = self.state.copy()
        state[33] = 1.0  # face detected
        state[34] = 0.3  # face x
        state[35] = 0.7  # face y
        state[36] = 0.9  # confidence
        state[38] = 1.0  # face count
        state[39] = 0.05  # low audio energy
        state[40] = 0.03  # low peak
        state[41] = 0.1  # low ZCR
        state[44] = 1.0  # interaction
        return state

    def scenario_audio_matters(self) -> np.ndarray:
        """Return a state where audio is informative and vision is not."""
        state = self.state.copy()
        state[33] = 0.0  # no face detected
        state[34] = 0.5  # default x
        state[35] = 0.5  # default y
        state[36] = 0.0  # no confidence
        state[38] = 0.0  # no face count
        state[39] = 0.7  # high audio energy
        state[40] = 0.6  # high peak
        state[41] = 0.4  # higher ZCR (speech-like)
        state[44] = 1.0  # interaction
        return state

    def scenario_both_matter(self) -> np.ndarray:
        """Return a state where both vision and audio are informative."""
        state = self.state.copy()
        state[33] = 1.0  # face detected
        state[34] = 0.5  # face centre
        state[35] = 0.4  # face slightly above centre
        state[36] = 0.85  # confidence
        state[38] = 1.0  # one face
        state[39] = 0.6  # moderate audio energy
        state[40] = 0.5  # moderate peak
        state[41] = 0.35  # moderate ZCR
        state[44] = 1.0  # interaction
        return state


__all__ = [
    "AUDIO_ENCODER_INPUT",
    "AUDIO_ENCODER_OUTPUT",
    "DEFAULT_HISTORY_LENGTH",
    "MULTIMODAL_BASE_SIZE",
    "MULTIMODAL_VERSION",
    "VISION_ENCODER_INPUT",
    "VISION_ENCODER_OUTPUT",
    "AudioEncoder",
    "HistoryBuffer",
    "MultimodalEncoder",
    "MultimodalEnvironment",
    "VisionEncoder",
    "multimodal_size",
]
