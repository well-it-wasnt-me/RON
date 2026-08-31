"""State encoder: converts DeskBot inputs into fixed-size numerical vectors.

The :class:`StateEncoder` is the bridge between DeskBot's perception and
state systems and the neural-network learning infrastructure.  It produces
a deterministic, fixed-size float vector from:

* **Vision** - face detection results (position, confidence, count, size).
* **Audio** - simple signal features (amplitude, energy, zero-crossing rate).
* **Robot state** - current emotion, behaviour state, servo positions,
  personality traits, interaction signals.

No pretrained models are used.  All features are computed from raw
perception data and internal state that already exists in DeskBot.
Missing inputs (no camera, no microphone) are handled gracefully by
filling with defaults.

Vector layout (v0 - 91 elements total):

    [0..9]   Emotion intensities (one per EmotionName, order matches enum)
    [10..17] Robot state one-hot (one per RobotState, order matches enum)
    [18..22] Personality traits (curiosity, energy, shyness, friendliness, playfulness)
    [23..32] Servo positions (10 slots: pan, tilt, left_arm, right_arm, + 6 reserved)
    [33]     Face detected (0.0 or 1.0)
    [34]     Face X position (normalised 0..1, 0.5 if no face)
    [35]     Face Y position (normalised 0..1, 0.5 if no face)
    [36]     Face confidence (0..1, 0 if no face)
    [37]     Face size (0..1, 0 if no face)
    [38]     Face count in scene (0..3, normalised by max_faces)
    [39]     Audio RMS energy (0..1, 0 if no audio)
    [40]     Audio peak amplitude (0..1, 0 if no audio)
    [41]     Audio zero-crossing rate (0..1, 0 if no audio)
    [42]     Speaking state (0.0 or 1.0)
    [43]     Listening state (0.0 or 1.0)
    [44]     Interaction flag (1.0 if face detected OR speech active)
    [45]     Idle seconds (normalised by 60.0, capped at 1.0)
    [46..50] Recent reward history (5 most recent reward values)
    [51]     Teaching context (0/1 - teaching mode active)
    [52]     Interaction active (0/1 - inside a teaching interaction/episode)
    [53]     Person present (0/1 - a face is present)
    [54..58] Gesture one-hot (none/wave/point/open_hand/other)
    [59]     Conversation turn count (normalised, capped at 10)
    [60]     Last action index (normalised by action_space_size)
    [61..90] Reserved for future features (zeros)

The encoder is **deterministic**: given the same inputs, it always
produces the same output.  This is critical for training stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from robot.behavior.state_machine import RobotState
from robot.events.events import EmotionName
from robot.learning.tensor import Tensor
from robot.logging import get_logger

_log = get_logger("learning.state_encoder")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Version of the encoding layout.  Increment this when the vector
# layout changes incompatibly. v2: the former zero-filled reserved block
# [51..64) now carries teaching/gesture/conversation context. STATE_SIZE
# and the 570-dim multimodal vector are unchanged (the slots merely carry
# meaning instead of zeros).
ENCODER_VERSION = 2

# Total size of the state vector.
STATE_SIZE = 91

# Indices for key sections of the vector.
_EMOTION_START = 0
_EMOTION_END = 10  # 10 emotions
_STATE_START = 10
_STATE_END = 18  # 8 states
_PERSONALITY_START = 18
_PERSONALITY_END = 23  # 5 traits
_SERVO_START = 23
_SERVO_END = 33  # 10 slots (4 used + 6 reserved)
_FACE_START = 33
_FACE_END = 39  # 6 face features
_AUDIO_START = 39
_AUDIO_END = 42  # 3 audio features
_FLAGS_START = 42
_FLAGS_END = 46  # 4 flags (speaking, listening, interaction, idle_time)
_REWARD_START = 46
_REWARD_END = 51  # 5 recent rewards

# Teaching / conversation / gesture context (carved out of the former
# 40-slot reserved block). Repurposing these slots does NOT change
# STATE_SIZE (still 91) or the multimodal 570-dim vector (history is just
# repeated 91-vectors). ENCODER_VERSION is bumped because the layout
# semantics of [51..64] change (was zeros, now meaningful).
_TEACHING_CONTEXT = 51  # 0/1 - teaching mode active
_INTERACTION_ACTIVE = 52  # 0/1 - inside a teaching interaction/episode
_PERSON_PRESENT = 53  # 0/1 - a person (face) is present
_GESTURE_START = 54  # 5-slot gesture one-hot: none/wave/point/open_hand/other
_GESTURE_END = 59  # exclusive
_CONVERSATION_TURN = 59  # normalized recent conversation turn count [0,1]
_LAST_ACTION_INDEX = 60  # last executed action index / action_space_size
_RESERVED2_START = 61  # remaining reserved (kept zero)
_RESERVED2_END = 91

# Recognised gesture names, one-hot encoded at [_GESTURE_START .. _GESTURE_END).
_GESTURE_NAMES: tuple[str, ...] = ("none", "wave", "point", "open_hand", "other")
# Default action-space size used to normalise ``last_action_index``. Matches
# the expanded 16-action :func:`deskbot_action_space`.
_DEFAULT_ACTION_SPACE_SIZE = 16

# Named servo slots in the vector.
_SERVO_SLOTS: dict[str, int] = {
    "pan": 0,
    "tilt": 1,
    "left_arm": 2,
    "right_arm": 3,
}

# Maximum number of faces to encode (for normalisation).
_MAX_FACES = 3


# ---------------------------------------------------------------------------
# Vision features
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class VisionFeatures:
    """Vision features extracted from face detection results.

    All values are normalised to [0, 1].  When no face is detected,
    all values default to 0 (or 0.5 for positions, meaning "centre
    unknown").

    Attributes
    ----------
    face_detected:
        1.0 if at least one face was detected, 0.0 otherwise.
    face_x:
        X position of the primary face (0..1, 0.5 if no face).
    face_y:
        Y position of the primary face (0..1, 0.5 if no face).
    face_confidence:
        Detection confidence (0..1, 0 if no face).
    face_size:
        Approximate face size as fraction of frame (0..1, 0 if no face).
    face_count:
        Number of faces in scene, normalised by MAX_FACES (0..1).
    """

    face_detected: float = 0.0
    face_x: float = 0.5
    face_y: float = 0.5
    face_confidence: float = 0.0
    face_size: float = 0.0
    face_count: float = 0.0

    @classmethod
    def from_face_event(
        cls,
        x: float = 0.5,
        y: float = 0.5,
        confidence: float = 0.0,
        face_count: int = 0,
    ) -> VisionFeatures:
        """Create from a FaceDetected event or face detector results."""
        return cls(
            face_detected=1.0 if face_count > 0 else 0.0,
            face_x=x,
            face_y=y,
            face_confidence=confidence,
            face_size=0.0,  # size not available from event; will be set from FaceDetectorResult
            face_count=min(face_count, _MAX_FACES) / _MAX_FACES,
        )

    @classmethod
    def no_face(cls) -> VisionFeatures:
        """Create a VisionFeatures representing no face detected."""
        return cls()

    def to_vector(self) -> list[float]:
        """Return the 6-element vision feature vector."""
        return [
            self.face_detected,
            self.face_x,
            self.face_y,
            self.face_confidence,
            self.face_size,
            self.face_count,
        ]


# ---------------------------------------------------------------------------
# Audio features
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AudioFeatures:
    """Audio features extracted from raw PCM samples.

    All values are normalised to [0, 1].  When no audio is available,
    all values default to 0.

    Attributes
    ----------
    rms_energy:
        Root mean square energy of the audio frame, normalised.
    peak_amplitude:
        Peak absolute amplitude of the audio frame, normalised.
    zero_crossing_rate:
        Fraction of samples that cross zero, normalised to [0, 1].
    """

    rms_energy: float = 0.0
    peak_amplitude: float = 0.0
    zero_crossing_rate: float = 0.0

    @classmethod
    def from_pcm(cls, pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> AudioFeatures:
        """Extract audio features from 16-bit PCM data.

        Parameters
        ----------
        pcm:
            Raw PCM bytes (signed 16-bit little-endian).
        sample_rate:
            Sample rate in Hz (unused for feature extraction, kept for
            future extensions).
        channels:
            Number of audio channels (1 for mono, 2 for stereo).
            If stereo, only the first channel is used.
        """
        import struct

        if not pcm or len(pcm) < 2:
            return cls()

        # Parse 16-bit signed samples
        n_samples = len(pcm) // 2
        try:
            samples = list(struct.unpack(f"<{n_samples}h", pcm[: n_samples * 2]))
        except struct.error:
            return cls()

        if not samples:
            return cls()

        # Normalise to [-1, 1] range (16-bit: max = 32767)
        max_val = 32767.0
        normalised = [s / max_val for s in samples]

        # RMS energy
        sum_sq = sum(s * s for s in normalised)
        rms = math.sqrt(sum_sq / len(normalised)) if normalised else 0.0

        # Peak amplitude
        peak = max(abs(s) for s in normalised) if normalised else 0.0

        # Zero-crossing rate
        if len(normalised) > 1:
            crossings = sum(
                1
                for i in range(1, len(normalised))
                if (normalised[i] >= 0) != (normalised[i - 1] >= 0)
            )
            zcr = crossings / (len(normalised) - 1)
        else:
            zcr = 0.0

        return cls(
            rms_energy=min(rms, 1.0),
            peak_amplitude=min(peak, 1.0),
            zero_crossing_rate=min(zcr, 1.0),
        )

    @classmethod
    def no_audio(cls) -> AudioFeatures:
        """Create an AudioFeatures representing silence / no microphone."""
        return cls()

    def to_vector(self) -> list[float]:
        """Return the 3-element audio feature vector."""
        return [self.rms_energy, self.peak_amplitude, self.zero_crossing_rate]


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StateEncoder:
    """Encode DeskBot's current state into a fixed-size numerical vector.

    The encoder is **deterministic**: the same inputs always produce the
    same output.  Missing inputs (no camera, no microphone) are filled
    with safe defaults.

    Parameters
    ----------
    max_faces:
        Maximum number of faces to normalise count against.
    """

    max_faces: int = _MAX_FACES

    # ----- Context (updated by external callers) -----
    emotions: dict[str, float] = field(default_factory=dict)
    state: RobotState = RobotState.IDLE
    personality: dict[str, float] = field(
        default_factory=lambda: {
            "curiosity": 0.7,
            "energy": 0.6,
            "shyness": 0.3,
            "friendliness": 0.8,
            "playfulness": 0.7,
        }
    )
    servos: dict[str, float] = field(default_factory=dict)
    vision: VisionFeatures = field(default_factory=VisionFeatures.no_face)
    audio: AudioFeatures = field(default_factory=AudioFeatures.no_audio)
    idle_seconds: float = 0.0
    recent_rewards: list[float] = field(default_factory=list)

    # ----- Teaching / conversation / gesture context -----
    # teaching_context: 0/1 - teaching mode active (NOT a RobotState; the
    #   flag rides in the repurposed reserved block to keep STATE_SIZE=91
    #   and the 8-wide state one-hot intact).
    # interaction_active: 0/1 - inside a teaching interaction/episode.
    # person_present: 0/1 - a person (face) is currently present.
    # gesture: one of _GESTURE_NAMES (one-hot encoded).
    # conversation_turn: count of recent conversation turns (normalised).
    # last_action_index: index of the last executed action (normalised).
    # action_space_size: size used to normalise last_action_index.
    teaching_context: bool = False
    interaction_active: bool = False
    person_present: bool = False
    gesture: str = "none"
    conversation_turn: int = 0
    last_action_index: int = -1
    action_space_size: int = _DEFAULT_ACTION_SPACE_SIZE

    def encode(self) -> list[float]:
        """Produce the fixed-size state vector.

        Returns a list of ``STATE_SIZE`` floats in [0, 1] range
        (except servo angles which are normalised to [-1, 1] and
        reward values which can be any float).
        """
        vec = [0.0] * STATE_SIZE
        self._encode_emotions(vec)
        self._encode_state(vec)
        self._encode_personality(vec)
        self._encode_servos(vec)
        self._encode_vision(vec)
        self._encode_audio(vec)
        self._encode_flags(vec)
        self._encode_rewards(vec)
        self._encode_teaching_context(vec)
        # Sanity: no NaN or inf
        for i, v in enumerate(vec):
            if math.isnan(v) or math.isinf(v):
                vec[i] = 0.0
        return vec

    def encode_tensor(self) -> Tensor:
        """Produce the state vector as a :class:`Tensor`."""
        return Tensor(self.encode())

    # ----- Sub-encoders -----

    def _encode_emotions(self, vec: list[float]) -> None:
        """Encode emotion intensities (10 values, one per EmotionName)."""
        for i, emo in enumerate(EmotionName):
            vec[_EMOTION_START + i] = float(self.emotions.get(emo.value, 0.0))

    def _encode_state(self, vec: list[float]) -> None:
        """Encode robot state as one-hot (8 values, one per RobotState)."""
        states = list(RobotState)
        try:
            idx = states.index(self.state)
            vec[_STATE_START + idx] = 1.0
        except ValueError:
            vec[_STATE_START + states.index(RobotState.IDLE)] = 1.0

    def _encode_personality(self, vec: list[float]) -> None:
        """Encode personality traits (5 values)."""
        trait_names = ["curiosity", "energy", "shyness", "friendliness", "playfulness"]
        for i, name in enumerate(trait_names):
            vec[_PERSONALITY_START + i] = max(0.0, min(1.0, float(self.personality.get(name, 0.5))))

    def _encode_servos(self, vec: list[float]) -> None:
        """Encode servo positions (10 slots, normalised to [-1, 1]).

        Known servo names (pan, tilt, left_arm, right_arm) are placed
        at fixed indices.  All others start at 0.0.  Angles are
        normalised from the typical servo range [0, 180] to [-1, 1]
        where 0 = centre (90°).
        """
        for name, slot_idx in _SERVO_SLOTS.items():
            angle = self.servos.get(name, 90.0)
            # Normalise from [0, 180] to [-1, 1], where 90° = 0.0
            vec[_SERVO_START + slot_idx] = max(-1.0, min(1.0, (angle - 90.0) / 90.0))

    def _encode_vision(self, vec: list[float]) -> None:
        """Encode vision features (6 values)."""
        vision_vec = self.vision.to_vector()
        for i, v in enumerate(vision_vec):
            vec[_FACE_START + i] = v

    def _encode_audio(self, vec: list[float]) -> None:
        """Encode audio features (3 values)."""
        audio_vec = self.audio.to_vector()
        for i, v in enumerate(audio_vec):
            vec[_AUDIO_START + i] = v

    def _encode_flags(self, vec: list[float]) -> None:
        """Encode binary flags and continuous values."""
        # Speaking state
        vec[_FLAGS_START + 0] = 1.0 if self.state == RobotState.SPEAKING else 0.0
        # Listening state
        vec[_FLAGS_START + 1] = 1.0 if self.state == RobotState.LISTENING else 0.0
        # Interaction flag (face detected OR in listening/thinking/curious state)
        interacting = self.vision.face_detected > 0.0 or self.state in (
            RobotState.LISTENING,
            RobotState.THINKING,
            RobotState.CURIOUS,
        )
        vec[_FLAGS_START + 2] = 1.0 if interacting else 0.0
        # Idle time normalised to [0, 1] (60 seconds = 1.0)
        vec[_FLAGS_START + 3] = min(self.idle_seconds / 60.0, 1.0)

    def _encode_rewards(self, vec: list[float]) -> None:
        """Encode recent reward history (5 values)."""
        for i in range(5):
            if i < len(self.recent_rewards):
                vec[_REWARD_START + i] = float(self.recent_rewards[i])
            else:
                vec[_REWARD_START + i] = 0.0

    def _encode_teaching_context(self, vec: list[float]) -> None:
        """Encode teaching/conversation/gesture context (slots [51..61]).

        These slots were previously zero-filled reserved space. They now
        carry the human-teaching context so the policy can condition on
        whether a teaching interaction is active, whether a person is
        present, and which gesture was observed - without changing
        ``STATE_SIZE`` or the multimodal 570-dim vector.
        """
        vec[_TEACHING_CONTEXT] = 1.0 if self.teaching_context else 0.0
        vec[_INTERACTION_ACTIVE] = 1.0 if self.interaction_active else 0.0
        vec[_PERSON_PRESENT] = 1.0 if self.person_present else 0.0
        # Gesture one-hot: default to "none" for an unknown gesture name.
        try:
            gidx = _GESTURE_NAMES.index(self.gesture)
        except ValueError:
            gidx = 0
        vec[_GESTURE_START + gidx] = 1.0
        # Normalised conversation turn count (cap at 10 turns -> 1.0).
        vec[_CONVERSATION_TURN] = min(self.conversation_turn / 10.0, 1.0)
        # Last action index normalised to [0, 1]; -1 (none) -> 0.0.
        if self.last_action_index < 0:
            vec[_LAST_ACTION_INDEX] = 0.0
        else:
            size = max(self.action_space_size, 1)
            vec[_LAST_ACTION_INDEX] = min(self.last_action_index / size, 1.0)

    # ----- Updaters (called by event handlers or simulation) -----

    def update_emotion(self, emotion: EmotionName, intensity: float = 1.0) -> None:
        """Update the emotion with the given intensity."""
        self.emotions[emotion.value] = max(0.0, min(1.0, intensity))

    def update_state(self, state: RobotState) -> None:
        """Update the current robot state."""
        self.state = state

    def update_servo(self, name: str, angle: float) -> None:
        """Update a servo position by name and angle."""
        self.servos[name] = angle

    def update_vision(
        self,
        face_detected: bool = False,
        face_x: float = 0.5,
        face_y: float = 0.5,
        face_confidence: float = 0.0,
        face_size: float = 0.0,
        face_count: int = 0,
    ) -> None:
        """Update vision features from face detection results."""
        self.vision = VisionFeatures(
            face_detected=1.0 if face_detected else 0.0,
            face_x=face_x,
            face_y=face_y,
            face_confidence=face_confidence,
            face_size=face_size,
            face_count=min(face_count, self.max_faces) / self.max_faces,
        )

    def update_audio(self, audio: AudioFeatures) -> None:
        """Update audio features."""
        self.audio = audio

    def update_idle(self, seconds: float) -> None:
        """Update idle seconds."""
        self.idle_seconds = seconds

    def push_reward(self, reward: float) -> None:
        """Add a reward to the recent history (keeps last 5)."""
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 5:
            self.recent_rewards = self.recent_rewards[-5:]

    # ----- Teaching / conversation / gesture updaters -----

    def update_teaching_context(self, active: bool) -> None:
        """Set whether teaching mode is active."""
        self.teaching_context = active

    def update_interaction_active(self, active: bool) -> None:
        """Set whether a teaching interaction/episode is in progress."""
        self.interaction_active = active

    def update_person_present(self, present: bool) -> None:
        """Set whether a person (face) is present."""
        self.person_present = present

    def update_gesture(self, gesture: str) -> None:
        """Set the observed gesture name.

        Unknown names are stored as-is but encode to the ``"none"`` slot.
        """
        self.gesture = gesture

    def update_conversation_turn(self, turn: int) -> None:
        """Set the recent conversation turn count."""
        self.conversation_turn = max(0, int(turn))

    def update_last_action(self, action_index: int, action_space_size: int | None = None) -> None:
        """Set the last executed action index (and optional action-space size)."""
        self.last_action_index = int(action_index)
        if action_space_size is not None and action_space_size > 0:
            self.action_space_size = int(action_space_size)

    def reset_teaching_context(self) -> None:
        """Clear only the teaching/gesture context (state stays)."""
        self.teaching_context = False
        self.interaction_active = False
        self.person_present = False
        self.gesture = "none"
        self.conversation_turn = 0
        self.last_action_index = -1
        self.action_space_size = _DEFAULT_ACTION_SPACE_SIZE

    def reset(self) -> None:
        """Reset all context to defaults."""
        self.emotions.clear()
        self.state = RobotState.IDLE
        self.personality = {
            "curiosity": 0.7,
            "energy": 0.6,
            "shyness": 0.3,
            "friendliness": 0.8,
            "playfulness": 0.7,
        }
        self.servos.clear()
        self.vision = VisionFeatures.no_face()
        self.audio = AudioFeatures.no_audio()
        self.idle_seconds = 0.0
        self.recent_rewards.clear()
        self.reset_teaching_context()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def state_size() -> int:
    """Return the size of the state vector produced by :class:`StateEncoder`."""
    return STATE_SIZE


def state_layout() -> dict[str, tuple[int, int]]:
    """Return a description of the state vector layout.

    Each entry maps a section name to a ``(start, end)`` tuple
    indicating the slice of the vector that section occupies.
    """
    return {
        "emotions": (_EMOTION_START, _EMOTION_END),
        "robot_state": (_STATE_START, _STATE_END),
        "personality": (_PERSONALITY_START, _PERSONALITY_END),
        "servos": (_SERVO_START, _SERVO_END),
        "vision": (_FACE_START, _FACE_END),
        "audio": (_AUDIO_START, _AUDIO_END),
        "flags": (_FLAGS_START, _FLAGS_END),
        "rewards": (_REWARD_START, _REWARD_END),
        "teaching_context": (_TEACHING_CONTEXT, _TEACHING_CONTEXT + 1),
        "interaction_active": (_INTERACTION_ACTIVE, _INTERACTION_ACTIVE + 1),
        "person_present": (_PERSON_PRESENT, _PERSON_PRESENT + 1),
        "gesture": (_GESTURE_START, _GESTURE_END),
        "conversation_turn": (_CONVERSATION_TURN, _CONVERSATION_TURN + 1),
        "last_action_index": (_LAST_ACTION_INDEX, _LAST_ACTION_INDEX + 1),
        "reserved": (_RESERVED2_START, _RESERVED2_END),
    }


__all__ = [
    "ENCODER_VERSION",
    "STATE_SIZE",
    "AudioFeatures",
    "StateEncoder",
    "VisionFeatures",
    "state_layout",
    "state_size",
]
