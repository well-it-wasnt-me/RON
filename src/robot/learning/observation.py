"""Typed observations that cleanly separate what the robot *sees* from what it *does*.

Explicit types for observations, actions, and rewards so that a single
transition can be inspected and clearly answer:

1. What did the robot know?   -> :class:`Observation`
2. What did it do?            -> :class:`LearningAction` (from ActionSpace)
3. What happened?             -> :class:`Observation` (next)
4. What reward did it receive? -> ``float``

Observations
------------

An :class:`Observation` is an immutable snapshot of everything the robot
perceived at a given moment:

* :class:`RobotObservation` - emotions, behaviour state, servo
  positions, personality traits, idle time.
* :class:`VisionObservation` - face detection results (reuses the
  existing :class:`VisionFeatures`).
* :class:`AudioObservation` - audio signal features (reuses the
  existing :class:`AudioFeatures`).

Reward history
~~~~~~~~~~~~~~

The :class:`RobotObservation` includes an optional ``recent_rewards``
tuple.  This is retained **deliberately** for temporal credit
assignment - the learning algorithm may need to know whether the robot
is on a positive or negative streak.  Only **past** rewards are
included; the current or future reward for the transition being
recorded is **never** part of the observation.  The tuple is empty by
default and is only populated when a reward model explicitly requests
it.

Actions
-------

Actions come from the project's existing :class:`ActionSpace` /
:class:`LearningAction` definitions.  No second action representation
is invented.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from robot.behavior.state_machine import RobotState
from robot.learning.state_encoder import AudioFeatures, VisionFeatures

# ---------------------------------------------------------------------------
# Sub-observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotObservation:
    """Internal robot state as observed at a given moment.

    Attributes
    ----------
    emotions:
        Mapping of emotion name -> intensity (0..1).
    state:
        Current behaviour state.
    personality:
        Mapping of trait name -> value (0..1).
    servos:
        Mapping of servo name -> angle (degrees).
    idle_seconds:
        How long the robot has been idle.
    recent_rewards:
        Tuple of recent past reward values.  Retained for temporal
        credit assignment; never contains the current or future reward.
        Empty by default.
    """

    emotions: dict[str, float] = field(default_factory=dict)
    state: RobotState = RobotState.IDLE
    personality: dict[str, float] = field(default_factory=dict)
    servos: dict[str, float] = field(default_factory=dict)
    idle_seconds: float = 0.0
    recent_rewards: tuple[float, ...] = ()
    # Teaching / conversation / gesture context (carried through the
    # encoder round-trip so it survives Observation.to_vector()). These
    # mirror the StateEncoder fields of the same name.
    teaching_context: bool = False
    interaction_active: bool = False
    person_present: bool = False
    gesture: str = "none"
    conversation_turn: int = 0
    last_action_index: int = -1
    action_space_size: int = 16


@dataclass(frozen=True)
class VisionObservation:
    """Vision observation wrapping :class:`VisionFeatures`.

    Face detection results: position, confidence, count.  This is what
    the robot *sees*, not what it *does*.  ``FaceDetected`` is an
    observation, never an action.
    """

    features: VisionFeatures = field(default_factory=VisionFeatures.no_face)

    @classmethod
    def no_face(cls) -> VisionObservation:
        return cls(features=VisionFeatures.no_face())

    @classmethod
    def from_face(
        cls,
        x: float = 0.5,
        y: float = 0.5,
        confidence: float = 0.0,
        face_count: int = 0,
    ) -> VisionObservation:
        return cls(
            features=VisionFeatures.from_face_event(
                x=x, y=y, confidence=confidence, face_count=face_count
            )
        )


@dataclass(frozen=True)
class AudioObservation:
    """Audio observation wrapping :class:`AudioFeatures`.

    Audio signal features: RMS energy, peak amplitude, zero-crossing
    rate.  This is what the robot *hears*, not what it *does*.
    ``SpeechRecognized`` is an observation, never an action.
    """

    features: AudioFeatures = field(default_factory=AudioFeatures.no_audio)

    @classmethod
    def no_audio(cls) -> AudioObservation:
        return cls(features=AudioFeatures.no_audio())


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """An immutable snapshot of everything the robot perceived at a moment.

    Combines robot state, vision, and audio sub-observations with a
    nanosecond timestamp.  This is the ``state`` / ``next_state`` in a
    transition.

    Attributes
    ----------
    robot:
        Internal robot state observation.
    vision:
        Vision (face detection) observation.
    audio:
        Audio signal observation.
    timestamp_ns:
        Monotonic nanosecond timestamp when the observation was taken.
    """

    robot: RobotObservation = field(default_factory=RobotObservation)
    vision: VisionObservation = field(default_factory=VisionObservation.no_face)
    audio: AudioObservation = field(default_factory=AudioObservation.no_audio)
    timestamp_ns: int = field(default_factory=time.monotonic_ns)

    @classmethod
    def from_encoder(cls, encoder: object) -> Observation:
        """Capture an :class:`Observation` from a :class:`StateEncoder`'s current context.

        The encoder's mutable fields (emotions, state, servos, vision,
        audio, etc.) are snapshotted into an immutable observation.
        """
        # Import here to avoid circular import at module level.
        from robot.learning.state_encoder import StateEncoder

        assert isinstance(encoder, StateEncoder)
        return cls(
            robot=RobotObservation(
                emotions=dict(encoder.emotions),
                state=encoder.state,
                personality=dict(encoder.personality),
                servos=dict(encoder.servos),
                idle_seconds=encoder.idle_seconds,
                recent_rewards=tuple(encoder.recent_rewards),
                teaching_context=encoder.teaching_context,
                interaction_active=encoder.interaction_active,
                person_present=encoder.person_present,
                gesture=encoder.gesture,
                conversation_turn=encoder.conversation_turn,
                last_action_index=encoder.last_action_index,
                action_space_size=encoder.action_space_size,
            ),
            vision=VisionObservation(features=encoder.vision),
            audio=AudioObservation(features=encoder.audio),
            timestamp_ns=time.monotonic_ns(),
        )

    def to_vector(self) -> list[float]:
        """Encode this observation into a flat float vector.

        Delegates to :class:`StateEncoder` to produce the same
        deterministic vector layout used throughout the learning system.
        """
        from robot.learning.state_encoder import StateEncoder

        enc = StateEncoder()
        enc.emotions = dict(self.robot.emotions)
        enc.state = self.robot.state
        enc.personality = dict(self.robot.personality)
        enc.servos = dict(self.robot.servos)
        enc.vision = self.vision.features
        enc.audio = self.audio.features
        enc.idle_seconds = self.robot.idle_seconds
        enc.recent_rewards = list(self.robot.recent_rewards)
        # Restore teaching/gesture context so the round-trip vector matches
        # a directly-encoded encoder (otherwise these slots silently zero out).
        enc.teaching_context = self.robot.teaching_context
        enc.interaction_active = self.robot.interaction_active
        enc.person_present = self.robot.person_present
        enc.gesture = self.robot.gesture
        enc.conversation_turn = self.robot.conversation_turn
        enc.last_action_index = self.robot.last_action_index
        enc.action_space_size = self.robot.action_space_size
        return enc.encode()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "robot": {
                "emotions": dict(self.robot.emotions),
                "state": self.robot.state.value,
                "personality": dict(self.robot.personality),
                "servos": dict(self.robot.servos),
                "idle_seconds": self.robot.idle_seconds,
                "recent_rewards": list(self.robot.recent_rewards),
                "teaching_context": self.robot.teaching_context,
                "interaction_active": self.robot.interaction_active,
                "person_present": self.robot.person_present,
                "gesture": self.robot.gesture,
                "conversation_turn": self.robot.conversation_turn,
                "last_action_index": self.robot.last_action_index,
                "action_space_size": self.robot.action_space_size,
            },
            "vision": {
                "face_detected": self.vision.features.face_detected,
                "face_x": self.vision.features.face_x,
                "face_y": self.vision.features.face_y,
                "face_confidence": self.vision.features.face_confidence,
                "face_size": self.vision.features.face_size,
                "face_count": self.vision.features.face_count,
            },
            "audio": {
                "rms_energy": self.audio.features.rms_energy,
                "peak_amplitude": self.audio.features.peak_amplitude,
                "zero_crossing_rate": self.audio.features.zero_crossing_rate,
            },
            "timestamp_ns": self.timestamp_ns,
        }


# ---------------------------------------------------------------------------
# Event -> Observation mapping
# ---------------------------------------------------------------------------


def event_to_observation_update(event: object, observation: Observation) -> Observation:
    """Apply a perception event to an observation, returning a new observation.

    Observation events (``FaceDetected``, ``GestureDetected``,
    ``SpeechRecognized``, ``EmotionChanged``, ``IdleTimeout``,
    ``ServoMoved``, ``StateChanged``)
    update the observation.  They are **never** treated as actions.

    Parameters
    ----------
    event:
        A DeskBot event payload (e.g. ``FaceDetected``).
    observation:
        The current observation to update.

    Returns
    -------
    Observation
        A new observation with the event applied.
    """
    import dataclasses

    from robot.events.events import (
        EmotionChanged,
        FaceDetected,
        GestureDetected,
        IdleTimeout,
        ServoMoved,
        SpeechRecognized,
        StateChanged,
    )

    robot = observation.robot
    vision = observation.vision
    audio = observation.audio

    if isinstance(event, FaceDetected):
        vision = VisionObservation.from_face(
            x=event.x, y=event.y, confidence=event.confidence, face_count=1
        )
        # A detected face means a person is present.
        robot = dataclasses.replace(robot, person_present=True)
    elif isinstance(event, GestureDetected):
        # A gesture is an observation: it updates the gesture one-hot and
        # marks a person present. It never creates a transition.
        robot = dataclasses.replace(robot, gesture=event.gesture, person_present=True)
    elif isinstance(event, EmotionChanged):
        new_emotions = dict(robot.emotions)
        new_emotions[event.current.value] = event.intensity
        robot = dataclasses.replace(robot, emotions=new_emotions)
    elif isinstance(event, StateChanged):
        robot = dataclasses.replace(robot, state=event.current)
    elif isinstance(event, ServoMoved):
        new_servos = dict(robot.servos)
        new_servos[event.name] = event.angle
        robot = dataclasses.replace(robot, servos=new_servos)
    elif isinstance(event, IdleTimeout):
        robot = dataclasses.replace(robot, idle_seconds=event.seconds_idle)
    elif isinstance(event, SpeechRecognized):
        # Speech recognition is an observation - it informs the robot
        # that interaction is happening but does not change the
        # encoder's structured fields directly.
        pass

    return Observation(
        robot=robot,
        vision=vision,
        audio=audio,
        timestamp_ns=time.monotonic_ns(),
    )


__all__ = [
    "AudioObservation",
    "Observation",
    "RobotObservation",
    "VisionObservation",
    "event_to_observation_update",
]
