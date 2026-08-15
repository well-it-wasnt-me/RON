"""All event payload types.

Events are immutable dataclasses. New payload types must be added here so
subscribers and publishers share a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot.behavior.state_machine import RobotState


class EmotionName(str, Enum):
    """High-level emotional states the face engine understands.

    The minimum set requested by the project spec.
    """

    NEUTRAL = "neutral"
    HAPPY = "happy"
    CURIOUS = "curious"
    THINKING = "thinking"
    SLEEPY = "sleepy"
    EMBARRASSED = "embarrassed"
    EXCITED = "excited"
    SAD = "sad"
    SURPRISED = "surprised"
    ANGRY = "angry"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class RobotStarted:
    when: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass(slots=True, frozen=True)
class RobotStopped:
    when: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    reason: str = "shutdown"


@dataclass(slots=True, frozen=True)
class RobotError:
    message: str
    component: str = "unknown"
    recoverable: bool = True


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class StateChanged:
    previous: RobotState
    current: RobotState


# ---------------------------------------------------------------------------
# Emotion
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class EmotionChanged:
    previous: EmotionName
    current: EmotionName
    intensity: float = 1.0


# ---------------------------------------------------------------------------
# Eye / animation
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class BlinkRequested:
    """Either eye or both. ``left`` and ``right`` may be False for a wink."""

    left: bool = True
    right: bool = True
    speed: float = 1.0


@dataclass(slots=True, frozen=True)
class LookRequested:
    """Request a glance in a direction (normalized -1..1)."""

    x: float = 0.0
    y: float = 0.0
    duration_s: float = 0.3


@dataclass(slots=True, frozen=True)
class DisplayUpdated:
    display: str  # "left" or "right"


@dataclass(slots=True, frozen=True)
class AnimationFinished:
    name: str
    interrupted: bool = False


# ---------------------------------------------------------------------------
# Servos
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class ServoMoved:
    name: str
    angle: float


# ---------------------------------------------------------------------------
# Idle
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class IdleTimeout:
    seconds_idle: float


# ---------------------------------------------------------------------------
# Personality
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class PersonalityChanged:
    trait: str
    value: float


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class FaceDetected:
    x: float  # normalized 0..1
    y: float
    confidence: float = 1.0


@dataclass(slots=True, frozen=True)
class SpeechRecognized:
    text: str
    confidence: float = 1.0
    language: str = "en"


@dataclass(slots=True, frozen=True)
class WakeWordDetected:
    phrase: str
    confidence: float = 1.0


@dataclass(slots=True, frozen=True)
class SoundEffectPlayed:
    """A sound effect was sent to the configured audio output."""

    name: str
    filename: str


# ---------------------------------------------------------------------------
# LLM streaming
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class LLMTokenReceived:
    """A single token from a streaming LLM response.

    Published for each token as the LLM generates its reply, enabling
    real-time face animation (thinking -> speaking).
    """

    token: str
    done: bool = False


@dataclass(slots=True, frozen=True)
class BotReply:
    """The full text reply the robot is speaking/returning.

    Published after the LLM finishes generating a response (both
    streaming and one-shot paths). Subscribers can use this to display
    the reply as text (terminal, display, WebSocket clients) when TTS
    is unavailable.
    """

    text: str
    user_text: str = ""


# Type alias for the entire event union.
Event = Any  # intentionally permissive - subscribers narrow by ``type(event)``

__all__ = [
    "AnimationFinished",
    "BlinkRequested",
    "BotReply",
    "DisplayUpdated",
    "EmotionChanged",
    "EmotionName",
    "Event",
    "FaceDetected",
    "IdleTimeout",
    "LLMTokenReceived",
    "LookRequested",
    "PersonalityChanged",
    "RobotError",
    "RobotStarted",
    "RobotStopped",
    "ServoMoved",
    "SoundEffectPlayed",
    "SpeechRecognized",
    "StateChanged",
    "WakeWordDetected",
]
