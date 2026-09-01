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
    x: float  # normalized 0..1, face centre
    y: float
    confidence: float = 1.0
    size: float = 0.0  # normalized 0..1, approx face size as a fraction of frame height
    known: bool = False  # True when the face is recognised/remembered (see FaceRecognized)


@dataclass(slots=True, frozen=True)
class GestureDetected:
    """A human hand gesture was observed (e.g. a wave).

    This is an *observation*, never an action. It updates the learning
    state encoder (gesture one-hot) but does not create a transition.
    Today the robot has no built-in vision gesture detector, so this
    event is produced by a synthetic injection channel (teaching API,
    CLI, constrained speech parser, or a test) rather than a CV model.
    """

    gesture: str  # one of: none, wave, point, open_hand, other
    confidence: float = 1.0
    x: float = 0.5  # normalized 0..1 gesture centroid, when available
    y: float = 0.5


@dataclass(slots=True, frozen=True)
class HumanFeedback:
    """Explicit human feedback on a recent robot action.

    A *post-hoc* signal: the human reacts to what the robot just did. It is an
    observation that the :class:`~robot.learning.feedback_service.FeedbackService`
    attributes to the most-recent eligible transition (by recency and, when
    available, ``interaction_id``). It never creates a transition by itself and
    never invents a reward — the attribution target must already exist in the
    recorder's working memory, else the feedback is dropped with a log line.

    Attributes
    ----------
    polarity:
        ``+1`` for positive ("good"), ``-1`` for negative ("no"/"wrong").
        Other integers are tolerated but the reward path clamps magnitude.
    magnitude:
        Strength of the feedback, multiplied with ``polarity`` to form the
        reward delta. Default ``1.0``.
    source:
        Origin of the feedback (``"speech"``, ``"api"``, ``"cli"`` …) so the
        dashboard can show how feedback was given.
    interaction_id:
        The teaching interaction the feedback belongs to, when known. Used to
        preferentially attribute feedback to a transition from the same
        interaction. ``None`` for ambient feedback.
    transition_id:
        The transition the feedback was attributed to. Filled in by the
        :class:`~robot.learning.feedback_service.FeedbackService` at attribution
        time; ``None`` until then.
    text:
        The raw human utterance that produced this feedback, for auditing.
    """

    polarity: int
    magnitude: float = 1.0
    source: str = "speech"
    interaction_id: str | None = None
    transition_id: str | None = None
    text: str = ""


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
    "GestureDetected",
    "HumanFeedback",
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
