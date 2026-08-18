"""Experience recorder that bridges DeskBot events to learning memory.

:class:`ExperienceRecorder` subscribes to the event bus, observes
state transitions and actions, and records them as :class:`Experience`
tuples in the memory layers.

The recorder uses :class:`StateEncoder` to convert the current robot
context into a fixed-size numerical vector suitable for neural-network
training.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    EmotionChanged,
    FaceDetected,
    IdleTimeout,
    ServoMoved,
    SpeechRecognized,
    StateChanged,
)
from robot.learning.experience import EpisodicMemory, Experience, ReplayBuffer, WorkingMemory
from robot.learning.state_encoder import StateEncoder
from robot.logging import get_logger

_log = get_logger("learning.recorder")


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

# Map known event types to indices (must be stable across versions)
_EVENT_TYPES = [
    "StateChanged",
    "EmotionChanged",
    "ServoMoved",
    "FaceDetected",
    "SpeechRecognized",
    "IdleTimeout",
]


def _action_vector_from_event(event_type: str, event: Any) -> list[float]:
    """Build a flat float vector describing an action.

    Current layout (v0):
        [event_type_onehot(6), event_params...]

    This is a minimal encoding; the full :class:`ActionEncoder` will
    be built in Phase 5.
    """
    onehot = [0.0] * len(_EVENT_TYPES)
    if event_type in _EVENT_TYPES:
        onehot[_EVENT_TYPES.index(event_type)] = 1.0

    # Encode key parameters
    params: list[float] = []
    if isinstance(event, EmotionChanged):
        from robot.events.events import EmotionName

        emo_onehot = [0.0] * len(EmotionName)
        with contextlib.suppress(ValueError, IndexError):
            emo_onehot[list(EmotionName).index(event.current)] = 1.0
        params.extend(emo_onehot)
        params.append(float(event.intensity))
    elif isinstance(event, ServoMoved):
        # Normalise angle to [-1, 1] range (assuming ±180 range)
        params.append(event.angle / 180.0)
    elif isinstance(event, FaceDetected):
        params.extend([event.x, event.y, event.confidence])
    elif isinstance(event, SpeechRecognized):
        # Simple presence signal
        params.append(1.0)
        params.append(event.confidence)
    elif isinstance(event, IdleTimeout):
        params.append(event.seconds_idle / 60.0)  # normalise to minutes

    return onehot + params


# ---------------------------------------------------------------------------
# Experience recorder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExperienceRecorder:
    """Bridges DeskBot events to experience memory.

    Subscribes to the event bus, observes state transitions and
    actions, and records them as :class:`Experience` tuples in the
    memory layers.

    The recorder maintains a :class:`StateEncoder` that captures the
    robot's current state.  When an event arrives, the encoder is
    updated and the current state vector is recorded.

    When ``on_experience_recorded`` is provided, the callback is
    invoked after each experience is stored, allowing an external
    system (e.g. :class:`LearningService`) to update its counters
    and trigger downstream processing.  This ensures a single
    authoritative ingestion path: every experience — whether
    recorded from an event or manually — flows through the same
    callback.

    Parameters
    ----------
    bus:
        The event bus to subscribe to.
    encoder:
        The state encoder. If None, a default one is created.
    working_memory:
        Short-term ring buffer.
    replay_buffer:
        Training replay buffer.
    episodic_memory:
        Persistent episodic memory (may be None if persistence is
        disabled).
    default_reward:
        Default reward for experiences where no explicit reward
        is provided.
    on_experience_recorded:
        Optional callback invoked after each experience is stored.
        Receives the :class:`Experience` as its sole argument.
    """

    bus: InMemoryEventBus
    encoder: StateEncoder = field(default_factory=StateEncoder)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    replay_buffer: ReplayBuffer = field(default_factory=ReplayBuffer)
    episodic_memory: EpisodicMemory | None = None
    default_reward: float = 0.0
    on_experience_recorded: Callable[[Experience], None] | None = field(default=None, repr=False)
    _pending_state: list[float] | None = field(default=None, init=False, repr=False)
    _pending_action: list[float] | None = field(default=None, init=False, repr=False)
    _pending_metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _subscribed: bool = field(default=False, init=False, repr=False)

    def attach(self) -> None:
        """Subscribe to the event bus."""
        if self._subscribed:
            return
        self.bus.subscribe(StateChanged, self._on_state_changed)
        self.bus.subscribe(EmotionChanged, self._on_emotion_changed)
        self.bus.subscribe(ServoMoved, self._on_servo_moved)
        self.bus.subscribe(FaceDetected, self._on_face_detected)
        self.bus.subscribe(SpeechRecognized, self._on_speech_recognized)
        self.bus.subscribe(IdleTimeout, self._on_idle_timeout)
        self._subscribed = True
        _log.info("experience_recorder.attached")

    def detach(self) -> None:
        """Unsubscribe from the event bus."""
        self.bus.unsubscribe(StateChanged, self._on_state_changed)
        self.bus.unsubscribe(EmotionChanged, self._on_emotion_changed)
        self.bus.unsubscribe(ServoMoved, self._on_servo_moved)
        self.bus.unsubscribe(FaceDetected, self._on_face_detected)
        self.bus.unsubscribe(SpeechRecognized, self._on_speech_recognized)
        self.bus.unsubscribe(IdleTimeout, self._on_idle_timeout)
        self._subscribed = False

    def update_context(self, **kwargs: Any) -> None:
        """Manually update the encoder context.

        Accepts the same keyword arguments as :class:`StateEncoder`
        attributes: ``state``, ``emotions``, ``servos``, ``personality``,
        ``vision``, ``audio``, ``idle_seconds``, ``recent_rewards``.
        """
        for key, value in kwargs.items():
            if key == "state":
                self.encoder.update_state(value)
            elif key == "emotions":
                self.encoder.emotions.update(value)
            elif key == "servos":
                self.encoder.servos.update(value)
            elif key == "personality":
                self.encoder.personality.update(value)
            elif key == "vision":
                self.encoder.vision = value
            elif key == "audio":
                self.encoder.audio = value
            elif key == "idle_seconds":
                self.encoder.idle_seconds = value
            else:
                # Store anything else in the encoder's internal dict
                # for backward compatibility
                pass

    def record(
        self,
        state: list[float],
        action: list[float],
        reward: float,
        next_state: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> Experience:
        """Manually record an experience tuple.

        This is the primary API for recording experiences.  It stores
        the experience in all configured memory layers.
        """
        exp = Experience(
            timestamp=datetime.now(tz=UTC),
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            metadata=metadata or {},
        )
        self._store(exp)
        return exp

    def record_with_encoder(
        self,
        action: list[float],
        reward: float,
        metadata: dict[str, Any] | None = None,
    ) -> Experience:
        """Record an experience using the current encoder state.

        Snapshots the current encoder state as ``state``, applies the
        action, then snapshots the updated encoder state as
        ``next_state``.  This is the preferred way to record during
        event processing.
        """
        state = self.encoder.encode()
        # Store the reward for the encoder's reward history
        self.encoder.push_reward(reward)
        next_state = self.encoder.encode()
        return self.record(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            metadata=metadata or {},
        )

    def _store(self, experience: Experience) -> None:
        """Store an experience in all memory layers and notify the callback."""
        self.working_memory.add(experience)
        self.replay_buffer.add(experience)
        if self.episodic_memory is not None:
            self.episodic_memory.add(experience)
        if self.on_experience_recorded is not None:
            self.on_experience_recorded(experience)

    # ------------------------------------------------------------------ handlers
    async def _on_state_changed(self, event: StateChanged) -> None:
        """Handle robot state transitions."""
        self.encoder.update_state(event.current)

        # If we have a pending action, complete the experience
        if self._pending_state is not None and self._pending_action is not None:
            next_state = self.encoder.encode()
            self.record(
                state=self._pending_state,
                action=self._pending_action,
                reward=self.default_reward,
                next_state=next_state,
                metadata=self._pending_metadata,
            )
            self._pending_state = None
            self._pending_action = None
            self._pending_metadata = {}

        # Record state change as an action
        action = _action_vector_from_event("StateChanged", event)
        self._pending_state = self.encoder.encode()
        self._pending_action = action
        self._pending_metadata = {
            "event_type": "StateChanged",
            "previous_state": event.previous.value,
            "current_state": event.current.value,
        }

    async def _on_emotion_changed(self, event: EmotionChanged) -> None:
        """Handle emotion changes."""
        # Snapshot state BEFORE applying the action so the experience
        # captures the transition (previous_state -> action -> next_state).
        state = self.encoder.encode()
        self.encoder.update_emotion(event.current, event.intensity)
        action = _action_vector_from_event("EmotionChanged", event)
        self.encoder.push_reward(self.default_reward)
        next_state = self.encoder.encode()
        self.record(
            state=state,
            action=action,
            reward=self.default_reward,
            next_state=next_state,
            metadata={
                "event_type": "EmotionChanged",
                "previous": event.previous.value,
                "current": event.current.value,
                "intensity": event.intensity,
            },
        )

    async def _on_servo_moved(self, event: ServoMoved) -> None:
        """Handle servo movements."""
        self.encoder.update_servo(event.name, event.angle)

    async def _on_face_detected(self, event: FaceDetected) -> None:
        """Handle face detection events."""
        # Snapshot state BEFORE applying the action.
        state = self.encoder.encode()
        self.encoder.update_vision(
            face_detected=True,
            face_x=event.x,
            face_y=event.y,
            face_confidence=event.confidence,
            face_count=1,
        )
        action = _action_vector_from_event("FaceDetected", event)
        self.encoder.push_reward(0.1)  # small positive reward for seeing a face
        next_state = self.encoder.encode()
        self.record(
            state=state,
            action=action,
            reward=0.1,  # small positive reward for seeing a face
            next_state=next_state,
            metadata={
                "event_type": "FaceDetected",
                "x": event.x,
                "y": event.y,
                "confidence": event.confidence,
            },
        )

    async def _on_speech_recognized(self, event: SpeechRecognized) -> None:
        """Handle speech recognition events."""
        # Snapshot state BEFORE applying the action.
        state = self.encoder.encode()
        action = _action_vector_from_event("SpeechRecognized", event)
        self.encoder.push_reward(0.05)  # small reward for interaction
        next_state = self.encoder.encode()
        self.record(
            state=state,
            action=action,
            reward=0.05,  # small reward for interaction
            next_state=next_state,
            metadata={
                "event_type": "SpeechRecognized",
                "text": event.text,
                "confidence": event.confidence,
            },
        )

    async def _on_idle_timeout(self, event: IdleTimeout) -> None:
        """Handle idle timeout events."""
        # Snapshot state BEFORE applying the action.
        state = self.encoder.encode()
        self.encoder.update_idle(event.seconds_idle)
        action = _action_vector_from_event("IdleTimeout", event)
        self.encoder.push_reward(-0.1)  # small negative reward for idling
        next_state = self.encoder.encode()
        self.record(
            state=state,
            action=action,
            reward=-0.1,  # small negative reward for idling
            next_state=next_state,
            metadata={
                "event_type": "IdleTimeout",
                "seconds_idle": event.seconds_idle,
            },
        )


__all__ = ["ExperienceRecorder"]
