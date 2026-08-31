"""Learning observation adapter: bridges runtime events to PreferenceLearner.

The adapter subscribes to the event bus and translates meaningful signals
into preference observations. It does NOT blindly turn every event into a
preference - only events that carry user-preference information are
forwarded to the learner.

Explicit preferences are extracted from speech events using the
:class:`PreferenceTracker`'s keyword matching. Behavioural observations
(rewarded actions, repeated interaction styles) are inferred from
reward signals and interaction patterns.

This module makes :class:`PreferenceLearner` the canonical learning
backend for preference observations, while :class:`PreferenceTracker`
remains responsible for the explicit linguistic extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    BotReply,
    EmotionChanged,
    FaceDetected,
    IdleTimeout,
    SpeechRecognized,
)
from robot.logging import get_logger

if TYPE_CHECKING:
    from robot.ai.preferences import PreferenceTracker
    from robot.learning.preference_learner import PreferenceLearner

_log = get_logger("learning.observation_adapter")


@dataclass(slots=True)
class LearningObservationAdapter:
    """Bridge runtime events to the canonical :class:`PreferenceLearner`.

    The adapter owns the event-bus subscription.  When no explicit
    ``PreferenceTracker`` is supplied, it creates one backed by the learner's
    store so speech preferences cannot silently stop at the conversation
    layer.
    """

    bus: InMemoryEventBus
    preference_learner: PreferenceLearner
    preference_tracker: PreferenceTracker | None = None
    _subscribed: bool = field(default=False, init=False, repr=False)
    _interaction_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the missing tracker and attach to the event bus."""
        if self.preference_tracker is None:
            from robot.ai.preferences import PreferenceTracker

            self.preference_tracker = PreferenceTracker(store=self.preference_learner.store)
        self.attach()

    def attach(self) -> None:
        """Subscribe to the event bus; safe to call more than once."""
        if self._subscribed:
            return
        self.bus.subscribe(SpeechRecognized, self._on_speech_recognized)
        self.bus.subscribe(FaceDetected, self._on_face_detected)
        self.bus.subscribe(EmotionChanged, self._on_emotion_changed)
        self.bus.subscribe(IdleTimeout, self._on_idle_timeout)
        self.bus.subscribe(BotReply, self._on_bot_reply)
        self._subscribed = True
        _log.info("learning.observation_adapter.attached")

    def detach(self) -> None:
        """Unsubscribe from the event bus; safe to call more than once."""
        if not self._subscribed:
            return
        self.bus.unsubscribe(SpeechRecognized, self._on_speech_recognized)
        self.bus.unsubscribe(FaceDetected, self._on_face_detected)
        self.bus.unsubscribe(EmotionChanged, self._on_emotion_changed)
        self.bus.unsubscribe(IdleTimeout, self._on_idle_timeout)
        self.bus.unsubscribe(BotReply, self._on_bot_reply)
        self._subscribed = False

    async def _on_speech_recognized(self, event: SpeechRecognized) -> None:
        """Extract explicit preferences and feed them into the learner."""
        self._interaction_count += 1

        if self.preference_tracker is not None:
            updated_prefs = self.preference_tracker.process_user_text(event.text)
            for pref in updated_prefs:
                self.preference_learner.observe(
                    category=pref.key,
                    value=pref.value,
                    reward=0.0,
                    source="explicit",
                )
                _log.debug(
                    "learning.preference_observed",
                    category=pref.key,
                    value=pref.value,
                    source="explicit",
                )

        text_len = len(event.text.strip())
        if text_len > 100:
            self.preference_learner.observe_interaction_style(
                style="detailed", reward=0.05, source="behavioral"
            )
        elif 0 < text_len < 10:
            self.preference_learner.observe_interaction_style(
                style="brief", reward=0.05, source="behavioral"
            )

    async def _on_face_detected(self, event: FaceDetected) -> None:
        """Observe face detection as a positive engagement signal."""
        self.preference_learner.observe(
            category="face_preference",
            value=f"pos_{event.x:.1f}_{event.y:.1f}",
            reward=0.1,
            source="behavioral",
        )

    async def _on_emotion_changed(self, event: EmotionChanged) -> None:
        """Track emotional responses as preference signals."""
        self.preference_learner.observe_emotional_response(
            emotion=event.current.value,
            reward=event.intensity * 0.1,
        )

    async def _on_idle_timeout(self, event: IdleTimeout) -> None:
        """Record reduced engagement after a long idle period."""
        if event.seconds_idle > 60:
            self.preference_learner.observe(
                category="interaction_time",
                value="infrequent",
                reward=-0.05,
                source="behavioral",
            )

    async def _on_bot_reply(self, event: BotReply) -> None:
        """Track successful interaction cycles."""
        self._interaction_count += 1
        if self._interaction_count % 5 == 0:
            self.preference_learner.observe_interaction_style(
                style="engaged", reward=0.05, source="behavioral"
            )

    def apply_decay(self) -> list[str]:
        """Apply time-based decay and return changed preference keys."""
        decayed = self.preference_learner.apply_decay()
        if decayed:
            _log.info("learning.preference_decay", count=len(decayed), keys=decayed[:5])
        return decayed


__all__ = ["LearningObservationAdapter"]
