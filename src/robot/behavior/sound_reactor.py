"""Sound reactor: play sound effects in response to robot events.

Subscribes to :class:`EmotionChanged` and :class:`StateChanged` events
on the bus and plays the matching WAV from ``assets/sounds/`` through the
:class:`SoundEffectsPlayer`.  This is what makes the robot audibly
expressive - a ``thinking`` chatter while it ponders, an ``angry`` or
``surprise`` blip when its emotion changes, a ``cute`` / ``very-cute``
sound when it gets happy or excited.

The sound name is the suffix of the WAV file name after ``-robot-``
(e.g. ``826372__charonfaustinus__small-robot-thinking.wav`` ->
``thinking``; ``...-robot-very-cute.wav`` -> ``very-cute``).  The
:indexing logic in :class:`SoundEffectsPlayer` already normalises these.

The reactor only plays a sound when one is actually available
(:meth:`SoundEffectsPlayer.has_sound`), so unmapped or missing sounds are
silently skipped.  It never raises - sound is an enhancement, not a
failure point.  ``talk`` sounds are intentionally **not** auto-played
because they would clash with TTS speech on the same audio output; they
remain available via the REST API and the LLM sound-effect tool.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from robot.behavior.state_machine import RobotState
from robot.events.bus import InMemoryEventBus
from robot.events.events import EmotionChanged, EmotionName, StateChanged
from robot.logging import get_logger
from robot.speech.sound_effects import SoundEffectsPlayer

_log = get_logger("behavior.sound_reactor")

# Emotion -> sound effect name.  Only emotions with a matching WAV are
# listed; the rest are left silent.
EMOTION_SOUNDS: Mapping[EmotionName, str] = {
    EmotionName.ANGRY: "angry",
    EmotionName.SURPRISED: "surprise",
    EmotionName.HAPPY: "cute",
    EmotionName.EXCITED: "very-cute",
    EmotionName.EMBARRASSED: "confused",
}

# Robot state -> sound effect name.
STATE_SOUNDS: Mapping[RobotState, str] = {
    RobotState.THINKING: "thinking",
}


@dataclass(slots=True)
class SoundReactor:
    """Play sound effects in reaction to robot events.

    Parameters
    ----------
    bus:
        The event bus to subscribe to.
    sound_effects:
        The player used to play the WAVs.
    enabled:
        Whether reactions are active.  When ``False``, :meth:`attach`
        is a no-op and no sounds are played.
    """

    bus: InMemoryEventBus
    sound_effects: SoundEffectsPlayer
    enabled: bool = True
    _subscribed: bool = field(default=False, init=False, repr=False)

    def attach(self) -> None:
        """Subscribe to emotion/state events on the bus."""
        if not self.enabled or self._subscribed:
            return
        self.bus.subscribe(EmotionChanged, self._on_emotion_changed)
        self.bus.subscribe(StateChanged, self._on_state_changed)
        self._subscribed = True
        _log.info(
            "sound_reactor.attached",
            sounds=self.sound_effects.list_sounds(),
        )

    def detach(self) -> None:
        """Unsubscribe from the bus."""
        if not self._subscribed:
            return
        self.bus.unsubscribe(EmotionChanged, self._on_emotion_changed)
        self.bus.unsubscribe(StateChanged, self._on_state_changed)
        self._subscribed = False

    async def _play(self, name: str) -> None:
        """Play a sound by name, skipping silently if unavailable."""
        if not self.sound_effects.has_sound(name):
            _log.debug("sound_reactor.no_sound", name=name)
            return
        try:
            await self.sound_effects.play(name)
        except Exception:
            _log.exception("sound_reactor.play_failed", name=name)

    async def _on_emotion_changed(self, event: EmotionChanged) -> None:
        """Play a sound matching the new emotion."""
        name = EMOTION_SOUNDS.get(event.current)
        if name is not None:
            await self._play(name)

    async def _on_state_changed(self, event: StateChanged) -> None:
        """Play a sound matching the new robot state."""
        name = STATE_SOUNDS.get(event.current)
        if name is not None:
            await self._play(name)


__all__ = ["EMOTION_SOUNDS", "STATE_SOUNDS", "SoundReactor"]
