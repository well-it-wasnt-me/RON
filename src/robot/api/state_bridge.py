"""Shared state bridge between DeskBotApp and the FastAPI layer.

The :class:`StateBridge` holds live references to the running robot's
components (state machine, event bus, conversation service, etc.) so
that API route handlers can read and manipulate real state instead of
placeholder data.

The bridge is attached to the FastAPI app's ``app.state`` namespace
and accessed by route handlers via ``request.app.state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot.ai.preferences import PreferenceTracker
    from robot.behavior.state_machine import StateMachine
    from robot.events.bus import InMemoryEventBus
    from robot.interfaces.audio import AudioOutput
    from robot.interfaces.camera import Camera
    from robot.interfaces.microphone import Microphone
    from robot.lifecycle.degradation import DegradationRegistry
    from robot.perception.perception_service import PerceptionService
    from robot.services.conversation_service import ConversationService
    from robot.speech.sound_effects import SoundEffectsPlayer
    from robot.speech.tts import TextToSpeech


@dataclass(slots=True)
class StateBridge:
    """Live reference holder for the FastAPI layer.

    Populated by :meth:`DeskBotApp.build` after all components are
    created. Routes read state from the bridge and publish events on
    the bus.
    """

    bus: InMemoryEventBus | None = None
    state_machine: StateMachine | None = None
    conversation: ConversationService | None = None
    tts: TextToSpeech | None = None
    perception: PerceptionService | None = None
    sound_effects: SoundEffectsPlayer | None = None
    preference_tracker: PreferenceTracker | None = None
    degradation: DegradationRegistry | None = None
    microphone: Microphone | None = None
    camera: Camera | None = None
    audio: AudioOutput | None = None
    _talking: bool = field(default=False, init=False)

    @property
    def is_ready(self) -> bool:
        """True when all core components are wired up."""
        return self.bus is not None and self.state_machine is not None


__all__ = ["StateBridge"]
