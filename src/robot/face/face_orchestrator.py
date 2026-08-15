"""Face orchestrator: emotion -> frame.

Maps EmotionChanged events to FaceModel + (optional) Look/Blink animations
so the face actually animates between emotions. Without this the face
stays stuck on NEUTRAL because no one translates emotion events into
face state.

Also subscribes to :class:`LLMTokenReceived` so the face animates
during streaming LLM responses:

* First token -> ``thinking`` emotion (eyes look up, slight smile).
* Subsequent tokens -> eyes drift periodically (``thinking dots`` effect).
* ``done=True`` -> ``happy`` emotion (transition to speaking).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    BlinkRequested,
    EmotionChanged,
    LLMTokenReceived,
    LookRequested,
    StateChanged,
    WakeWordDetected,
)
from robot.face.animations import SpeakingAnimation, ThinkingDotsAnimation, WakeAnimation
from robot.face.animator import FaceAnimator
from robot.face.emotions import EmotionEngine
from robot.logging import get_logger

_log = get_logger("face.orchestrator")


# Map robot state -> face emotion (so state changes move the face too).
_STATE_TO_EMOTION: dict[str, str] = {
    "boot": "neutral",
    "idle": "neutral",
    "curious": "curious",
    "listening": "curious",
    "thinking": "thinking",
    "speaking": "happy",
    "sleeping": "sleepy",
    "error": "angry",
}


@dataclass(slots=True)
class FaceOrchestrator:
    """Bridges EmotionChanged + StateChanged + LLMTokenReceived events
    to the FaceAnimator.

    Subscribes to the event bus and translates each event into one or more
    face commands (``set_emotion``, ``blink``, ``look``, etc.).

    The orchestrator does NOT own the FaceAnimator - it borrows a
    reference and calls the same public methods the rest of the app uses.
    """

    bus: InMemoryEventBus
    face_animator: FaceAnimator
    emotions: EmotionEngine
    blink_for_emotion: Callable[[str], bool] = field(default=lambda _emotion: False, init=False)
    # Whether to play the visual wake animation. When False, the face
    # transitions directly to the 'curious' emotion on wake word detection.
    wake_animation_enabled: bool = field(default=True)
    # Timestamp of the last LLM token received, for periodic eye drift.
    _last_token_time: float = field(default=0.0, init=False)
    # Whether we are currently in a streaming LLM response.
    _streaming: bool = field(default=False, init=False)
    # Count of tokens received in current streaming response.
    _token_count: int = field(default=0, init=False)
    # Accumulated reply text from LLM streaming, used to create
    # SpeakingAnimation when transitioning to SPEAKING.
    _reply_text: str = field(default="", init=False)

    def attach(self) -> None:
        self.bus.subscribe(EmotionChanged, self._on_emotion)
        self.bus.subscribe(StateChanged, self._on_state)
        self.bus.subscribe(LLMTokenReceived, self._on_llm_token)
        self.bus.subscribe(WakeWordDetected, self._on_wake_word)

    def detach(self) -> None:
        self.bus.unsubscribe(EmotionChanged, self._on_emotion)
        self.bus.unsubscribe(StateChanged, self._on_state)
        self.bus.unsubscribe(LLMTokenReceived, self._on_llm_token)
        self.bus.unsubscribe(WakeWordDetected, self._on_wake_word)

    async def _on_emotion(self, event: EmotionChanged) -> None:
        emotion = event.current.value if hasattr(event.current, "value") else str(event.current)
        try:
            self.face_animator.set_emotion(emotion, intensity=event.intensity)
        except Exception:
            _log.exception("face_orchestrator.set_emotion_failed", emotion=emotion)
        # Trigger a blink when emotion changes (the BlinkRequested
        # subscriber handles the actual eye animation).
        with contextlib.suppress(Exception):
            await self.bus.publish(BlinkRequested(speed=1.5))

    async def _on_state(self, event: StateChanged) -> None:
        emotion = _STATE_TO_EMOTION.get(event.current.value, "neutral")
        try:
            self.face_animator.set_emotion(emotion)
        except Exception:
            _log.exception("face_orchestrator.set_state_emotion_failed", emotion=emotion)

        # Create speaking animation when transitioning to SPEAKING.
        if event.current.value == "speaking" and self._reply_text:
            self.face_animator.set_speaking_animation(SpeakingAnimation(text=self._reply_text))
            _log.debug("face_orchestrator.speaking_animation_started")

        # Clear speaking animation and reply text when returning to IDLE.
        if event.current.value == "idle":
            self.face_animator.set_speaking_animation(None)
            self._reply_text = ""
            _log.debug("face_orchestrator.speaking_animation_cleared")

        # For the listening state, glance slightly toward the user.
        if event.current.value == "listening":
            with contextlib.suppress(Exception):
                await self.bus.publish(LookRequested(x=0.0, y=0.1, duration_s=0.4))

    async def _on_llm_token(self, event: LLMTokenReceived) -> None:
        """Animate the face during streaming LLM token generation.

        * First token: create :class:`ThinkingDotsAnimation` and set
          emotion to ``thinking``.
        * Subsequent tokens: the thinking animation auto-advances via
          :meth:`FaceAnimator.step` each frame.
        * ``done=True``: clear the thinking animation and transition
          to ``happy`` (speaking).
        """
        if not self._streaming:
            # First token in a new streaming response.
            self._streaming = True
            self._token_count = 0
            self._reply_text = ""
            self._last_token_time = time.monotonic()
            # Start the thinking dots animation.
            self.face_animator.set_thinking_animation(ThinkingDotsAnimation())
            try:
                self.face_animator.set_emotion("thinking", intensity=0.8)
            except Exception:
                _log.exception("face_orchestrator.thinking_emotion_failed")
            _log.debug("face_orchestrator.streaming_start")

        self._token_count += 1
        self._reply_text += event.token

        if event.done:
            # Streaming complete - clear thinking animation and transition
            # to "happy" (speaking) emotion.
            self._streaming = False
            self._token_count = 0
            self.face_animator.set_thinking_animation(None)
            try:
                self.face_animator.set_emotion("happy", intensity=1.0)
            except Exception:
                _log.exception("face_orchestrator.speaking_emotion_failed")
            # Blink to signal the transition.
            with contextlib.suppress(Exception):
                await self.bus.publish(BlinkRequested(speed=1.0))
            _log.debug("face_orchestrator.streaming_done")

    async def _on_wake_word(self, event: WakeWordDetected) -> None:
        """Play the wake animation when a wake word is detected.

        The wake animation takes full control of the face for ~1 second,
        then completes naturally.  The subsequent :class:`EmotionChanged`
        event (``curious``) will set the emotion while the wake animation
        plays; the wake animation has higher priority in
        :meth:`FaceAnimator.step`.

        When ``wake_animation_enabled`` is ``False``, the wake animation
        is skipped and the face transitions directly to ``curious``.
        """
        if self.wake_animation_enabled:
            self.face_animator.set_wake_animation(WakeAnimation())
            _log.debug("face_orchestrator.wake_animation_started")


__all__ = ["_STATE_TO_EMOTION", "FaceOrchestrator"]
