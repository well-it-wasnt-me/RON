"""Reactions: map events to behavior actions.

The reaction engine subscribes to events on the bus and produces
:class:`BehaviorAction` objects. It is intentionally stateless; the
state machine is the only source of truth for what the robot is currently
doing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from robot.behavior.actions import (
    BehaviorAction,
    CelebrateAction,
    LookAroundAction,
    RequestBlinkAction,
    RequestServoMoveAction,
)
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    FaceDetected,
    SpeechRecognized,
    StateChanged,
    WakeWordDetected,
)
from robot.logging import get_logger

_log = get_logger("behavior.reactions")


@dataclass(slots=True)
class ReactionEngine:
    bus: InMemoryEventBus
    state_machine: StateMachine
    _outbox: list[BehaviorAction] = field(default_factory=list)

    def attach(self) -> None:
        self.bus.subscribe(WakeWordDetected, self._on_wake_word)
        self.bus.subscribe(FaceDetected, self._on_face)
        self.bus.subscribe(SpeechRecognized, self._on_speech)
        self.bus.subscribe(StateChanged, self._on_state_change)

    def detach(self) -> None:
        self.bus.unsubscribe(WakeWordDetected, self._on_wake_word)
        self.bus.unsubscribe(FaceDetected, self._on_face)
        self.bus.unsubscribe(SpeechRecognized, self._on_speech)
        self.bus.unsubscribe(StateChanged, self._on_state_change)

    def drain(self) -> list[BehaviorAction]:
        out = list(self._outbox)
        self._outbox.clear()
        return out

    # ------------------------------------------------------------------ handlers
    async def _on_wake_word(self, event: WakeWordDetected) -> None:
        _log.info("reaction.wake_word", phrase=event.phrase)
        self._outbox.append(RequestBlinkAction(speed=0.7))
        self._outbox.append(RequestServoMoveAction(servo="pan", angle=0.0))

    async def _on_face(self, event: FaceDetected) -> None:
        self._outbox.append(LookAroundAction(points=2))

    async def _on_speech(self, event: SpeechRecognized) -> None:
        _log.info("reaction.speech", text=event.text)
        self._outbox.append(RequestBlinkAction(speed=1.0))

    async def _on_state_change(self, event: StateChanged) -> None:
        if event.current is RobotState.SPEAKING and event.previous is not RobotState.SPEAKING:
            self._outbox.append(CelebrateAction(intensity=0.4))


__all__ = ["ReactionEngine"]
