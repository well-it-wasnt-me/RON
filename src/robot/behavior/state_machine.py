"""Robot state machine.

States and allowed transitions live here. Components can ask the machine to
transition; the machine raises :class:`StateTransitionError` if the move is
illegal. Each transition publishes a :class:`StateChanged` event on the bus.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from robot.errors import StateTransitionError
from robot.events.bus import InMemoryEventBus
from robot.events.events import StateChanged
from robot.logging import get_logger

_log = get_logger("behavior.state_machine")


class RobotState(str, Enum):
    BOOT = "boot"
    IDLE = "idle"
    CURIOUS = "curious"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    SLEEPING = "sleeping"
    ERROR = "error"


# Default transition table. ``Any`` means "from any state" is not allowed; the
# table is enumerated explicitly to keep behaviour predictable.
_ALLOWED: dict[RobotState, set[RobotState]] = {
    RobotState.BOOT: {RobotState.IDLE, RobotState.ERROR},
    RobotState.IDLE: {
        RobotState.CURIOUS,
        RobotState.LISTENING,
        RobotState.SPEAKING,
        RobotState.SLEEPING,
        RobotState.ERROR,
    },
    RobotState.CURIOUS: {
        RobotState.IDLE,
        RobotState.LISTENING,
        RobotState.SPEAKING,
        RobotState.ERROR,
    },
    RobotState.LISTENING: {RobotState.THINKING, RobotState.IDLE, RobotState.ERROR},
    RobotState.THINKING: {RobotState.SPEAKING, RobotState.IDLE, RobotState.ERROR},
    RobotState.SPEAKING: {RobotState.IDLE, RobotState.LISTENING, RobotState.ERROR},
    RobotState.SLEEPING: {RobotState.IDLE, RobotState.LISTENING, RobotState.ERROR},
    RobotState.ERROR: {RobotState.IDLE, RobotState.SLEEPING},
}

EntryHook = Callable[[], None]
ExitHook = Callable[[], None]


@dataclass(slots=True)
class StateTransition:
    previous: RobotState
    current: RobotState


@dataclass(slots=True)
class StateMachine:
    """Cooperative async state machine.

    Entry/exit hook exceptions are logged by default.  Set
    ``strict=True`` to propagate them, which is useful for tests
    and for critical hooks whose failure must not be silently
    swallowed.
    """

    bus: InMemoryEventBus
    strict: bool = False
    _state: RobotState = RobotState.BOOT
    _entry_hooks: dict[RobotState, list[EntryHook]] = field(default_factory=dict)
    _exit_hooks: dict[RobotState, list[ExitHook]] = field(default_factory=dict)

    @property
    def state(self) -> RobotState:
        return self._state

    def on_enter(self, state: RobotState, hook: EntryHook) -> None:
        self._entry_hooks.setdefault(state, []).append(hook)

    def on_exit(self, state: RobotState, hook: ExitHook) -> None:
        self._exit_hooks.setdefault(state, []).append(hook)

    async def transition(self, target: RobotState) -> None:
        if target is self._state:
            return
        if target not in _ALLOWED[self._state]:
            raise StateTransitionError(f"illegal transition {self._state.value} -> {target.value}")
        previous = self._state

        # Run exit hooks — log errors; in strict mode, propagate them.
        for hook in self._exit_hooks.get(previous, ()):
            try:
                hook()
            except Exception:
                _log.exception(
                    "state_machine.exit_hook_failed",
                    previous=previous.value,
                    target=target.value,
                    hook=getattr(hook, "__qualname__", repr(hook)),
                )
                if self.strict:
                    raise

        self._state = target
        _log.info("state.transition", previous=previous.value, current=target.value)
        await self.bus.publish(StateChanged(previous=previous, current=target))

        # Run entry hooks — log errors; in strict mode, propagate them.
        for hook in self._entry_hooks.get(target, ()):
            try:
                hook()
            except Exception:
                _log.exception(
                    "state_machine.entry_hook_failed",
                    previous=previous.value,
                    current=target.value,
                    hook=getattr(hook, "__qualname__", repr(hook)),
                )
                if self.strict:
                    raise

    def can_transition(self, target: RobotState) -> bool:
        # Self-transitions are always allowed (no-op).
        if target is self._state:
            return True
        return target in _ALLOWED[self._state]


__all__ = ["RobotState", "StateMachine", "StateTransition"]
