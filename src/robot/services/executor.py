"""Turns :class:`BehaviorAction` objects into hardware commands."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from robot.behavior.actions import (
    BehaviorAction,
    CelebrateAction,
    LookAroundAction,
    RequestBlinkAction,
    RequestLookAction,
    RequestServoMoveAction,
    RequestSleepAction,
)
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    BlinkRequested,
    EmotionChanged,
    EmotionName,
    LookRequested,
    ServoMoved,
)
from robot.interfaces.servo import ServoController
from robot.logging import get_logger

_log = get_logger("services.executor")


@dataclass(slots=True)
class ActionExecutor:
    """Translate behavior actions into bus events and servo commands.

    Keeping the executor as a thin mapping means tests can verify the wiring
    without spinning up the full app.

    The executor depends on the :class:`ServoController` protocol only; the
    concrete backend (mock, GPIO, PCA9685) is injected at construction time.
    """

    bus: InMemoryEventBus
    servo_controller: ServoController

    executed: list[BehaviorAction] = field(default_factory=list)

    async def execute(self, actions: Iterable[BehaviorAction]) -> None:
        for action in actions:
            await self._execute_one(action)
            self.executed.append(action)

    async def execute_one(self, action: BehaviorAction) -> None:
        await self._execute_one(action)
        self.executed.append(action)

    async def _execute_one(self, action: BehaviorAction) -> None:
        if isinstance(action, RequestBlinkAction):
            await self.bus.publish(
                BlinkRequested(left=action.left, right=action.right, speed=action.speed)
            )
        elif isinstance(action, RequestLookAction):
            await self.bus.publish(
                LookRequested(x=action.x, y=action.y, duration_s=action.duration_s)
            )
        elif isinstance(action, RequestServoMoveAction):
            servo = self.servo_controller.get(action.servo)
            await servo.move_to(action.angle, action.duration_s)
            await self.bus.publish(ServoMoved(name=servo.name, angle=action.angle))
        elif isinstance(action, LookAroundAction):
            await self.bus.publish(LookRequested(x=0.5, y=0.0, duration_s=0.3))
        elif isinstance(action, CelebrateAction):
            await self.bus.publish(
                EmotionChanged(
                    previous=EmotionName.NEUTRAL,
                    current=EmotionName.HAPPY,
                    intensity=action.intensity,
                )
            )
        elif isinstance(action, RequestSleepAction):
            _log.info("executor.sleep_requested", duration_s=action.duration_s)
        else:
            _log.warning("executor.unknown_action", action=action.name)


__all__ = ["ActionExecutor"]
