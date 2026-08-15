"""Idle behaviour: produces actions while the robot is in :data:`RobotState.IDLE`."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import anyio

from robot.behavior.actions import (
    BehaviorAction,
    LookAroundAction,
    RequestBlinkAction,
    RequestLookAction,
)
from robot.behavior.personality import Personality, PersonalityTrait
from robot.behavior.state_machine import RobotState, StateMachine
from robot.logging import get_logger
from robot.utils.clock import Clock
from robot.utils.random_source import RandomSource

_log = get_logger("behavior.idle")


@dataclass(slots=True)
class IdleBehavior:
    """Long-running coroutine that emits idle actions.

    The behavior is driven by the personality weights and the random source -
    deterministic in tests, lively in production.
    """

    state_machine: StateMachine
    personality: Personality
    rng: RandomSource
    clock: Clock
    min_idle_s: float = 2.0
    max_idle_s: float = 6.0
    _outbox: list[BehaviorAction] = field(default_factory=list)
    _stopped: bool = True

    async def run(self) -> None:
        """Emit idle actions forever (or until cancelled)."""
        self._stopped = False
        try:
            while not self._stopped:
                if self.state_machine.state is not RobotState.IDLE:
                    await self.clock.sleep(0.2)
                    continue
                await self.clock.sleep(self.rng.uniform(self.min_idle_s, self.max_idle_s))
                action = self._choose_action()
                if action is not None:
                    self._outbox.append(action)
                    _log.debug("idle.action", action=action.name, payload=action.payload)
        finally:
            self._stopped = True

    def stop(self) -> None:
        self._stopped = True

    def drain(self) -> list[BehaviorAction]:
        out = list(self._outbox)
        self._outbox.clear()
        return out

    # ------------------------------------------------------------------ internals
    def _choose_action(self) -> BehaviorAction | None:
        roll = self.rng.random()
        # Probability of an idle action, modulated by personality.
        energy = self.personality.value(PersonalityTrait.ENERGY)
        if roll > 0.3 + 0.5 * energy:
            return None

        curiosity = self.personality.value(PersonalityTrait.CURIOSITY)
        playfulness = self.personality.value(PersonalityTrait.PLAYFULNESS)
        action_roll = self.rng.random()
        if action_roll < 0.45:
            return RequestBlinkAction(left=True, right=True, speed=1.0)
        if action_roll < 0.75:
            x = self.rng.uniform(-curiosity, curiosity)
            y = self.rng.uniform(-0.3, 0.3)
            return RequestLookAction(x=x, y=y, duration_s=0.4)
        if action_roll < 0.95:
            return LookAroundAction(points=self.rng.randint(2, 4))
        if playfulness > 0.6:
            return RequestBlinkAction(left=True, right=False, speed=1.5)  # wink
        return None


@contextlib.asynccontextmanager
async def idle_loop(behavior: IdleBehavior) -> AsyncIterator[None]:
    async with anyio.create_task_group() as tg:
        tg.start_soon(behavior.run)
        try:
            yield
        finally:
            behavior.stop()


__all__ = ["IdleBehavior", "idle_loop"]
