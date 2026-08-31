"""Turns :class:`BehaviorAction` objects into hardware commands.

The executor is the **single concrete execution point** for runtime robot actions.
Keeping the executor as a thin mapping means tests can verify the wiring without
spinning up the full app.  The executor depends on the :class:`ServoController`
protocol only; the concrete backend (mock, GPIO, PCA9685) is injected at
construction time.

When an :class:`ExperienceRecorder` is wired in (``experience_recorder``), each
*mappable* action is wrapped in the learning transition lifecycle —
``recorder.begin_transition()`` before execution and
``recorder.complete_transition()`` after the outcome is observed — so a real
state → action → outcome → next-state experience is stored through the existing
:class:`TransitionStore`.  Observation events continue to update the encoder only;
they never create transitions.  Learning instrumentation failures are swallowed
so they can never crash or block the robot.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
from robot.learning.action_mapping import behavior_action_to_index
from robot.learning.observation import Observation
from robot.logging import get_logger

if TYPE_CHECKING:
    from robot.learning.recorder import ExperienceRecorder

_log = get_logger("services.executor")


@dataclass(slots=True)
class ActionExecutor:
    """Translate behavior actions into bus events and servo commands.

    Keeping the executor as a thin mapping means tests can verify the wiring
    without spinning up the full app.

    The executor depends on the :class:`ServoController` protocol only; the
    concrete backend (mock, GPIO, PCA9685) is injected at construction time.

    When ``experience_recorder`` is set, each executable action is wrapped in the
    learning transition lifecycle so real actions produce stored experiences.
    """

    bus: InMemoryEventBus
    servo_controller: ServoController

    executed: list[BehaviorAction] = field(default_factory=list)
    #: Optional experience recorder. When set, mappable actions open a transition
    #: before execution and complete it (with reward + execution outcome) after.
    #: Left ``None`` when on-device learning is disabled — behaviour is unchanged.
    experience_recorder: ExperienceRecorder | None = None

    async def execute(self, actions: Iterable[BehaviorAction]) -> None:
        for action in actions:
            await self.execute_one(action)

    async def execute_one(self, action: BehaviorAction) -> None:
        await self._execute_with_learning(action)
        self.executed.append(action)

    async def _execute_with_learning(self, action: BehaviorAction) -> None:
        """Execute one action, wrapping it in the learning transition lifecycle.

        The pre-action state is captured *before* execution; the post-action
        ``next_state`` is captured *after* the outcome is observed (i.e. after the
        bus events published by :meth:`_execute_one` have updated the encoder).
        Any learning-side failure is logged and swallowed so it cannot block the
        robot.  Hardware execution failures are recorded with
        ``execution_success=False`` and then re-raised so the existing error
        propagation (e.g. ``_drain_behaviors`` logging + continue) is preserved.
        """
        recorder = self.experience_recorder
        action_index = self._action_index_for(action, recorder)

        pending = None
        observation_before: Observation | None = None
        if action_index is not None and recorder is not None:
            try:
                observation_before = Observation.from_encoder(recorder.encoder)
                pending = recorder.begin_transition(action_index)
            except Exception:
                _log.exception("executor.learning_begin_failed", action=action.name)
                pending = None
                observation_before = None

        success = True
        reason = ""
        try:
            await self._execute_one(action)
        except Exception as exc:
            success = False
            reason = repr(exc)
            raise
        finally:
            if pending is not None and recorder is not None:
                try:
                    reward = self._compute_reward(recorder, observation_before, action_index)
                    recorder.complete_transition(
                        pending,
                        reward=reward,
                        execution_success=success,
                        execution_failure_reason=reason,
                        metadata={"behavior_action_name": action.name},
                    )
                except Exception:
                    _log.exception("executor.learning_complete_failed", action=action.name)

    def _action_index_for(
        self, action: BehaviorAction, recorder: ExperienceRecorder | None
    ) -> int | None:
        """Resolve the action to an :class:`ActionSpace` index, or ``None``."""
        if recorder is None:
            return None
        return behavior_action_to_index(action, recorder.action_space)

    def _compute_reward(
        self,
        recorder: ExperienceRecorder,
        observation_before: Observation | None,
        action_index: int | None,
    ) -> float:
        """Compute the transition reward using the recorder's existing reward model.

        Falls back to ``recorder.default_reward`` if the typed observations are
        unavailable or the reward model raises.  No reward value is invented —
        only the repository's existing :class:`RewardModel` components are used.
        """
        if observation_before is None or action_index is None:
            return recorder.default_reward
        try:
            observation_after = Observation.from_encoder(recorder.encoder)
            return float(
                recorder.reward_model.compute_for_action_index(
                    observation=observation_before,
                    action_index=action_index,
                    next_observation=observation_after,
                    action_space=recorder.action_space,
                )
            )
        except Exception:
            _log.exception("executor.learning_reward_failed")
            return recorder.default_reward

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
