"""Reward model: compute reward from a transition, separate from the recorder.

Phase 2 of the production learning plan requires reward calculation to
be moved out of the recorder.  The :class:`RewardModel` computes the
scalar reward for a transition after the outcome has been observed.

The reward policy is configurable: different reward components can be
added or removed without touching the recorder or the transition store.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from robot.learning.action_learning import ActionSpace, LearningAction
from robot.learning.observation import Observation
from robot.logging import get_logger

_log = get_logger("learning.reward")


# ---------------------------------------------------------------------------
# Reward component protocol
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RewardContext:
    """Context passed to a reward component.

    Attributes
    ----------
    observation:
        The observation before the action was taken.
    action:
        The action that was selected and executed.
    next_observation:
        The observation after the action was executed.
    events:
        Additional events that occurred during the transition
        (e.g. user feedback, system events).
    """

    observation: Observation
    action: LearningAction
    next_observation: Observation
    events: list[Any] = field(default_factory=list)


RewardComponent = Callable[[RewardContext], float]


# ---------------------------------------------------------------------------
# Built-in reward components
# ---------------------------------------------------------------------------


def face_engagement_reward(ctx: RewardContext) -> float:
    """Reward for engaging with a detected face.

    Positive reward when a face is detected and the action is
    interaction-oriented (look_center, celebrate, look_around).
    """
    face_detected = ctx.next_observation.vision.features.face_detected > 0.0
    if not face_detected:
        return 0.0

    interaction_actions = {"look_center", "celebrate", "look_around", "blink", "wink"}
    if ctx.action.name in interaction_actions:
        return 0.1
    return 0.0


def idle_penalty_reward(ctx: RewardContext) -> float:
    """Penalise idling when stimuli are present."""
    face = ctx.next_observation.vision.features.face_detected > 0.0
    audio = ctx.next_observation.audio.features.rms_energy > 0.1
    if (face or audio) and ctx.action.name == "sleep":
        return -0.5
    if not face and not audio and ctx.action.name == "sleep":
        return 0.2  # energy saving
    return 0.0


def interaction_reward(ctx: RewardContext) -> float:
    """Small positive reward for any interaction when a face is present."""
    face = ctx.next_observation.vision.features.face_detected > 0.0
    if face and ctx.action.name in {"celebrate", "look_center"}:
        return 0.05
    return 0.0


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RewardModel:
    """Computes the scalar reward for a transition.

    The model applies a list of reward components, each contributing a
    float.  The final reward is the sum of all components, clamped to
    ``[-max_abs_reward, max_abs_reward]``.

    Parameters
    ----------
    components:
        List of reward components.  Each is a callable that takes a
        :class:`RewardContext` and returns a float.
    max_abs_reward:
        Maximum absolute reward value (clamp).
    default_reward:
        Fallback reward when no components are configured.
    """

    components: list[RewardComponent] = field(
        default_factory=lambda: [face_engagement_reward, idle_penalty_reward, interaction_reward]
    )
    max_abs_reward: float = 2.0
    default_reward: float = 0.0

    def compute(
        self,
        observation: Observation,
        action: LearningAction,
        next_observation: Observation,
        events: list[Any] | None = None,
    ) -> float:
        """Compute the scalar reward for a transition.

        Parameters
        ----------
        observation:
            Observation before the action.
        action:
            The action that was executed.
        next_observation:
            Observation after the action (the outcome).
        events:
            Additional events that occurred during the transition.

        Returns
        -------
        float
            The scalar reward.
        """
        if not self.components:
            return self.default_reward

        ctx = RewardContext(
            observation=observation,
            action=action,
            next_observation=next_observation,
            events=events or [],
        )

        total = 0.0
        for component in self.components:
            try:
                total += float(component(ctx))
            except Exception:
                _log.exception("reward.component_error", component=component.__name__)

        # Clamp
        if total > self.max_abs_reward:
            total = self.max_abs_reward
        elif total < -self.max_abs_reward:
            total = -self.max_abs_reward

        return total

    def compute_for_action_index(
        self,
        observation: Observation,
        action_index: int,
        next_observation: Observation,
        action_space: ActionSpace,
        events: list[Any] | None = None,
    ) -> float:
        """Compute reward using an action index from the action space."""
        action = action_space.get(action_index)
        return self.compute(observation, action, next_observation, events)


__all__ = [
    "RewardComponent",
    "RewardContext",
    "RewardModel",
    "face_engagement_reward",
    "idle_penalty_reward",
    "interaction_reward",
]
