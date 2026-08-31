"""Phase 6: the ``human_feedback_reward`` reward component.

The component sums ``polarity*magnitude`` of every :class:`HumanFeedback`
present in ``ctx.events`` and clamps to ``[-1, 1]``. With no human feedback it
returns ``0.0`` (it never invents a signal). It is composable: the default
:class:`RewardModel` includes it alongside the engagement/penalty components.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from robot.events.events import HumanFeedback
from robot.learning.action_learning import deskbot_action_space
from robot.learning.observation import Observation
from robot.learning.reward import (
    RewardContext,
    RewardModel,
    human_feedback_reward,
)

_ACTION = deskbot_action_space().get(13)  # wave


def _ctx(events: Sequence[object]) -> RewardContext:
    obs = Observation()
    return RewardContext(
        observation=obs,
        action=_ACTION,
        next_observation=obs,
        events=list(events),
    )


class TestHumanFeedbackComponent:
    def test_no_events_is_zero(self) -> None:
        assert human_feedback_reward(_ctx([])) == 0.0

    def test_only_non_feedback_events_is_zero(self) -> None:
        # Non-HumanFeedback events are ignored, never treated as feedback.
        assert human_feedback_reward(_ctx(["something", 42, object()])) == 0.0

    def test_single_positive(self) -> None:
        fb = HumanFeedback(polarity=1, magnitude=1.0, source="speech")
        assert human_feedback_reward(_ctx([fb])) == pytest.approx(1.0)

    def test_single_negative(self) -> None:
        fb = HumanFeedback(polarity=-1, magnitude=1.0, source="speech")
        assert human_feedback_reward(_ctx([fb])) == pytest.approx(-1.0)

    def test_magnitude_scales(self) -> None:
        fb = HumanFeedback(polarity=1, magnitude=0.5, source="speech")
        assert human_feedback_reward(_ctx([fb])) == pytest.approx(0.5)

    def test_multiple_feedback_sum_then_clamp(self) -> None:
        # Two positive + one negative = 1.0 + 1.0 - 1.0 = 1.0 (within clamp).
        fbs = [
            HumanFeedback(polarity=1, magnitude=1.0, source="speech"),
            HumanFeedback(polarity=1, magnitude=1.0, source="speech"),
            HumanFeedback(polarity=-1, magnitude=1.0, source="speech"),
        ]
        assert human_feedback_reward(_ctx(fbs)) == pytest.approx(1.0)

    def test_positive_clamped_to_one(self) -> None:
        # 1.0 + 0.7 = 1.7 -> clamped to 1.0.
        fbs = [
            HumanFeedback(polarity=1, magnitude=1.0, source="speech"),
            HumanFeedback(polarity=1, magnitude=0.7, source="speech"),
        ]
        assert human_feedback_reward(_ctx(fbs)) == pytest.approx(1.0)

    def test_negative_clamped_to_neg_one(self) -> None:
        fbs = [
            HumanFeedback(polarity=-1, magnitude=1.0, source="speech"),
            HumanFeedback(polarity=-1, magnitude=0.8, source="speech"),
        ]
        assert human_feedback_reward(_ctx(fbs)) == pytest.approx(-1.0)

    def test_ignores_non_feedback_among_feedback(self) -> None:
        fb = HumanFeedback(polarity=1, magnitude=1.0, source="speech")
        # Mixed: only the HumanFeedback contributes.
        assert human_feedback_reward(_ctx(["noise", fb, 42])) == pytest.approx(1.0)


class TestComposition:
    def test_default_model_includes_human_feedback(self) -> None:
        """The default RewardModel has the human_feedback component wired in."""
        names = {c.__name__ for c in RewardModel().components}
        assert "human_feedback_reward" in names

    def test_feedback_component_contributes_alongside_others(self) -> None:
        """End-to-end: a HumanFeedback event raises the computed reward."""
        model = RewardModel()
        obs = Observation()
        fb = HumanFeedback(polarity=1, magnitude=1.0, source="speech")
        without = model.compute(obs, _ACTION, obs, events=[])
        with_fb = model.compute(obs, _ACTION, obs, events=[fb])
        # Feedback is non-negative, so the amended reward is >= the base.
        assert with_fb >= without
        assert with_fb == pytest.approx(without + 1.0)
