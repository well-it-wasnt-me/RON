"""Phase 2: forward + reverse action mapping for the new behaviour actions.

Forward: ``behavior_action_to_index`` resolves a concrete
:class:`BehaviorAction` to its action-space index.

Reverse: ``action_index_to_behavior_action`` resolves an index back to a
concrete action (used by both the demonstration and policy-proposal paths
so every learned action flows through the same executor).

The round-trip must be stable for all 5 new action types, and
``MoveArmAction`` must resolve to ``move_left_arm``/``move_right_arm`` by
servo name.
"""

from __future__ import annotations

import pytest

from robot.behavior.actions import (
    BehaviorAction,
    ChangeEmotionAction,
    MoveArmAction,
    SetStateAction,
    SpeakAction,
    WaveAction,
)
from robot.learning.action_learning import ActionSpace, deskbot_action_space
from robot.learning.action_mapping import (
    action_index_to_behavior_action,
    behavior_action_to_index,
    behavior_action_to_name,
)


@pytest.fixture
def space() -> ActionSpace:
    return deskbot_action_space()


class TestForwardMapping:
    def test_wave_maps(self, space: ActionSpace) -> None:
        assert behavior_action_to_name(WaveAction()) == "wave"
        assert behavior_action_to_index(WaveAction(), space) == 13

    def test_speak_maps(self, space: ActionSpace) -> None:
        assert behavior_action_to_name(SpeakAction(text="hi")) == "speak"
        assert behavior_action_to_index(SpeakAction(text="hi"), space) == 10

    def test_change_emotion_maps(self, space: ActionSpace) -> None:
        assert behavior_action_to_name(ChangeEmotionAction(emotion="sad")) == "change_emotion"
        assert behavior_action_to_index(ChangeEmotionAction(emotion="sad"), space) == 11

    def test_set_state_maps(self, space: ActionSpace) -> None:
        assert behavior_action_to_name(SetStateAction(state="curious")) == "set_state"
        assert behavior_action_to_index(SetStateAction(state="curious"), space) == 12

    def test_move_arm_resolves_by_servo(self, space: ActionSpace) -> None:
        left = MoveArmAction(servo="left_arm", angle=45.0)
        right = MoveArmAction(servo="right_arm", angle=120.0)
        assert behavior_action_to_name(left) == "move_left_arm"
        assert behavior_action_to_name(right) == "move_right_arm"
        assert behavior_action_to_index(left, space) == 14
        assert behavior_action_to_index(right, space) == 15

    def test_move_arm_unknown_servo_is_none(self, space: ActionSpace) -> None:
        bad = MoveArmAction(servo="flerp", angle=45.0)
        assert behavior_action_to_name(bad) is None
        assert behavior_action_to_index(bad, space) is None


class TestReverseMapping:
    def test_wave_round_trip(self, space: ActionSpace) -> None:
        action = action_index_to_behavior_action(13, space)
        assert isinstance(action, WaveAction)

    def test_speak_round_trip_with_override(self, space: ActionSpace) -> None:
        action = action_index_to_behavior_action(10, space, {"text": "good robot"})
        assert isinstance(action, SpeakAction)
        assert action.text == "good robot"

    def test_change_emotion_round_trip_with_override(self, space: ActionSpace) -> None:
        action = action_index_to_behavior_action(
            11, space, {"emotion": "sad", "intensity": 0.5}
        )
        assert isinstance(action, ChangeEmotionAction)
        assert action.emotion == "sad"
        assert action.intensity == 0.5

    def test_set_state_round_trip_with_override(self, space: ActionSpace) -> None:
        action = action_index_to_behavior_action(12, space, {"state": "curious"})
        assert isinstance(action, SetStateAction)
        assert action.state == "curious"

    def test_move_left_arm_round_trip_with_angle(self, space: ActionSpace) -> None:
        action = action_index_to_behavior_action(14, space, {"angle": 30.0})
        assert isinstance(action, MoveArmAction)
        assert action.servo == "left_arm"
        assert action.angle == 30.0

    def test_move_right_arm_round_trip_default(self, space: ActionSpace) -> None:
        action = action_index_to_behavior_action(15, space)
        assert isinstance(action, MoveArmAction)
        assert action.servo == "right_arm"
        assert action.angle == 90.0  # registered default

    def test_out_of_range_index_returns_none(self, space: ActionSpace) -> None:
        assert action_index_to_behavior_action(99, space) is None
        assert action_index_to_behavior_action(-1, space) is None

    def test_invalid_emotion_returns_none(self, space: ActionSpace) -> None:
        """An override with an invalid emotion must not produce an action."""
        assert action_index_to_behavior_action(11, space, {"emotion": "flerp"}) is None

    def test_invalid_state_returns_none(self, space: ActionSpace) -> None:
        assert action_index_to_behavior_action(12, space, {"state": "flerp"}) is None

    def test_out_of_range_arm_angle_returns_none(self, space: ActionSpace) -> None:
        assert action_index_to_behavior_action(14, space, {"angle": 999.0}) is None
        assert action_index_to_behavior_action(15, space, {"angle": -5.0}) is None


class TestExistingActionsStillRoundTrip:
    """The reverse mapping must still cover the original 10 actions."""

    def test_celebrate_round_trip(self, space: ActionSpace) -> None:
        from robot.behavior.actions import CelebrateAction

        action = action_index_to_behavior_action(7, space)
        assert isinstance(action, CelebrateAction)

    def test_look_round_trip(self, space: ActionSpace) -> None:
        from robot.behavior.actions import RequestLookAction

        action = action_index_to_behavior_action(2, space)  # look_center
        assert isinstance(action, RequestLookAction)


class TestRoundTripIdentity:
    """Forward then reverse yields a behaviourally-equivalent action."""

    @pytest.mark.parametrize(
        "action",
        [
            WaveAction(),
            SpeakAction(text="hello"),
            ChangeEmotionAction(emotion="happy"),
            SetStateAction(state="idle"),
            MoveArmAction(servo="left_arm", angle=90.0),
            MoveArmAction(servo="right_arm", angle=90.0),
        ],
    )
    def test_round_trip_name(self, action: BehaviorAction, space: ActionSpace) -> None:
        idx = behavior_action_to_index(action, space)
        assert idx is not None
        resolved = action_index_to_behavior_action(idx, space)
        assert resolved is not None
        assert behavior_action_to_name(resolved) == behavior_action_to_name(action)
