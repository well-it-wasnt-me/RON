"""Phase 2: the expanded 16-action DeskBot action space.

Pins that the teaching-loop interaction actions are registered with stable
indices and validated params, and that the safety validator admits them.
"""

from __future__ import annotations

from robot.learning.action_learning import deskbot_action_space
from robot.learning.safety import ActionSafetyValidator


def test_action_space_has_16_actions() -> None:
    space = deskbot_action_space()
    assert space.size == 16


def test_new_action_indices_are_stable() -> None:
    """The 6 new actions occupy indices 10..15 in a fixed order."""
    space = deskbot_action_space()
    expected = {
        10: "speak",
        11: "change_emotion",
        12: "set_state",
        13: "wave",
        14: "move_left_arm",
        15: "move_right_arm",
    }
    for idx, name in expected.items():
        assert space.get(idx).name == name
        assert space.get_by_name(name).index == idx


def test_new_actions_carry_validated_params() -> None:
    space = deskbot_action_space()
    speak = space.get_by_name("speak")
    assert speak.params["text"] == "hello"
    emo = space.get_by_name("change_emotion")
    assert emo.params["emotion"] == "happy"
    assert emo.params["intensity"] == 1.0
    state = space.get_by_name("set_state")
    assert state.params["state"] == "idle"
    wave = space.get_by_name("wave")
    assert wave.params == {}
    left = space.get_by_name("move_left_arm")
    assert left.params["servo"] == "left_arm"
    assert left.params["angle"] == 90.0
    right = space.get_by_name("move_right_arm")
    assert right.params["servo"] == "right_arm"
    assert right.params["angle"] == 90.0


def test_action_vector_is_16_wide() -> None:
    space = deskbot_action_space()
    vec = space.action_vector(13)  # wave
    assert len(vec) == 16
    assert vec[13] == 1.0
    assert sum(vec) == 1.0


def test_safety_validator_admits_new_actions() -> None:
    """The ActionSafetyValidator must allow the new action names."""
    validator = ActionSafetyValidator()
    for name in ("speak", "change_emotion", "set_state", "wave", "move_left_arm", "move_right_arm"):
        ok, reason = validator.validate_action(name, {})
        assert ok, f"{name} should be allowed: {reason}"


def test_safety_validator_rejects_unknown_action() -> None:
    validator = ActionSafetyValidator()
    ok, reason = validator.validate_action("fly", {})
    assert not ok
    assert "unknown" in reason


def test_existing_action_indices_unchanged() -> None:
    """Expanding the space must not shift the original 10 indices."""
    space = deskbot_action_space()
    original = {
        0: "look_left",
        1: "look_right",
        2: "look_center",
        3: "look_up",
        4: "look_down",
        5: "blink",
        6: "wink",
        7: "celebrate",
        8: "sleep",
        9: "look_around",
    }
    for idx, name in original.items():
        assert space.get(idx).name == name
