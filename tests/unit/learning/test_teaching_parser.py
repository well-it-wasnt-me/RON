"""Phase 8: the constrained teaching-instruction parser (no LLM).

``parse_teaching_instruction`` recognises only the narrow form
``"when I <gesture>, <action>"`` and resolves ``<action>`` against the
registered action-space names. It must return ``None`` for anything that is
not a teaching instruction so the utterance falls through to the normal LLM
conversation turn — the LLM never decides what action the robot learns.
"""

from __future__ import annotations

import pytest

from robot.learning.action_learning import deskbot_action_space
from robot.learning.teaching_parser import (
    DemonstrationSpec,
    parse_teaching_instruction,
)


@pytest.fixture
def action_space() -> object:
    return deskbot_action_space()


def test_basic_wave_instruction(action_space: object) -> None:
    """'when I wave, wave back' -> trigger=wave, action=wave (index 13)."""
    spec = parse_teaching_instruction("ron when i wave, wave back", action_space)  # type: ignore[arg-type]
    assert spec is not None
    assert spec.trigger_gesture == "wave"
    assert spec.desired_action == "wave"
    assert spec.desired_action_index == 13


def test_case_insensitive(action_space: object) -> None:
    spec = parse_teaching_instruction("When I WAVE, Wave", action_space)  # type: ignore[arg-type]
    assert spec is not None
    assert spec.trigger_gesture == "wave"
    assert spec.desired_action == "wave"


def test_multi_word_action_spoken_form(action_space: object) -> None:
    """'look left' (spoken) resolves to the registered look_left action."""
    spec = parse_teaching_instruction("when i point, look left", action_space)  # type: ignore[arg-type]
    assert spec is not None
    assert spec.trigger_gesture == "point"
    assert spec.desired_action == "look_left"
    assert spec.desired_action_index == 0


def test_move_left_arm_phrase(action_space: object) -> None:
    """Longest action phrase wins: 'move left arm' over a shorter prefix."""
    spec = parse_teaching_instruction("when i wave, move left arm", action_space)  # type: ignore[arg-type]
    assert spec is not None
    assert spec.desired_action == "move_left_arm"
    assert spec.desired_action_index == 14


def test_non_teaching_utterance_returns_none(action_space: object) -> None:
    """A plain question is not a teaching instruction."""
    assert parse_teaching_instruction("what is the weather today", action_space) is None  # type: ignore[arg-type]


def test_empty_text_returns_none(action_space: object) -> None:
    assert parse_teaching_instruction("", action_space) is None  # type: ignore[arg-type]


def test_unknown_action_returns_none(action_space: object) -> None:
    """An action name that is not registered yields None (fall through)."""
    assert parse_teaching_instruction("when i wave, fly away", action_space) is None  # type: ignore[arg-type]


def test_stopword_gesture_rejected(action_space: object) -> None:
    """A stopword in the gesture slot is not a valid trigger."""
    # 'back' is a stopword, not a gesture.
    assert parse_teaching_instruction("when i back, wave", action_space) is None  # type: ignore[arg-type]


def test_returns_dataclass_instance(action_space: object) -> None:
    spec = parse_teaching_instruction("when i wave, wave", action_space)  # type: ignore[arg-type]
    assert isinstance(spec, DemonstrationSpec)
