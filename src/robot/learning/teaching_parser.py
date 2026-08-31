"""Constrained parser for human teaching instructions (no LLM).

``parse_teaching_instruction`` recognises a single, deliberately narrow form of
spoken instruction:

    "RON, when I <gesture>, <action>"

e.g. "ron when i wave, wave back" -> ``DemonstrationSpec(trigger_gesture="wave",
desired_action="wave")``.

This is a *constrained* parser built on a fixed regex and the action space's
registered names — it is **not** unrestricted LLM generation. The action half
must literally match one of the action-space names (the 16 built-ins); the
gesture half is a single word. Anything else returns ``None`` so the utterance
falls through to the normal conversation turn. This keeps the LLM out of the
safety-critical "what action should the robot learn" decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from robot.learning.action_learning import ActionSpace

#: ``when I <gesture> [,] [you] <action> [back|too|...]``. The gesture is a
#: single word; the action must be one of the action-space names, matched as a
#: whole word so "waving" does not satisfy "wave". Matching is case-insensitive
#: and tolerates a leading "ron" / "hey ron".
_INSTRUCTION_RE = re.compile(
    r"\bwhen\s+i\s+(?P<gesture>[a-z_]+)\b.*?\b(?P<action>[a-z_]+)\b",
    re.IGNORECASE,
)

#: Stopwords that may appear in the action slot but are not actions themselves
#: (e.g. "when I wave, you wave back" — "back" is not an action). The action
#: must be a registered action name, so these are only a fallback guard.
_ACTION_STOPWORDS = frozenset({"back", "too", "please", "then", "do", "should"})


@dataclass(frozen=True, slots=True)
class DemonstrationSpec:
    """A parsed teaching demonstration.

    Attributes
    ----------
    trigger_gesture:
        The gesture name that should trigger the demonstration (e.g. "wave").
        Compared verbatim (case-insensitive) against a ``GestureDetected``
        event's ``gesture`` field.
    desired_action:
        The action-space name to demonstrate when the trigger fires (e.g.
        "wave").
    desired_action_index:
        The resolved action-space index for ``desired_action``.
    """

    trigger_gesture: str
    desired_action: str
    desired_action_index: int


def parse_teaching_instruction(
    text: str,
    action_space: ActionSpace,
) -> DemonstrationSpec | None:
    """Parse a constrained teaching instruction, or return ``None``.

    Returns ``None`` when the utterance is not a teaching instruction, the
    gesture is missing, or the named action is not a registered action-space
    action. Never raises.
    """
    if not text:
        return None

    # Map lowercased name -> true name + index for a case-insensitive lookup.
    name_to_index: dict[str, tuple[str, int]] = {}
    for i in range(action_space.size):
        nm = action_space.get(i).name
        name_to_index[nm.lower()] = (nm, i)

    for match in _INSTRUCTION_RE.finditer(text):
        gesture = match.group("gesture").lower().strip()
        # The action slot is greedy-tolerant: scan the trailing tokens after the
        # gesture for the first registered action name. Action names may be
        # multi-word ("look left", "move left arm"), so match spoken forms
        # (underscores as spaces), longest first.
        tail_start = match.end("gesture")
        action_name = _first_action_in_rest(text[tail_start:], name_to_index)
        if action_name is None:
            continue
        true_name, idx = name_to_index[action_name]
        if not gesture or gesture in _ACTION_STOPWORDS:
            continue
        return DemonstrationSpec(
            trigger_gesture=gesture,
            desired_action=true_name,
            desired_action_index=idx,
        )
    return None


def _first_action_in_rest(
    rest: str,
    name_to_index: dict[str, tuple[str, int]],
) -> str | None:
    """Return the first registered action name appearing in ``rest``.

    Action names may be spoken with spaces (``"look left"`` for
    ``look_left``); both the underscore form and the space form are accepted,
    matched as whole phrases (longest names first so ``move left arm`` wins
    over a shorter prefix).
    """
    rest_lower = rest.lower()
    # Build candidate matchers sorted by length descending.
    candidates = sorted(name_to_index.keys(), key=len, reverse=True)
    for name in candidates:
        spoken = name.replace("_", " ")
        pattern = r"\b" + re.escape(spoken).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, rest_lower):
            return name
    return None


__all__ = ["DemonstrationSpec", "parse_teaching_instruction"]
