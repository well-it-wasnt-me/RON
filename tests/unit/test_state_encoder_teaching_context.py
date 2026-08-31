"""Phase 1: teaching/gesture/conversation context in the state vector.

These tests pin the load-bearing invariants of the repurposed reserved
block ``[51..61)``:

* ``STATE_SIZE`` stays 91 and ``multimodal_size(5)`` stays 570 — repurposing
  the reserved slots must NOT change vector sizes.
* The new context fields round-trip through :class:`Observation` (the
  ``from_encoder`` → ``to_vector`` path rebuilds a *fresh* encoder, so new
  fields must be copied both directions or they silently zero out).
* The gesture slot is a proper one-hot with exactly one ``1.0``.
"""

from __future__ import annotations

from robot.learning.multimodal import multimodal_size
from robot.learning.observation import Observation
from robot.learning.state_encoder import (
    ENCODER_VERSION,
    STATE_SIZE,
    StateEncoder,
)


def test_state_size_unchanged_at_91() -> None:
    """Repurposing reserved slots does not change STATE_SIZE."""
    assert STATE_SIZE == 91
    assert len(StateEncoder().encode()) == 91


def test_multimodal_size_unchanged_at_570() -> None:
    """The 570-dim multimodal vector is preserved (history repeats 91-vectors)."""
    assert multimodal_size(5) == 570


def test_encoder_version_bumped() -> None:
    """The layout semantics of [51..61) changed; the version must reflect that."""
    assert ENCODER_VERSION >= 2


def test_teaching_context_slots_encode() -> None:
    """The teaching-context flags land at slots [51..54)."""
    enc = StateEncoder()
    enc.update_teaching_context(True)
    enc.update_interaction_active(True)
    enc.update_person_present(True)
    vec = enc.encode()
    assert vec[51] == 1.0  # teaching_context
    assert vec[52] == 1.0  # interaction_active
    assert vec[53] == 1.0  # person_present
    # Defaults are 0.0.
    vec0 = StateEncoder().encode()
    assert vec0[51] == 0.0
    assert vec0[52] == 0.0
    assert vec0[53] == 0.0


def test_gesture_one_hot_has_exactly_one_set() -> None:
    """Each gesture name produces a one-hot with exactly one 1.0 at [54..59)."""
    gestures = ["none", "wave", "point", "open_hand", "other"]
    for i, gesture in enumerate(gestures):
        enc = StateEncoder()
        enc.update_gesture(gesture)
        vec = enc.encode()
        onehot = vec[54:59]
        assert sum(1.0 for v in onehot if v == 1.0) == 1, (
            f"gesture {gesture!r} should set exactly one one-hot slot, got {onehot}"
        )
        assert onehot[i] == 1.0, f"gesture {gesture!r} should set slot {i}, got {onehot}"
        # All other slots in the one-hot are 0.0.
        assert sum(onehot) == 1.0


def test_unknown_gesture_falls_back_to_none_slot() -> None:
    """An unrecognised gesture name encodes to the 'none' one-hot slot."""
    enc = StateEncoder()
    enc.update_gesture("flerp")
    vec = enc.encode()
    assert vec[54] == 1.0  # none
    assert sum(vec[54:59]) == 1.0


def test_conversation_turn_and_last_action_normalised() -> None:
    """Slots [59] and [60] are normalised counts in [0, 1]."""
    enc = StateEncoder()
    enc.update_conversation_turn(5)
    enc.update_last_action(8, action_space_size=16)
    vec = enc.encode()
    assert vec[59] == 0.5  # 5 / 10
    assert vec[60] == 0.5  # 8 / 16

    # Capping: 20 turns → 1.0.
    enc.update_conversation_turn(20)
    assert enc.encode()[59] == 1.0

    # Negative last-action (none) → 0.0.
    enc2 = StateEncoder()
    enc2.update_last_action(-1)
    assert enc2.encode()[60] == 0.0


def test_reserved_slots_after_context_stay_zero() -> None:
    """Slots [61..91) remain reserved/zero."""
    enc = StateEncoder()
    enc.update_teaching_context(True)
    enc.update_gesture("wave")
    vec = enc.encode()
    assert all(v == 0.0 for v in vec[61:91])


def test_observation_round_trip_preserves_teaching_context() -> None:
    """New context fields survive the Observation from_encoder→to_vector round-trip.

    ``Observation.to_vector`` rebuilds a fresh ``StateEncoder`` and copies
    fields back. If a new field is not copied in both directions it
    silently zeros out (size-91 still passes, but the value is lost).
    """
    enc = StateEncoder()
    enc.update_teaching_context(True)
    enc.update_interaction_active(True)
    enc.update_person_present(True)
    enc.update_gesture("wave")
    enc.update_conversation_turn(3)
    enc.update_last_action(13, action_space_size=16)

    direct = enc.encode()
    via_observation = Observation.from_encoder(enc).to_vector()

    assert len(via_observation) == STATE_SIZE
    # The whole vector must match — this is the load-bearing guard.
    assert via_observation == direct
    # Specifically the teaching-context slots:
    assert via_observation[51] == 1.0
    assert via_observation[52] == 1.0
    assert via_observation[53] == 1.0
    # wave is index 1 in (none, wave, point, open_hand, other) → slot 55.
    assert via_observation[55] == 1.0
    assert sum(via_observation[54:59]) == 1.0
    assert via_observation[59] == 0.3  # 3 / 10
    assert via_observation[60] == 13 / 16


def test_reset_clears_teaching_context() -> None:
    """reset() restores the teaching-context fields to defaults."""
    enc = StateEncoder()
    enc.update_teaching_context(True)
    enc.update_interaction_active(True)
    enc.update_person_present(True)
    enc.update_gesture("wave")
    enc.update_conversation_turn(7)
    enc.update_last_action(2)
    enc.reset()
    vec = enc.encode()
    assert vec[51] == 0.0
    assert vec[52] == 0.0
    assert vec[53] == 0.0
    assert vec[54] == 1.0  # gesture back to "none"
    assert vec[59] == 0.0
    assert vec[60] == 0.0


def test_reset_teaching_context_keeps_state() -> None:
    """reset_teaching_context() clears only the context, not emotions/state."""
    enc = StateEncoder()
    from robot.events.events import EmotionName

    enc.update_emotion(EmotionName.HAPPY, 0.9)
    enc.update_gesture("wave")
    enc.reset_teaching_context()
    vec = enc.encode()
    # Emotion survives.
    happy_idx = list(EmotionName).index(EmotionName.HAPPY)
    assert vec[happy_idx] == 0.9
    # Gesture cleared to none.
    assert vec[54] == 1.0
