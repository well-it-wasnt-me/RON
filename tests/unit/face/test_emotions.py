"""Tests for the :class:`EmotionEngine`."""

from __future__ import annotations

import pytest

from robot.face.emotions import EmotionEngine
from robot.face.model import FaceModel


def test_engine_has_all_spec_emotions() -> None:
    e = EmotionEngine()
    for required in [
        "neutral",
        "happy",
        "curious",
        "thinking",
        "sleepy",
        "embarrassed",
        "excited",
        "sad",
        "surprised",
        "angry",
    ]:
        assert required in e.available(), f"missing emotion {required!r}"


def test_unknown_emotion_raises() -> None:
    e = EmotionEngine()
    with pytest.raises(KeyError):
        e.build("nope")


def test_neutral_model_is_default() -> None:
    e = EmotionEngine(width=64, height=64)
    m = e.build("neutral")
    assert isinstance(m, FaceModel)
    assert m.width == 64


def test_happy_includes_a_smile() -> None:
    from robot.face.components import MouthShape

    e = EmotionEngine()
    m = e.build("happy")
    assert m.mouth.shape is MouthShape.SMILE


def test_surprised_includes_wide_open_mouth() -> None:
    from robot.face.components import MouthShape

    e = EmotionEngine()
    m = e.build("surprised")
    assert m.mouth.shape is MouthShape.WIDE_OPEN


def test_angry_includes_angry_eyebrows() -> None:
    from robot.face.components import EyebrowShape

    e = EmotionEngine()
    m = e.build("angry")
    assert m.left_eyebrow.shape is EyebrowShape.ANGRY
    assert m.right_eyebrow.shape is EyebrowShape.ANGRY


def test_emotions_produce_body_hints() -> None:
    from robot.face.model import ArmPose, HeadTilt

    e = EmotionEngine()
    for name in e.available():
        m = e.build(name)
        # Each emotion emits a body hint with valid head_tilt + arm_pose
        assert isinstance(m.body_hint.head_tilt, HeadTilt)
        assert isinstance(m.body_hint.arm_pose, ArmPose)
        assert 0.0 <= m.body_hint.intensity <= 1.0


def test_custom_emotion_can_be_registered() -> None:
    from robot.face.components import (
        Eye,
        Eyebrow,
        EyebrowShape,
        Eyelids,
        Gaze,
        Mouth,
        MouthShape,
    )
    from robot.face.emotions import _EmotionDef
    from robot.face.model import ArmPose, BodyLanguageHint, HeadTilt

    e = EmotionEngine()
    definition = _EmotionDef(
        eyes=(Eye(gaze=Gaze(0.0, 0.0)), Eye(gaze=Gaze(0.0, 0.0))),
        eyelids=Eyelids(),
        eyebrows=(Eyebrow(shape=EyebrowShape.NEUTRAL), Eyebrow(shape=EyebrowShape.NEUTRAL)),
        mouth=Mouth(shape=MouthShape.SMILE, openness=0.3, width=0.5),
        cheeks=type("Cheeks", (), {"state": "none", "intensity": 0.0})(),
        overlay=type(
            "Overlay",
            (),
            {"kind": "none", "position": None, "size": 0, "rotation": 0, "color": (0, 0, 0)},
        )(),
        body_hint=BodyLanguageHint(
            head_tilt=HeadTilt.NEUTRAL, arm_pose=ArmPose.RELAXED, intensity=0.5
        ),
    )
    e.register("custom", definition)
    assert "custom" in e.available()
