"""Tests for the :class:`FaceModel`."""

from __future__ import annotations

from robot.face.components import (
    Cheeks,
    CheekState,
    Eye,
    Eyebrow,
    EyebrowShape,
    Eyelids,
    Gaze,
    Mouth,
    MouthShape,
)
from robot.face.model import (
    ArmPose,
    BodyLanguageHint,
    FaceModel,
    HeadTilt,
)


def test_default_model_is_neutral() -> None:
    m = FaceModel()
    assert m.width == 240
    assert m.height == 240
    assert m.left_eye.openness == 1.0
    assert m.eyelids.top == 0.0
    assert m.cheeks.state is CheekState.NONE


def test_model_is_immutable() -> None:
    m = FaceModel()
    try:
        m.width = 100  # type: ignore[misc]
    except Exception:
        return  # expected
    # If we get here, the model is mutable - fail
    raise AssertionError("FaceModel should be frozen")


def test_with_eyes_replaces_one_eye() -> None:
    m = FaceModel()
    new_left = Eye(gaze=Gaze(0.5, 0.5))
    m2 = m.with_eyes(left_eye=new_left)
    assert m2.left_eye.gaze.x == 0.5
    assert m2.right_eye is m.right_eye  # unchanged


def test_with_eyelids() -> None:
    m = FaceModel()
    m2 = m.with_eyelids(Eyelids(top=0.5))
    assert m2.eyelids.top == 0.5
    assert m2.eyelids.bottom == 0.0


def test_with_eyebrows() -> None:
    m = FaceModel()
    angry = Eyebrow(shape=EyebrowShape.ANGRY)
    m2 = m.with_eyebrows(left=angry)
    assert m2.left_eyebrow.shape is EyebrowShape.ANGRY
    assert m2.right_eyebrow.shape is m.right_eyebrow.shape


def test_with_mouth() -> None:
    m = FaceModel()
    m2 = m.with_mouth(Mouth(shape=MouthShape.SMILE, openness=0.4, width=0.7))
    assert m2.mouth.shape is MouthShape.SMILE


def test_with_cheeks() -> None:
    m = FaceModel()
    m2 = m.with_cheeks(Cheeks(state=CheekState.BRIGHT, intensity=0.9))
    assert m2.cheeks.state is CheekState.BRIGHT


def test_with_body_hint() -> None:
    m = FaceModel()
    hint = BodyLanguageHint(head_tilt=HeadTilt.EXCITED, arm_pose=ArmPose.WIDE)
    m2 = m.with_body_hint(hint)
    assert m2.body_hint.head_tilt is HeadTilt.EXCITED
    assert m2.body_hint.arm_pose is ArmPose.WIDE


def test_with_transform() -> None:
    m = FaceModel()
    m2 = m.with_transform(bounce=0.5, squash=1.2)
    assert m2.bounce == 0.5
    assert m2.squash == 1.2
