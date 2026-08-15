"""Tests for the Face components (eyes, eyebrows, mouth, overlays, …)."""

from __future__ import annotations

from robot.face.components import (
    Accessory,
    AccessoryKind,
    Cheeks,
    CheekState,
    Eye,
    Eyebrow,
    EyebrowShape,
    Eyelids,
    FacePalette,
    Gaze,
    Mouth,
    MouthShape,
    Overlay,
    OverlayKind,
    Point,
)


def test_gaze_clamps() -> None:
    g = Gaze(x=2.0, y=-2.0)
    assert g.x == 2.0  # raw values are not clamped (the model clamps elsewhere)
    assert g.y == -2.0
    c = g.clamped()
    assert c.x == 1.0
    assert c.y == -1.0


def test_eye_defaults() -> None:
    e = Eye()
    assert e.gaze.x == 0.0
    assert e.openness == 1.0
    assert e.pupil_dilation == 0.5


def test_eyelids_default_open() -> None:
    e = Eyelids()
    assert e.top == 0.0
    assert e.bottom == 0.0


def test_eyebrow_shapes() -> None:
    for shape in EyebrowShape:
        b = Eyebrow(shape=shape)
        assert b.shape is shape


def test_mouth_shapes() -> None:
    for shape in MouthShape:
        m = Mouth(shape=shape)
        assert m.shape is shape


def test_overlay_kinds() -> None:
    for kind in OverlayKind:
        o = Overlay(kind=kind)
        assert o.kind is kind


def test_accessory_kinds() -> None:
    for kind in AccessoryKind:
        a = Accessory(kind=kind)
        assert a.kind is kind


def test_cheek_states() -> None:
    for state in CheekState:
        c = Cheeks(state=state)
        assert c.state is state


def test_palette_defaults() -> None:
    p = FacePalette()
    assert p.background == (10, 10, 20)
    assert p.sclera == (245, 245, 235)


def test_point() -> None:
    pt = Point(0.5, -0.5)
    assert pt.x == 0.5
    assert pt.y == -0.5
