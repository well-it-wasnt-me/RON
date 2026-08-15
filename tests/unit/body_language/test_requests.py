"""Tests for the body-language request types and the calibration registry."""

from __future__ import annotations

import pytest

from robot.body_language.requests import (
    DEFAULT_CALIBRATION,
    ArmsOpen,
    ArmsRelax,
    Celebrate,
    HeadNod,
    HeadTiltRequest,
    LookLeft,
    LookRight,
    ServoCalibration,
    ServoFrame,
    Shrug,
    Wave,
)


def test_calibration_clamps() -> None:
    cal = ServoCalibration("pan", min_angle=30.0, max_angle=150.0, center_angle=90.0)
    assert cal.clamp(20.0) == 30.0
    assert cal.clamp(200.0) == 150.0
    assert cal.clamp(90.0) == 90.0


def test_calibration_normalised_round_trip() -> None:
    cal = ServoCalibration("pan", min_angle=30.0, max_angle=150.0, center_angle=90.0)
    assert cal.normalised(cal.center_angle) == pytest.approx(0.0, abs=1e-6)
    assert cal.normalised(150.0) == pytest.approx(1.0, abs=1e-6)
    assert cal.normalised(30.0) == pytest.approx(-1.0, abs=1e-6)


def test_calibration_inverted() -> None:
    cal = ServoCalibration("pan", min_angle=30.0, max_angle=150.0, center_angle=90.0, inverted=True)
    assert cal.from_normalised(1.0) == pytest.approx(30.0, abs=1e-6)
    assert cal.from_normalised(-1.0) == pytest.approx(150.0, abs=1e-6)


def test_default_calibration_has_four_servos() -> None:
    assert set(DEFAULT_CALIBRATION.keys()) == {"pan", "tilt", "left_arm", "right_arm"}


def test_look_left_turns_pan() -> None:
    req = LookLeft(amount=30.0)
    frames = req.frames()
    assert len(frames) >= 1
    assert frames[-1].targets["pan"] < 90.0  # turned left


def test_look_right_turns_pan() -> None:
    req = LookRight(amount=30.0)
    frames = req.frames()
    assert frames[-1].targets["pan"] > 90.0


def test_head_nod_has_two_frames() -> None:
    req = HeadNod(amplitude=15.0)
    frames = req.frames()
    assert len(frames) == 2
    assert frames[0].targets["tilt"] != frames[1].targets["tilt"]


def test_arms_relax_centre() -> None:
    req = ArmsRelax()
    frames = req.frames()
    assert frames[-1].targets["left_arm"] == 90.0
    assert frames[-1].targets["right_arm"] == 90.0


def test_arms_open_spreads() -> None:
    req = ArmsOpen(amount=20.0)
    frames = req.frames()
    assert frames[-1].targets["left_arm"] < 90.0
    assert frames[-1].targets["right_arm"] > 90.0


def test_wave_is_four_frames() -> None:
    req = Wave()
    frames = req.frames()
    assert len(frames) == 4


def test_celebrate_raises_then_returns() -> None:
    req = Celebrate()
    frames = req.frames()
    assert len(frames) == 2
    assert frames[0].targets["left_arm"] != frames[1].targets["left_arm"]


def test_shrug_raises_both_arms() -> None:
    req = Shrug()
    frames = req.frames()
    assert frames[-1].targets["left_arm"] < 90.0
    assert frames[-1].targets["right_arm"] < 90.0


def test_head_tilt_() -> None:
    req = HeadTiltRequest(direction="left", amount=15.0)
    frames = req.frames()
    assert frames[-1].targets["tilt"] < 90.0
    req = HeadTiltRequest(direction="right", amount=15.0)
    frames = req.frames()
    assert frames[-1].targets["tilt"] > 90.0


def test_servo_frame_defaults() -> None:
    f = ServoFrame()
    assert f.targets == {}
    assert f.duration_s == 0.4
    assert f.get("missing", 90.0) == 90.0
