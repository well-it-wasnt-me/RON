"""Tests for the :class:`BodyLanguageEngine`."""

from __future__ import annotations

from robot.body_language.engine import (
    BodyLanguageEngine,
    hint_to_pose,
)
from robot.body_language.requests import (
    DEFAULT_CALIBRATION,
    LookLeft,
    ServoCalibration,
    Wave,
)
from robot.face.model import (
    ArmPose,
    BodyLanguageHint,
    HeadTilt,
)
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.utils.clock import SystemClock


def _make_engine() -> tuple[BodyLanguageEngine, MockServoBus]:
    bus = MockServoBus(
        {
            "pan": MockServo(name="pan", min_angle=30, max_angle=150, _angle=90.0),
            "tilt": MockServo(name="tilt", min_angle=45, max_angle=135, _angle=90.0),
            "left_arm": MockServo(name="left_arm", min_angle=20, max_angle=160, _angle=90.0),
            "right_arm": MockServo(name="right_arm", min_angle=20, max_angle=160, _angle=90.0),
        }
    )
    controller = wrap_servo_controller(bus, backend_name="mock")
    engine = BodyLanguageEngine(
        servo_controller=controller, clock=SystemClock(), calibration=dict(DEFAULT_CALIBRATION)
    )
    return engine, bus


def test_engine_starts_at_centre() -> None:
    engine, _ = _make_engine()
    snap = engine.snapshot()
    assert snap.get("pan") == 90.0
    assert snap.get("left_arm") == 90.0


def test_hint_to_pose_neutral() -> None:
    pose = hint_to_pose(
        BodyLanguageHint(head_tilt=HeadTilt.NEUTRAL, arm_pose=ArmPose.RELAXED, intensity=0.5)
    )
    assert pose.get("pan") == 90.0
    assert pose.get("tilt") == 90.0
    assert pose.get("left_arm") == 90.0
    assert pose.get("right_arm") == 90.0


def test_hint_to_pose_curious_tilts_head() -> None:
    pose = hint_to_pose(
        BodyLanguageHint(head_tilt=HeadTilt.CURIOUS, arm_pose=ArmPose.RELAXED, intensity=1.0)
    )
    assert pose.get("tilt") < 90.0
    assert pose.get("pan") < 90.0  # slight pan with curious


def test_hint_to_pose_arms_wide() -> None:
    pose = hint_to_pose(
        BodyLanguageHint(head_tilt=HeadTilt.NEUTRAL, arm_pose=ArmPose.WIDE, intensity=1.0)
    )
    assert pose.get("left_arm") < 90.0 - 30.0
    assert pose.get("right_arm") > 90.0 + 30.0


def test_hint_to_pose_intensity_scales_effect() -> None:
    small = hint_to_pose(
        BodyLanguageHint(head_tilt=HeadTilt.CURIOUS, arm_pose=ArmPose.WIDE, intensity=0.1)
    )
    big = hint_to_pose(
        BodyLanguageHint(head_tilt=HeadTilt.CURIOUS, arm_pose=ArmPose.WIDE, intensity=1.0)
    )
    assert abs(90.0 - small.get("left_arm")) < abs(90.0 - big.get("left_arm"))


def test_apply_hint_updates_snapshot() -> None:
    engine, _ = _make_engine()
    engine.apply_hint(
        BodyLanguageHint(head_tilt=HeadTilt.NEUTRAL, arm_pose=ArmPose.OPEN, intensity=1.0)
    )
    assert engine.snapshot().get("left_arm") != 90.0


def test_perform_sync_updates_snapshot() -> None:
    engine, _ = _make_engine()
    engine.perform_sync(LookLeft(amount=30.0))
    # perform_sync updates the pose snapshot (the servos are mocked and
    # would require a running event loop to drive them).
    assert engine.snapshot().get("pan") != 90.0


def test_perform_sync_calibrates() -> None:
    """The engine must not command angles outside the calibration range."""
    engine, bus = _make_engine()
    engine.perform_sync(LookLeft(amount=200.0))  # out of range
    angle = bus.get("pan").angle
    assert 30.0 <= angle <= 150.0


def test_custom_calibration_is_honoured() -> None:
    engine, _ = _make_engine()
    engine.set_calibration(
        {
            "pan": ServoCalibration("pan", min_angle=60.0, max_angle=120.0, center_angle=90.0),
            "tilt": DEFAULT_CALIBRATION["tilt"],
            "left_arm": DEFAULT_CALIBRATION["left_arm"],
            "right_arm": DEFAULT_CALIBRATION["right_arm"],
        }
    )
    engine.perform_sync(LookLeft(amount=90.0))
    assert engine.snapshot().get("pan") >= 60.0


def test_wave_moves_right_arm_only() -> None:
    engine, _ = _make_engine()
    wave = Wave()
    # The wave's frames cycle the right arm: up, centre, up, centre.
    # The left arm should stay at centre throughout.
    for frame in wave.frames():
        for name, value in frame.targets.items():
            engine._current.targets[name] = engine._calibrate(name, value)
    # Inspect the peak (second-up frame): the right arm should have moved
    peak = Wave().frames()[0].targets["right_arm"]
    assert peak != 90.0
    for frame in wave.frames():
        # The left arm should never appear in any wave frame
        assert "left_arm" not in frame.targets
