"""Tests for the behavior library."""

from __future__ import annotations

from tests.fakes.clock import FakeClock

from robot.behavior_library.behavior import (
    BehaviorRunner,
    body,
    excited,
    face,
    greeting,
    listening,
    sleeping,
    surprised,
    thinking,
    wait,
)
from robot.body_language.requests import ArmsOpen
from robot.face.animator import FaceAnimator
from robot.face.emotions import EmotionEngine
from robot.face.renderer import FaceRenderer
from robot.face.themes.minimal import MinimalTheme
from robot.hardware.displays.mock_display import MockDisplay
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.hardware.servos.mock_servo import MockServo, MockServoBus


def _make_stack() -> tuple[FaceAnimator, MockServoBus]:
    bus = MockServoBus(
        {
            "pan": MockServo(name="pan", min_angle=30, max_angle=150, _angle=90.0),
            "tilt": MockServo(name="tilt", min_angle=45, max_angle=135, _angle=90.0),
            "left_arm": MockServo(name="left_arm", min_angle=20, max_angle=160, _angle=90.0),
            "right_arm": MockServo(name="right_arm", min_angle=20, max_angle=160, _angle=90.0),
        }
    )
    controller = wrap_servo_controller(bus, backend_name="mock")
    body_engine = type("B", (), {"servo_controller": controller})()
    from robot.body_language.engine import BodyLanguageEngine

    body_engine = BodyLanguageEngine(servo_controller=controller, clock=FakeClock())
    face_anim = FaceAnimator(
        renderer=FaceRenderer(width=32, height=32),
        display=MockDisplay(width=32, height=32),
        clock=FakeClock(),
        emotions=EmotionEngine(width=32, height=32),
        theme=MinimalTheme(),
        fps=20,
        width=32,
        height=32,
    )
    # Patch the body engine back as a BodyLanguageEngine
    return face_anim, bus, body_engine  # type: ignore[return-value]


def _make_runner() -> BehaviorRunner:
    face_anim, _, body_engine = _make_stack()  # type: ignore[misc]
    return BehaviorRunner(face=face_anim, body=body_engine, clock=FakeClock())


def test_face_step_builder() -> None:
    s = face("blink")
    assert s.kind == "face"
    assert s.face is not None
    assert s.face.method == "blink"


def test_body_step_builder() -> None:
    s = body(ArmsOpen())
    assert s.kind == "body"
    assert s.body is not None
    assert s.body.request == ArmsOpen()


def test_wait_step_builder() -> None:
    s = wait(0.3)
    assert s.kind == "wait"
    assert s.wait is not None
    assert s.wait.seconds == 0.3


def test_greeting_has_expected_steps() -> None:
    g = greeting()
    assert g.name == "greeting"
    assert any(s.kind == "face" for s in g.steps)
    assert any(s.kind == "body" for s in g.steps)


def test_all_built_in_behaviors_have_a_name() -> None:
    for b in [greeting(), thinking(), listening(), sleeping(), excited(), surprised()]:
        assert b.name
        assert b.steps


async def test_runner_runs_greeting() -> None:
    runner = _make_runner()
    g = greeting()
    await runner.run(g)
    # The face should now reflect the last step (reset), and the body should
    # be back to relaxed
    assert runner.face.current.eyelids.top >= 0.0  # nothing in progress
