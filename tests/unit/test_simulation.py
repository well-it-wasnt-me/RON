"""Tests for the :class:`SimulationDriver`."""

from __future__ import annotations

from robot.body_language.requests import ArmsOpen
from robot.face.themes import get_theme
from robot.simulation.driver import SimulationDriver


def test_simulation_driver_builds() -> None:
    d = SimulationDriver(width=120, height=160, face_size=120, fps=15)
    assert d.face is not None
    assert d.body is not None


def test_simulation_step_produces_composite_frame() -> None:
    d = SimulationDriver(width=120, height=160, face_size=120, fps=15)
    frame = d.step()
    assert frame.width == 120
    assert frame.height == 160
    assert len(frame.pixels) == 120 * 160 * 3


def test_simulation_uses_the_same_engines() -> None:
    """The simulation must NOT have its own face/body logic - it composes
    the production engines so behaviour is identical to hardware mode."""
    from robot.face.components import MouthShape

    d = SimulationDriver(width=120, height=160, face_size=120, fps=15)
    # Set an emotion; both the face.current and a rendered frame should
    # reflect it. Step several times to let the 0.3s timeline finish.
    d.face.set_emotion("happy")
    for _ in range(10):
        d.step()
    assert d.face.current.mouth.shape is MouthShape.SMILE


def test_simulation_theme_can_be_changed() -> None:
    d = SimulationDriver(width=120, height=160, face_size=120, fps=15)
    d.theme = get_theme("cute")
    d.face.theme = d.theme
    for _ in range(5):
        frame = d.step()
    # Cute theme has a different background colour than the default
    assert tuple(frame.pixels[:3]) == (255, 240, 245)


def test_simulation_body_changes_pose() -> None:
    d = SimulationDriver(width=120, height=160, face_size=120, fps=15)
    d.body.perform_sync(ArmsOpen(amount=15.0))
    assert d.body.snapshot().get("left_arm") != 90.0
