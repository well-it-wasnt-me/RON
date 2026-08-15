"""Simulation driver: face + body + servo overlay, no hardware required.

The driver wires the same :class:`FaceAnimator` and
:class:`BodyLanguageEngine` used in production into a 30 FPS loop that
composites the face with a stick-figure servo visualisation and
displays the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from robot.body_language.engine import BodyLanguageEngine
from robot.body_language.requests import (
    DEFAULT_CALIBRATION,
    ServoCalibration,
)
from robot.events.bus import InMemoryEventBus
from robot.face.animator import FaceAnimator
from robot.face.emotions import EmotionEngine
from robot.face.renderer import FaceRenderer
from robot.face.themes import Theme
from robot.hardware.displays.mock_display import MockDisplay
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.interfaces.display import Display, EyeFrame
from robot.logging import get_logger
from robot.simulation.overlay import ServoOverlay
from robot.utils.clock import Clock, SystemClock

_log = get_logger("simulation.driver")


@dataclass(slots=True)
class SimulationDriver:
    """Run the full robot stack against an in-memory mock display."""

    DEFAULT_FPS: ClassVar[int] = 30

    bus: InMemoryEventBus = field(default_factory=InMemoryEventBus)
    width: int = 240
    height: int = 320  # extra height for the body diagram
    face_size: int = 240  # the face is square, drawn in the upper portion
    fps: int = DEFAULT_FPS
    clock: Clock = field(default_factory=SystemClock)
    theme: Theme | None = None

    _face: FaceAnimator = field(init=False)
    _body: BodyLanguageEngine = field(init=False)
    _display: MockDisplay = field(init=False)
    _overlay: ServoOverlay = field(init=False)
    _renderer: FaceRenderer = field(init=False)
    _stopped: bool = field(default=True, init=False)
    _servo_bus: MockServoBus | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        # The mock display receives the composite (face + overlay) frame
        self._display = MockDisplay(width=self.width, height=self.height)
        # The face engine is told the upper portion is the face
        self._renderer = FaceRenderer(width=self.face_size, height=self.face_size)
        # The mock servo bus is calibrated with the default limits
        self._servo_bus = MockServoBus(
            {
                "pan": MockServo(name="pan", min_angle=30.0, max_angle=150.0, _angle=90.0),
                "tilt": MockServo(name="tilt", min_angle=45.0, max_angle=135.0, _angle=90.0),
                "left_arm": MockServo(
                    name="left_arm", min_angle=20.0, max_angle=160.0, _angle=90.0
                ),
                "right_arm": MockServo(
                    name="right_arm", min_angle=20.0, max_angle=160.0, _angle=90.0
                ),
            }
        )
        servo_controller = wrap_servo_controller(self._servo_bus, backend_name="mock")
        self._body = BodyLanguageEngine(
            servo_controller=servo_controller,
            clock=self.clock,
            calibration=dict(DEFAULT_CALIBRATION),
        )
        # The face animator runs on the (smaller) face panel
        from robot.face.themes.minimal import MinimalTheme

        self._face = FaceAnimator(
            renderer=self._renderer,
            display=MockDisplay(width=self.face_size, height=self.face_size),
            clock=self.clock,
            emotions=EmotionEngine(width=self.face_size, height=self.face_size),
            theme=self.theme or MinimalTheme(),
            fps=self.fps,
            bus=self.bus,
            width=self.face_size,
            height=self.face_size,
        )
        self._overlay = ServoOverlay(calibration=dict(DEFAULT_CALIBRATION))

    # ------------------------------------------------------------------ properties
    @property
    def face(self) -> FaceAnimator:
        return self._face

    @property
    def body(self) -> BodyLanguageEngine:
        return self._body

    @property
    def display(self) -> Display:
        return self._display

    @property
    def is_running(self) -> bool:
        return not self._stopped

    # ------------------------------------------------------------------ public
    def step(self) -> EyeFrame:
        """Advance one frame and return the composite frame."""
        face_frame = self._face.step(drift=True)
        pose = self._body.snapshot()
        # The face is square; composite it into a taller canvas that
        # leaves room for the body diagram below.
        out = bytearray(self.width * self.height * 3)
        # Center the face in the upper portion of the canvas
        x_offset = (self.width - self.face_size) // 2
        for y in range(self.face_size):
            src_start = y * self.face_size * 3
            src_end = src_start + self.face_size * 3
            dst_start = (y * self.width + x_offset) * 3
            out[dst_start : dst_start + self.face_size * 3] = face_frame.pixels[src_start:src_end]
        composite_pixels = self._overlay.composite(out, self.width, self.height, pose)
        return EyeFrame(
            width=self.width,
            height=self.height,
            pixels=bytes(composite_pixels),
        )

    async def _step_async(self) -> EyeFrame:
        frame = self.step()
        await self._display.show(frame)
        return frame

    async def run_forever(self) -> None:
        self._stopped = False
        frame_interval = 1.0 / self.fps
        try:
            while not self._stopped:
                await self._step_async()
                await self.clock.sleep(frame_interval)
        finally:
            self._stopped = True

    def stop(self) -> None:
        self._stopped = True

    def set_calibration(self, calibration: dict[str, ServoCalibration]) -> None:
        self._body.set_calibration(calibration)
        self._overlay = ServoOverlay(calibration=calibration)


__all__ = ["SimulationDriver"]
