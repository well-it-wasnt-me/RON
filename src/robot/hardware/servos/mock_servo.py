"""Mock servo bus used for tests and headless development."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from robot.errors import ServoError
from robot.interfaces.servo import Servo
from robot.logging import get_logger

_log = get_logger("hardware.servos.mock")


@dataclass(slots=True)
class MockServo:
    """Records every move command."""

    name: str
    min_angle: float = 0.0
    max_angle: float = 180.0
    _angle: float = 90.0
    _released: bool = False
    history: list[tuple[float, float]] = field(default_factory=list)  # (angle, duration_s)

    @property
    def angle(self) -> float:
        return self._angle

    async def move_to(self, angle: float, duration_s: float = 0.4) -> None:
        if not (self.min_angle <= angle <= self.max_angle):
            raise ServoError(f"angle {angle} out of range [{self.min_angle}, {self.max_angle}]")
        self._angle = angle
        self._released = False
        self.history.append((angle, duration_s))
        _log.debug("servo.move_to", name=self.name, angle=angle, duration_s=duration_s)

    async def release(self) -> None:
        self._released = True


class MockServoBus:
    """Holds a collection of :class:`MockServo` instances by logical name."""

    def __init__(self, servos: Mapping[str, MockServo] | None = None) -> None:
        self._servos: dict[str, MockServo] = dict(servos or {})

    def add(self, servo: MockServo) -> MockServo:
        self._servos[servo.name] = servo
        return servo

    def get(self, name: str) -> MockServo:
        try:
            return self._servos[name]
        except KeyError as exc:
            raise ServoError(f"servo not found: {name!r}") from exc

    def all(self) -> list[MockServo]:
        return list(self._servos.values())

    def all_angles(self) -> dict[str, float]:
        return {name: s.angle for name, s in self._servos.items()}

    async def release_all(self) -> None:
        for servo in self._servos.values():
            await servo.release()


# Make MockServo satisfy the Servo Protocol at runtime.
_ = Servo


__all__ = ["MockServo", "MockServoBus"]
