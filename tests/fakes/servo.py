"""Fake servos and fake servo backends for tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from robot.errors import ServoError
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.interfaces.servo import ServoController


@dataclass
class FakeServo:
    name: str
    min_angle: float = 0.0
    max_angle: float = 180.0
    _angle: float = 90.0
    moves: list[tuple[float, float]] = field(default_factory=list)
    released: bool = False

    @property
    def angle(self) -> float:
        return self._angle

    async def move_to(self, angle: float, duration_s: float = 0.4) -> None:
        if not (self.min_angle <= angle <= self.max_angle):
            raise ServoError(f"angle {angle} out of range")
        self._angle = angle
        self.moves.append((angle, duration_s))

    async def release(self) -> None:
        self.released = True


class FakeServoBus:
    def __init__(self) -> None:
        self.servos: dict[str, FakeServo] = {}

    def add(self, servo: FakeServo) -> FakeServo:
        self.servos[servo.name] = servo
        return servo

    def get(self, name: str) -> FakeServo:
        if name not in self.servos:
            raise ServoError(f"servo not found: {name!r}")
        return self.servos[name]

    def all(self) -> list[FakeServo]:
        return list(self.servos.values())


@dataclass
class FakeGpioServo:
    """Minimal fake matching gpiozero.Servo's value/detach/close API."""

    pin: int
    min_pulse_width: float
    max_pulse_width: float
    frame_width: float
    _value: float | None = None
    detached: bool = False
    closed: bool = False
    write_count: int = 0

    @property
    def value(self) -> float | None:
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        if not -1.0 <= value <= 1.0:
            raise ValueError("value out of range")
        self._value = value
        self.detached = False
        self.write_count += 1

    def detach(self) -> None:
        self.detached = True

    def close(self) -> None:
        self.closed = True


class FakeGpioServoFactory:
    """Drop-in factory for ``gpiozero.Servo`` in unit tests."""

    def __init__(self) -> None:
        self.servos: dict[int, FakeGpioServo] = {}
        self.calls: list[tuple[int, float, float, float, int]] = []

    def __call__(self, pin: int, channel: Any, frequency: int) -> FakeGpioServo:
        servo = FakeGpioServo(
            pin=pin,
            min_pulse_width=channel.min_pulse_us / 1_000_000.0,
            max_pulse_width=channel.max_pulse_us / 1_000_000.0,
            frame_width=1.0 / frequency,
        )
        self.calls.append(
            (pin, servo.min_pulse_width, servo.max_pulse_width, servo.frame_width, frequency)
        )
        self.servos[pin] = servo
        return servo


@dataclass
class FakePcaDevice:
    frequency: int = 50
    duties: dict[int, float] = field(default_factory=dict)
    closed: bool = False

    def duty(self, channel: int, fraction: float) -> None:
        self.duties[channel] = float(fraction)

    def close(self) -> None:
        self.closed = True


def make_fake_pca_factory() -> tuple[Callable[[Any], FakePcaDevice], list[FakePcaDevice]]:
    created: list[FakePcaDevice] = []

    def factory(config: object) -> FakePcaDevice:
        device = FakePcaDevice(frequency=getattr(config, "frequency", 50))
        created.append(device)
        return device

    return factory, created


def make_servo_controller_from_fakes(bus: FakeServoBus | None = None) -> ServoController:
    return wrap_servo_controller(bus if bus is not None else FakeServoBus(), backend_name="fake")


__all__ = [
    "FakeGpioServo",
    "FakeGpioServoFactory",
    "FakePcaDevice",
    "FakeServo",
    "FakeServoBus",
    "make_fake_pca_factory",
    "make_servo_controller_from_fakes",
]
