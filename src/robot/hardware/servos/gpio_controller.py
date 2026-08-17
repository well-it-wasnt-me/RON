"""Raspberry Pi GPIO servo controller.

This implementation drives hobby servos through :class:`gpiozero.Servo`.
Using gpiozero's native servo abstraction keeps pulse generation and timing in
one well-tested hardware layer instead of duplicating it with
``PWMOutputDevice`` and hand-written duty-cycle calculations.

Wiring:

* Head pan  -> GPIO 12
* Head tilt -> GPIO 13
* Left arm -> GPIO 18
* Right arm -> GPIO 19

Servos must be powered from an external 5V supply with a common ground to the
Pi. The GPIO pins carry only the PWM control signal.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import ClassVar, Protocol, cast

from robot.config import GPIOServoConfig, ServoChannelConfig
from robot.errors import ServoError
from robot.interfaces.servo import Servo
from robot.logging import get_logger

_log = get_logger("hardware.servos.gpio")


class _ServoDevice(Protocol):
    """Minimal subset of :class:`gpiozero.Servo` used by the controller."""

    @property
    def value(self) -> float | None: ...

    @value.setter
    def value(self, value: float) -> None: ...

    def detach(self) -> None: ...

    def close(self) -> None: ...


def _default_servo_factory(
    pin: int, channel: ServoChannelConfig, frequency: int
) -> _ServoDevice:  # pragma: no cover - hardware
    """Create a real gpiozero ``Servo`` for one GPIO pin."""
    try:
        from gpiozero import Servo as GpioZeroServo
    except Exception as exc:
        raise ServoError(
            f"gpiozero is not available on this platform: {exc!r}. "
            "Install the 'hardware' extra or run with DESKBOT_SERVOS__BACKEND=mock."
        ) from exc

    if frequency <= 0:
        raise ServoError(f"servo frequency must be positive, got {frequency}")

    return cast(
        "_ServoDevice",
        GpioZeroServo(
            pin,
            initial_value=None,
            min_pulse_width=channel.min_pulse_us / 1_000_000.0,
            max_pulse_width=channel.max_pulse_us / 1_000_000.0,
            frame_width=1.0 / frequency,
        ),
    )


@dataclass(slots=True)
class _GPIOServo:
    """Logical servo wrapper around a gpiozero ``Servo``."""

    COMMAND_EPSILON: ClassVar[float] = 0.005

    name: str
    channel: ServoChannelConfig
    device: _ServoDevice
    _angle: float = 0.0
    _released: bool = True
    _commanded_value: float | None = None

    @property
    def angle(self) -> float:
        return self._angle

    async def move_to(self, angle: float, duration_s: float = 0.4) -> None:
        """Move the servo to an angle using gpiozero's [-1, 1] value range."""
        min_angle = self.channel.min_angle_deg
        max_angle = self.channel.max_angle_deg
        if not min_angle <= angle <= max_angle:
            raise ServoError(
                f"angle {angle} out of range [{min_angle}, {max_angle}] for servo {self.name!r}"
            )

        span = max_angle - min_angle
        if span <= 0:
            raise ServoError(
                f"invalid angle range [{min_angle}, {max_angle}] for servo {self.name!r}"
            )

        fraction = (angle - min_angle) / span
        if self.channel.inverted:
            fraction = 1.0 - fraction

        value = (fraction * 2.0) - 1.0

        if (
            not self._released
            and self._commanded_value is not None
            and abs(value - self._commanded_value) < self.COMMAND_EPSILON
        ):
            self._angle = angle
            return

        self.device.value = value
        self._angle = angle
        self._commanded_value = value
        self._released = False
        _log.debug(
            "servo.move_to",
            backend="gpio",
            name=self.name,
            angle=angle,
            value=value,
        )

        if duration_s and duration_s > 0:
            await asyncio.sleep(duration_s)

    async def release(self) -> None:
        with contextlib.suppress(Exception):
            self.device.detach()
        self._released = True
        self._commanded_value = None


class RaspberryPiGPIOServoController:
    """Drive hobby servos directly from the Pi GPIO header using gpiozero."""

    BACKEND_NAME: str = "gpio"

    def __init__(
        self,
        config: GPIOServoConfig,
        *,
        servo_factory: Callable[[int, ServoChannelConfig, int], _ServoDevice] | None = None,
    ) -> None:
        self._config = config
        self._servo_factory = servo_factory or _default_servo_factory
        self._servos: dict[str, _GPIOServo] = {}
        self._closed = False
        self._initialise()

    @property
    def backend_name(self) -> str:
        return self.BACKEND_NAME

    def get(self, name: str) -> Servo:
        try:
            return self._servos[name]
        except KeyError as exc:
            raise ServoError(f"servo not found: {name!r}") from exc

    def all(self) -> list[Servo]:
        return list(self._servos.values())

    async def release_all(self) -> None:
        for servo in self._servos.values():
            await servo.release()

    async def close(self) -> None:
        if self._closed:
            return
        for servo in self._servos.values():
            with contextlib.suppress(Exception):
                servo.device.close()
        self._closed = True
        _log.info("servo.gpio.closed")

    def _initialise(self) -> None:
        """Build one gpiozero Servo per logical servo name."""
        created: list[_GPIOServo] = []
        try:
            for logical_name, channel in self._resolve_channels():
                pin_number = self._resolve_pin(logical_name)
                self._validate_pin(pin_number)
                device = self._servo_factory(pin_number, channel, self._config.frequency)
                servo = _GPIOServo(
                    name=logical_name,
                    channel=channel,
                    device=device,
                    _angle=channel.center_angle_deg,
                )
                self._servos[logical_name] = servo
                created.append(servo)
        except Exception:
            for servo in created:
                with contextlib.suppress(Exception):
                    servo.device.close()
            raise
        _log.info(
            "servo.gpio.initialised",
            count=len(self._servos),
            frequency=self._config.frequency,
        )

    def _resolve_channels(self) -> Iterator[tuple[str, ServoChannelConfig]]:
        defaults = {
            "pan": ServoChannelConfig(min_angle_deg=-90.0, max_angle_deg=90.0),
            "tilt": ServoChannelConfig(min_angle_deg=-30.0, max_angle_deg=30.0),
            "left_arm": ServoChannelConfig(min_angle_deg=0.0, max_angle_deg=180.0),
            "right_arm": ServoChannelConfig(min_angle_deg=0.0, max_angle_deg=180.0),
        }
        for name, cfg in self._config.channels.items():
            yield name, cfg
        for name, cfg in defaults.items():
            if name not in self._config.channels:
                yield name, cfg

    def _resolve_pin(self, logical_name: str) -> int:
        mapping = {
            "pan": self._config.pins.pan,
            "tilt": self._config.pins.tilt,
            "left_arm": self._config.pins.left_arm,
            "right_arm": self._config.pins.right_arm,
        }
        if logical_name in mapping:
            return int(mapping[logical_name])
        channel = self._config.channels.get(logical_name)
        if channel is not None and channel.gpio_pin is not None:
            return int(channel.gpio_pin)
        raise ServoError(
            f"no GPIO pin configured for servo {logical_name!r}; "
            f"set 'servos.gpio.pins.{logical_name}' or channel.gpio_pin"
        )

    @staticmethod
    def _validate_pin(pin: int) -> None:
        if not (2 <= pin <= 27):
            raise ServoError(
                f"GPIO pin {pin} is outside the valid BCM range [2, 27] for the Pi 5 header"
            )


__all__ = ["RaspberryPiGPIOServoController"]
