"""PCA9685 servo controller (stub, real driver to follow).

The PCA9685 is a 16-channel, 12-bit PWM/Servo driver controlled over I2C.
This module ships a *runtime skeleton* that satisfies the
:class:`~robot.interfaces.servo.ServoController` protocol but does not
import the third-party ``adafruit_pca9685`` library yet. It is fully
testable with a fake I2C bus and the existing servo bus tests are kept
green by a ``mock``-mode fall-back.

When the hardware becomes available, drop the real driver into this file
*behind* the same :class:`PCA9685ServoController` class - the public
contract is fixed by the protocol.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

from robot.config import PCA9685ServoConfig, ServoChannelConfig
from robot.errors import ServoError
from robot.interfaces.servo import Servo
from robot.logging import get_logger

_log = get_logger("hardware.servos.pca9685")


# ---------------------------------------------------------------------------
# I2C abstraction - the only place that knows about the PCA9685 SDK.
# ---------------------------------------------------------------------------
class _Pca9685Device(Protocol):
    """Minimal subset of a PCA9685 driver we depend on."""

    @property
    def frequency(self) -> int: ...

    @frequency.setter
    def frequency(self, hz: int) -> None: ...

    def duty(self, channel: int, fraction: float) -> None: ...

    def close(self) -> None: ...


def _default_device_factory(
    config: PCA9685ServoConfig,
) -> _Pca9685Device:  # pragma: no cover - hardware
    """Create a real PCA9685 device on the configured I2C bus.

    The implementation is intentionally deferred to keep this module
    importable on non-Pi machines; uncomment when the SDK is available.
    """
    try:
        # The exact SDK call is intentionally not pinned here so this file
        # builds without the optional dependency installed. Once the driver
        # is wired in production, replace this body with the real import.
        raise NotImplementedError(
            "PCA9685 SDK is not bundled. Install adafruit-circuitpython-pca9685 "
            "and adafruit-blinka, then re-enable the driver here."
        )
    except Exception as exc:
        raise ServoError(
            f"PCA9685 backend is not available in this environment: {exc!r}. "
            "Set DESKBOT_SERVOS__BACKEND=mock or =gpio."
        ) from exc


# ---------------------------------------------------------------------------
# Single servo
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _Pca9685Servo:
    name: str
    channel_index: int
    channel: ServoChannelConfig
    device: _Pca9685Device
    _angle: float = 0.0

    @property
    def angle(self) -> float:
        return self._angle

    @staticmethod
    def angle_to_duty_fraction(
        angle_deg: float,
        *,
        min_angle: float,
        max_angle: float,
        min_pulse_us: int,
        max_pulse_us: int,
        period_us: float,
        inverted: bool,
    ) -> float:
        if max_angle == min_angle:
            raise ServoError(f"servo {min_angle=} equals max_angle; cannot interpolate")
        clamped = max(min(angle_deg, max_angle), min_angle)
        fraction = (clamped - min_angle) / (max_angle - min_angle)
        if inverted:
            fraction = 1.0 - fraction
        pulse_us = min_pulse_us + fraction * (max_pulse_us - min_pulse_us)
        duty = pulse_us / period_us
        return max(0.0, min(1.0, duty))

    async def move_to(self, angle: float, duration_s: float = 0.4) -> None:
        if not (self.channel.min_angle_deg <= angle <= self.channel.max_angle_deg):
            raise ServoError(
                f"angle {angle} out of range "
                f"[{self.channel.min_angle_deg}, {self.channel.max_angle_deg}] "
                f"for servo {self.name!r}"
            )
        period_us = 1_000_000.0 / float(self.device.frequency)
        duty = self.angle_to_duty_fraction(
            angle,
            min_angle=self.channel.min_angle_deg,
            max_angle=self.channel.max_angle_deg,
            min_pulse_us=self.channel.min_pulse_us,
            max_pulse_us=self.channel.max_pulse_us,
            period_us=period_us,
            inverted=self.channel.inverted,
        )
        self.device.duty(self.channel_index, duty)
        self._angle = angle
        _log.debug(
            "servo.move_to",
            backend="pca9685",
            name=self.name,
            channel=self.channel_index,
            angle=angle,
            duty=duty,
        )

    async def release(self) -> None:
        with contextlib.suppress(Exception):
            self.device.duty(self.channel_index, 0.0)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class PCA9685ServoController:
    """Drive up to 16 servos through a PCA9685 board over I2C."""

    BACKEND_NAME: str = "pca9685"

    def __init__(
        self,
        config: PCA9685ServoConfig,
        *,
        device_factory: Callable[[PCA9685ServoConfig], _Pca9685Device] | None = None,
    ) -> None:
        self._config = config
        self._device_factory: Callable[[PCA9685ServoConfig], _Pca9685Device] = (
            device_factory or _default_device_factory
        )
        self._device: _Pca9685Device | None = None
        self._servos: dict[str, _Pca9685Servo] = {}
        self._closed = False
        self._initialise()

    # ------------------------------------------------------------------ public
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
                if self._device is not None:
                    self._device.duty(servo.channel_index, 0.0)
        if self._device is not None:
            with contextlib.suppress(Exception):
                self._device.close()
        self._closed = True
        _log.info("servo.pca9685.closed")

    # ------------------------------------------------------------------ internals
    def _initialise(self) -> None:
        self._device = self._device_factory(self._config)
        try:
            self._device.frequency = self._config.frequency
        except Exception as exc:
            raise ServoError(f"failed to set PCA9685 frequency: {exc!r}") from exc
        for logical_name, channel in self._resolve_channels():
            self._servos[logical_name] = _Pca9685Servo(
                name=logical_name,
                channel_index=channel.channel,
                channel=channel,
                device=self._device,
            )
        _log.info(
            "servo.pca9685.initialised",
            count=len(self._servos),
            frequency=self._config.frequency,
            address=f"0x{self._config.address:02X}",
        )

    def _resolve_channels(self) -> Iterator[tuple[str, ServoChannelConfig]]:
        defaults: dict[str, tuple[int, ServoChannelConfig]] = {
            "pan": (0, ServoChannelConfig(min_angle_deg=-90.0, max_angle_deg=90.0, channel=0)),
            "tilt": (1, ServoChannelConfig(min_angle_deg=-30.0, max_angle_deg=30.0, channel=1)),
            "left_arm": (2, ServoChannelConfig(min_angle_deg=0.0, max_angle_deg=180.0, channel=2)),
            "right_arm": (3, ServoChannelConfig(min_angle_deg=0.0, max_angle_deg=180.0, channel=3)),
        }
        if not self._config.channels:
            for name, (idx, cfg) in defaults.items():
                yield (
                    name,
                    ServoChannelConfig(
                        min_angle_deg=cfg.min_angle_deg,
                        max_angle_deg=cfg.max_angle_deg,
                        chip=0,
                        channel=idx,
                    ),
                )
            return
        for name, cfg in self._config.channels.items():
            yield name, cfg


__all__ = ["PCA9685ServoController"]
