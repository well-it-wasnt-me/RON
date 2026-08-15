"""Servo controller factory.

Selects a :class:`ServoController` implementation at application boot based
on the value of ``config.servos.backend``. The factory fails fast: a bad or
unavailable backend raises :class:`ServoError` instead of silently falling
back to another one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from robot.config import ServosConfig
from robot.errors import ServoError
from robot.interfaces.servo import ServoController
from robot.logging import get_logger

_log = get_logger("hardware.servos.factory")


ServoFactory = Callable[[int, object, int], object]
PcaDeviceFactory = Callable[[object], object]


class ServoControllerFactory:
    """Build a :class:`ServoController` from a :class:`ServosConfig`."""

    def __init__(
        self,
        config: ServosConfig,
        *,
        servo_factory: ServoFactory | None = None,
        pca_device_factory: PcaDeviceFactory | None = None,
    ) -> None:
        self._config = config
        self._servo_factory = servo_factory
        self._pca_device_factory = pca_device_factory

    def build(self) -> ServoController:
        """Instantiate the controller for the configured backend."""
        backend = self._config.backend
        _log.info("servo.backend_selected", backend=backend)
        match backend:
            case "mock":
                from robot.hardware.servos.mock_servo import MockServoBus

                return self._build_mock(MockServoBus)
            case "gpio":
                from robot.hardware.servos.gpio_controller import RaspberryPiGPIOServoController

                return self._build_gpio(RaspberryPiGPIOServoController)
            case "pca9685":
                from robot.hardware.servos.pca9685_controller import PCA9685ServoController

                return self._build_pca9685(PCA9685ServoController)
            case _:
                raise ServoError(
                    f"unknown servo backend {backend!r}; expected one of 'mock', 'gpio', 'pca9685'"
                )

    def _build_mock(self, mock_cls: type) -> ServoController:
        from robot.hardware.servos.adapter import wrap_servo_controller
        from robot.hardware.servos.mock_servo import MockServo, MockServoBus

        bus = MockServoBus(
            {
                "pan": MockServo(name="pan", min_angle=-90.0, max_angle=90.0),
                "tilt": MockServo(name="tilt", min_angle=-30.0, max_angle=30.0),
                "left_arm": MockServo(name="left_arm", min_angle=0.0, max_angle=180.0),
                "right_arm": MockServo(name="right_arm", min_angle=0.0, max_angle=180.0),
            }
        )
        return wrap_servo_controller(bus, backend_name="mock")

    def _build_gpio(self, controller_cls: type) -> ServoController:
        try:
            if self._servo_factory is not None:
                return cast(
                    "ServoController",
                    controller_cls(self._config.gpio, servo_factory=self._servo_factory),
                )
            return cast("ServoController", controller_cls(self._config.gpio))
        except ServoError:
            raise
        except Exception as exc:
            _log.exception("servo.gpio.init_failed")
            raise ServoError(f"GPIO backend initialisation failed: {exc!r}") from exc

    def _build_pca9685(self, controller_cls: type) -> ServoController:
        try:
            if self._pca_device_factory is not None:
                return cast(
                    "ServoController",
                    controller_cls(self._config.pca9685, device_factory=self._pca_device_factory),
                )
            return cast("ServoController", controller_cls(self._config.pca9685))
        except ServoError:
            raise
        except Exception as exc:
            _log.exception("servo.pca9685.init_failed")
            raise ServoError(f"PCA9685 backend initialisation failed: {exc!r}") from exc


__all__ = ["ServoControllerFactory"]
