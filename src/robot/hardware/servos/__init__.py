"""Servo drivers.

Three interchangeable backends are available through the
:class:`~robot.hardware.servos.factory.ServoControllerFactory`:

* ``mock`` - in-memory :class:`MockServoBus` (default, for tests and dev).
* ``gpio`` - :class:`RaspberryPiGPIOServoController` (Pi 5 header).
* ``pca9685`` - :class:`PCA9685ServoController` (16-channel I2C board).

The original :class:`MockServoBus` and :class:`MockServo` are still
imported from :mod:`robot.hardware.servos.mock_servo` for backward
compatibility - see the original scaffold.
"""

from robot.hardware.servos.adapter import wrap_servo_controller
from robot.hardware.servos.factory import ServoControllerFactory
from robot.hardware.servos.gpio_controller import RaspberryPiGPIOServoController
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.hardware.servos.pca9685_controller import PCA9685ServoController

__all__ = [
    "MockServo",
    "MockServoBus",
    "PCA9685ServoController",
    "RaspberryPiGPIOServoController",
    "ServoControllerFactory",
    "wrap_servo_controller",
]
