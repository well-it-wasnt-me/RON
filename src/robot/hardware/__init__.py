"""Concrete hardware implementations live here.

Every module in this package must:

* Live behind one of the protocols defined in :mod:`robot.interfaces`.
* Fail gracefully (raise a typed :class:`HardwareError` and let the caller
  decide whether to retry, log, or ignore).
* Be safe to import on non-Pi platforms - heavy drivers should be imported
  lazily inside the constructor.
"""

from robot.hardware.audio.mock_audio import MockAudioOutput
from robot.hardware.audio.usb_speaker import UsbSpeaker
from robot.hardware.displays.factory import DisplayFactory
from robot.hardware.displays.gc9a01 import FakeSpiTransport, GC9A01Display
from robot.hardware.displays.mock_display import MockDisplay
from robot.hardware.sensors.mock_camera import MockCamera
from robot.hardware.sensors.mock_microphone import MockMicrophone
from robot.hardware.servos.factory import ServoControllerFactory
from robot.hardware.servos.gpio_controller import RaspberryPiGPIOServoController
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.hardware.servos.pca9685_controller import PCA9685ServoController

__all__ = [
    "DisplayFactory",
    "FakeSpiTransport",
    "GC9A01Display",
    "MockAudioOutput",
    "MockCamera",
    "MockDisplay",
    "MockMicrophone",
    "MockServo",
    "MockServoBus",
    "PCA9685ServoController",
    "RaspberryPiGPIOServoController",
    "ServoControllerFactory",
    "UsbSpeaker",
]
