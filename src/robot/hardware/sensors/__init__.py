"""Sensors: microphones, cameras, IMUs, etc."""

from robot.hardware.sensors.mock_camera import MockCamera
from robot.hardware.sensors.mock_microphone import MockMicrophone
from robot.hardware.sensors.rtsp_camera import RtspCamera
from robot.hardware.sensors.rtsp_microphone import RtspMicrophone
from robot.hardware.sensors.usb_camera import UsbCamera
from robot.hardware.sensors.usb_microphone import UsbMicrophone

__all__ = [
    "MockCamera",
    "MockMicrophone",
    "RtspCamera",
    "RtspMicrophone",
    "UsbCamera",
    "UsbMicrophone",
]
