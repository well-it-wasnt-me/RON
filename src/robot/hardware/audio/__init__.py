"""Audio drivers."""

from robot.hardware.audio.mock_audio import MockAudioOutput
from robot.hardware.audio.usb_speaker import UsbSpeaker

__all__ = ["MockAudioOutput", "UsbSpeaker"]
