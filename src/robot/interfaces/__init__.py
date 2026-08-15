"""Hardware and infrastructure interfaces.

Every concrete implementation lives in :mod:`robot.hardware`. The rest of the
codebase interacts only with the protocols declared here.
"""

from robot.interfaces.audio import AudioOutput
from robot.interfaces.camera import Camera, Frame
from robot.interfaces.clock import Clock as IClock
from robot.interfaces.display import Display, EyeFrame
from robot.interfaces.event_bus import EventBus, EventHandler
from robot.interfaces.llm import LLM, Message, Role
from robot.interfaces.microphone import AudioChunk, Microphone
from robot.interfaces.random_source import RandomSource as IRandomSource
from robot.interfaces.servo import Servo, ServoController

__all__ = [
    "LLM",
    "AudioChunk",
    "AudioOutput",
    "Camera",
    "Display",
    "EventBus",
    "EventHandler",
    "EyeFrame",
    "Frame",
    "IClock",
    "IRandomSource",
    "Message",
    "Microphone",
    "Role",
    "Servo",
    "ServoController",
]
