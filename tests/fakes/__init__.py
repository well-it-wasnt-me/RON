"""Fake implementations of every hardware interface.

These are intentionally separate from :mod:`robot.hardware.*` mocks so that
test code does not depend on production code paths.
"""

from tests.fakes.audio import FakeAudioOutput
from tests.fakes.bus import RecordingBus
from tests.fakes.camera import FakeCamera
from tests.fakes.clock import FakeClock
from tests.fakes.display import FakeDisplay
from tests.fakes.llm import FakeLLM
from tests.fakes.microphone import FakeMicrophone
from tests.fakes.random import FakeRandom
from tests.fakes.servo import (
    FakePcaDevice,
    FakeServo,
    FakeServoBus,
    make_fake_pca_factory,
    make_servo_controller_from_fakes,
)

__all__ = [
    "FakeAudioOutput",
    "FakeCamera",
    "FakeClock",
    "FakeDisplay",
    "FakeLLM",
    "FakeMicrophone",
    "FakePcaDevice",
    "FakeRandom",
    "FakeServo",
    "FakeServoBus",
    "RecordingBus",
    "make_fake_pca_factory",
    "make_servo_controller_from_fakes",
]
