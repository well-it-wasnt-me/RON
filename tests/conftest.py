"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import pytest

from robot.behavior.state_machine import StateMachine
from robot.config import AppSettings
from robot.events.bus import InMemoryEventBus
from robot.hardware.audio.mock_audio import MockAudioOutput
from robot.hardware.displays.mock_display import MockDisplay
from robot.hardware.sensors.mock_camera import MockCamera
from robot.hardware.sensors.mock_microphone import MockMicrophone
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.interfaces.servo import ServoController
from robot.utils.clock import SystemClock
from robot.utils.random_source import SystemRandomSource


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(_env_file=None, env="testing", log_level="WARNING", use_mocks=True)


@pytest.fixture
def bus() -> InMemoryEventBus:
    return InMemoryEventBus()


@pytest.fixture
def state_machine(bus: InMemoryEventBus) -> StateMachine:
    return StateMachine(bus=bus)


@pytest.fixture
def clock() -> SystemClock:
    return SystemClock()


@pytest.fixture
def rng() -> SystemRandomSource:
    return SystemRandomSource(seed=1234)


@pytest.fixture
def display() -> MockDisplay:
    return MockDisplay(width=64, height=64)


@pytest.fixture
def mock_eyes() -> MockDisplay:
    """Backwards-compatible alias for the single display."""
    return MockDisplay(width=64, height=64)


@pytest.fixture
def servo_bus() -> MockServoBus:
    return MockServoBus(
        {
            "head_pan": MockServo(name="head_pan", min_angle=-90.0, max_angle=90.0),
            "head_tilt": MockServo(name="head_tilt", min_angle=-30.0, max_angle=30.0),
        }
    )


@pytest.fixture
def servo_controller() -> ServoController:
    bus = MockServoBus(
        {
            "head_pan": MockServo(name="head_pan", min_angle=-90.0, max_angle=90.0),
            "head_tilt": MockServo(name="head_tilt", min_angle=-30.0, max_angle=30.0),
            "eye_left": MockServo(name="eye_left", min_angle=0.0, max_angle=180.0),
            "eye_right": MockServo(name="eye_right", min_angle=0.0, max_angle=180.0),
        }
    )
    return wrap_servo_controller(bus, backend_name="mock")


@pytest.fixture
def audio_output() -> MockAudioOutput:
    return MockAudioOutput(sample_rate=48_000, channels=1)


@pytest.fixture
def microphone() -> MockMicrophone:
    return MockMicrophone(sample_rate=16_000, channels=1, frame_ms=10)


@pytest.fixture
def camera() -> MockCamera:
    return MockCamera(width=320, height=240)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
