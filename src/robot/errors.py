"""Custom exception hierarchy for DeskBot.

The base class :class:`DeskBotError` is the root of every exception raised by
this codebase. Catching it at the application boundary is the recommended way
to keep the robot alive in the face of subsystem failures.

Conventions:

* Hardware errors are non-fatal - they should be caught and reported via the
  event bus and the logger.
* Configuration errors *are* fatal at boot time.
* Programming errors (e.g. invalid state transitions) should be loud.
"""

from __future__ import annotations


class DeskBotError(Exception):
    """Base class for every exception raised by DeskBot."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ConfigurationError(DeskBotError):
    """A configuration value is missing, malformed, or contradictory."""


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
class HardwareError(DeskBotError):
    """A hardware component failed, timed out, or returned invalid data."""


class DisplayError(HardwareError):
    """Failure in a display driver (e.g. SPI write error)."""


class ServoError(HardwareError):
    """Failure controlling a servo (e.g. PCA9685 communication error)."""


class AudioError(HardwareError):
    """Failure in the audio output path."""


class MicrophoneError(HardwareError):
    """Failure capturing audio from the microphone."""


class CameraError(HardwareError):
    """Failure capturing an image from the camera."""


class SensorError(HardwareError):
    """Generic sensor failure (IMU, distance, light, ...)."""


# ---------------------------------------------------------------------------
# AI / speech
# ---------------------------------------------------------------------------
class LLMError(DeskBotError):
    """The language model provider returned an error or timed out."""


class SpeechError(DeskBotError):
    """Generic TTS/STT failure."""


class WakeWordError(SpeechError):
    """Wake-word engine reported an error."""


# ---------------------------------------------------------------------------
# State / behaviour
# ---------------------------------------------------------------------------
class StateTransitionError(DeskBotError):
    """An illegal state-machine transition was requested."""


class AnimationError(DeskBotError):
    """An animation could not be scheduled or completed."""


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
class DependencyResolutionError(DeskBotError):
    """The dependency container could not satisfy a request."""


class LifecycleError(DeskBotError):
    """The application lifecycle was misused (double start, double stop, ...)."""


__all__ = [
    "AnimationError",
    "AudioError",
    "CameraError",
    "ConfigurationError",
    "DependencyResolutionError",
    "DeskBotError",
    "DisplayError",
    "HardwareError",
    "LLMError",
    "LifecycleError",
    "MicrophoneError",
    "SensorError",
    "ServoError",
    "SpeechError",
    "StateTransitionError",
    "WakeWordError",
]
