"""Typed Pydantic response/request schemas for the DeskBot REST API.

These models give FastAPI's OpenAPI/Swagger UI rich, structured request
and response examples instead of the generic ``additionalProperties: {}``
blobs that result from annotating endpoints with ``dict[str, Any]``.

Every response model uses ``model_config = ConfigDict(extra="allow")`` so
that runtime responses carrying extra/dynamic fields are still serialised
correctly (no 422), while the declared fields and the explicit ``examples``
make the ``/docs`` page genuinely usable for "Try it out".
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _ex(*examples: dict[str, Any]) -> ConfigDict:
    """Build a config that allows extra fields and attaches examples."""
    return ConfigDict(extra="allow", json_schema_extra={"examples": list(examples)})


# ---------------------------------------------------------------------------
# Generic acknowledgement
# ---------------------------------------------------------------------------


class OkResponse(BaseModel):
    """Generic ``{status: "ok"}`` acknowledgement."""

    model_config = _ex({"status": "ok"})

    status: str = Field(default="ok", description="Outcome of the request.")
    detail: str | None = Field(default=None, description="Optional human-readable detail.")


# ---------------------------------------------------------------------------
# Health & version
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Overall health, possibly with per-component degradation info."""

    model_config = _ex(
        {"status": "ok"},
        {
            "status": "degraded",
            "components": {
                "camera": {"status": "degraded", "reason": "device not found"},
                "microphone": {"status": "ok"},
            },
        },
    )

    status: str = Field(description='"ok" or "degraded".')


class VersionResponse(BaseModel):
    """API name and version."""

    model_config = _ex({"version": "0.1.0", "name": "DeskBot API"})

    version: str
    name: str


# ---------------------------------------------------------------------------
# State / config / perception / audio / conversation
# ---------------------------------------------------------------------------


class StateResponse(BaseModel):
    """Current robot state."""

    model_config = _ex(
        {"state": "idle"}, {"state": "unknown", "detail": "DeskBot app not attached"}
    )

    state: str = Field(
        description="Robot state (idle, curious, listening, thinking, speaking, sleeping) or 'unknown'."
    )
    detail: str | None = None


class ConfigResponse(BaseModel):
    """Current (masked) configuration. Free-form nested object."""

    model_config = _ex(
        {
            "env": "development",
            "hardware": "mock",
            "displays": {"backend": "gc9a01", "width": 240, "height": 240, "rotation": 0},
            "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-x****"},
        }
    )


class PerceptionResponse(BaseModel):
    """Perception (face detection) status."""

    model_config = _ex(
        {"enabled": True, "running": True, "scan_interval_s": 0.5, "max_faces": 3},
        {"enabled": False},
    )

    enabled: bool
    running: bool | None = None
    scan_interval_s: float | None = None
    max_faces: int | None = None


class AudioStatusResponse(BaseModel):
    """Audio output status."""

    model_config = _ex(
        {"enabled": True, "talking": False},
        {"enabled": False, "detail": "DeskBot app not attached"},
    )

    enabled: bool
    talking: bool | None = None
    detail: str | None = None


class ConversationStatusResponse(BaseModel):
    """Conversation pipeline status."""

    model_config = _ex(
        {
            "enabled": True,
            "state": "idle",
            "wake_checker": "OpenWakeWordChecker",
            "stt": "WhisperSTT",
            "tts": "OpenAITTS",
            "llm": "OpenAILLM",
        },
        {"enabled": False},
    )

    enabled: bool
    state: str | None = None
    wake_checker: str | None = None
    stt: str | None = None
    tts: str | None = None
    llm: str | None = None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


class SpeakRequest(BaseModel):
    """Request body for the speak command."""

    model_config = _ex({"text": "Hello! How are you today?"})

    text: str = Field(..., min_length=1, max_length=1000, description="Text to speak.")


class EmotionRequest(BaseModel):
    """Request body for the emotion command."""

    model_config = _ex({"emotion": "happy", "intensity": 0.8})

    emotion: str = Field(..., description="Emotion name (neutral, happy, curious, etc.).")
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)


class StateRequest(BaseModel):
    """Request body for the state transition command."""

    model_config = _ex({"state": "curious"})

    state: str = Field(
        ..., description="Target state (idle, curious, listening, thinking, speaking, sleeping)."
    )


class CommandResponse(BaseModel):
    """Generic command acknowledgement with the echoed payload."""

    model_config = _ex(
        {"status": "ok", "text": "Hello!"},
        {"status": "ok", "emotion": "happy"},
        {"status": "ok", "state": "curious"},
        {"status": "error", "detail": "DeskBot app not attached"},
    )

    status: str = Field(description='"ok" or "error".')
    detail: str | None = None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class ServoInfo(BaseModel):
    """A single servo's name and current angle."""

    model_config = _ex({"name": "pan", "angle": 90.0})

    name: str
    angle: float


class ServoListResponse(BaseModel):
    """All servos and their angles."""

    model_config = _ex(
        {"servos": [{"name": "pan", "angle": 90.0}, {"name": "tilt", "angle": 90.0}]}
    )

    servos: list[ServoInfo]


class ServoMoveResponse(BaseModel):
    """Acknowledgement of a servo move command."""

    model_config = _ex({"name": "pan", "angle": 45.0, "duration_s": 0.4})

    name: str
    angle: float
    duration_s: float


class ServoReleaseResponse(BaseModel):
    """Acknowledgement of a servo release command."""

    model_config = _ex({"name": "pan", "released": True})

    name: str
    released: bool


class ReleaseAllResponse(BaseModel):
    """Acknowledgement of release-all command."""

    model_config = _ex({"released": True})

    released: bool


class CalibrationStep(BaseModel):
    """One step of a calibration sequence."""

    model_config = _ex({"position": "min", "angle": 0.0, "actual_angle": 0.0})

    position: str
    angle: float
    actual_angle: float


class CalibrateServoResponse(BaseModel):
    """Result of a servo calibration sequence."""

    model_config = _ex(
        {
            "servo": "pan",
            "sequence": [
                {"position": "min", "angle": 0.0, "actual_angle": 0.0},
                {"position": "centre", "angle": 90.0, "actual_angle": 90.0},
            ],
        }
    )

    servo: str
    sequence: list[CalibrationStep]


class DisplayConfigResponse(BaseModel):
    """Display configuration."""

    model_config = _ex(
        {
            "backend": "gc9a01",
            "width": 240,
            "height": 240,
            "rotation": 0,
            "spi_hz": 40000000,
            "invert": False,
        }
    )

    backend: str
    width: int
    height: int
    rotation: int
    spi_hz: int
    invert: bool


class TestPatternResponse(BaseModel):
    """Acknowledgement of a display test pattern."""

    model_config = _ex({"pattern": "gradient", "width": 240, "height": 240})

    pattern: str
    width: int
    height: int


class ClearDisplayResponse(BaseModel):
    """Acknowledgement of display clear."""

    model_config = _ex({"cleared": True})

    cleared: bool


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


class PreferenceItem(BaseModel):
    """A single user preference."""

    model_config = _ex(
        {
            "key": "tts:voice",
            "value": "alloy",
            "confidence": 0.9,
            "source": "explicit",
            "updated_at": "2026-08-12T10:30:00Z",
        }
    )

    key: str
    value: Any
    confidence: float
    source: str
    updated_at: str


class PreferenceListResponse(BaseModel):
    """All stored preferences."""

    model_config = _ex(
        {
            "preferences": [
                {
                    "key": "tts:voice",
                    "value": "alloy",
                    "confidence": 0.9,
                    "source": "explicit",
                    "updated_at": "2026-08-12T10:30:00Z",
                }
            ]
        }
    )

    preferences: list[PreferenceItem]


class PreferenceDeleteResponse(BaseModel):
    """Acknowledgement of a preference deletion."""

    model_config = _ex({"status": "deleted", "key": "tts:voice"})

    status: str
    key: str


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------


class LearningStatusResponse(BaseModel):
    """Snapshot of the background learning service."""

    model_config = _ex(
        {
            "enabled": True,
            "total_experiences": 1280,
            "new_experiences_since_train": 12,
            "training_cycles_completed": 8,
            "current_model_loss": 0.0123,
            "candidate_model_loss": 0.0118,
            "last_training_time": "2026-08-12T10:29:55Z",
            "last_training_duration_s": 1.842,
            "is_training": False,
            "promotions": 5,
            "rollbacks": 3,
            "model_version": 5,
        }
    )

    enabled: bool
    total_experiences: int
    new_experiences_since_train: int
    training_cycles_completed: int
    current_model_loss: float | None = None
    candidate_model_loss: float | None = None
    last_training_time: str | None = None
    last_training_duration_s: float = 0.0
    is_training: bool = False
    promotions: int = 0
    rollbacks: int = 0
    model_version: int = 0


class LearnedPreferenceItem(BaseModel):
    """A single learned preference with confidence."""

    model_config = _ex(
        {
            "key": "preferred_action:celebrate",
            "category": "preferred_action",
            "value": "celebrate",
            "confidence": 0.75,
            "observation_count": 12,
            "avg_reward": 0.82,
            "source": "behavioral",
            "last_observed": "2026-08-12T10:25:00Z",
        }
    )

    key: str
    category: str
    value: str
    confidence: float
    observation_count: int
    avg_reward: float
    source: str
    last_observed: str


class LearningPreferencesResponse(BaseModel):
    """All learned preferences and the total tracked pattern count."""

    model_config = _ex(
        {
            "preferences": [
                {
                    "key": "preferred_action:celebrate",
                    "category": "preferred_action",
                    "value": "celebrate",
                    "confidence": 0.75,
                    "observation_count": 12,
                    "avg_reward": 0.82,
                    "source": "behavioral",
                    "last_observed": "2026-08-12T10:25:00Z",
                }
            ],
            "total_patterns": 17,
        }
    )

    preferences: list[LearnedPreferenceItem]
    total_patterns: int = 0


class LearningScheduleSchema(BaseModel):
    """Training schedule configuration."""

    model_config = _ex(
        {"min_new_experiences": 32, "train_interval_s": 30.0, "min_experiences_for_training": 64}
    )

    min_new_experiences: int
    train_interval_s: float
    min_experiences_for_training: int


class LearningResourceLimitsSchema(BaseModel):
    """Resource limits for background training."""

    model_config = _ex(
        {
            "batch_size": 32,
            "training_epochs_per_cycle": 5,
            "max_cpu_fraction": 0.3,
            "max_model_params": 500000,
            "eval_sample_size": 128,
        }
    )

    batch_size: int
    training_epochs_per_cycle: int
    max_cpu_fraction: float
    max_model_params: int
    eval_sample_size: int


class LearningCheckpointSchema(BaseModel):
    """Checkpoint management configuration."""

    model_config = _ex(
        {"checkpoint_dir": "~/.deskbot/checkpoints", "keep_last_n": 5, "promote_threshold": 1.0}
    )

    checkpoint_dir: str
    keep_last_n: int
    promote_threshold: float


class LearningConfigResponse(BaseModel):
    """Learning service configuration."""

    model_config = _ex(
        {
            "schedule": {
                "min_new_experiences": 32,
                "train_interval_s": 30.0,
                "min_experiences_for_training": 64,
            },
            "resource_limits": {
                "batch_size": 32,
                "training_epochs_per_cycle": 5,
                "max_cpu_fraction": 0.3,
                "max_model_params": 500000,
                "eval_sample_size": 128,
            },
            "checkpoint_config": {
                "checkpoint_dir": "~/.deskbot/checkpoints",
                "keep_last_n": 5,
                "promote_threshold": 1.0,
            },
        }
    )

    schedule: LearningScheduleSchema
    resource_limits: LearningResourceLimitsSchema
    checkpoint_config: LearningCheckpointSchema


class ForceTrainResponse(BaseModel):
    """Result of a forced training cycle."""

    model_config = _ex({"triggered": True, "training_cycles_completed": 9, "is_training": False})

    triggered: bool
    training_cycles_completed: int
    is_training: bool


# ---------------------------------------------------------------------------
# Settings: request models
# ---------------------------------------------------------------------------


class MicTestRequest(BaseModel):
    """Parameters for a microphone record-and-playback test."""

    model_config = _ex({"duration_s": 3.0, "play_back": True})

    duration_s: float = Field(default=3.0, ge=0.5, le=10.0)
    play_back: bool = Field(default=True)


class ToneRequest(BaseModel):
    """Parameters for an audio test tone."""

    model_config = _ex({"frequency_hz": 440.0, "duration_s": 1.0, "volume": 0.5})

    frequency_hz: float = Field(default=440.0, ge=20.0, le=20_000.0)
    duration_s: float = Field(default=1.0, ge=0.1, le=10.0)
    volume: float = Field(default=0.5, ge=0.0, le=1.0)


class AudioDeviceTestRequest(BaseModel):
    """Play a test tone through a specific output device (runtime test)."""

    model_config = _ex(
        {"device": "default", "frequency_hz": 440.0, "duration_s": 1.0, "volume": 0.5}
    )

    device: str | int = Field(default="default")
    frequency_hz: float = Field(default=440.0, ge=20.0, le=20_000.0)
    duration_s: float = Field(default=1.0, ge=0.1, le=10.0)
    volume: float = Field(default=0.5, ge=0.0, le=1.0)


class AudioSwitchRequest(BaseModel):
    """Switch the active audio output device at runtime."""

    model_config = _ex({"device": "usb_headset"})

    device: str | int = Field(default="default")


class TTSTestRequest(BaseModel):
    """Speak a test phrase through the TTS engine."""

    model_config = _ex(
        {"text": "Hello! This is a test of the text to speech system.", "direct": True}
    )

    text: str = Field(default="Hello! This is a test of the text to speech system.")
    direct: bool = Field(
        default=True,
        description="When True, speak directly via TTS (no LLM). When False, go through the LLM pipeline.",
    )


class LLMTestRequest(BaseModel):
    """Send a test prompt to the LLM and return the response."""

    model_config = _ex({"prompt": "Hello! Who are you?"})

    prompt: str = Field(default="Hello! Who are you?")


# ---------------------------------------------------------------------------
# Settings: response models
# ---------------------------------------------------------------------------


class CameraInfoResponse(BaseModel):
    """Camera type and resolution."""

    model_config = _ex(
        {"type": "UsbCamera", "width": 640, "height": 480, "is_mock": False, "captured": 1024}
    )

    type: str
    width: int | None = None
    height: int | None = None
    is_mock: bool | None = None
    captured: int | None = None


class MicrophoneInfoResponse(BaseModel):
    """Microphone type and sample rate."""

    model_config = _ex({"type": "UsbMicrophone", "sample_rate": 16000, "is_mock": False})

    type: str
    sample_rate: int | None = None
    is_mock: bool | None = None


class MicLevelResponse(BaseModel):
    """Current microphone RMS level."""

    model_config = _ex({"level": 0.18})

    level: float | None = None


class AudioInfoResponse(BaseModel):
    """Audio output type, sample rate, and channels."""

    model_config = _ex(
        {
            "type": "UsbSpeaker",
            "sample_rate": 48000,
            "channels": 2,
            "is_mock": False,
            "output_device": "default",
        }
    )

    type: str
    sample_rate: int | None = None
    channels: int | None = None
    is_mock: bool | None = None
    output_device: str | None = None


class AudioDevice(BaseModel):
    """A single sounddevice entry."""

    model_config = _ex({"name": "hw:1,0", "index": 2, "channels": 2, "default": True})

    name: str | None = None
    index: int | None = None
    channels: int | None = None
    default: bool | None = None


class AudioDevicesResponse(BaseModel):
    """List of available audio devices."""

    model_config = _ex(
        {
            "devices": [{"name": "hw:1,0", "index": 2, "channels": 2, "default": True}],
            "available": True,
        },
        {"devices": [], "available": False, "error": "sounddevice not installed"},
    )

    devices: list[AudioDevice] = []
    available: bool = True
    error: str | None = None


class ToneResponse(BaseModel):
    """Acknowledgement of a test tone."""

    model_config = _ex(
        {"status": "ok", "frequency_hz": 440.0, "duration_s": 1.0, "sample_rate": 48000}
    )

    status: str
    frequency_hz: float | None = None
    duration_s: float | None = None
    sample_rate: int | None = None


class AudioTestDeviceResponse(BaseModel):
    """Acknowledgement of a device test tone."""

    model_config = _ex(
        {"status": "ok", "device": "default", "frequency_hz": 440.0, "duration_s": 1.0}
    )

    status: str
    device: str | int | None = None
    frequency_hz: float | None = None
    duration_s: float | None = None


class AudioSwitchResponse(BaseModel):
    """Acknowledgement of an audio device switch."""

    model_config = _ex({"status": "ok", "device": "usb_headset", "type": "UsbSpeaker"})

    status: str
    device: str | int | None = None
    type: str | None = None


class TTSTestResponse(BaseModel):
    """Acknowledgement of a TTS test."""

    model_config = _ex(
        {"status": "ok", "text": "Hello!", "engine": "OpenAITTS"},
        {"status": "ok", "text": "Hello!", "via": "llm_pipeline"},
    )

    status: str
    text: str | None = None
    engine: str | None = None
    via: str | None = None


class LLMTestResponse(BaseModel):
    """Result of an LLM test prompt."""

    model_config = _ex(
        {
            "status": "ok",
            "prompt": "Hello! Who are you?",
            "response": "I'm DeskBot, your desktop companion!",
            "engine": "OpenAILLM",
        },
        {"status": "error", "prompt": "Hello!", "error": "No LLM configured"},
    )

    status: str
    prompt: str | None = None
    response: str | None = None
    engine: str | None = None
    error: str | None = None


class SoundEffectResponse(BaseModel):
    """Acknowledgement of a sound effect play."""

    model_config = _ex(
        {"status": "ok", "name": "talk"},
        {"status": "error", "name": "unknown", "error": "not found"},
    )

    status: str
    name: str | None = None
    error: str | None = None


class SoundEffectsListResponse(BaseModel):
    """List of available sound effects."""

    model_config = _ex(
        {"available": True, "effects": ["talk", "thinking", "cute"]},
        {"available": False, "effects": []},
    )

    available: bool
    effects: list[str] = []


class SettingsInfoResponse(BaseModel):
    """Hardware & subsystem overview (free-form nested)."""

    model_config = _ex(
        {
            "ready": True,
            "camera": {
                "type": "UsbCamera",
                "width": 640,
                "height": 480,
                "is_mock": False,
                "captured": 1024,
            },
            "microphone": {"type": "UsbMicrophone", "sample_rate": 16000},
            "audio": {"type": "UsbSpeaker", "sample_rate": 48000},
            "display": {"backend": "gc9a01", "width": 240, "height": 240, "rotation": 0, "fps": 30},
            "servos": {"count": 4, "servos": [{"name": "pan", "angle": 90.0}]},
        },
        {"ready": False},
    )

    ready: bool = False


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


class SystemInfoResponse(BaseModel):
    """System information for the dashboard."""

    model_config = _ex(
        {
            "hostname": "deskbot",
            "platform": "Linux-6.1.0-aarch64",
            "machine": "aarch64",
            "processor": "aarch64",
            "python_version": "3.12.4",
            "cpu_count": 4,
            "uptime_s": 3600.5,
            "uptime_human": "1h 0m",
            "pid": 12345,
            "app_version": "0.1.0",
            "env": "production",
            "health": {"status": "ok"},
        }
    )

    hostname: str
    platform: str
    machine: str
    processor: str
    python_version: str
    cpu_count: int
    uptime_s: float
    uptime_human: str
    pid: int
    app_version: str
    env: str
    health: dict[str, Any] = {}


class LogEntry(BaseModel):
    """A single log entry."""

    model_config = _ex(
        {
            "timestamp": "2026-08-12T10:30:00Z",
            "level": "INFO",
            "logger": "api.commands",
            "event": "speak",
            "data": {"text": "hi"},
        }
    )

    timestamp: str
    level: str
    logger: str
    event: str
    data: dict[str, Any] = {}


class LogsResponse(BaseModel):
    """Recent log entries."""

    model_config = _ex(
        {
            "count": 2,
            "entries": [
                {
                    "timestamp": "2026-08-12T10:30:00Z",
                    "level": "INFO",
                    "logger": "api.commands",
                    "event": "speak",
                    "data": {"text": "hi"},
                }
            ],
        }
    )

    count: int
    entries: list[LogEntry] = []


class BluetoothResponse(BaseModel):
    """Bluetooth speaker status."""

    model_config = _ex(
        {
            "available": True,
            "type": "BluetoothSpeaker",
            "is_bluetooth": True,
            "connected": True,
            "device_name": "DeskBot BT",
        },
        {"available": False},
    )

    available: bool
    type: str | None = None
    is_bluetooth: bool | None = None
    connected: bool | None = None
    sink_name: str | None = None
    device_mac: str | None = None
    device_name: str | None = None
    auto_connect: bool | None = None
    playing: bool | None = None
    sample_rate: int | None = None
    channels: int | None = None


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class FrameStatsResponse(BaseModel):
    """Frame budget statistics."""

    model_config = _ex(
        {
            "target_fps": 30,
            "actual_fps": 29.4,
            "avg_frame_time_ms": 34.0,
            "p50_frame_time_ms": 33.0,
            "p95_frame_time_ms": 38.0,
            "p99_frame_time_ms": 42.0,
            "dropped_frames": 3,
            "total_frames": 1200,
        },
        {"enabled": False},
    )

    enabled: bool | None = None
    target_fps: int | None = None
    actual_fps: float | None = None
    avg_frame_time_ms: float | None = None
    p50_frame_time_ms: float | None = None
    p95_frame_time_ms: float | None = None
    p99_frame_time_ms: float | None = None
    dropped_frames: int | None = None
    total_frames: int | None = None


class PerformanceSummaryResponse(BaseModel):
    """Combined profiler summary."""

    model_config = _ex(
        {
            "frames": {
                "target_fps": 30,
                "actual_fps": 29.4,
                "avg_frame_time_ms": 34.0,
                "p50_frame_time_ms": 33.0,
                "p95_frame_time_ms": 38.0,
                "p99_frame_time_ms": 42.0,
                "dropped_frames": 3,
                "total_frames": 1200,
            },
            "servos": {"pan": {"avg_ms": 2.1, "max_ms": 5.0, "count": 100}},
            "bus": {"events_per_s": 120.0, "avg_latency_ms": 0.4},
        }
    )

    frames: dict[str, Any] = {}
    servos: dict[str, Any] = {}
    bus: dict[str, Any] = {}


class ServoProfilerStatsResponse(BaseModel):
    """Per-servo latency statistics (free-form, keyed by servo name)."""

    model_config = _ex(
        {
            "pan": {"avg_ms": 2.1, "max_ms": 5.0, "count": 100},
            "tilt": {"avg_ms": 1.8, "max_ms": 4.2, "count": 100},
        },
        {"enabled": False},
    )

    enabled: bool | None = None


class BusProfilerStatsResponse(BaseModel):
    """Event bus throughput and latency statistics."""

    model_config = _ex(
        {"events_per_s": 120.0, "avg_latency_ms": 0.4, "total_events": 5400, "subscribers": 12},
        {"enabled": False},
    )

    enabled: bool | None = None


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class ConfigValidationErrorItem(BaseModel):
    """A single validation error."""

    model_config = _ex(
        {
            "field": "displays.width",
            "message": "Input should be greater than 0",
            "type": "greater_than",
            "input": -1,
        }
    )

    field: str
    message: str
    type: str
    input: Any = None


class ConfigValidateResponse(BaseModel):
    """Result of a config validation request."""

    model_config = _ex(
        {"valid": True},
        {
            "valid": False,
            "errors": [
                {
                    "field": "displays.width",
                    "message": "Input should be greater than 0",
                    "type": "greater_than",
                    "input": -1,
                }
            ],
        },
    )

    valid: bool
    errors: list[ConfigValidationErrorItem] = []


class ConfigValidateRequest(BaseModel):
    """A partial or full configuration override to validate.

    Any subset of the :class:`AppSettings` fields may be supplied; omitted
    fields fall back to the current defaults during validation.
    """

    model_config = _ex(
        {"env": "development", "displays": {"backend": "gc9a01", "width": 240, "height": 240}},
        {"servos": {"pan": {"min": 0, "max": 180}}},
    )


class ConfigSchemaResponse(BaseModel):
    """The full JSON Schema for the :class:`AppSettings` model (free-form)."""

    model_config = _ex(
        {
            "properties": {"env": {"type": "string", "default": "development"}},
            "title": "AppSettings",
            "type": "object",
        }
    )


__all__ = [
    "AudioDevice",
    "AudioDeviceTestRequest",
    "AudioDevicesResponse",
    "AudioInfoResponse",
    "AudioStatusResponse",
    "AudioSwitchRequest",
    "AudioSwitchResponse",
    "AudioTestDeviceResponse",
    "BluetoothResponse",
    "BusProfilerStatsResponse",
    "CalibrateServoResponse",
    "CalibrationStep",
    "ClearDisplayResponse",
    "CommandResponse",
    "ConfigResponse",
    "ConfigSchemaResponse",
    "ConfigValidateRequest",
    "ConfigValidateResponse",
    "ConfigValidationErrorItem",
    "ConversationStatusResponse",
    "DisplayConfigResponse",
    "EmotionRequest",
    "ForceTrainResponse",
    "FrameStatsResponse",
    "HealthResponse",
    "LLMTestRequest",
    "LLMTestResponse",
    "LearnedPreferenceItem",
    "LearningCheckpointSchema",
    "LearningConfigResponse",
    "LearningPreferencesResponse",
    "LearningResourceLimitsSchema",
    "LearningScheduleSchema",
    "LearningStatusResponse",
    "LogEntry",
    "LogsResponse",
    "MicLevelResponse",
    "MicTestRequest",
    "MicrophoneInfoResponse",
    "OkResponse",
    "PerceptionResponse",
    "PerformanceSummaryResponse",
    "PreferenceDeleteResponse",
    "PreferenceItem",
    "PreferenceListResponse",
    "ReleaseAllResponse",
    "ServoInfo",
    "ServoListResponse",
    "ServoMoveResponse",
    "ServoProfilerStatsResponse",
    "ServoReleaseResponse",
    "SettingsInfoResponse",
    "SoundEffectResponse",
    "SoundEffectsListResponse",
    "SpeakRequest",
    "StateRequest",
    "StateResponse",
    "SystemInfoResponse",
    "TTSTestRequest",
    "TTSTestResponse",
    "TestPatternResponse",
    "ToneRequest",
    "ToneResponse",
    "VersionResponse",
]
