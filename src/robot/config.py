"""Pydantic Settings configuration for DeskBot.

All runtime knobs live here. Values are sourced from:

1. Defaults declared on each model.
2. Environment variables prefixed with ``DESKBOT_``.
3. An optional ``.env`` file (use ``.env.example`` as a template).
4. A future YAML file referenced via ``DESKBOT_CONFIG_FILE``.

Pin numbers, bus numbers, sample rates, persona traits, and LLM credentials
must NEVER appear anywhere else in the codebase - inject this settings object
into the components that need them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from robot.errors import ConfigurationError

Env = Literal["development", "testing", "production"]

ServoBackend = Literal["mock", "gpio", "pca9685"]


class PersonalityConfig(BaseSettings):
    """Personality traits in the [0.0, 1.0] range."""

    model_config = SettingsConfigDict(env_prefix="DESKBOT_PERSONALITY__", extra="ignore")

    curiosity: float = Field(default=0.7, ge=0.0, le=1.0)
    energy: float = Field(default=0.6, ge=0.0, le=1.0)
    shyness: float = Field(default=0.3, ge=0.0, le=1.0)
    friendliness: float = Field(default=0.8, ge=0.0, le=1.0)
    playfulness: float = Field(default=0.7, ge=0.0, le=1.0)


#: SPI mode the GC9A01 driver uses. 0 = CPOL=0/CPHA=0 (Waveshare default).
SPI_MODE_0: int = 0
SPI_MODE_3: int = 3


class FaceConfig(BaseSettings):
    """Face rendering configuration.

    ``theme`` selects which built-in :class:`Theme` the renderer uses.
    The default ``"vector"`` is the Anki Vector 2.0 minimalist face
    (two glowing dots + a line); set to ``"minimal"``, ``"cute"``,
    ``"pixel"``, ``"retro_lcd"``, or ``"wireframe"`` for
    the other built-ins.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_FACE__", extra="ignore")

    theme: Literal["vector", "minimal", "cute", "pixel", "retro_lcd", "wireframe"] = "vector"


class DisplayConfig(BaseSettings):
    """Single circular TFT display configuration (e.g. GC9A01).

    ``backend`` selects which display driver to use:

    * ``"mock"`` - in-memory display used by tests and headless dev.
    * ``"gc9a01"`` - real GC9A01 over SPI (Pi 5 prototype).

    Hardware wiring (matches ``docs/wiring.md``):

    * ``bus`` + ``device`` select ``/dev/spidev<bus>.<device>``.
    * ``spi_hz`` is the SPI clock rate in Hz. Use 8 MHz for first bring-up;
      bump to 16 / 32 MHz once colour fills work.
    * ``spi_mode`` is the SPI mode (0 for GC9A01). Explicit, do not rely
      on whatever the OS default happens to be.
    * ``dc_pin`` drives the GC9A01's D/C line. REQUIRED for hardware.
    * ``reset_pin`` drives the panel RESET line. Strongly recommended.
      If unset, the driver assumes a wired-OR reset (panel always out of
      reset) - fine for some boards, but the safest choice is to wire
      RST to a free GPIO and set it here.
    * ``backlight_pin`` drives the panel BL/LED line. If unset, the driver
      assumes the backlight is hard-wired to 3V3 (always on). Do NOT
      drive a backlight pin without first confirming the panel's
      backlight voltage / current requirements.
    * ``invert`` controls the GC9A01's ``INVON``/``INVOFF`` at init.
      Some Waveshare panels want inversion ON, others want OFF - the
      diagnostic CLI exposes a flag so both can be tested.
    * ``chunk_bytes`` is the maximum SPI payload per ``writebytes``
      call. The Linux kernel caps each transaction at 4096 bytes by
      default; bump it only if you've raised ``SPI_MSGSIZ``.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_DISPLAYS__", extra="ignore")

    # Display backend selector:
    #   "mock"            - in-memory display (tests, headless dev)
    #   "gc9a01"          - raw SPI + GPIO driver (the original implementation)
    #   "circuitpython"   - Adafruit CircuitPython displayio + GC9A01A driver
    #                        (verified working on this exact panel)
    #   "cp" / "displayio" - aliases for "circuitpython"
    backend: Literal["mock", "gc9a01", "circuitpython", "cp", "displayio"] = "mock"
    bus: int = Field(default=0, ge=0)
    device: int = Field(default=0, ge=0)
    spi_hz: int = Field(default=8_000_000, ge=100_000, le=125_000_000)
    spi_mode: int = Field(default=SPI_MODE_0)

    @field_validator("spi_mode")
    @classmethod
    def _check_spi_mode(cls, value: int) -> int:
        if value not in (0, 1, 2, 3):
            raise ConfigurationError(f"invalid spi_mode {value}; must be 0..3")
        return value

    # BCM GPIO number for the GC9A01's DC (data/command) pin.
    # Required for hardware mode; ignored by mock. Default matches wiring.md.
    dc_pin: int | None = Field(default=25, ge=2, le=27)
    # BCM GPIO number for the GC9A01's RST pin. Optional: if None we
    # assume the pin is tied high on the board.
    reset_pin: int | None = Field(default=24, ge=2, le=27)
    # BCM GPIO number for the GC9A01's BL (backlight) pin. Optional:
    # if None we leave it alone (panel is hard-wired to 3V3).
    backlight_pin: int | None = Field(default=None, ge=2, le=27)
    # Whether the panel needs colour inversion at boot (INVON vs INVOFF).
    invert: bool = True
    # Maximum payload size per SPI write. The kernel cap is 4096 bytes.
    chunk_bytes: int = Field(default=4096, ge=64, le=65_536)
    width: int = Field(default=240, gt=0)
    height: int = Field(default=240, gt=0)
    fps: int = Field(default=30, gt=0, le=240)
    rotation: int = Field(default=0, ge=0, le=3)

    @field_validator("rotation")
    @classmethod
    def _check_rotation(cls, value: int) -> int:
        if value not in (0, 1, 2, 3):
            raise ConfigurationError(f"invalid rotation {value}; must be 0..3")
        return value

    @field_validator("dc_pin", "reset_pin", "backlight_pin")
    @classmethod
    def _warn_pins_must_not_collide(cls, value: int | None) -> int | None:
        # The check happens in ``validate_pins`` (cross-field); this hook
        # only ensures the type is honoured.
        return value

    def validate_pins(self) -> None:
        """Cross-field check that GPIO pins don't collide.

        Called by :func:`robot.hardware.displays.factory.DisplayFactory`
        before opening the SPI device.
        """
        pins = [p for p in (self.dc_pin, self.reset_pin, self.backlight_pin) if p is not None]
        if len(set(pins)) != len(pins):
            dupes = sorted({p for p in pins if pins.count(p) > 1})
            raise ConfigurationError(
                f"display GPIO pins must be unique (got duplicate(s) for {dupes}); "
                "verify dc_pin / reset_pin / backlight_pin."
            )

    def effective_spi_hz(self) -> int:
        """Return the SPI clock that should actually be programmed."""
        return self.spi_hz


class ServoChannelConfig(BaseSettings):
    """Parameters for a single servo channel, shared across all backends.

    The GPIO / PCA9685 specific fields (``chip`` and ``channel``) are only
    used by their respective backends. ``min_pulse_us`` and ``max_pulse_us``
    apply to all PWM backends.
    """

    model_config = SettingsConfigDict(extra="ignore")

    min_pulse_us: int = Field(default=500, gt=0)
    max_pulse_us: int = Field(default=2500, gt=0)
    min_angle_deg: float = Field(default=0.0, ge=-360.0, le=360.0)
    max_angle_deg: float = Field(default=180.0, ge=-360.0, le=360.0)
    inverted: bool = False

    @property
    def center_angle_deg(self) -> float:
        """Midpoint between ``min_angle_deg`` and ``max_angle_deg``."""
        return (self.min_angle_deg + self.max_angle_deg) / 2.0

    # PCA9685-specific (ignored by the GPIO backend).
    chip: int = Field(default=0, ge=0)
    channel: int = Field(default=0, ge=0, le=15)
    # GPIO-specific (ignored by the PCA9685 backend).
    gpio_pin: int | None = Field(default=None, ge=2, le=27)


class GPIOServoMapping(BaseSettings):
    """Map of logical servo name to BCM GPIO pin number."""

    model_config = SettingsConfigDict(env_prefix="DESKBOT_SERVOS__GPIO__", extra="ignore")

    pan: int = Field(default=12, ge=2, le=27)
    tilt: int = Field(default=13, ge=2, le=27)
    left_arm: int = Field(default=18, ge=2, le=27)
    right_arm: int = Field(default=19, ge=2, le=27)


class GPIOServoConfig(BaseSettings):
    """Configuration block for the GPIO servo backend."""

    model_config = SettingsConfigDict(env_prefix="DESKBOT_SERVOS__GPIO__", extra="ignore")

    frequency: int = Field(default=50, gt=0, le=1000)
    pins: GPIOServoMapping = Field(default_factory=GPIOServoMapping)
    # Logical servo name -> ServoChannelConfig. The factory pre-fills defaults
    # for pan/tilt/left_arm/right_arm when keys are missing.
    channels: dict[str, ServoChannelConfig] = Field(default_factory=dict)


class PCA9685ServoConfig(BaseSettings):
    """Configuration block for the PCA9685 servo backend."""

    model_config = SettingsConfigDict(env_prefix="DESKBOT_SERVOS__PCA9685__", extra="ignore")

    address: int = Field(default=0x40, ge=0x03, le=0x77)
    bus: int = Field(default=1, ge=0)
    frequency: int = Field(default=50, gt=0, le=1000)
    channels: dict[str, ServoChannelConfig] = Field(default_factory=dict)


class ServosConfig(BaseSettings):
    """Configuration for every servo the robot owns.

    ``backend`` selects which :class:`~robot.interfaces.servo.ServoController`
    implementation is used at runtime. The two backends are configured via
    the ``gpio`` and ``pca9685`` sub-blocks respectively; both are present
    at all times so a config switch only requires changing ``backend``.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_SERVOS__", extra="ignore")

    backend: ServoBackend = "mock"
    frequency: int = Field(default=50, gt=0, le=1000)
    gpio: GPIOServoConfig = Field(default_factory=GPIOServoConfig)
    pca9685: PCA9685ServoConfig = Field(default_factory=PCA9685ServoConfig)
    # Legacy single-board configuration kept for backward compatibility
    # with the original scaffold; both new backends ignore it.
    head_pan: ServoChannelConfig = Field(
        default_factory=lambda: ServoChannelConfig(
            min_pulse_us=500, max_pulse_us=2500, min_angle_deg=30, max_angle_deg=150, gpio_pin=12
        )
    )
    head_tilt: ServoChannelConfig = Field(
        default_factory=lambda: ServoChannelConfig(
            min_pulse_us=500, max_pulse_us=2500, min_angle_deg=45, max_angle_deg=135, gpio_pin=13
        )
    )
    left_arm: ServoChannelConfig = Field(
        default_factory=lambda: ServoChannelConfig(
            min_pulse_us=500, max_pulse_us=2500, min_angle_deg=20, max_angle_deg=160, gpio_pin=18
        )
    )
    right_arm: ServoChannelConfig = Field(
        default_factory=lambda: ServoChannelConfig(
            min_pulse_us=500, max_pulse_us=2500, min_angle_deg=20, max_angle_deg=160, gpio_pin=19
        )
    )

    def channels_for_backend(self) -> dict[str, ServoChannelConfig]:
        """Return the channel config the active backend should use."""
        match self.backend:
            case "gpio":
                return self.gpio.channels
            case "pca9685":
                return self.pca9685.channels
            case _:
                return {}


class AudioConfig(BaseSettings):
    """Audio output (speaker) configuration.

    The ``backend`` field selects which audio driver to use:

    * ``"mock"`` - in-memory audio that records bytes but produces no
      sound (safe for tests and headless dev).
    * ``"usb"`` - ``UsbSpeaker`` using ``sounddevice`` / ALSA.
      Good for USB speakers and the 3.5 mm headphone jack.
    * ``"bluetooth"`` - ``BluetoothSpeaker`` using PulseAudio /
      PipeWire to route audio to a paired Bluetooth A2DP sink.

    When ``backend`` is ``"bluetooth"``, set ``bluetooth_mac`` or
    ``bluetooth_name`` so the speaker can auto-discover the correct
    PulseAudio sink. If neither is set, it falls back to the first
    Bluetooth sink it finds.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_AUDIO__", extra="ignore")

    backend: Literal["usb", "bluetooth", "mock"] = "mock"
    output_device: str = Field(default="default")
    sample_rate: int = Field(default=48_000, gt=0)
    channels: int = Field(default=1, ge=1, le=8)
    dtype: str = Field(default="int16")
    # Bluetooth-specific (only used when backend="bluetooth").
    bluetooth_mac: str = Field(
        default="",
        description="MAC address of the Bluetooth A2DP device (e.g. 'AA:BB:CC:DD:EE:FF').",
    )
    bluetooth_name: str = Field(
        default="", description="Friendly name substring of the Bluetooth device."
    )
    bluetooth_auto_connect: bool = Field(
        default=True, description="Whether to auto-connect to the Bluetooth sink on first play()."
    )


class MicrophoneConfig(BaseSettings):
    """Microphone capture configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DESKBOT_MICROPHONE__",
        extra="ignore",
    )

    backend: Literal["usb", "rtsp"] = "usb"

    input_device: str = Field(default="default")
    sample_rate: int = Field(default=16_000, gt=0)
    channels: int = Field(default=1, ge=1, le=8)
    frame_ms: int = Field(default=30, gt=0)
    dtype: str = Field(default="int16")

    rtsp_transport: Literal["tcp", "udp"] = Field(
        default="tcp",
        description="Transport used by the RTSP microphone backend.",
    )


class CameraConfig(BaseSettings):
    """Camera capture configuration.

    ``backend`` selects which camera driver to use:

    * ``"mock"`` - in-memory camera for tests and headless dev.
    * ``"usb"`` - :class:`~robot.hardware.sensors.usb_camera.UsbCamera`
      over a local V4L2 device (e.g. USB webcam).
    * ``"rtsp"`` - :class:`~robot.hardware.sensors.rtsp_camera.RtspCamera`
      over an RTSP stream.  Requires ``rtsp_url``.

    When ``backend`` is ``"rtsp"``, set ``rtsp_url`` to the full RTSP
    stream URL (e.g. ``rtsp://192.168.1.50:554/stream``).
    The ``device`` field is ignored in this mode.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_CAMERA__", extra="ignore")

    backend: Literal["mock", "usb", "rtsp"] = "mock"
    device: int = Field(default=0, ge=0)
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)
    fps: int = Field(default=30, gt=0)
    rtsp_url: str = Field(
        default="",
        description="RTSP stream URL (required when backend='rtsp').",
    )


class PerceptionConfig(BaseSettings):
    """Perception (face detection) configuration."""

    model_config = SettingsConfigDict(env_prefix="DESKBOT_PERCEPTION__", extra="ignore")

    enabled: bool = Field(default=True)
    scan_interval_s: float = Field(default=0.5, gt=0.0)
    idle_scan_interval_s: float = Field(
        default=2.0, gt=0.0, description="Scan interval when robot is IDLE (slower to save CPU)."
    )
    curious_scan_interval_s: float = Field(
        default=0.3,
        gt=0.0,
        description="Scan interval when robot is CURIOUS (faster for tracking).",
    )
    max_faces: int = Field(default=3, ge=0)
    score_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Minimum confidence for YuNet face detection. Lower = more detections but more false positives.",
    )
    scale_factor: float = Field(default=1.3, gt=1.0)
    min_neighbors: int = Field(default=4, ge=1)


class LLMConfig(BaseSettings):
    """Language model provider configuration."""

    model_config = SettingsConfigDict(env_prefix="DESKBOT_LLM__", extra="ignore")

    provider: Literal["mock", "openai", "ollama", "custom"] = "mock"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, gt=0)
    timeout_s: float = Field(default=15.0, gt=0.0)


class PiperConfig(BaseSettings):
    """Piper TTS configuration."""

    model_config = SettingsConfigDict(env_prefix="DESKBOT_TTS__PIPER__", extra="ignore")

    model: str = Field(
        default="en_US-lessac-medium",
        description="Piper voice model name. Available models: "
        "https://github.com/rhasspy/piper/blob/master/VOICES.md",
    )
    download_dir: str = Field(
        default="",
        description="Directory for Piper model files. Empty string uses ~/.local/share/piper/.",
    )
    use_cuda: bool = Field(
        default=False,
        description="Use GPU (CUDA) for inference if available.",
    )
    speaker_id: int | None = Field(
        default=None,
        description="Speaker ID for multi-speaker models (0-based).",
    )
    noise_scale: float | None = Field(default=None, ge=0.0, le=2.0)
    length_scale: float | None = Field(default=None, gt=0.0, le=3.0)
    noise_w_scale: float | None = Field(default=None, ge=0.0, le=2.0)


class ElevenLabsConfig(BaseSettings):
    """Configuration for ElevenLabs cloud TTS.

    ElevenLabs provides high-quality cloud-based speech synthesis with
    many natural-sounding voices. A paid plan is required for production
    use; a free tier is available for testing (10 000 characters/month).

    Find voice IDs at: https://api.elevenlabs.io/v1/voices

    Available models:
    - ``eleven_monolingual_v1`` - English only, lowest latency.
    - ``eleven_multilingual_v1`` - Multi-language, good quality.
    - ``eleven_multilingual_v2`` - Multi-language, highest quality (default).
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_TTS__ELEVENLABS__", extra="ignore")

    api_key: str = Field(
        default="",
        description="ElevenLabs API key. Required when provider='elevenlabs'.",
    )
    voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM",
        description="ElevenLabs voice ID. Default is 'Rachel'.",
    )
    model_id: str = Field(
        default="eleven_multilingual_v2",
        description="ElevenLabs model ID. 'eleven_multilingual_v2' recommended.",
    )
    stability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice stability. Higher = more consistent.",
    )
    similarity_boost: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Voice clarity + similarity. Higher = more similar to original.",
    )


class TTSConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DESKBOT_TTS__", extra="ignore")

    provider: Literal["mock", "piper", "elevenlabs", "openai", "espeak"] = "mock"
    voice: str = "default"
    rate: float = Field(default=1.0, gt=0.0)
    piper: PiperConfig = Field(default_factory=PiperConfig)
    elevenlabs: ElevenLabsConfig = Field(default_factory=ElevenLabsConfig)


class STTConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DESKBOT_STT__", extra="ignore")

    provider: Literal["mock", "whisper", "vosk", "google"] = "mock"
    model: str = "base"
    language: str = "en"


class WakeWordConfig(BaseSettings):
    """Wake-word detection configuration.

    ``provider`` selects which wake-word engine to use:

    * ``"mock"`` - :class:`MockWakeWordChecker` that triggers after N chunks.
      Good for integration tests. NOT a production detector.
    * ``"openwakeword"`` - model-based detection provided by the optional
      ``deskbot[wakeword]`` dependency. This is the recommended production
      backend.
    * ``"porcupine"`` - Picovoice Porcupine engine (requires ``pvporcupine``).
    * ``"snowboy"`` - Snowboy hotword engine (requires ``snowboy``).

    Energy / RMS / volume is **not** a valid wake-word provider: it was
    removed because loud audio is not a wake phrase. Selecting it is a
    configuration validation error.

    When ``provider`` is ``"porcupine"``, set ``porcupine_access_key`` and
    optionally ``porcupine_keyword`` or ``porcupine_model_path``.
    When ``provider`` is ``"snowboy"``, set ``snowboy_model_path``.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_WAKEWORD__", extra="ignore")

    provider: Literal["mock", "openwakeword", "porcupine", "snowboy"] = "mock"
    phrase: str = "hey deskbot"
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Wake-word confidence threshold for model-based detectors.",
    )
    model_path: str = Field(
        default="",
        description="Optional path to a custom openWakeWord ONNX model.",
    )
    # Porcupine-specific configuration.
    porcupine_access_key: str = Field(
        default="",
        description="Picovoice AccessKey for Porcupine wake-word detection. "
        "Get a free key at https://console.picovoice.ai/",
    )
    porcupine_keyword: str = Field(
        default="picovoice",
        description="Built-in Porcupine keyword (e.g. 'picovoice', 'hey google'). "
        "Ignored when porcupine_model_path is set.",
    )
    porcupine_model_path: str = Field(
        default="",
        description="Path to a custom Porcupine wake-word model (.ppn file). "
        "When set, takes priority over porcupine_keyword.",
    )
    # Snowboy-specific configuration.
    snowboy_model_path: str = Field(
        default="",
        description="Path to a Snowboy model file (.pmdl or .umdl). "
        "Required when provider is 'snowboy'.",
    )
    wake_animation: bool = Field(
        default=True,
        description="Whether to play the visual wake animation when a wake word is detected. "
        "When False, the face transitions directly to the 'curious' emotion.",
    )


class ApiConfig(BaseSettings):
    """REST API and WebSocket server configuration.

    By default the API binds to ``127.0.0.1`` so it is only reachable
    from the local machine.  Set ``host`` to ``0.0.0.0`` to expose it
    on the network - **only do this behind a reverse proxy or with
    ``api_key`` set**, since the control endpoints can drive servos
    and speakers.

    When ``api_key`` is non-empty, all mutating/control endpoints
    require an ``Authorization: Bearer <key>`` header (or
    ``?api_key=<key>`` query parameter).  Read-only health endpoints
    remain open.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_API__", extra="ignore")

    enabled: bool = Field(default=True, description="Whether to start the REST API server.")
    host: str = Field(default="127.0.0.1", description="Bind address for the API server.")
    port: int = Field(default=8000, ge=0, le=65535, description="Port for the API server.")
    api_key: str = Field(
        default="",
        description="Shared secret for control endpoints. When empty, no auth is enforced "
        "(only safe when bound to 127.0.0.1).",
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8000"],
        description="Allowed CORS origins. Default allows only the local dashboard.",
    )


class PerformanceConfig(BaseSettings):
    """Performance profiling configuration.

    When ``enabled`` is ``False``, all profilers are no-ops with zero
    overhead. Individual profilers can be toggled independently.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_PERFORMANCE__", extra="ignore")

    enabled: bool = Field(default=True, description="Master switch for performance profiling.")
    frame_profiling: bool = Field(default=True, description="Enable frame budget monitoring.")
    servo_profiling: bool = Field(default=True, description="Enable servo latency profiling.")
    bus_profiling: bool = Field(default=True, description="Enable event bus throughput profiling.")
    bus_sample_rate: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Fraction of bus events to sample (0.1 = 10%).",
    )
    report_interval_frames: int = Field(
        default=100,
        gt=0,
        description="Publish FrameStatsReport event every N frames.",
    )


class SoundsConfig(BaseSettings):
    """Sound effects configuration."""

    model_config = SettingsConfigDict(env_prefix="DESKBOT_SOUNDS__", extra="ignore")

    enabled: bool = Field(default=True, description="Whether to play sound effects.")
    volume: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Sound effects volume (0.0-1.0)."
    )
    reactions_enabled: bool = Field(
        default=True,
        description="Whether to automatically play sound effects in reaction to "
        "emotion/state changes (e.g. 'thinking' while pondering, 'angry'/'surprise' "
        "when emotions change).",
    )


class MemoryConfig(BaseSettings):
    """Configuration for short-term conversational memory."""

    model_config = SettingsConfigDict(env_prefix="DESKBOT_MEMORY__", extra="ignore")

    enabled: bool = Field(default=True, description="Whether conversation memory is used.")
    capacity: int = Field(default=1024, ge=1, le=10_000)
    recall_limit: int = Field(default=5, ge=1, le=50)


class VectorMemoryConfig(BaseSettings):
    """Configuration for semantic (vector-based) memory.

    When ``enabled`` is ``True``, the conversation service uses
    :class:`VectorMemory` instead of the keyword-based :class:`Memory`,
    enabling semantic search over past interactions.

    The ``backend`` field selects the embedding provider:

    * ``"none"`` -- uses :class:`NoOpEmbedding` (zero vectors); search
      falls back to chronological order. No extra dependencies needed.
    * ``"sentence_transformers"`` -- uses :class:`SentenceTransformerEmbedding`
      for real semantic similarity. Requires ``pip install sentence-transformers``.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_VECTOR_MEMORY__", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Whether to use vector-based semantic memory instead of keyword search.",
    )
    backend: Literal["none", "sentence_transformers"] = Field(
        default="none",
        description="Embedding backend: 'none' (NoOpEmbedding) or 'sentence_transformers'.",
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name (only used when backend='sentence_transformers').",
    )
    capacity: int = Field(default=2048, ge=1, le=10_000)
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for search_similar results.",
    )
    recall_limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of entries returned by semantic recall.",
    )


class PreferencesConfig(BaseSettings):
    """Configuration for user preference learning.

    When ``enabled`` is ``True``, the conversation service extracts
    user preferences from utterances (e.g. "my name is Alice",
    "be funny") and injects them into the system prompt so the
    robot can personalise its responses.

    The ``store`` field selects the persistence backend:

    * ``"memory"`` -- in-memory store, lost on restart (good for dev).
    * ``"sqlite"`` -- SQLite-backed store, survives restarts.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_PREFERENCES__", extra="ignore")

    enabled: bool = Field(
        default=True,
        description="Whether to learn and apply user preferences.",
    )
    store: Literal["memory", "sqlite"] = Field(
        default="memory",
        description="Preference store backend: 'memory' (no persistence) or 'sqlite'.",
    )
    db_path: str = Field(
        default="~/.deskbot/preferences.db",
        description="Path to the SQLite database file (only used when store='sqlite').",
    )


class ConversationConfig(BaseSettings):
    """Conversation persistence configuration."""

    model_config = SettingsConfigDict(env_prefix="DESKBOT_CONVERSATION__", extra="ignore")

    store: Literal["memory", "sqlite"] = Field(
        default="memory",
        description="Conversation persistence backend: 'memory' (no persistence) or 'sqlite'.",
    )
    db_path: str = Field(
        default="~/.deskbot/conversations.db",
        description="Path to the SQLite database file (only used when store='sqlite').",
    )
    conversation_id: str = Field(
        default="default",
        description="ID for the default conversation.",
    )


class ToolConfig(BaseSettings):
    """Configuration for LLM function / tool calling.

    When ``enabled`` is ``True``, the conversation service will include
    tool definitions in LLM calls and dispatch tool calls from LLM
    responses through the :class:`ToolExecutor`.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_TOOLS__", extra="ignore")

    enabled: bool = Field(
        default=True,
        description="Whether to include tool definitions in LLM calls.",
    )


class PluginConfig(BaseSettings):
    """Configuration for the plugin system.

    When ``enabled`` is ``True``, the application will discover and load
    plugins from ``deskbot.plugins`` entry points at startup.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_PLUGINS__", extra="ignore")

    enabled: bool = Field(
        default=True,
        description="Whether to enable the plugin system.",
    )
    discover_entry_points: bool = Field(
        default=True,
        description="Whether to discover plugins from the deskbot.plugins entry point group.",
    )
    plugin_packages: list[str] = Field(
        default_factory=list,
        description="Explicit list of plugin package names to import and register.",
    )


class MqttConfig(BaseSettings):
    """MQTT bridge configuration.

    When ``enabled`` is ``True``, the app starts an MQTT bridge that
    publishes DeskBot events to a broker and subscribes to command
    topics for remote control.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_MQTT__", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Whether to start the MQTT bridge.",
    )
    host: str = Field(default="localhost", description="MQTT broker host.")
    port: int = Field(default=1883, ge=1, le=65535, description="MQTT broker port.")
    username: str = Field(default="", description="MQTT broker username (empty = no auth).")
    password: str = Field(default="", description="MQTT broker password.")
    topic_prefix: str = Field(default="deskbot", description="MQTT topic prefix.")
    keepalive: int = Field(
        default=60, ge=10, le=300, description="MQTT keepalive interval in seconds."
    )
    qos: int = Field(default=1, ge=0, le=2, description="MQTT QoS level (0, 1, or 2).")
    publish_events: bool = Field(
        default=True, description="Whether to publish local events to MQTT."
    )
    subscribe_commands: bool = Field(
        default=True, description="Whether to subscribe to command topics."
    )
    heartbeat_interval: int = Field(
        default=30, ge=5, le=300, description="Heartbeat interval in seconds."
    )


class HomeAssistantConfig(BaseSettings):
    """Home Assistant MQTT discovery configuration.

    When ``enabled`` is ``True``, the app starts a Home Assistant
    bridge that publishes MQTT Auto Discovery payloads so the robot
    appears as a native device in Home Assistant without manual YAML.

    Requires ``paho-mqtt`` (same dependency as the MQTT bridge).
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_HOMEASSISTANT__", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Whether to start the Home Assistant MQTT bridge.",
    )
    host: str = Field(default="homeassistant.local", description="MQTT broker host.")
    port: int = Field(default=1883, ge=1, le=65535, description="MQTT broker port.")
    username: str = Field(default="", description="MQTT broker username (empty = no auth).")
    password: str = Field(default="", description="MQTT broker password.")
    discovery_prefix: str = Field(
        default="homeassistant",
        description="HA MQTT discovery prefix (typically 'homeassistant').",
    )
    device_id: str = Field(default="deskbot", description="HA device identifier.")
    device_name: str = Field(default="DeskBot", description="HA device display name.")
    device_manufacturer: str = Field(
        default="DeskBot Contributors",
        description="HA device manufacturer.",
    )
    device_model: str = Field(
        default="Desktop Companion Robot",
        description="HA device model.",
    )
    qos: int = Field(default=1, ge=0, le=2, description="MQTT QoS level.")


class LearningConfig(BaseSettings):
    """Configuration for local on-device learning.

    When ``enabled`` is ``True``, the experience recorder subscribes
    to the event bus and records observations/actions as experience
    tuples.  These are used by the learning system for training the
    world model and action/value functions.

    The ``store`` field selects the persistence backend:

    * ``"memory"`` -- in-memory store, lost on restart (good for dev).
    * ``"sqlite"`` -- SQLite-backed store, survives restarts.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_LEARNING__", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Whether to enable experience recording and learning.",
    )
    store: Literal["memory", "sqlite"] = Field(
        default="memory",
        description="Experience store backend: 'memory' (no persistence) or 'sqlite'.",
    )
    db_path: str = Field(
        default="~/.deskbot/experiences.db",
        description="Path to the SQLite database file (only used when store='sqlite').",
    )
    working_memory_capacity: int = Field(
        default=256,
        ge=1,
        le=10_000,
        description="Maximum number of experiences in working memory.",
    )
    replay_buffer_capacity: int = Field(
        default=10_000,
        ge=100,
        le=1_000_000,
        description="Maximum number of experiences in the replay buffer.",
    )
    episodic_capacity: int = Field(
        default=10_000,
        ge=100,
        le=1_000_000,
        description="Maximum number of experiences in episodic memory.",
    )
    replay_seed: int = Field(
        default=42,
        description="Random seed for replay buffer sampling (reproducibility).",
    )

    # Training schedule
    min_new_experiences: int = Field(
        default=32,
        ge=1,
        description="Minimum new experiences since last training before triggering a cycle.",
    )
    train_interval_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Minimum seconds between training cycles.",
    )
    min_experiences_for_training: int = Field(
        default=64,
        ge=1,
        description="Minimum total experiences in replay buffer before any training.",
    )

    # Resource limits
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Mini-batch size for training.",
    )
    training_epochs_per_cycle: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of training epochs per background training cycle.",
    )
    eval_sample_size: int = Field(
        default=128,
        ge=16,
        le=10_000,
        description="Number of experiences to sample for candidate evaluation.",
    )
    max_cpu_fraction: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description="Target maximum CPU fraction for the training thread (soft limit).",
    )
    max_model_params: int = Field(
        default=500_000,
        ge=1000,
        description="Maximum trainable parameters per model.",
    )

    # Checkpointing
    checkpoint_dir: str = Field(
        default="~/.deskbot/checkpoints",
        description="Directory for model checkpoints.",
    )
    keep_last_n_checkpoints: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of checkpoints to keep on disk.",
    )
    promote_threshold: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Candidate is promoted if its loss <= current_loss * this threshold. "
        "1.0 means promote if equal or better; 0.95 means require at least 5% improvement.",
    )

    # Multimodal encoding
    use_multimodal: bool = Field(
        default=False,
        description="When True, use the MultimodalEncoder (trainable vision/audio sub-encoders "
        "+ temporal history) instead of the plain StateEncoder. Produces a richer state "
        "vector at the cost of more parameters and compute.",
    )
    multimodal_history_length: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Number of recent state snapshots included in the multimodal vector "
        "for temporal context. 0 disables history (uses only the base + sub-encoder outputs).",
    )


class TeachingConfig(BaseSettings):
    """Configuration for the human teaching loop (Phases 8-10).

    Teaching is a *context flag*, not a :class:`RobotState` — it never appears
    in the state one-hot, so ``STATE_SIZE`` (91 / 570 multimodal) is unchanged.
    When ``enabled`` is ``False`` the teaching controller stays inactive and
    no gesture triggers fire.
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_TEACHING__", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Whether the teaching loop is armed (gesture-triggered demonstrate/practice).",
    )
    feedback_window_s: float = Field(
        default=5.0,
        gt=0.0,
        description="Maximum age (s) of a transition to be eligible for feedback attribution.",
    )
    staleness_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Broader staleness bound for applying recorded feedback to a transition.",
    )
    practice_epsilon: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Exploration epsilon the policy uses in practice mode.",
    )
    cooldown_s: float = Field(
        default=0.2,
        gt=0.0,
        description="Safety-gate cooldown (s) between the same action during teaching.",
    )
    min_experiences_for_practice: int = Field(
        default=64,
        ge=0,
        description="Total experiences required before practice mode trusts a policy proposal; "
        "below this it falls back to demonstration.",
    )


class TelegramConfig(BaseSettings):
    """Telegram bot bridge configuration.

    When ``enabled`` is ``True``, the app starts a Telegram bot that lets
    you chat with DeskBot and control every aspect via Telegram messages.

    Requires ``httpx`` (already a core API dependency).

    Create a bot token by talking to ``@BotFather`` on Telegram, then set::

        DESKBOT_TELEGRAM__ENABLED=true
        DESKBOT_TELEGRAM__BOT_TOKEN=123456:ABC-DEF...
        DESKBOT_TELEGRAM__ALLOWED_USER_IDS=[123456789]
    """

    model_config = SettingsConfigDict(env_prefix="DESKBOT_TELEGRAM__", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Whether to start the Telegram bot bridge.",
    )
    bot_token: str = Field(
        default="",
        description="Telegram bot token from @BotFather.",
    )
    allowed_user_ids: list[int] = Field(
        default_factory=list,
        description="If non-empty, only these Telegram user IDs may interact with the bot.",
    )
    chat_timeout_s: float = Field(
        default=60.0,
        gt=0.0,
        description="Seconds to wait for a BotReply event before timing out.",
    )
    api_base: str = Field(
        default="https://api.telegram.org",
        description="Telegram Bot API base URL (override for self-hosted instances).",
    )


class AppSettings(BaseSettings):
    """Root configuration for the DeskBot application."""

    model_config = SettingsConfigDict(
        env_prefix="DESKBOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the YAML source between env and dotenv.

        Precedence (lowest first): init < YAML < dotenv < env. Env vars win
        over everything so ``DESKBOT_DISPLAYS__BACKEND=gc9a01`` always
        beats a ``display: { backend: circuitpython }`` line in the YAML.
        """
        # Precedence (lowest first -> highest last):
        # YAML < dotenv < env < init.
        # Env vars win over YAML and the .env file.
        return (
            env_settings,
            dotenv_settings,
            _yaml_config_source(settings_cls),
            init_settings,
            file_secret_settings,
        )

    env: Env = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    timezone: str = "UTC"
    hardware: Literal["mock", "real"] = "mock"
    config_file: Path | None = None
    assets_dir: Path = Field(default=Path("assets"))
    use_mocks: bool = True

    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    displays: DisplayConfig = Field(default_factory=DisplayConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)
    servos: ServosConfig = Field(default_factory=ServosConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    microphone: MicrophoneConfig = Field(default_factory=MicrophoneConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)
    sounds: SoundsConfig = Field(default_factory=SoundsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    vector_memory: VectorMemoryConfig = Field(default_factory=VectorMemoryConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    teaching: TeachingConfig = Field(default_factory=TeachingConfig)
    preferences: PreferencesConfig = Field(default_factory=PreferencesConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    mqtt: MqttConfig = Field(default_factory=MqttConfig)
    homeassistant: HomeAssistantConfig = Field(default_factory=HomeAssistantConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @field_validator("timezone")
    @classmethod
    def _check_timezone(cls, value: str) -> str:
        if not value:
            raise ConfigurationError("timezone must not be empty")
        return value

    def is_production(self) -> bool:
        return self.env == "production"

    def is_development(self) -> bool:
        return self.env == "development"


def load_settings() -> AppSettings:
    """Build an :class:`AppSettings` instance from environment + .env."""
    return AppSettings()


__all__ = [
    "SPI_MODE_0",
    "SPI_MODE_3",
    "ApiConfig",
    "AppSettings",
    "AudioConfig",
    "CameraConfig",
    "ConversationConfig",
    "DisplayConfig",
    "ElevenLabsConfig",
    "GPIOServoConfig",
    "GPIOServoMapping",
    "LLMConfig",
    "LearningConfig",
    "MemoryConfig",
    "MicrophoneConfig",
    "PCA9685ServoConfig",
    "PerceptionConfig",
    "PerformanceConfig",
    "PersonalityConfig",
    "PiperConfig",
    "PreferencesConfig",
    "STTConfig",
    "ServoBackend",
    "ServoChannelConfig",
    "ServosConfig",
    "SoundsConfig",
    "TTSConfig",
    "TelegramConfig",
    "VectorMemoryConfig",
    "WakeWordConfig",
    "load_settings",
]


def _yaml_config_source(settings_cls: type[BaseSettings]) -> YamlConfigSettingsSource:
    """Return a YAML config source if ``DESKBOT_CONFIG_FILE`` is set.

    The file path comes from the ``DESKBOT_CONFIG_FILE`` env var. When
    set, the YAML is parsed and each top-level key is mapped to the
    corresponding nested settings class. Environment variables still
    take precedence - the YAML source is consulted only when a key is
    missing in the environment.

    If no YAML file is configured we return a YamlConfigSettingsSource
    pointing at the current directory. pydantic-settings requires every
    source to be callable; the source becomes a no-op when the path
    doesn't exist.
    """
    from pydantic_settings.sources.providers.yaml import YamlConfigSettingsSource

    path_str = os.environ.get("DESKBOT_CONFIG_FILE")
    path = Path(path_str).expanduser() if path_str else Path()
    return YamlConfigSettingsSource(settings_cls, yaml_file=path)
