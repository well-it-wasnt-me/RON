"""Application bootstrap.

:class:`DeskBotApp` wires every component together using a
:class:`Container`. It owns the long-running tasks (event loop, eye
animator, idle behaviour, services) and exposes a single ``run()`` coroutine.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import anyio

from robot.ai.conversation import ConversationManager
from robot.ai.conversation_store import ConversationStore
from robot.ai.llm_mock import MockLLM
from robot.ai.llm_ollama import OllamaLLM
from robot.ai.llm_openai import OpenAILLM
from robot.ai.memory_factory import create_memory
from robot.ai.preferences import (
    InMemoryPreferenceStore,
    PreferenceTracker,
    SqlitePreferenceStore,
)
from robot.ai.prompts import system_prompt
from robot.ai.tools.executor import ToolExecutor
from robot.ai.tools.registry import BUILTIN_TOOLS, ToolRegistry
from robot.api.calibration import set_calibration_state
from robot.api.state_bridge import StateBridge
from robot.behavior.actions import BehaviorAction
from robot.behavior.idle import IdleBehavior
from robot.behavior.perception_behavior import PerceptionBehavior
from robot.behavior.personality import Personality
from robot.behavior.reactions import ReactionEngine
from robot.behavior.sound_reactor import SoundReactor
from robot.behavior.state_machine import RobotState, StateMachine
from robot.config import AppSettings, TTSConfig, WakeWordConfig, load_settings
from robot.dependency_container import Container
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    EmotionChanged,
    EmotionName,
    RobotStarted,
    RobotStopped,
)
from robot.eye_engine.animator import EyeDisplayAnimator
from robot.face.animator import FaceAnimator
from robot.face.emotions import EmotionEngine
from robot.face.face_orchestrator import FaceOrchestrator
from robot.face.renderer import FaceRenderer
from robot.face.themes.base import Theme
from robot.hardware.audio.mock_audio import MockAudioOutput
from robot.hardware.displays.factory import DisplayFactory
from robot.hardware.displays.mock_display import MockDisplay
from robot.hardware.sensors.mock_camera import MockCamera
from robot.hardware.sensors.mock_microphone import MockMicrophone
from robot.hardware.sensors.rtsp_camera import RtspCamera
from robot.hardware.sensors.rtsp_microphone import RtspMicrophone
from robot.hardware.sensors.usb_camera import UsbCamera
from robot.hardware.sensors.usb_microphone import UsbMicrophone
from robot.hardware.servos.factory import ServoControllerFactory
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.interfaces.audio import AudioOutput
from robot.interfaces.camera import Camera
from robot.interfaces.display import Display
from robot.interfaces.llm import LLM
from robot.interfaces.microphone import Microphone
from robot.interfaces.servo import ServoController
from robot.learning.experience import (
    EpisodicMemory,
    ReplayBuffer,
    SqliteExperienceStore,
    WorkingMemory,
)
from robot.learning.learning_service import (
    CheckpointConfig,
    LearningSchedule,
    LearningService,
    ResourceLimits,
)
from robot.learning.observation_adapter import LearningObservationAdapter
from robot.learning.preference_learner import PreferenceLearner
from robot.learning.safety import LearningSafetyManager
from robot.lifecycle import Lifecycle
from robot.lifecycle.degradation import DegradationEntry, DegradationRegistry, safe_init
from robot.logging import configure_logging, get_logger
from robot.perception import PerceptionService
from robot.performance.bus_profiler import BusProfiler
from robot.performance.frame_profiler import FrameProfiler
from robot.performance.servo_profiler import ServoProfiler
from robot.plugins.registry import PluginError, PluginRegistry
from robot.services.conversation_service import ConversationService
from robot.services.executor import ActionExecutor
from robot.speech.sound_effects import SoundEffectsPlayer
from robot.speech.stt import MockSTT, SpeechToText
from robot.speech.stt_whisper import StreamingWhisperAdapter, WhisperSTT
from robot.speech.tts import MockTTS, TextToSpeech
from robot.speech.tts_espeak import EspeakNGTTS
from robot.speech.tts_openai import OpenAITTS
from robot.speech.tts_piper import PiperTTS
from robot.speech.wakeword import (
    MockWakeWordChecker,
    NullWakeWordChecker,
    WakeWordChecker,
)
from robot.utils.clock import SystemClock
from robot.utils.random_source import SystemRandomSource

_log = get_logger("app")


@dataclass(slots=True)
class DeskBotApp:
    """The high-level DeskBot application.

    Tests construct the app directly with hand-rolled dependencies; production
    uses :meth:`from_settings` to build the default mock stack.
    """

    settings: AppSettings
    container: Container
    bus: InMemoryEventBus
    state_machine: StateMachine
    lifecycle: Lifecycle
    personality: Personality
    display: Display
    servo_controller: ServoController
    face_animator: FaceAnimator | None = None
    eye_animator: EyeDisplayAnimator | None = None  # back-compat
    idle: IdleBehavior | None = None
    reactions: ReactionEngine | None = None
    executor: ActionExecutor | None = None
    conversation: ConversationService | None = None
    perception: PerceptionService | None = None
    _perception_behavior: PerceptionBehavior | None = None
    _microphone: Microphone | None = field(default=None, init=False, repr=False)
    _camera: Camera | None = field(default=None, init=False, repr=False)
    _audio: AudioOutput | None = field(default=None, init=False, repr=False)
    _sound_effects: SoundEffectsPlayer | None = field(default=None, init=False, repr=False)
    _sound_reactor: SoundReactor | None = field(default=None, init=False, repr=False)
    _learning_service: LearningService | None = field(default=None, init=False, repr=False)
    _safety_manager: LearningSafetyManager | None = field(default=None, init=False, repr=False)
    _observation_adapter: LearningObservationAdapter | None = field(
        default=None, init=False, repr=False
    )
    _api_bridge: StateBridge | None = field(default=None, init=False, repr=False)
    _api_server: object | None = field(default=None, init=False, repr=False)
    _api_thread: object | None = field(default=None, init=False, repr=False)
    _plugin_registry: PluginRegistry | None = field(default=None, init=False, repr=False)
    _mqtt_bridge: object | None = field(default=None, init=False, repr=False)
    _ha_bridge: object | None = field(default=None, init=False, repr=False)
    _telegram_bridge: object | None = field(default=None, init=False, repr=False)
    _task_group: anyio.abc.TaskGroup | None = field(default=None, init=False, repr=False)
    _drain_stop: bool = field(default=False, init=False, repr=False)
    _degradation: DegradationRegistry | None = field(default=None, init=False, repr=False)
    _frame_profiler: FrameProfiler | None = field(default=None, init=False, repr=False)
    _servo_profiler: ServoProfiler | None = field(default=None, init=False, repr=False)
    _bus_profiler: BusProfiler | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> DeskBotApp:
        """Build the default mock-stack application."""
        settings = settings or load_settings()
        configure_logging(settings)
        return cls.build(settings)

    @classmethod
    def build(cls, settings: AppSettings) -> DeskBotApp:  # noqa: PLR0912
        """Build the application.

        The servo backend is selected by ``settings.servos.backend``. The
        rest of the wiring is identical across backends.

        All hardware components use :func:`safe_init` so that failures
        fall back to mocks and the degradation is recorded in the
        :class:`DegradationRegistry`.  The robot must **never crash** due
        to a missing hardware component.
        """
        bus = InMemoryEventBus()
        state_machine = StateMachine(bus=bus)
        personality = Personality.from_config(settings.personality)
        clock = SystemClock()
        rng = SystemRandomSource()
        degradation = DegradationRegistry()

        # Display - selected by configuration (mock / gc9a01 / circuitpython).
        # Safe-init: fall back to MockDisplay if the real driver fails.
        display: Display = safe_init(
            factory=lambda: DisplayFactory(settings.displays).build(),
            component="display",
            fallback=lambda: MockDisplay(
                width=settings.displays.width, height=settings.displays.height
            ),
            registry=degradation,
            original_backend=settings.displays.backend,
            fallback_backend="mock",
        )

        # Servos - selected by configuration.
        # Safe-init: fall back to MockServoBus if the real driver fails.
        servo_controller: ServoController = safe_init(
            factory=lambda: ServoControllerFactory(settings.servos).build(),
            component="servos",
            fallback=_mock_servo_bus,
            registry=degradation,
            original_backend=settings.servos.backend,
            fallback_backend="mock",
        )

        # Audio output - selected by settings.audio.backend.
        audio: AudioOutput = _build_audio(settings, degradation)

        # Sound effects - play WAV files from assets/sounds/ through AudioOutput.
        sound_effects = SoundEffectsPlayer(
            audio=audio,
            enabled=settings.sounds.enabled,
            volume=settings.sounds.volume,
            bus=bus,
        )
        _log.info("sound_effects.enabled", enabled=settings.sounds.enabled)

        # Sound reactor: play emotion/state sound effects automatically.
        sound_reactor = SoundReactor(
            bus=bus,
            sound_effects=sound_effects,
            enabled=settings.sounds.enabled and settings.sounds.reactions_enabled,
        )
        sound_reactor.attach()
        _log.info("sound_reactor.enabled", enabled=sound_reactor.enabled)

        # Local learning service (on-device "brain").  Disabled by default;
        # enable with DESKBOT_LEARNING__ENABLED=true.  Started in
        # _on_startup so the background training thread only runs while
        # the app is live.
        learning_stack = _build_learning_service(settings, bus)
        learning_service: LearningService | None = None
        safety_manager: LearningSafetyManager | None = None
        observation_adapter: LearningObservationAdapter | None = None
        if learning_stack is not None:
            learning_service, safety_manager, observation_adapter = learning_stack

        # Sensors - the microphone may be a physical USB device or the
        # audio track of the configured RTSP camera stream.
        microphone: Microphone
        camera: Camera

        if settings.hardware == "real":
            match settings.microphone.backend:
                case "usb":
                    microphone = safe_init(  # type: ignore[assignment]
                        factory=lambda: UsbMicrophone(
                            input_device=settings.microphone.input_device,
                            _sample_rate_field=settings.microphone.sample_rate,
                            channels=settings.microphone.channels,
                            frame_ms=settings.microphone.frame_ms,
                        ),
                        component="microphone",
                        fallback=lambda: MockMicrophone(
                            sample_rate=settings.microphone.sample_rate,
                            channels=settings.microphone.channels,
                            frame_ms=settings.microphone.frame_ms,
                        ),
                        registry=degradation,
                        original_backend="usb",
                        fallback_backend="mock",
                    )

                case "rtsp":
                    if settings.camera.backend != "rtsp" or not settings.camera.rtsp_url:
                        raise ValueError(
                            "microphone.backend='rtsp' requires "
                            "camera.backend='rtsp' and camera.rtsp_url"
                        )

                    microphone = safe_init(  # type: ignore[assignment]
                        factory=lambda: RtspMicrophone(
                            url=settings.camera.rtsp_url,
                            output_sample_rate=settings.microphone.sample_rate,
                            channels=settings.microphone.channels,
                            frame_ms=settings.microphone.frame_ms,
                            transport=settings.microphone.rtsp_transport,
                        ),
                        component="microphone",
                        fallback=lambda: MockMicrophone(
                            sample_rate=settings.microphone.sample_rate,
                            channels=settings.microphone.channels,
                            frame_ms=settings.microphone.frame_ms,
                        ),
                        registry=degradation,
                        original_backend="rtsp",
                        fallback_backend="mock",
                    )

                case _:
                    raise ValueError(
                        f"unsupported microphone backend: {settings.microphone.backend!r}"
                    )

            if settings.camera.backend == "rtsp" and settings.camera.rtsp_url:
                camera = safe_init(
                    factory=lambda: RtspCamera(  # type: ignore[assignment]
                        url=settings.camera.rtsp_url,
                        width=settings.camera.width,
                        height=settings.camera.height,
                        fps=settings.camera.fps,
                    ),
                    component="camera",
                    fallback=lambda: MockCamera(
                        width=settings.camera.width,
                        height=settings.camera.height,
                    ),
                    registry=degradation,
                    original_backend="rtsp",
                    fallback_backend="mock",
                )
            else:
                camera = safe_init(
                    factory=lambda: UsbCamera(  # type: ignore[assignment]
                        device=settings.camera.device,
                        width=settings.camera.width,
                        height=settings.camera.height,
                        fps=settings.camera.fps,
                    ),
                    component="camera",
                    fallback=lambda: MockCamera(
                        width=settings.camera.width,
                        height=settings.camera.height,
                    ),
                    registry=degradation,
                    original_backend="usb",
                    fallback_backend="mock",
                )

        else:
            microphone = MockMicrophone(
                sample_rate=settings.microphone.sample_rate,
                channels=settings.microphone.channels,
                frame_ms=settings.microphone.frame_ms,
            )

            if settings.camera.backend == "rtsp" and settings.camera.rtsp_url:
                camera = safe_init(
                    factory=lambda: RtspCamera(  # type: ignore[assignment]
                        url=settings.camera.rtsp_url,
                        width=settings.camera.width,
                        height=settings.camera.height,
                        fps=settings.camera.fps,
                    ),
                    component="camera",
                    fallback=lambda: MockCamera(
                        width=settings.camera.width,
                        height=settings.camera.height,
                    ),
                    registry=degradation,
                    original_backend="rtsp",
                    fallback_backend="mock",
                )
            else:
                camera = MockCamera(
                    width=settings.camera.width,
                    height=settings.camera.height,
                )

        # Face engine (the new, complete face renderer)
        face_renderer = FaceRenderer(width=settings.displays.width, height=settings.displays.height)
        # Legacy eye engine — no longer constructed in the production path.
        # The EyeDisplayAnimator field is kept as None for back-compat with
        # tests that check ``app.eye_animator is not None``. Use ``face_animator``
        # for all production rendering. See findings.md L25.
        # Theme selection: Vector 2.0 minimalist face is the new default
        face_theme = _resolve_face_theme(getattr(settings, "face", None))
        _log.info(
            "face.theme_active",
            theme=type(face_theme).__name__,
            mode=getattr(face_theme.palette, "mode", "face"),
        )
        print(
            f"[deskbot] face theme = {type(face_theme).__name__} "
            f"(mode={getattr(face_theme.palette, 'mode', 'face')!r})",
            flush=True,
        )
        face_animator = FaceAnimator(
            renderer=face_renderer,
            display=display,
            clock=clock,
            emotions=EmotionEngine(width=settings.displays.width, height=settings.displays.height),
            theme=face_theme,
            fps=settings.displays.fps,
            bus=bus,
            width=settings.displays.width,
            height=settings.displays.height,
        )
        eye_animator = None  # Legacy eye engine removed from production path

        # Behavior
        idle = IdleBehavior(
            state_machine=state_machine,
            personality=personality,
            rng=rng,
            clock=clock,
        )
        reactions = ReactionEngine(bus=bus, state_machine=state_machine)
        executor = ActionExecutor(bus=bus, servo_controller=servo_controller)

        # Services - LLM/STT/TTS/wake-word backend selection.
        conversation = _build_ai_stack(
            bus=bus,
            state_machine=state_machine,
            microphone=microphone,
            settings=settings,
            servo_controller=servo_controller,
            audio=audio,
            degradation=degradation,
        )

        # Lifecycle
        lifecycle = Lifecycle(bus=bus)
        app = cls(
            settings=settings,
            container=_build_container(
                bus=bus,
                state_machine=state_machine,
                personality=personality,
                display=display,
                servo_controller=servo_controller,
                audio=audio,
                face_animator=face_animator,
                microphone=microphone,
                camera=camera,
                eye_animator=eye_animator,
                idle=idle,
                reactions=reactions,
                executor=executor,
                conversation=conversation,
            ),
            bus=bus,
            state_machine=state_machine,
            lifecycle=lifecycle,
            personality=personality,
            display=display,
            servo_controller=servo_controller,
            face_animator=face_animator,
            eye_animator=eye_animator,
            idle=idle,
            reactions=reactions,
            executor=executor,
        )
        # Subscribe internal services and lifecycle hooks
        reactions.attach()
        conversation.attach()

        # Face orchestrator
        face_orchestrator = FaceOrchestrator(
            bus=bus,
            face_animator=face_animator,
            emotions=EmotionEngine(width=settings.displays.width, height=settings.displays.height),
            wake_animation_enabled=settings.wakeword.wake_animation,
        )
        face_orchestrator.attach()

        # Perception service: face detection on camera frames.
        if settings.perception.enabled and not isinstance(camera, MockCamera):
            try:
                from robot.perception.face_detector import create_face_detector

                face_detector = create_face_detector(
                    max_faces=settings.perception.max_faces,
                    score_threshold=settings.perception.score_threshold,
                    scale_factor=settings.perception.scale_factor,
                    min_neighbors=settings.perception.min_neighbors,
                )
                perception = PerceptionService(
                    camera=camera,
                    bus=bus,
                    state_machine=state_machine,
                    face_detector=face_detector,
                    scan_interval_s=settings.perception.scan_interval_s,
                    idle_scan_interval_s=settings.perception.idle_scan_interval_s,
                    curious_scan_interval_s=settings.perception.curious_scan_interval_s,
                    max_faces=settings.perception.max_faces,
                )
                perception_behavior = PerceptionBehavior(
                    bus=bus,
                    state_machine=state_machine,
                    idle_timeout_s=5.0,
                )
                perception_behavior.attach()
                _log.info(
                    "perception.enabled",
                    scan_interval_s=settings.perception.scan_interval_s,
                    idle_scan_interval_s=settings.perception.idle_scan_interval_s,
                    curious_scan_interval_s=settings.perception.curious_scan_interval_s,
                )
            except Exception:
                _log.warning("perception.disabled", reason="face_detector_unavailable")
                perception = None
                perception_behavior = None
        else:
            perception = None
            perception_behavior = None
            if settings.perception.enabled and isinstance(camera, MockCamera):
                _log.info("perception.disabled", reason="mock_camera")
        app.perception = perception
        app._perception_behavior = perception_behavior

        # Plugin system
        plugin_registry = PluginRegistry(bus=bus)
        if settings.plugins.enabled and settings.plugins.discover_entry_points:
            discovered = plugin_registry.discover_entry_points()
            for plugin in discovered:
                with contextlib.suppress(PluginError):
                    plugin_registry.register(plugin)
        app._plugin_registry = plugin_registry

        # Capture references to hardware instances
        app._microphone = microphone
        app._camera = camera
        app._audio = audio
        app._sound_effects = sound_effects
        app._sound_reactor = sound_reactor
        app._learning_service = learning_service
        app._safety_manager = safety_manager
        app._observation_adapter = observation_adapter
        app.conversation = conversation
        app._degradation = degradation

        # Performance profiling
        perf = settings.performance
        if perf.enabled:
            if perf.frame_profiling:
                app._frame_profiler = FrameProfiler(
                    target_fps=settings.displays.fps,
                    window=100,
                    report_interval=perf.report_interval_frames,
                    bus=bus,
                    enabled=True,
                )
                # Wire the frame profiler into the face animator for per-frame timing
                if app.face_animator is not None:
                    app.face_animator._frame_profiler = app._frame_profiler
            if perf.servo_profiling:
                app._servo_profiler = ServoProfiler(
                    bus=bus,
                    window=100,
                    enabled=True,
                )
                app._servo_profiler.start()
            if perf.bus_profiling:
                app._bus_profiler = BusProfiler(
                    bus=bus,
                    sample_rate=perf.bus_sample_rate,
                    window=500,
                    enabled=True,
                )
                app._bus_profiler.start()

        # Wire the state bridge so the REST API can read live state.
        api_bridge = StateBridge(
            bus=bus,
            state_machine=state_machine,
            conversation=conversation,
            tts=conversation.tts,
            perception=perception,
            sound_effects=sound_effects,
            preference_tracker=conversation.preference_tracker,
            degradation=degradation,
            microphone=microphone,
            camera=camera,
            audio=audio,
        )
        app._api_bridge = api_bridge
        _log_backend_status(
            settings=settings,
            tts=conversation.tts,
            audio=audio,
            degradation=degradation,
        )

        # Wire the WebSocket event streamer to the bus so connected WS
        # clients receive every event (state changes, emotions, bot
        # replies, LLM tokens, etc.) in real time.
        from robot.api.ws import get_streamer

        bus.subscribe(object, get_streamer().on_event)

        lifecycle.add_startup(app._on_startup)
        lifecycle.add_shutdown(app._on_shutdown)
        return app

    # ------------------------------------------------------------------ lifecycle
    async def _on_startup(self) -> None:
        if self.conversation is not None:
            await self.conversation.conversation.load()
        # Load and start plugins.
        if self._plugin_registry is not None and self._plugin_registry.plugin_count > 0:
            await self._plugin_registry.load_all()
            await self._plugin_registry.start_all()
        # Start MQTT bridge if configured.
        if self.settings.mqtt.enabled:
            try:
                from robot.services.mqtt_bridge import (
                    MqttBridge as MqttBridgeImpl,
                    MqttConfig as MqttBridgeConfig,
                )

                mqtt_config = MqttBridgeConfig(
                    host=self.settings.mqtt.host,
                    port=self.settings.mqtt.port,
                    username=self.settings.mqtt.username,
                    password=self.settings.mqtt.password,
                    topic_prefix=self.settings.mqtt.topic_prefix,
                    keepalive=self.settings.mqtt.keepalive,
                    qos=self.settings.mqtt.qos,
                    publish_events=self.settings.mqtt.publish_events,
                    subscribe_commands=self.settings.mqtt.subscribe_commands,
                    heartbeat_interval=self.settings.mqtt.heartbeat_interval,
                )
                self._mqtt_bridge = MqttBridgeImpl(bus=self.bus, config=mqtt_config)
                await self._mqtt_bridge.start()
                _log.info("mqtt.bridge_started", host=mqtt_config.host, port=mqtt_config.port)
            except ImportError:
                _log.warning("mqtt.paho_mqtt_not_installed", msg="Install paho-mqtt to enable MQTT")
                self._mqtt_bridge = None
            except Exception:
                _log.exception("mqtt.bridge_start_failed")
                self._mqtt_bridge = None
        # Start Home Assistant bridge if configured.
        if self.settings.homeassistant.enabled:
            try:
                from robot.services.home_assistant import (
                    HomeAssistantBridge as HABridge,
                    HomeAssistantConfig as HABridgeConfig,
                )

                ha_config = HABridgeConfig(
                    host=self.settings.homeassistant.host,
                    port=self.settings.homeassistant.port,
                    username=self.settings.homeassistant.username,
                    password=self.settings.homeassistant.password,
                    discovery_prefix=self.settings.homeassistant.discovery_prefix,
                    device_id=self.settings.homeassistant.device_id,
                    device_name=self.settings.homeassistant.device_name,
                    device_manufacturer=self.settings.homeassistant.device_manufacturer,
                    device_model=self.settings.homeassistant.device_model,
                    qos=self.settings.homeassistant.qos,
                )
                self._ha_bridge = HABridge(bus=self.bus, config=ha_config)
                await self._ha_bridge.start()
                _log.info("ha.bridge_started", host=ha_config.host, port=ha_config.port)
            except ImportError:
                _log.warning(
                    "ha.paho_mqtt_not_installed",
                    msg="Install paho-mqtt to enable Home Assistant integration",
                )
                self._ha_bridge = None
            except Exception:
                _log.exception("ha.bridge_start_failed")
                self._ha_bridge = None
        # Start Telegram bridge if configured.
        if self.settings.telegram.enabled and self.settings.telegram.bot_token:
            try:
                from robot.services.telegram_bridge import (
                    TelegramBridge as TelegramBridgeImpl,
                    TelegramConfig as TelegramBridgeCfg,
                )

                tg_config = TelegramBridgeCfg(
                    bot_token=self.settings.telegram.bot_token,
                    enabled=self.settings.telegram.enabled,
                    allowed_user_ids=self.settings.telegram.allowed_user_ids,
                    chat_timeout_s=self.settings.telegram.chat_timeout_s,
                    api_base=self.settings.telegram.api_base,
                )
                self._telegram_bridge = TelegramBridgeImpl(
                    config=tg_config,
                    bus=self.bus,
                    app=self,
                )
                await self._telegram_bridge.start()
                _log.info("telegram.bridge_started")
            except ImportError:
                _log.warning(
                    "telegram.httpx_not_installed",
                    msg="Install httpx to enable the Telegram bridge",
                )
                self._telegram_bridge = None
            except Exception:
                _log.exception("telegram.bridge_start_failed")
                self._telegram_bridge = None
        # Start the local learning service (background training thread).
        if self._learning_service is not None:
            with contextlib.suppress(Exception):
                self._learning_service.load_latest_checkpoint()
            with contextlib.suppress(Exception):
                self._learning_service.start()
                _log.info("learning.service_started")
        await self.bus.publish(RobotStarted())
        await self.state_machine.transition(RobotState.IDLE)
        await self.bus.publish(
            EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.NEUTRAL)
        )

    async def _on_shutdown(self) -> None:
        # Detach the sound reactor.
        if self._sound_reactor is not None:
            self._sound_reactor.detach()
            self._sound_reactor = None
        # Stop the local learning service.
        if self._learning_service is not None:
            with contextlib.suppress(Exception):
                self._learning_service.stop()
                _log.info("learning.service_stopped")
            self._learning_service = None
            self._safety_manager = None
        # Stop and unload plugins.
        if self._plugin_registry is not None and self._plugin_registry.plugin_count > 0:
            await self._plugin_registry.stop_all()
            await self._plugin_registry.unload_all()
        # Stop MQTT bridge.
        if self._mqtt_bridge is not None:
            await self._mqtt_bridge.stop()  # type: ignore[attr-defined]
            self._mqtt_bridge = None
        # Stop Home Assistant bridge.
        if self._ha_bridge is not None:
            await self._ha_bridge.stop()  # type: ignore[attr-defined]
            self._ha_bridge = None
        # Stop Telegram bridge.
        if self._telegram_bridge is not None:
            await self._telegram_bridge.stop()  # type: ignore[attr-defined]
            self._telegram_bridge = None
        # Stop performance profilers.
        if self._servo_profiler is not None:
            self._servo_profiler.stop()
        if self._bus_profiler is not None:
            self._bus_profiler.stop()
        await self.bus.publish(RobotStopped(reason="shutdown"))
        with contextlib.suppress(Exception):
            await self.servo_controller.close()
        # Close audio output if it has a close method.
        if self._audio is not None:
            with contextlib.suppress(Exception):
                await self._audio.close()
        if self.conversation is not None:
            with contextlib.suppress(Exception):
                await self.conversation.conversation.close()
        self._close_stores()
        await self._stop_api()
        await self.bus.close()

    def _close_stores(self) -> None:
        """Close SQLite stores that were opened during build().

        This is called from both :meth:`_on_shutdown` (async path) and
        tests that build the app without running it.  Without this,
        SQLite connections leak and trigger ``ResourceWarning``.
        """
        if self.conversation is not None:
            # Preference store
            tracker = self.conversation.preference_tracker
            if tracker is not None:
                store = getattr(tracker, "_store", None)
                if store is not None and hasattr(store, "close"):
                    with contextlib.suppress(Exception):
                        store.close()
            # Conversation store
            conv = self.conversation.conversation
            store = getattr(conv, "store", None)
            if store is not None and hasattr(store, "close"):
                with contextlib.suppress(Exception):
                    store.close()
        # Learning service episodic memory
        if self._learning_service is not None:
            ep_mem = getattr(self._learning_service, "episodic_memory", None)
            if ep_mem is not None:
                store = getattr(ep_mem, "store", None)
                if store is not None and hasattr(store, "close"):
                    with contextlib.suppress(Exception):
                        store.close()
            # Preference learner store
            pref_learner = getattr(self._learning_service, "preference_learner", None)
            if pref_learner is not None:
                store = getattr(pref_learner, "_store", None)
                if store is not None and hasattr(store, "close"):
                    with contextlib.suppress(Exception):
                        store.close()

    async def _drain_behaviors(self) -> None:
        """Periodically drain idle/reaction outboxes and execute the actions.

        This closes the loop between :class:`IdleBehavior`/`ReactionEngine`
        (which produce :class:`BehaviorAction` objects) and
        :class:`ActionExecutor` (which translates them into bus events
        and servo commands). Without this drainer the behavior layer
        queues actions that are never executed.
        """
        while not self._drain_stop:
            try:
                await anyio.sleep(0.1)
                actions: list[BehaviorAction] = []
                if self.idle is not None:
                    actions.extend(self.idle.drain())
                if self.reactions is not None:
                    actions.extend(self.reactions.drain())
                if actions and self.executor is not None:
                    await self.executor.execute(actions)
            except anyio.get_cancelled_exc_class():
                raise
            except Exception:
                _log.exception("app.behavior_drain_failed")

    # ------------------------------------------------------------------ API
    def _start_api(self) -> None:
        """Start the REST API server in a background thread.

        Creates the FastAPI app, wires the state bridge, and starts
        uvicorn on the configured host:port.
        """
        if not self.settings.api.enabled:
            return
        import threading

        import uvicorn

        from robot.api.app import create_app

        app = create_app(settings=self.settings)
        app.state.bridge = self._api_bridge
        app.state.deskbot = self
        app.state.frame_profiler = self._frame_profiler
        app.state.servo_profiler = self._servo_profiler
        app.state.bus_profiler = self._bus_profiler
        app.state.learning_service = self._learning_service
        app.state.safety_manager = self._safety_manager

        # Wire calibration routes to the real servo controller, display,
        # and settings so the web panel's calibration page works.
        set_calibration_state(
            servo_controller=self.servo_controller,
            display=self.display,
            settings=self.settings,
        )

        host = self.settings.api.host
        port = self.settings.api.port

        _log.info("api.starting", host=host, port=port)

        config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)

        def _run_server() -> None:
            server.run()

        api_thread = threading.Thread(target=_run_server, name="DeskBot-API", daemon=True)
        api_thread.start()
        self._api_server = server
        self._api_thread = api_thread
        _log.info("api.started", host=host, port=port, url=f"http://{host}:{port}")

    async def _stop_api(self) -> None:
        """Stop the REST API server if running."""
        # Unwire calibration routes so a stale module-level reference
        # doesn't point at freed hardware after shutdown.
        set_calibration_state()
        if self._api_server is not None:
            _log.info("api.stopping")
            self._api_server.should_exit = True  # type: ignore[attr-defined]
            # Join the API thread with a short timeout so in-flight requests
            # can finish without hanging shutdown.
            thread = getattr(self, "_api_thread", None)
            if thread is not None:
                thread.join(timeout=5.0)
            self._api_server = None
            self._api_thread = None

    # ------------------------------------------------------------------ run
    @contextlib.asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        """Run the application until the surrounding task group is cancelled."""
        async with self.lifecycle.running() as tg:
            self._task_group = tg
            if self.face_animator is not None:
                tg.start_soon(self.face_animator.run_forever)
            if self.idle is not None:
                tg.start_soon(self.idle.run)
            # Drain queued behavior actions from idle/reaction engines
            # and execute them through the ActionExecutor. Without this,
            # idle blinks/glances and event reactions are silently discarded.
            if self.executor is not None:
                tg.start_soon(self._drain_behaviors)
            if self.conversation is not None:
                self.conversation.start_audio_loop()
            # Start perception (face detection) if available.
            if self.perception is not None:
                tg.start_soon(self.perception.start)
            # Hardware banner - now uses the degradation registry.
            from robot.cli.doctor import _hardware_banner

            _hardware_banner(
                self.settings, self.display, microphone=self._microphone, camera=self._camera
            )
            if self._degradation is not None:
                _log.info("degradation.summary", summary=self._degradation.summary())

            # Start the REST API server if configured.
            if self.settings.api.enabled:
                self._start_api()

            try:
                yield
            finally:
                self._drain_stop = True
                if self.idle is not None:
                    self.idle.stop()
                if self.face_animator is not None:
                    self.face_animator.stop()
                if self.conversation is not None:
                    self.conversation.stop_audio_loop()
                if self.perception is not None:
                    await self.perception.stop()
                await self._stop_api()
                self._task_group = None


# ---------------------------------------------------------------------------
# Audio builder (extracted from build() for readability)
# ---------------------------------------------------------------------------


def _build_audio(
    settings: AppSettings,
    degradation: DegradationRegistry,
) -> AudioOutput:
    """Build the audio output, with safe-init fallbacks for every backend."""
    if settings.audio.backend == "bluetooth":
        return safe_init(
            factory=lambda: _bluetooth_speaker(settings),
            component="audio",
            fallback=lambda: MockAudioOutput(
                sample_rate=settings.audio.sample_rate,
                channels=settings.audio.channels,
            ),
            registry=degradation,
            original_backend="bluetooth",
            fallback_backend="mock",
        )
    if settings.audio.backend == "usb":
        return safe_init(
            factory=lambda: _usb_speaker(settings),
            component="audio",
            fallback=lambda: MockAudioOutput(
                sample_rate=settings.audio.sample_rate,
                channels=settings.audio.channels,
            ),
            registry=degradation,
            original_backend="usb",
            fallback_backend="mock",
        )
    # Default: mock audio
    degradation.record(
        DegradationEntry(
            component="audio",
            status="ok",
            original_backend="mock",
            fallback_backend="mock",
        )
    )
    return MockAudioOutput(
        sample_rate=settings.audio.sample_rate,
        channels=settings.audio.channels,
    )


def _bluetooth_speaker(settings: AppSettings) -> AudioOutput:
    """Import and construct a BluetoothSpeaker (may raise ImportError)."""
    from robot.hardware.audio.bluetooth_speaker import BluetoothSpeaker

    return BluetoothSpeaker(
        device_mac=settings.audio.bluetooth_mac,
        device_name=settings.audio.bluetooth_name,
        sample_rate=settings.audio.sample_rate,
        channels=settings.audio.channels,
        auto_connect=settings.audio.bluetooth_auto_connect,
    )


def _usb_speaker(settings: AppSettings) -> AudioOutput:
    """Import and construct a UsbSpeaker (may raise RuntimeError/ImportError)."""
    from robot.hardware.audio.usb_speaker import UsbSpeaker

    return UsbSpeaker(
        output_device=settings.audio.output_device,
        _sample_rate=settings.audio.sample_rate,
        channels=settings.audio.channels,
    )


def _mock_servo_bus() -> ServoController:
    """Build a mock servo controller as the fallback for safe-init."""
    from robot.hardware.servos.adapter import wrap_servo_controller

    bus = MockServoBus(
        {
            "pan": MockServo(name="pan", min_angle=-90.0, max_angle=90.0),
            "tilt": MockServo(name="tilt", min_angle=-30.0, max_angle=30.0),
            "left_arm": MockServo(name="left_arm", min_angle=0.0, max_angle=180.0),
            "right_arm": MockServo(name="right_arm", min_angle=0.0, max_angle=180.0),
        }
    )
    return wrap_servo_controller(bus, backend_name="mock")


# ---------------------------------------------------------------------------
# AI stack builder
# ---------------------------------------------------------------------------


def _build_ai_stack(  # noqa: PLR0912
    *,
    bus: InMemoryEventBus,
    state_machine: StateMachine,
    microphone: Microphone | None,
    settings: AppSettings,
    servo_controller: ServoController | None = None,
    audio: AudioOutput | None = None,
    degradation: DegradationRegistry | None = None,
) -> ConversationService:
    """Build the LLM/STT/TTS/wake-word stack from configuration.

    The defaults match the original mock stack. When the user sets
    ``DESKBOT_LLM__PROVIDER=openai``, ``DESKBOT_STT__PROVIDER=whisper``,
    ``DESKBOT_TTS__PROVIDER=openai`` and provides API keys, the real
    OpenAI-compatible backends are used instead.
    """
    _reg = degradation or DegradationRegistry()

    # LLM
    llm_cfg = settings.llm
    if llm_cfg.provider == "openai":
        llm: LLM = safe_init(
            factory=lambda: OpenAILLM(  # type: ignore[assignment]
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url or "https://api.openai.com/v1",
                model=llm_cfg.model,
                temperature=llm_cfg.temperature,
                max_tokens=llm_cfg.max_tokens,
                timeout_s=llm_cfg.timeout_s,
            ),
            component="llm",
            fallback=lambda: _mock_llm(llm_cfg.provider),
            registry=_reg,
            original_backend="openai",
            fallback_backend="mock",
        )
    elif llm_cfg.provider == "ollama":
        llm = safe_init(
            factory=lambda: OllamaLLM(  # type: ignore[assignment]
                model=llm_cfg.model or "llama3.2",
                base_url=llm_cfg.base_url or "http://localhost:11434",
                timeout_s=llm_cfg.timeout_s,
                temperature=llm_cfg.temperature,
            ),
            component="llm",
            fallback=lambda: _mock_llm(llm_cfg.provider),
            registry=_reg,
            original_backend="ollama",
            fallback_backend="mock",
        )
    else:
        llm = _mock_llm(llm_cfg.provider)

    # STT
    stt_cfg = settings.stt
    if stt_cfg.provider == "whisper":
        stt: SpeechToText = safe_init(
            factory=lambda: StreamingWhisperAdapter(  # type: ignore[assignment]
                WhisperSTT(
                    api_key=llm_cfg.api_key,
                    base_url=llm_cfg.base_url or "https://api.openai.com/v1",
                    model=stt_cfg.model,
                    language=stt_cfg.language,
                )
            ),
            component="stt",
            fallback=MockSTT,
            registry=_reg,
            original_backend="whisper",
            fallback_backend="mock",
        )
    else:
        stt = MockSTT()

    # TTS
    tts_cfg = settings.tts
    if tts_cfg.provider == "openai":
        tts: TextToSpeech = safe_init(
            factory=lambda: OpenAITTS(  # type: ignore[assignment]
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url or "https://api.openai.com/v1",
                model="tts-1",
                voice=tts_cfg.voice,
                audio=audio,
            ),
            component="tts",
            fallback=MockTTS,
            registry=_reg,
            original_backend="openai",
            fallback_backend="mock",
        )
    elif tts_cfg.provider == "piper":
        piper = tts_cfg.piper
        tts = safe_init(
            factory=lambda: PiperTTS(  # type: ignore[assignment]
                model=piper.model,
                download_dir=piper.download_dir or None,
                use_cuda=piper.use_cuda,
                speaker_id=piper.speaker_id,
                noise_scale=piper.noise_scale,
                length_scale=piper.length_scale,
                noise_w_scale=piper.noise_w_scale,
                audio=audio,
            ),
            component="tts",
            fallback=MockTTS,
            registry=_reg,
            original_backend="piper",
            fallback_backend="mock",
        )
    elif tts_cfg.provider == "espeak":
        tts = safe_init(
            factory=lambda: EspeakNGTTS(  # type: ignore[assignment]
                voice=tts_cfg.voice if tts_cfg.voice != "default" else "en",
                audio=audio,
            ),
            component="tts",
            fallback=MockTTS,
            registry=_reg,
            original_backend="espeak",
            fallback_backend="mock",
        )
    elif tts_cfg.provider == "elevenlabs":
        tts = safe_init(
            factory=lambda: _elevenlabs_tts(tts_cfg, audio),
            component="tts",
            fallback=MockTTS,
            registry=_reg,
            original_backend="elevenlabs",
            fallback_backend="mock",
        )
    else:
        _reg.record(
            DegradationEntry(
                component="tts",
                status="ok",
                original_backend="mock",
                fallback_backend="mock",
            )
        )
        tts = MockTTS()

    # Conversation persistence
    conv_cfg = settings.conversation
    if conv_cfg.store == "sqlite":
        from robot.ai.conversation_sqlite import SqliteConversationStore

        conv_store: ConversationStore | None = SqliteConversationStore(db_path=conv_cfg.db_path)
    else:
        conv_store = None
    conversation = ConversationManager(
        llm=llm,
        system_prompt=system_prompt(),
        store=conv_store,
        conversation_id=conv_cfg.conversation_id,
    )

    # Wake-word checker
    wakeword_cfg = settings.wakeword
    wake_checker: WakeWordChecker
    if wakeword_cfg.provider == "mock":
        wake_checker = MockWakeWordChecker(phrase=wakeword_cfg.phrase)
    elif wakeword_cfg.provider == "openwakeword":
        wake_checker = safe_init(
            factory=lambda: _openwakeword_checker(wakeword_cfg),
            component="wakeword",
            fallback=NullWakeWordChecker,
            registry=_reg,
            original_backend="openwakeword",
            fallback_backend="null",
        )
    elif wakeword_cfg.provider == "porcupine":
        wake_checker = safe_init(
            factory=lambda: _porcupine_checker(wakeword_cfg),
            component="wakeword",
            fallback=NullWakeWordChecker,
            registry=_reg,
            original_backend="porcupine",
            fallback_backend="null",
        )
    elif wakeword_cfg.provider == "snowboy":
        wake_checker = safe_init(
            factory=lambda: _snowboy_checker(wakeword_cfg),
            component="wakeword",
            fallback=NullWakeWordChecker,
            registry=_reg,
            original_backend="snowboy",
            fallback_backend="null",
        )
    else:
        wake_checker = NullWakeWordChecker()  # type: ignore[unreachable]

    # Tool calling setup
    tool_registry: ToolRegistry | None = None
    tool_executor: ToolExecutor | None = None
    if settings.tools.enabled:
        tool_registry = ToolRegistry()
        for definition in BUILTIN_TOOLS.values():
            tool_registry.add(definition, handler=_noop_tool_handler)
        tool_executor = ToolExecutor(
            registry=tool_registry,
            bus=bus,
            servo_controller=servo_controller,
            tts=tts,
            audio=audio,
        )

    memory = create_memory(settings) if settings.memory.enabled else None

    # Preference tracker
    preference_tracker: PreferenceTracker | None = None
    if settings.preferences.enabled:
        if settings.preferences.store == "sqlite":
            _pref_store: Any = SqlitePreferenceStore(db_path=settings.preferences.db_path)
        else:
            _pref_store = InMemoryPreferenceStore()
        preference_tracker = PreferenceTracker(store=cast("Any", _pref_store))

    return ConversationService(
        bus=bus,
        state_machine=state_machine,
        stt=stt,
        tts=tts,
        llm=llm,
        conversation=conversation,
        microphone=microphone,
        wake_checker=wake_checker,
        memory=memory,
        memory_recall_limit=settings.memory.recall_limit,
        preference_tracker=preference_tracker,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        audio=audio,
    )


# ---------------------------------------------------------------------------
# Small factory helpers (lazy-imported inside safe_init lambdas)
# ---------------------------------------------------------------------------


def _mock_llm(provider: str) -> MockLLM:
    """Build a MockLLM with canned responses."""
    mock = MockLLM(name=f"mock-llm:{provider}")
    mock.register("hello", "Hi there!")
    mock.register("how are you", "Feeling very electric today.")
    mock.register("what time is it", "I lost my watch on day one, sorry.")
    mock.register("who are you", "I'm DeskBot, a small desktop companion.")
    return mock


def _log_backend_status(
    *,
    settings: AppSettings,
    tts: TextToSpeech,
    audio: AudioOutput,
    degradation: DegradationRegistry,
) -> None:
    """Log configured vs active audio backends so degraded mocks are obvious."""
    entries = {entry.component: entry for entry in degradation.report()}
    tts_entry = entries.get("tts")
    audio_entry = entries.get("audio")
    _log.info(
        "tts.backend.active",
        configured_provider=settings.tts.provider,
        active_backend=type(tts).__name__,
        degraded=tts_entry.status == "degraded" if tts_entry else False,
        error=tts_entry.error if tts_entry else None,
        mock_backend=isinstance(tts, MockTTS),
    )
    _log.info(
        "audio.backend.active",
        configured_backend=settings.audio.backend,
        active_backend=type(audio).__name__,
        degraded=audio_entry.status == "degraded" if audio_entry else False,
        error=audio_entry.error if audio_entry else None,
        mock_backend=isinstance(audio, MockAudioOutput),
    )
    if isinstance(audio, MockAudioOutput) and settings.hardware == "real":
        _log.warning(
            "audio.backend.mock_in_production",
            message="Audio output is MockAudioOutput in a real hardware run; "
            "no physical speech will be produced. "
            "Set DESKBOT_AUDIO__BACKEND=usb to enable the speaker.",
        )
    if isinstance(tts, MockTTS) and settings.hardware == "real":
        _log.warning(
            "tts.backend.mock_in_production",
            message="TTS is MockTTS in a real hardware run; "
            "no physical speech will be produced. "
            "Set DESKBOT_TTS__PROVIDER to a real backend (openai, piper, espeak, elevenlabs).",
        )


def _elevenlabs_tts(tts_cfg: TTSConfig, audio: AudioOutput | None) -> TextToSpeech:
    """Import and construct ElevenLabsTTS (may raise ImportError)."""
    from robot.speech.tts_elevenlabs import ElevenLabsTTS

    cfg = tts_cfg
    return ElevenLabsTTS(
        api_key=cfg.elevenlabs.api_key,
        voice_id=cfg.elevenlabs.voice_id,
        model_id=cfg.elevenlabs.model_id,
        audio=audio,
        stability=cfg.elevenlabs.stability,
        similarity_boost=cfg.elevenlabs.similarity_boost,
    )


def _openwakeword_checker(wakeword_cfg: WakeWordConfig) -> WakeWordChecker:
    """Import and construct OpenWakeWordChecker (may raise ImportError).

    Eagerly validates the ``openwakeword`` import so that
    :func:`safe_init` can catch the failure and fall back gracefully
    instead of deferring it to every ``check()`` call.
    """
    import importlib.util

    if importlib.util.find_spec("openwakeword") is None:
        raise ImportError("openwakeword is not installed")

    from robot.speech.wakeword_openwakeword import OpenWakeWordChecker

    cfg = wakeword_cfg
    return OpenWakeWordChecker(
        phrase=cfg.phrase,
        threshold=cfg.threshold,
        model_path=cfg.model_path or None,
    )


def _porcupine_checker(wakeword_cfg: WakeWordConfig) -> WakeWordChecker:
    """Import and construct PorcupineWakeWordChecker (may raise ImportError)."""
    from robot.speech.wakeword_porcupine import PorcupineWakeWordChecker

    cfg = wakeword_cfg
    return PorcupineWakeWordChecker(
        access_key=cfg.porcupine_access_key,
        keyword=cfg.porcupine_keyword,
        model_path=cfg.porcupine_model_path or None,
        sensitivity=cfg.threshold,
    )


def _snowboy_checker(wakeword_cfg: WakeWordConfig) -> WakeWordChecker:
    """Import and construct SnowboyWakeWordChecker (may raise ImportError)."""
    from robot.speech.wakeword_snowboy import SnowboyWakeWordChecker

    cfg = wakeword_cfg
    return SnowboyWakeWordChecker(
        model_path=cfg.snowboy_model_path,
        sensitivity=cfg.threshold,
    )


async def _noop_tool_handler(**kwargs: object) -> dict[str, str]:
    """Placeholder handler for built-in tools. Real dispatch goes through ToolExecutor."""
    return {"status": "ok", "note": "handled by executor"}


def _resolve_face_theme(face_config: object | None) -> Theme:
    """Pick a Theme instance based on the FaceConfig.

    Falls back to :class:`VectorTheme` (the Anki Vector 2.0 minimalist
    face) when no configuration is provided or when an unknown theme
    name is requested.
    """
    from robot.face.themes import BUILTIN_THEMES

    requested = "vector"
    if face_config is not None:
        configured_theme = getattr(face_config, "theme", None)
        if configured_theme is not None:
            requested = str(configured_theme) or "vector"
    theme_cls = BUILTIN_THEMES.get(requested)
    if theme_cls is None:
        _log.warning("face.unknown_theme", theme=requested, fallback="vector")
        theme_cls = BUILTIN_THEMES["vector"]
    return theme_cls()


def _build_learning_service(
    settings: AppSettings,
    bus: InMemoryEventBus,
) -> tuple[LearningService, LearningSafetyManager, LearningObservationAdapter] | None:
    """Build the on-device learning service from settings.

    Returns ``None`` when learning is disabled.  When enabled, constructs
    the :class:`LearningService` (world model, action learner, experience
    recorder, optional SQLite-backed episodic memory, and a
    :class:`PreferenceLearner`) plus a :class:`LearningSafetyManager`.
    The service is **not** started here; :meth:`DeskBotApp._on_startup`
    starts the background training thread.
    """
    cfg = settings.learning
    if not cfg.enabled:
        return None

    schedule = LearningSchedule(
        min_new_experiences=cfg.min_new_experiences,
        train_interval_s=cfg.train_interval_s,
        min_experiences_for_training=cfg.min_experiences_for_training,
    )
    resource_limits = ResourceLimits(
        max_cpu_fraction=cfg.max_cpu_fraction,
        batch_size=cfg.batch_size,
        max_model_params=cfg.max_model_params,
        training_epochs_per_cycle=cfg.training_epochs_per_cycle,
        eval_sample_size=cfg.eval_sample_size,
    )
    checkpoint_config = CheckpointConfig(
        checkpoint_dir=cfg.checkpoint_dir,
        keep_last_n=cfg.keep_last_n_checkpoints,
        promote_threshold=cfg.promote_threshold,
    )

    # Persistent episodic memory (survives restarts) when using SQLite.
    episodic_memory: EpisodicMemory | None = None
    if cfg.store == "sqlite":
        episodic_memory = EpisodicMemory(
            store=SqliteExperienceStore(db_path=cfg.db_path),
            capacity=cfg.episodic_capacity,
        )

    # Preference learner with a matching persistence backend.
    if cfg.store == "sqlite":
        pref_db = str(Path(cfg.db_path).expanduser().parent / "learned_preferences.db")
        pref_store: object = SqlitePreferenceStore(db_path=pref_db)
    else:
        pref_store = InMemoryPreferenceStore()
    preference_learner = PreferenceLearner(store=pref_store)  # type: ignore[arg-type]
    with contextlib.suppress(Exception):
        preference_learner.load_from_store()

    # Create the observation adapter that bridges events to PreferenceLearner.
    # The PreferenceTracker from the conversation service will also feed
    # extracted preferences through this adapter.
    observation_adapter = LearningObservationAdapter(
        bus=bus,
        preference_learner=preference_learner,
        preference_tracker=None,  # wired later in _build_ai_stack
    )

    service = LearningService(
        bus=bus,
        schedule=schedule,
        resource_limits=resource_limits,
        checkpoint_config=checkpoint_config,
        episodic_memory=episodic_memory,
        preference_learner=preference_learner,
        working_memory=WorkingMemory(capacity=cfg.working_memory_capacity),
        replay_buffer=ReplayBuffer(capacity=cfg.replay_buffer_capacity, seed=cfg.replay_seed),
        use_multimodal=cfg.use_multimodal,
        multimodal_history_length=cfg.multimodal_history_length,
    )
    safety = LearningSafetyManager(checkpoint_manager=service.checkpoint_mgr)
    _log.info(
        "learning.built",
        store=cfg.store,
        episodic=episodic_memory is not None,
        preferences=preference_learner.total_patterns,
        multimodal=cfg.use_multimodal,
        multimodal_history=cfg.multimodal_history_length if cfg.use_multimodal else 0,
    )
    return service, safety, observation_adapter


def _build_container(
    *,
    bus: InMemoryEventBus,
    state_machine: StateMachine,
    personality: Personality,
    display: Display,
    servo_controller: ServoController,
    audio: AudioOutput,
    microphone: Microphone,
    camera: Camera,
    face_animator: FaceAnimator,
    eye_animator: EyeDisplayAnimator | None,
    idle: IdleBehavior,
    reactions: ReactionEngine,
    executor: ActionExecutor,
    conversation: Any,
) -> Container:
    container = Container()
    container.register_instance(InMemoryEventBus, bus)
    container.register_instance(StateMachine, state_machine)
    container.register_instance(Personality, personality)
    container.register_instance(Display, display)  # type: ignore[type-abstract]
    container.register_instance(ServoController, servo_controller)  # type: ignore[type-abstract]
    container.register_instance(AudioOutput, audio)  # type: ignore[type-abstract]
    container.register_instance(Microphone, microphone)  # type: ignore[type-abstract]
    container.register_instance(Camera, camera)  # type: ignore[type-abstract]
    container.register_instance(FaceAnimator, face_animator)
    if eye_animator is not None:
        container.register_instance(EyeDisplayAnimator, eye_animator)
    container.register_instance(IdleBehavior, idle)
    container.register_instance(ReactionEngine, reactions)
    container.register_instance(ActionExecutor, executor)
    container.register_instance(MockLLM, conversation.llm)
    return container


__all__ = ["DeskBotApp"]
