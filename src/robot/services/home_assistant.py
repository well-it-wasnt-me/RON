"""Home Assistant integration - exposes DeskBot as an HA entity.

The :class:`HomeAssistantBridge` publishes the robot's state, emotion,
and sensor data to a Home Assistant MQTT broker using the
`MQTT Statestream <https://www.home-assistant.io/integrations/mqtt/>`_
discovery protocol. HA will automatically create entities for the robot
without manual configuration.

This bridge also subscribes to HA command topics so HA automations can
control the robot (change emotion, trigger speech, etc.).

Requires the ``paho-mqtt`` package.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

from robot.behavior.state_machine import RobotState
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    EmotionChanged,
    EmotionName,
    FaceDetected,
    SoundEffectPlayed,
    StateChanged,
    WakeWordDetected,
)
from robot.logging import get_logger

_log = get_logger("services.home_assistant")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class HomeAssistantConfig:
    """Home Assistant MQTT discovery configuration."""

    host: str = "homeassistant.local"
    port: int = 1883
    username: str = ""
    password: str = ""
    discovery_prefix: str = "homeassistant"
    device_id: str = "deskbot"
    device_name: str = "DeskBot"
    device_manufacturer: str = "DeskBot Contributors"
    device_model: str = "Desktop Companion Robot"
    qos: int = 1


# ---------------------------------------------------------------------------
# HA Discovery payload helpers
# ---------------------------------------------------------------------------
def _build_device_info(config: HomeAssistantConfig) -> dict[str, Any]:
    """Build the HA device info block."""
    return {
        "identifiers": [config.device_id],
        "name": config.device_name,
        "manufacturer": config.device_manufacturer,
        "model": config.device_model,
        "sw_version": "0.1.0",
    }


def _build_sensor_config(
    config: HomeAssistantConfig,
    sensor_name: str,
    icon: str = "mdi:robot",
    unit: str | None = None,
) -> dict[str, Any]:
    """Build an MQTT discovery config payload for a sensor."""
    base_topic = f"{config.discovery_prefix}/sensor/{config.device_id}/{sensor_name}"
    return {
        "name": f"{config.device_name} {sensor_name.replace('_', ' ').title()}",
        "unique_id": f"{config.device_id}_{sensor_name}",
        "state_topic": f"{base_topic}/state",
        "command_topic": f"{base_topic}/set",
        "availability_topic": f"{config.discovery_prefix}/sensor/{config.device_id}/availability",
        "device": _build_device_info(config),
        "icon": icon,
        "unit_of_measurement": unit,
    }


def _build_select_config(
    config: HomeAssistantConfig,
    entity_name: str,
    options: list[str],
    icon: str = "mdi:robot",
) -> dict[str, Any]:
    """Build an MQTT discovery config payload for a select entity."""
    base_topic = f"{config.discovery_prefix}/select/{config.device_id}/{entity_name}"
    return {
        "name": f"{config.device_name} {entity_name.replace('_', ' ').title()}",
        "unique_id": f"{config.device_id}_{entity_name}",
        "state_topic": f"{base_topic}/state",
        "command_topic": f"{base_topic}/set",
        "availability_topic": f"{config.discovery_prefix}/select/{config.device_id}/availability",
        "options": options,
        "device": _build_device_info(config),
        "icon": icon,
    }


# ---------------------------------------------------------------------------
# Home Assistant Bridge
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class HomeAssistantBridge:
    """Bridges DeskBot state to Home Assistant via MQTT discovery."""

    bus: InMemoryEventBus
    config: HomeAssistantConfig = field(default_factory=HomeAssistantConfig)
    _client: Any = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False)
    _current_state: str = "idle"
    _current_emotion: str = "neutral"
    # Store task references to prevent GC of pending asyncio tasks.
    _pending_tasks: list[asyncio.Task[None]] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Connect to MQTT, publish discovery configs, and subscribe to events."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise ImportError(
                "paho-mqtt is required for HomeAssistantBridge. Install with: pip install paho-mqtt"
            ) from exc

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{self.config.device_id}-ha-bridge",
        )

        if self.config.username:
            self._client.username_pw_set(self.config.username, self.config.password)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        avail_topic = f"{self.config.discovery_prefix}/sensor/{self.config.device_id}/availability"
        self._client.will_set(avail_topic, payload="offline", qos=self.config.qos, retain=True)

        _log.info("ha_bridge.connecting", host=self.config.host, port=self.config.port)
        self._client.connect_async(self.config.host, self.config.port, keepalive=60)
        self._client.loop_start()

        self.bus.subscribe(StateChanged, self._on_state_changed)
        self.bus.subscribe(EmotionChanged, self._on_emotion_changed)
        self.bus.subscribe(FaceDetected, self._on_face_detected)
        self.bus.subscribe(WakeWordDetected, self._on_wake_word)
        self.bus.subscribe(SoundEffectPlayed, self._on_sound_effect)

    async def stop(self) -> None:
        """Disconnect from MQTT."""
        if self._client is not None:
            avail_topic = (
                f"{self.config.discovery_prefix}/sensor/{self.config.device_id}/availability"
            )
            with contextlib.suppress(Exception):
                self._client.publish(
                    avail_topic, payload="offline", qos=self.config.qos, retain=True
                )
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False

    # ------------------------------------------------------------------ MQTT callbacks
    def _on_connect(
        self,
        client: object,
        userdata: object,
        flags: object,
        rc: int,
        properties: object | None = None,
    ) -> None:
        self._connected = True
        _log.info("ha_bridge.connected", host=self.config.host)

        self._publish_discovery()

        avail_topic = f"{self.config.discovery_prefix}/sensor/{self.config.device_id}/availability"
        self._client.publish(avail_topic, payload="online", qos=self.config.qos, retain=True)

        prefix = self.config.discovery_prefix
        device = self.config.device_id
        self._client.subscribe(f"{prefix}/select/{device}/emotion/set", qos=self.config.qos)
        self._client.subscribe(f"{prefix}/select/{device}/state/set", qos=self.config.qos)

    def _on_disconnect(
        self,
        client: object,
        userdata: object,
        flags: object,
        rc: int,
        properties: object | None = None,
    ) -> None:
        self._connected = False
        if rc != 0:
            _log.warning("ha_bridge.unexpected_disconnect", rc=rc)

    def _on_message(self, client: object, userdata: object, msg: object) -> None:
        """Handle incoming MQTT commands from HA."""
        topic = getattr(msg, "topic", "")
        payload = getattr(msg, "payload", b"")
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="replace")

        _log.info("ha_bridge.command_received", topic=topic, payload=payload[:200])

        try:
            data = json.loads(payload) if payload.strip() else {}
        except json.JSONDecodeError:
            data = {"raw": payload}

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        device = self.config.device_id

        if f"select/{device}/emotion/set" in topic:
            try:
                emotion = EmotionName(data.get("emotion", payload.strip().lower()))
                t = loop.create_task(
                    self.bus.publish(
                        EmotionChanged(previous=EmotionName(self._current_emotion), current=emotion)
                    )
                )
                self._pending_tasks.append(t)
                t.add_done_callback(
                    lambda _: self._pending_tasks.remove(t) if t in self._pending_tasks else None
                )
            except ValueError:
                _log.warning("ha_bridge.invalid_emotion", emotion=data.get("emotion", payload))

        elif f"select/{device}/state/set" in topic:
            try:
                state = RobotState(data.get("state", "idle"))
                t = loop.create_task(
                    self.bus.publish(
                        StateChanged(previous=RobotState(self._current_state), current=state)
                    )
                )
                self._pending_tasks.append(t)
                t.add_done_callback(
                    lambda _: self._pending_tasks.remove(t) if t in self._pending_tasks else None
                )
            except ValueError:
                _log.warning("ha_bridge.invalid_state", state=data.get("state", payload))

    # ------------------------------------------------------------------ Discovery
    def _publish_discovery(self) -> None:
        """Publish MQTT Auto Discovery configs for all entities."""
        prefix = self.config.discovery_prefix
        device = self.config.device_id

        state_config = _build_select_config(
            self.config, "state", [s.value for s in RobotState], icon="mdi:robot-outline"
        )
        self._client.publish(
            f"{prefix}/select/{device}/state/config",
            payload=json.dumps(state_config),
            qos=self.config.qos,
            retain=True,
        )

        emotion_config = _build_select_config(
            self.config, "emotion", [e.value for e in EmotionName], icon="mdi:emoticon-outline"
        )
        self._client.publish(
            f"{prefix}/select/{device}/emotion/config",
            payload=json.dumps(emotion_config),
            qos=self.config.qos,
            retain=True,
        )

        wake_config = _build_sensor_config(self.config, "wake_word", icon="mdi:microphone")
        self._client.publish(
            f"{prefix}/sensor/{device}/wake_word/config",
            payload=json.dumps(wake_config),
            qos=self.config.qos,
            retain=True,
        )

        face_config = _build_sensor_config(
            self.config, "face_detected", icon="mdi:face-recognition"
        )
        self._client.publish(
            f"{prefix}/sensor/{device}/face_detected/config",
            payload=json.dumps(face_config),
            qos=self.config.qos,
            retain=True,
        )

        sound_config = _build_sensor_config(self.config, "sound_effect", icon="mdi:speaker")
        self._client.publish(
            f"{prefix}/sensor/{device}/sound_effect/config",
            payload=json.dumps(sound_config),
            qos=self.config.qos,
            retain=True,
        )

        _log.info("ha_bridge.discovery_published")

    # ------------------------------------------------------------------ Event handlers
    async def _on_state_changed(self, event: StateChanged) -> None:
        self._current_state = (
            event.current.value if hasattr(event.current, "value") else str(event.current)
        )
        self._publish_state("state", self._current_state)

    async def _on_emotion_changed(self, event: EmotionChanged) -> None:
        self._current_emotion = (
            event.current.value if hasattr(event.current, "value") else str(event.current)
        )
        self._publish_state("emotion", self._current_emotion)

    async def _on_face_detected(self, event: FaceDetected) -> None:
        self._publish_state("face_detected", f"detected:{event.confidence:.0%}")

    async def _on_wake_word(self, event: WakeWordDetected) -> None:
        self._publish_state("wake_word", f"{event.phrase}:{event.confidence:.0%}")

    async def _on_sound_effect(self, event: SoundEffectPlayed) -> None:
        self._publish_state("sound_effect", event.name)

    # ------------------------------------------------------------------ MQTT publish helpers
    def _publish_state(self, entity: str, value: str) -> None:
        """Publish a state value to the appropriate MQTT topic."""
        if not self._connected or self._client is None:
            return

        prefix = self.config.discovery_prefix
        device = self.config.device_id
        topic = f"{prefix}/select/{device}/{entity}/state"
        if entity in ("wake_word", "face_detected", "sound_effect"):
            topic = f"{prefix}/sensor/{device}/{entity}/state"

        with contextlib.suppress(Exception):
            self._client.publish(topic, payload=value, qos=self.config.qos, retain=False)


__all__ = ["HomeAssistantBridge", "HomeAssistantConfig"]
