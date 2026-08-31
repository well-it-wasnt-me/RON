"""MQTT bridge - publishes DeskBot events to an MQTT broker and receives commands.

Requires the ``paho-mqtt`` package (v2.1+).

.. warning::
    Any MQTT client that can publish to the command topics can drive the
    robot (change emotion, trigger speech, move servos). The broker must
    be on a trusted/isolated network, or ACLs must restrict publish
    access to the command topics to a single privileged client.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    BlinkRequested,
    EmotionChanged,
    EmotionName,
    FaceDetected,
    LLMTokenReceived,
    RobotError,
    RobotStarted,
    RobotStopped,
    ServoMoved,
    SoundEffectPlayed,
    SpeechRecognized,
    StateChanged,
    WakeWordDetected,
)
from robot.logging import get_logger

_log = get_logger("services.mqtt_bridge")


def _make_task_done_callback(tasks: list[Any], logger: object, log_prefix: str = "mqtt") -> Any:
    """Create a done-callback that removes a task and logs exceptions."""

    def _done(task: asyncio.Task[None]) -> None:
        if task in tasks:
            tasks.remove(task)
        exc = task.exception()
        if exc is not None:
            _log.warning(f"{log_prefix}.task_exception", error=str(exc))

    return _done


@dataclass(slots=True)
class MqttConfig:
    """MQTT connection configuration."""

    host: str = "localhost"
    port: int = 1883
    username: str = ""
    password: str = ""
    topic_prefix: str = "deskbot"
    keepalive: int = 60
    qos: int = 1
    publish_events: bool = True
    subscribe_commands: bool = True
    heartbeat_interval: int = 30


def _serialise_event(event: object) -> dict[str, Any]:
    """Convert an event dataclass to a JSON-serialisable dict.

    Uses the same robust dataclass-field walker as the WebSocket streamer
    so enums and datetimes are handled correctly.
    """
    from dataclasses import fields as _fields
    from datetime import date
    from enum import Enum

    result: dict[str, Any] = {"type": type(event).__name__}
    try:
        for f in _fields(event):  # type: ignore[arg-type]
            value = getattr(event, f.name)
            if isinstance(value, Enum):
                result[f.name] = value.value
            elif isinstance(value, (datetime, date)):
                result[f.name] = value.isoformat()
            else:
                result[f.name] = value
    except TypeError:
        # Not a dataclass - fall back to dir()-walking with private attr filter.
        for attr in dir(event):
            if attr.startswith("_"):
                continue
            value = getattr(event, attr, None)
            if callable(value):
                continue
            if isinstance(value, datetime):
                result[attr] = value.isoformat()
            elif value is not None and hasattr(value, "value"):
                result[attr] = value.value
            else:
                result[attr] = value
    return result


def _event_type_name(event: object) -> str:
    """Return the short type name for topic routing."""
    return type(event).__name__


@dataclass(slots=True)
class MqttBridge:
    """Bridges local events to/from an MQTT broker."""

    bus: InMemoryEventBus
    config: MqttConfig = field(default_factory=MqttConfig)
    _client: Any = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False)
    _pending_tasks: list[asyncio.Task[None]] = field(default_factory=list, init=False, repr=False)

    async def start(self) -> None:
        """Connect to the MQTT broker and subscribe to events."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise ImportError(
                "paho-mqtt is required for MqttBridge. Install with: pip install paho-mqtt"
            ) from exc

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,  # type: ignore[attr-defined]
            client_id="deskbot-bridge",
        )

        if self.config.username:
            self._client.username_pw_set(self.config.username, self.config.password)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        lwt_topic = f"{self.config.topic_prefix}/status"
        self._client.will_set(lwt_topic, payload="offline", qos=self.config.qos, retain=True)

        _log.info("mqtt.connecting", host=self.config.host, port=self.config.port)
        self._client.connect_async(
            self.config.host, self.config.port, keepalive=self.config.keepalive
        )
        self._client.loop_start()

        event_types = [
            RobotStarted,
            RobotStopped,
            StateChanged,
            EmotionChanged,
            BlinkRequested,
            FaceDetected,
            SpeechRecognized,
            WakeWordDetected,
            ServoMoved,
            SoundEffectPlayed,
            LLMTokenReceived,
            RobotError,
        ]
        for event_type in event_types:
            self.bus.subscribe(event_type, self._on_local_event)

    async def stop(self) -> None:
        """Disconnect from the MQTT broker."""
        if self._client is not None:
            status_topic = f"{self.config.topic_prefix}/status"
            with contextlib.suppress(Exception):
                self._client.publish(
                    status_topic, payload="offline", qos=self.config.qos, retain=True
                )
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False
            _log.info("mqtt.disconnected")

    def _on_connect(
        self,
        client: object,
        userdata: object,
        flags: object,
        rc: int,
        properties: object | None = None,
    ) -> None:
        self._connected = True
        _log.info("mqtt.connected", host=self.config.host, port=self.config.port)

        if self.config.subscribe_commands:
            prefix = self.config.topic_prefix
            topics = [
                f"{prefix}/commands/emotion",
                f"{prefix}/commands/state",
                f"{prefix}/commands/speak",
                f"{prefix}/commands/servo",
            ]
            for topic in topics:
                self._client.subscribe(topic, qos=self.config.qos)
                _log.info("mqtt.subscribed", topic=topic)

        status_topic = f"{self.config.topic_prefix}/status"
        self._client.publish(status_topic, payload="online", qos=self.config.qos, retain=True)

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
            _log.warning("mqtt.unexpected_disconnect", rc=rc)

    def _on_message(self, client: object, userdata: object, msg: object) -> None:
        """Handle incoming MQTT command messages."""
        try:
            topic = getattr(msg, "topic", "")
            payload = getattr(msg, "payload", b"")
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", errors="replace")

            _log.info("mqtt.command_received", topic=topic, payload=payload[:200])

            try:
                data = json.loads(payload) if payload.strip() else {}
            except json.JSONDecodeError:
                data = {"raw": payload}

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return

            if "emotion" in topic:
                emotion = data.get("emotion", "neutral")
                intensity = float(data.get("intensity", 1.0))
                t = loop.create_task(
                    self.bus.publish(
                        EmotionChanged(
                            previous=EmotionName.NEUTRAL,
                            current=EmotionName(emotion),
                            intensity=intensity,
                        )
                    )
                )
                self._pending_tasks.append(t)
                t.add_done_callback(_make_task_done_callback(self._pending_tasks, _log, "mqtt"))
            elif "state" in topic:
                from robot.behavior.state_machine import RobotState

                state = data.get("state", "idle")
                t = loop.create_task(
                    self.bus.publish(
                        StateChanged(previous=RobotState.IDLE, current=RobotState(state))
                    )
                )
                self._pending_tasks.append(t)
                t.add_done_callback(_make_task_done_callback(self._pending_tasks, _log, "mqtt"))
            elif "speak" in topic:
                text = data.get("text", payload)
                t = loop.create_task(self.bus.publish(SpeechRecognized(text=text, confidence=1.0)))
                self._pending_tasks.append(t)
                t.add_done_callback(_make_task_done_callback(self._pending_tasks, _log, "mqtt"))
            elif "servo" in topic:
                name = data.get("name", "pan")
                angle = float(data.get("angle", 90.0))
                t = loop.create_task(self.bus.publish(ServoMoved(name=name, angle=angle)))
                self._pending_tasks.append(t)
                t.add_done_callback(_make_task_done_callback(self._pending_tasks, _log, "mqtt"))

        except Exception:
            _log.exception("mqtt.command_error")

    async def _on_local_event(self, event: object) -> None:
        """Publish a local event to MQTT."""
        if not self.config.publish_events or not self._connected or self._client is None:
            return

        topic = f"{self.config.topic_prefix}/events/{_event_type_name(event)}"
        try:
            payload = json.dumps(_serialise_event(event))
            self._client.publish(topic, payload=payload, qos=self.config.qos, retain=False)
        except Exception:
            _log.exception("mqtt.publish_failed", topic=topic)


__all__ = ["MqttBridge", "MqttConfig"]
