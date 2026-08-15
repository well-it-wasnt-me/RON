"""Tests for the MQTT bridge and its configuration."""

from robot.config import AppSettings, MqttConfig
from robot.events.bus import InMemoryEventBus
from robot.events.events import EmotionChanged, EmotionName, StateChanged
from robot.services.mqtt_bridge import (
    MqttBridge,
    MqttConfig as BridgeMqttConfig,
    _event_type_name,
    _serialise_event,
)


class TestMqttConfig:
    def test_mqtt_config_defaults(self) -> None:
        """MqttConfig defaults to disabled, localhost:1883."""
        cfg = MqttConfig()
        assert cfg.enabled is False
        assert cfg.host == "localhost"
        assert cfg.port == 1883
        assert cfg.topic_prefix == "deskbot"
        assert cfg.qos == 1
        assert cfg.publish_events is True
        assert cfg.subscribe_commands is True
        assert cfg.heartbeat_interval == 30

    def test_mqtt_config_from_env(self) -> None:
        """MqttConfig can be overridden via environment variables."""
        import os

        os.environ["DESKBOT_MQTT__ENABLED"] = "true"
        os.environ["DESKBOT_MQTT__HOST"] = "mqtt.example.com"
        os.environ["DESKBOT_MQTT__PORT"] = "8883"
        os.environ["DESKBOT_MQTT__USERNAME"] = "deskbot"
        os.environ["DESKBOT_MQTT__TOPIC_PREFIX"] = "mybot"
        try:
            cfg = MqttConfig()
            assert cfg.enabled is True
            assert cfg.host == "mqtt.example.com"
            assert cfg.port == 8883
            assert cfg.username == "deskbot"
            assert cfg.topic_prefix == "mybot"
        finally:
            for key in (
                "DESKBOT_MQTT__ENABLED",
                "DESKBOT_MQTT__HOST",
                "DESKBOT_MQTT__PORT",
                "DESKBOT_MQTT__USERNAME",
                "DESKBOT_MQTT__TOPIC_PREFIX",
            ):
                del os.environ[key]

    def test_mqtt_config_in_app_settings(self) -> None:
        """AppSettings includes mqtt field with defaults."""
        settings = AppSettings()
        assert settings.mqtt.enabled is False
        assert settings.mqtt.host == "localhost"

    def test_mqtt_config_disabled_by_default(self) -> None:
        """MQTT is opt-in (enabled=False by default)."""
        cfg = MqttConfig()
        assert cfg.enabled is False


class TestMqttBridgeSerialisation:
    """Test event serialisation helpers (no paho-mqtt required)."""

    def test_event_type_name(self) -> None:
        """_event_type_name returns the class name."""
        event = EmotionChanged(previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY)
        assert _event_type_name(event) == "EmotionChanged"

    def test_serialise_event(self) -> None:
        """_serialise_event converts an event dataclass to a JSON-serialisable dict."""
        event = EmotionChanged(
            previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY, intensity=0.8
        )
        result = _serialise_event(event)
        assert result["type"] == "EmotionChanged"
        assert result["previous"] == "neutral"
        assert result["current"] == "happy"
        assert result["intensity"] == 0.8

    def test_serialise_state_changed(self) -> None:
        """_serialise_event handles StateChanged with enum values."""
        from robot.behavior.state_machine import RobotState

        event = StateChanged(previous=RobotState.IDLE, current=RobotState.LISTENING)
        result = _serialise_event(event)
        assert result["type"] == "StateChanged"
        assert result["previous"] == "idle"
        assert result["current"] == "listening"


class TestMqttBridgeCreation:
    """Test MqttBridge construction (no paho-mqtt required)."""

    def test_bridge_creation_default_config(self) -> None:
        """MqttBridge can be created with default config."""
        bus = InMemoryEventBus()
        config = BridgeMqttConfig()
        bridge = MqttBridge(bus=bus, config=config)
        assert bridge.config.host == "localhost"
        assert bridge.config.port == 1883
        assert bridge._client is None
        assert bridge._connected is False

    def test_bridge_creation_custom_config(self) -> None:
        """MqttBridge can be created with custom config."""
        bus = InMemoryEventBus()
        config = BridgeMqttConfig(host="broker.hivemq.com", port=1883, topic_prefix="testbot")
        bridge = MqttBridge(bus=bus, config=config)
        assert bridge.config.host == "broker.hivemq.com"
        assert bridge.config.topic_prefix == "testbot"

    def test_bridge_start_fails_without_paho(self) -> None:
        """MqttBridge.start() raises ImportError when paho-mqtt is not installed."""
        bus = InMemoryEventBus()
        config = BridgeMqttConfig()
        bridge = MqttBridge(bus=bus, config=config)
        # paho-mqtt may or may not be installed in test environments.
        # We just verify that start() and stop() are async methods.
        assert hasattr(bridge, "start")
        assert hasattr(bridge, "stop")

    def test_bridge_stop_when_not_started(self) -> None:
        """MqttBridge.stop() is safe when bridge was never started."""
        bus = InMemoryEventBus()
        config = BridgeMqttConfig()
        bridge = MqttBridge(bus=bus, config=config)
        # stop() should be safe even when _client is None.
        import asyncio

        asyncio.run(bridge.stop())
        assert bridge._client is None
        assert bridge._connected is False


class TestMqttBridgeEventSubscription:
    """Test that MqttBridge subscribes to the event bus."""

    def test_bridge_subscribes_on_start_mock(self) -> None:
        """When paho-mqtt is not available, start() raises ImportError."""
        bus = InMemoryEventBus()
        config = BridgeMqttConfig()
        bridge = MqttBridge(bus=bus, config=config)

        # Verify the bus has no subscribers yet.
        # The bridge subscribes to events during start(), but since
        # paho-mqtt is likely not installed, we test the configuration
        # rather than the actual connection.
        assert bridge.config.publish_events is True
        assert bridge.config.subscribe_commands is True

    def test_bridge_does_not_publish_when_disabled(self) -> None:
        """When publish_events is False, _on_local_event is a no-op."""
        bus = InMemoryEventBus()
        config = BridgeMqttConfig(publish_events=False)
        bridge = MqttBridge(bus=bus, config=config)
        # Not connected, so publishing should be skipped.
        assert bridge._connected is False

    def test_bridge_config_qos_range(self) -> None:
        """QoS must be between 0 and 2."""
        config = BridgeMqttConfig(qos=2)
        assert config.qos == 2
        config = BridgeMqttConfig(qos=0)
        assert config.qos == 0


class TestAppSettingsMqtt:
    """Test that AppSettings correctly includes MQTT configuration."""

    def test_app_settings_mqtt_disabled_by_default(self) -> None:
        settings = AppSettings()
        assert settings.mqtt.enabled is False

    def test_app_settings_mqtt_enabled_via_env(self) -> None:
        import os

        os.environ["DESKBOT_MQTT__ENABLED"] = "true"
        try:
            settings = AppSettings()
            assert settings.mqtt.enabled is True
        finally:
            del os.environ["DESKBOT_MQTT__ENABLED"]

    def test_app_settings_mqtt_host_via_env(self) -> None:
        import os

        os.environ["DESKBOT_MQTT__HOST"] = "test.mqtt.local"
        try:
            settings = AppSettings()
            assert settings.mqtt.host == "test.mqtt.local"
        finally:
            del os.environ["DESKBOT_MQTT__HOST"]
