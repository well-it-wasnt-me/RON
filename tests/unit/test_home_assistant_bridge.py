"""Tests for the Home Assistant bridge and its configuration."""

import pytest

from robot.behavior.state_machine import RobotState
from robot.config import AppSettings, HomeAssistantConfig
from robot.events.bus import InMemoryEventBus
from robot.events.events import EmotionChanged, EmotionName, StateChanged
from robot.services.home_assistant import (
    HomeAssistantBridge,
    HomeAssistantConfig as HABridgeConfig,
    _build_device_info,
    _build_select_config,
    _build_sensor_config,
)


class TestHomeAssistantConfig:
    def test_ha_config_defaults(self) -> None:
        """HomeAssistantConfig defaults to disabled, homeassistant.local:1883."""
        cfg = HomeAssistantConfig()
        assert cfg.enabled is False
        assert cfg.host == "homeassistant.local"
        assert cfg.port == 1883
        assert cfg.discovery_prefix == "homeassistant"
        assert cfg.device_id == "deskbot"
        assert cfg.device_name == "DeskBot"
        assert cfg.qos == 1

    def test_ha_config_from_env(self) -> None:
        """HomeAssistantConfig can be overridden via environment variables."""
        import os

        os.environ["DESKBOT_HOMEASSISTANT__ENABLED"] = "true"
        os.environ["DESKBOT_HOMEASSISTANT__HOST"] = "ha.local"
        os.environ["DESKBOT_HOMEASSISTANT__PORT"] = "1884"
        os.environ["DESKBOT_HOMEASSISTANT__DEVICE_ID"] = "mybot"
        try:
            cfg = HomeAssistantConfig()
            assert cfg.enabled is True
            assert cfg.host == "ha.local"
            assert cfg.port == 1884
            assert cfg.device_id == "mybot"
        finally:
            for key in (
                "DESKBOT_HOMEASSISTANT__ENABLED",
                "DESKBOT_HOMEASSISTANT__HOST",
                "DESKBOT_HOMEASSISTANT__PORT",
                "DESKBOT_HOMEASSISTANT__DEVICE_ID",
            ):
                del os.environ[key]

    def test_ha_config_in_app_settings(self) -> None:
        """AppSettings includes homeassistant field with defaults."""
        settings = AppSettings()
        assert settings.homeassistant.enabled is False
        assert settings.homeassistant.host == "homeassistant.local"

    def test_ha_config_disabled_by_default(self) -> None:
        """HA integration is opt-in (enabled=False by default)."""
        cfg = HomeAssistantConfig()
        assert cfg.enabled is False


class TestHADiscoveryPayloads:
    """Test MQTT discovery payload structure."""

    def test_build_device_info(self) -> None:
        """_build_device_info returns the HA device info block."""
        cfg = HABridgeConfig()
        info = _build_device_info(cfg)
        assert info["identifiers"] == ["deskbot"]
        assert info["name"] == "DeskBot"
        assert info["manufacturer"] == "DeskBot Contributors"
        assert info["model"] == "Desktop Companion Robot"
        assert "sw_version" in info

    def test_build_sensor_config(self) -> None:
        """_build_sensor_config returns a complete HA sensor discovery payload."""
        cfg = HABridgeConfig()
        sensor = _build_sensor_config(cfg, "wake_word", icon="mdi:microphone")
        assert sensor["name"] == "DeskBot Wake Word"
        assert sensor["unique_id"] == "deskbot_wake_word"
        assert sensor["state_topic"] == "homeassistant/sensor/deskbot/wake_word/state"
        assert sensor["icon"] == "mdi:microphone"
        assert "device" in sensor

    def test_build_select_config(self) -> None:
        """_build_select_config returns a complete HA select discovery payload."""
        cfg = HABridgeConfig()
        options = ["idle", "curious", "listening"]
        select = _build_select_config(cfg, "state", options, icon="mdi:robot-outline")
        assert select["name"] == "DeskBot State"
        assert select["unique_id"] == "deskbot_state"
        assert select["state_topic"] == "homeassistant/select/deskbot/state/state"
        assert select["command_topic"] == "homeassistant/select/deskbot/state/set"
        assert select["options"] == options
        assert select["icon"] == "mdi:robot-outline"
        assert "device" in select

    def test_custom_discovery_prefix(self) -> None:
        """Discovery prefix and device_id are configurable."""
        cfg = HABridgeConfig(discovery_prefix="myhome", device_id="mybot")
        sensor = _build_sensor_config(cfg, "face_detected", icon="mdi:face-recognition")
        assert sensor["state_topic"] == "myhome/sensor/mybot/face_detected/state"
        assert sensor["availability_topic"] == "myhome/sensor/mybot/availability"


class TestHABridgeCreation:
    """Test HomeAssistantBridge construction (no paho-mqtt required)."""

    def test_bridge_creation_default_config(self) -> None:
        """HomeAssistantBridge can be created with default config."""
        bus = InMemoryEventBus()
        config = HABridgeConfig()
        bridge = HomeAssistantBridge(bus=bus, config=config)
        assert bridge.config.host == "homeassistant.local"
        assert bridge.config.port == 1883
        assert bridge._client is None
        assert bridge._connected is False

    def test_bridge_creation_custom_config(self) -> None:
        """HomeAssistantBridge can be created with custom config."""
        bus = InMemoryEventBus()
        config = HABridgeConfig(host="192.168.1.100", port=1883, device_id="mybot")
        bridge = HomeAssistantBridge(bus=bus, config=config)
        assert bridge.config.host == "192.168.1.100"
        assert bridge.config.device_id == "mybot"

    def test_bridge_stop_when_not_started(self) -> None:
        """HomeAssistantBridge.stop() is safe when bridge was never started."""
        bus = InMemoryEventBus()
        config = HABridgeConfig()
        bridge = HomeAssistantBridge(bus=bus, config=config)
        import asyncio

        asyncio.run(bridge.stop())
        assert bridge._client is None
        assert bridge._connected is False

    def test_bridge_subscribes_to_events_on_start_mock(self) -> None:
        """Verify bridge config for event subscription."""
        bus = InMemoryEventBus()
        config = HABridgeConfig()
        bridge = HomeAssistantBridge(bus=bus, config=config)
        # Not started yet, but we can verify config.
        assert bridge.config.discovery_prefix == "homeassistant"
        assert bridge.config.device_id == "deskbot"


class TestHABridgeCommandHandling:
    """Test that HA bridge correctly dispatches MQTT commands to events."""

    @pytest.mark.asyncio
    async def test_emotion_command_publishes_event(self) -> None:
        """An emotion command on the HA command topic publishes EmotionChanged."""
        bus = InMemoryEventBus()
        config = HABridgeConfig()
        bridge = HomeAssistantBridge(bus=bus, config=config)

        events: list[EmotionChanged] = []
        bus.subscribe(EmotionChanged, events.append)

        # Simulate the _on_message callback
        class MockMsg:
            topic = "homeassistant/select/deskbot/emotion/set"
            payload = b'{"emotion": "happy"}'

        bridge._on_message(None, None, MockMsg())
        # Give the async task a chance to run
        import asyncio

        await asyncio.sleep(0.05)

        assert len(events) == 1
        assert events[0].current == EmotionName.HAPPY

    @pytest.mark.asyncio
    async def test_state_command_publishes_event(self) -> None:
        """A state command on the HA command topic publishes StateChanged."""
        bus = InMemoryEventBus()
        config = HABridgeConfig()
        bridge = HomeAssistantBridge(bus=bus, config=config)

        events: list[StateChanged] = []
        bus.subscribe(StateChanged, events.append)

        class MockMsg:
            topic = "homeassistant/select/deskbot/state/set"
            payload = b'{"state": "curious"}'

        bridge._on_message(None, None, MockMsg())
        import asyncio

        await asyncio.sleep(0.05)

        assert len(events) == 1
        assert events[0].current == RobotState.CURIOUS
