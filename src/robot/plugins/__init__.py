"""Plugin system for DeskBot.

The plugin system allows extending DeskBot with new capabilities at runtime.
Plugins can subscribe to events, register new commands, and extend the
application lifecycle.

Three built-in plugins are provided:

* :class:`MqttBridgePlugin` - publishes/subscribes events via MQTT.
* :class:`HomeAssistantPlugin` - exposes the robot as an HA entity.
* :class:`CalibrationPlugin` - serves a web-based calibration dashboard.

Third-party plugins can be registered via entry points or manually.
"""

from robot.plugins.plugin import Plugin, PluginInfo, PluginState
from robot.plugins.registry import PluginRegistry

__all__ = [
    "Plugin",
    "PluginInfo",
    "PluginRegistry",
    "PluginState",
]
