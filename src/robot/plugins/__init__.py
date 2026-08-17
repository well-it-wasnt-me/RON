"""Plugin system for DeskBot.

The plugin system allows extending DeskBot with new capabilities at runtime.
Plugins can subscribe to events, register new commands, and extend the
application lifecycle.

The plugin system is the extension point for third-party capabilities.
Third-party plugins can be registered via ``deskbot.plugins`` entry points
or manually via :class:`PluginRegistry`.

Note: the MQTT bridge and Home Assistant discovery are app-level services
(:mod:`robot.services.mqtt_bridge`, :mod:`robot.services.home_assistant`)
wired directly into ``DeskBotApp``, not plugins.
"""

from robot.plugins.plugin import Plugin, PluginInfo, PluginState
from robot.plugins.registry import PluginRegistry

__all__ = [
    "Plugin",
    "PluginInfo",
    "PluginRegistry",
    "PluginState",
]
