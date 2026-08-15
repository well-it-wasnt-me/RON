"""Plugin registry - loads, starts, stops, and discovers plugins.

The registry manages the full lifecycle of every plugin and ensures
that dependency ordering is respected. It also supports discovering
plugins from ``deskbot.plugins`` entry points.
"""

from __future__ import annotations

from collections import OrderedDict

from robot.errors import DeskBotError
from robot.events.bus import InMemoryEventBus
from robot.logging import get_logger
from robot.plugins.plugin import Plugin, PluginInfo, PluginState

_log = get_logger("plugins.registry")


class PluginError(DeskBotError):
    """Plugin lifecycle or dependency error."""


class PluginRegistry:
    """Manages plugin lifecycle: load, start, stop, unload.

    Usage::

        registry = PluginRegistry(bus=event_bus)
        registry.register(my_plugin)
        await registry.load_all()
        await registry.start_all()
        # ... later ...
        await registry.stop_all()
        await registry.unload_all()
    """

    def __init__(self, bus: InMemoryEventBus) -> None:
        self._bus = bus
        self._plugins: OrderedDict[str, Plugin] = OrderedDict()
        self._states: dict[str, PluginState] = OrderedDict()

    # ------------------------------------------------------------------ register
    def register(self, plugin: Plugin) -> None:
        """Register a plugin by name.

        Raises :class:`PluginError` if a plugin with the same name is
        already registered.
        """
        name = plugin.info.name
        if name in self._plugins:
            raise PluginError(f"plugin {name!r} is already registered")
        self._plugins[name] = plugin
        self._states[name] = PluginState.UNLOADED

    def unregister(self, name: str) -> None:
        """Remove a plugin. Must be in UNLOADED or ERRORED state."""
        state = self._states.get(name)
        if state is None:
            raise PluginError(f"plugin {name!r} is not registered")
        if state not in (PluginState.UNLOADED, PluginState.ERRORED):
            raise PluginError(
                f"cannot unregister plugin {name!r} in state {state.value}; "
                "stop and unload it first"
            )
        del self._plugins[name]
        del self._states[name]

    # ------------------------------------------------------------------ lifecycle
    async def load_all(self) -> None:
        """Load all registered plugins in dependency order."""
        ordered = self._topological_sort()
        for name in ordered:
            plugin = self._plugins[name]
            try:
                await plugin.load()
                self._states[name] = PluginState.LOADED
                _log.info("plugin.loaded", name=name)
            except Exception:
                self._states[name] = PluginState.ERRORED
                _log.exception("plugin.load_failed", name=name)

    async def start_all(self) -> None:
        """Start all loaded plugins."""
        for name, plugin in self._plugins.items():
            if self._states[name] == PluginState.LOADED:
                try:
                    await plugin.start()
                    self._states[name] = PluginState.STARTED
                    _log.info("plugin.started", name=name)
                except Exception:
                    self._states[name] = PluginState.ERRORED
                    _log.exception("plugin.start_failed", name=name)

    async def stop_all(self) -> None:
        """Stop all started plugins in reverse order."""
        for name in reversed(list(self._plugins.keys())):
            if self._states[name] == PluginState.STARTED:
                try:
                    await self._plugins[name].stop()
                    self._states[name] = PluginState.STOPPED
                    _log.info("plugin.stopped", name=name)
                except Exception:
                    self._states[name] = PluginState.ERRORED
                    _log.exception("plugin.stop_failed", name=name)

    async def unload_all(self) -> None:
        """Unload all plugins in reverse order."""
        for name in reversed(list(self._plugins.keys())):
            if self._states[name] in (
                PluginState.LOADED,
                PluginState.STOPPED,
                PluginState.UNLOADED,
            ):
                try:
                    await self._plugins[name].unload()
                    self._states[name] = PluginState.UNLOADED
                    _log.info("plugin.unloaded", name=name)
                except Exception:
                    self._states[name] = PluginState.ERRORED
                    _log.exception("plugin.unload_failed", name=name)

    # ------------------------------------------------------------------ query
    def get(self, name: str) -> Plugin:
        """Return a plugin by name."""
        if name not in self._plugins:
            raise PluginError(f"plugin {name!r} not found")
        return self._plugins[name]

    def list_plugins(self) -> list[PluginInfo]:
        """Return info for all registered plugins."""
        return [p.info for p in self._plugins.values()]

    def state_of(self, name: str) -> PluginState:
        """Return the current state of a plugin."""
        return self._states.get(name, PluginState.UNLOADED)

    @property
    def plugin_count(self) -> int:
        return len(self._plugins)

    # ------------------------------------------------------------------ discovery
    def discover_entry_points(self) -> list[Plugin]:
        """Discover plugins from the ``deskbot.plugins`` entry point group.

        Returns a list of instantiated plugins. Call :meth:`register`
        to add them to this registry.
        """
        plugins: list[Plugin] = []
        try:
            # Python 3.12+ uses importlib.metadata directly.
            from importlib.metadata import entry_points

            eps = entry_points(group="deskbot.plugins")
            for ep in eps:
                try:
                    factory = ep.load()
                    plugin: Plugin = factory(bus=self._bus)
                    plugins.append(plugin)
                    _log.info("plugin.discovered", name=ep.name, module=ep.value)
                except Exception:
                    _log.exception("plugin.discovery_failed", entry_point=ep.name)
        except Exception:
            _log.warning("plugin.entry_points_unavailable")
        return plugins

    # ------------------------------------------------------------------ internal
    def _topological_sort(self) -> list[str]:
        """Sort plugins so dependencies come first."""
        visited: set[str] = set()
        order: list[str] = []
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise PluginError(f"circular dependency involving {name!r}")
            visiting.add(name)
            plugin = self._plugins.get(name)
            if plugin is not None:
                for dep in plugin.info.depends:
                    if dep not in self._plugins:
                        raise PluginError(
                            f"plugin {name!r} depends on {dep!r} which is not registered"
                        )
                    visit(dep)
            visiting.discard(name)
            visited.add(name)
            order.append(name)

        for name in self._plugins:
            visit(name)
        return order


__all__ = ["PluginError", "PluginRegistry"]
