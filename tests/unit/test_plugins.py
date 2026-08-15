"""Tests for the plugin system."""

import pytest

from robot.events.bus import InMemoryEventBus
from robot.plugins.plugin import PluginInfo, PluginState
from robot.plugins.registry import PluginError, PluginRegistry


class SimplePlugin:
    """A minimal test plugin."""

    def __init__(self, bus: InMemoryEventBus | None = None) -> None:
        self.started = False
        self.stopped = False
        self.loaded = False

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(name="simple", version="1.0.0", description="A test plugin")

    async def load(self) -> None:
        self.loaded = True

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def unload(self) -> None:
        pass


class DependentPlugin:
    """A plugin that depends on SimplePlugin."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(name="dependent", version="1.0.0", depends=("simple",))

    async def load(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def unload(self) -> None:
        pass


class TestPluginInfo:
    def test_equality(self) -> None:
        a = PluginInfo(name="test", version="1.0")
        b = PluginInfo(name="test", version="2.0")
        assert a == b  # Equality is by name only

    def test_hash(self) -> None:
        a = PluginInfo(name="test", version="1.0")
        b = PluginInfo(name="test", version="2.0")
        assert hash(a) == hash(b)


class TestPluginRegistry:
    def test_register_and_list(self) -> None:
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        plugin = SimplePlugin(bus=bus)
        registry.register(plugin)
        assert registry.plugin_count == 1
        info_list = registry.list_plugins()
        assert len(info_list) == 1
        assert info_list[0].name == "simple"

    def test_register_duplicate_raises(self) -> None:
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        plugin = SimplePlugin(bus=bus)
        registry.register(plugin)
        with pytest.raises(PluginError):
            registry.register(SimplePlugin(bus=bus))

    def test_unregister(self) -> None:
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        plugin = SimplePlugin(bus=bus)
        registry.register(plugin)
        registry.unregister("simple")
        assert registry.plugin_count == 0

    async def test_unregister_loaded_raises(self) -> None:
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        plugin = SimplePlugin(bus=bus)
        registry.register(plugin)
        await registry.load_all()
        with pytest.raises(PluginError):
            registry.unregister("simple")

    def test_get_plugin(self) -> None:
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        plugin = SimplePlugin(bus=bus)
        registry.register(plugin)
        result = registry.get("simple")
        assert result is plugin

    def test_get_missing_raises(self) -> None:
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        with pytest.raises(PluginError):
            registry.get("missing")

    async def test_load_all(self) -> None:
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        plugin = SimplePlugin(bus=bus)
        registry.register(plugin)
        await registry.load_all()
        assert registry.state_of("simple") == PluginState.LOADED

    async def test_start_all(self) -> None:
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        plugin = SimplePlugin(bus=bus)
        registry.register(plugin)
        await registry.load_all()
        await registry.start_all()
        assert registry.state_of("simple") == PluginState.STARTED
        assert plugin.started

    async def test_stop_all(self) -> None:
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        plugin = SimplePlugin(bus=bus)
        registry.register(plugin)
        await registry.load_all()
        await registry.start_all()
        await registry.stop_all()
        assert registry.state_of("simple") == PluginState.STOPPED
        assert plugin.stopped

    def test_dependency_ordering(self) -> None:
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        simple = SimplePlugin(bus=bus)
        dependent = DependentPlugin()
        registry.register(dependent)
        registry.register(simple)
        # The topological sort should put simple before dependent.
        order = registry._topological_sort()
        assert order.index("simple") < order.index("dependent")

    def test_circular_dependency_raises(self) -> None:
        bus = InMemoryEventBus()

        class PluginA:
            @property
            def info(self) -> PluginInfo:
                return PluginInfo(name="a", version="1.0", depends=("b",))

            async def load(self) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def unload(self) -> None:
                pass

        class PluginB:
            @property
            def info(self) -> PluginInfo:
                return PluginInfo(name="b", version="1.0", depends=("a",))

            async def load(self) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def unload(self) -> None:
                pass

        registry = PluginRegistry(bus)
        registry.register(PluginA())
        registry.register(PluginB())
        with pytest.raises(PluginError, match="circular"):
            registry._topological_sort()

    def test_missing_dependency_raises(self) -> None:
        bus = InMemoryEventBus()

        class PluginWithMissingDep:
            @property
            def info(self) -> PluginInfo:
                return PluginInfo(name="orphan", version="1.0", depends=("nonexistent",))

            async def load(self) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def unload(self) -> None:
                pass

        registry = PluginRegistry(bus)
        registry.register(PluginWithMissingDep())
        with pytest.raises(PluginError, match="not registered"):
            registry._topological_sort()


# ---------------------------------------------------------------------------
# PluginConfig tests
# ---------------------------------------------------------------------------
class TestPluginConfig:
    def test_plugin_config_defaults(self) -> None:
        from robot.config import PluginConfig

        pc = PluginConfig()
        assert pc.enabled is True
        assert pc.discover_entry_points is True
        assert pc.plugin_packages == []

    def test_plugin_config_from_env(self) -> None:
        import os

        from robot.config import PluginConfig

        os.environ["DESKBOT_PLUGINS__ENABLED"] = "false"
        os.environ["DESKBOT_PLUGINS__DISCOVER_ENTRY_POINTS"] = "false"
        try:
            pc = PluginConfig()
            assert pc.enabled is False
            assert pc.discover_entry_points is False
        finally:
            del os.environ["DESKBOT_PLUGINS__ENABLED"]
            del os.environ["DESKBOT_PLUGINS__DISCOVER_ENTRY_POINTS"]

    def test_plugin_config_in_app_settings(self) -> None:
        from robot.config import AppSettings

        settings = AppSettings()
        assert settings.plugins.enabled is True
        assert settings.plugins.discover_entry_points is True

    def test_plugin_config_disabled_via_env(self) -> None:
        import os

        from robot.config import AppSettings

        os.environ["DESKBOT_PLUGINS__ENABLED"] = "false"
        try:
            settings = AppSettings()
            assert settings.plugins.enabled is False
        finally:
            del os.environ["DESKBOT_PLUGINS__ENABLED"]


# ---------------------------------------------------------------------------
# Plugin lifecycle wiring tests
# ---------------------------------------------------------------------------
class TestPluginLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self) -> None:
        """Test load -> start -> stop -> unload lifecycle."""
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        plugin = SimplePlugin(bus=bus)
        registry.register(plugin)

        await registry.load_all()
        assert registry.state_of("simple") == PluginState.LOADED
        assert plugin.loaded is True

        await registry.start_all()
        assert registry.state_of("simple") == PluginState.STARTED
        assert plugin.started is True

        await registry.stop_all()
        assert registry.state_of("simple") == PluginState.STOPPED
        assert plugin.stopped is True

        await registry.unload_all()
        assert registry.state_of("simple") == PluginState.UNLOADED

    @pytest.mark.asyncio
    async def test_plugin_that_fails_start_does_not_crash(self) -> None:
        """A plugin that raises in start() gets ERRORED state but doesn't crash the registry."""

        class FailingPlugin:
            @property
            def info(self) -> PluginInfo:
                return PluginInfo(name="failing", version="1.0.0")

            async def load(self) -> None:
                pass

            async def start(self) -> None:
                raise RuntimeError("start failed!")

            async def stop(self) -> None:
                pass

            async def unload(self) -> None:
                pass

        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        registry.register(FailingPlugin())

        await registry.load_all()
        assert registry.state_of("failing") == PluginState.LOADED

        await registry.start_all()
        # Should be errored, not crashed.
        assert registry.state_of("failing") == PluginState.ERRORED

    @pytest.mark.asyncio
    async def test_empty_registry_is_safe(self) -> None:
        """An empty plugin registry should not crash on any lifecycle call."""
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)

        await registry.load_all()
        await registry.start_all()
        await registry.stop_all()
        await registry.unload_all()
        assert registry.plugin_count == 0

    @pytest.mark.asyncio
    async def test_discover_entry_points_with_no_plugins(self) -> None:
        """discover_entry_points returns an empty list when no entry points exist."""
        bus = InMemoryEventBus()
        registry = PluginRegistry(bus)
        discovered = registry.discover_entry_points()
        assert discovered == []


class TestPluginRegistryExtended:
    """Extended tests for :class:`PluginRegistry`."""

    def test_list_plugins_empty(self) -> None:
        from robot.events.bus import InMemoryEventBus
        from robot.plugins.registry import PluginRegistry

        bus = InMemoryEventBus()
        registry = PluginRegistry(bus=bus)
        assert registry.list_plugins() == []

    def test_plugin_count_empty(self) -> None:
        from robot.events.bus import InMemoryEventBus
        from robot.plugins.registry import PluginRegistry

        bus = InMemoryEventBus()
        registry = PluginRegistry(bus=bus)
        assert registry.plugin_count == 0

    def test_state_of_unknown_plugin(self) -> None:
        from robot.events.bus import InMemoryEventBus
        from robot.plugins.registry import PluginRegistry, PluginState  # type: ignore[attr-defined]

        bus = InMemoryEventBus()
        registry = PluginRegistry(bus=bus)
        state = registry.state_of("unknown")
        assert state == PluginState.UNLOADED

    def test_get_unknown_plugin_raises(self) -> None:
        from robot.events.bus import InMemoryEventBus
        from robot.plugins.registry import PluginError, PluginRegistry

        bus = InMemoryEventBus()
        registry = PluginRegistry(bus=bus)
        try:
            registry.get("unknown")
            pytest.fail("Expected PluginError")
        except PluginError:
            pass  # expected

    @pytest.mark.asyncio
    async def test_load_all_empty(self) -> None:
        from robot.events.bus import InMemoryEventBus
        from robot.plugins.registry import PluginRegistry

        bus = InMemoryEventBus()
        registry = PluginRegistry(bus=bus)
        await registry.load_all()
        # Should not raise

    @pytest.mark.asyncio
    async def test_start_all_empty(self) -> None:
        from robot.events.bus import InMemoryEventBus
        from robot.plugins.registry import PluginRegistry

        bus = InMemoryEventBus()
        registry = PluginRegistry(bus=bus)
        await registry.start_all()
        # Should not raise

    @pytest.mark.asyncio
    async def test_stop_all_empty(self) -> None:
        from robot.events.bus import InMemoryEventBus
        from robot.plugins.registry import PluginRegistry

        bus = InMemoryEventBus()
        registry = PluginRegistry(bus=bus)
        await registry.stop_all()
        # Should not raise

    @pytest.mark.asyncio
    async def test_unload_all_empty(self) -> None:
        from robot.events.bus import InMemoryEventBus
        from robot.plugins.registry import PluginRegistry

        bus = InMemoryEventBus()
        registry = PluginRegistry(bus=bus)
        await registry.unload_all()
        # Should not raise
