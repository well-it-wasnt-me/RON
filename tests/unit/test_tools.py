"""Tests for the tool calling system."""

import pytest

from robot.ai.tools.executor import ToolExecutionError, ToolExecutor
from robot.ai.tools.registry import BUILTIN_TOOLS, ToolError, ToolRegistry
from robot.ai.tools.schema import ToolDefinition, ToolParameter, ToolParameterType
from robot.events.bus import InMemoryEventBus


async def _noop_handler(**kwargs: object) -> dict[str, str]:
    return {"status": "ok"}


class TestToolDefinition:
    def test_to_openai_schema(self) -> None:
        defn = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters=(
                ToolParameter(
                    name="query", type=ToolParameterType.STRING, description="Search query"
                ),
                ToolParameter(
                    name="count",
                    type=ToolParameterType.INTEGER,
                    description="Number of results",
                    required=False,
                    default=5,
                ),
            ),
        )
        schema = defn.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "count" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]
        assert "count" not in schema["function"]["parameters"]["required"]

    def test_enum_parameters(self) -> None:
        defn = ToolDefinition(
            name="choose",
            description="Pick an option",
            parameters=(
                ToolParameter(
                    name="option",
                    type=ToolParameterType.STRING,
                    description="Option",
                    enum=("a", "b", "c"),
                ),
            ),
        )
        schema = defn.to_openai_schema()
        assert schema["function"]["parameters"]["properties"]["option"]["enum"] == ["a", "b", "c"]


class TestToolRegistry:
    def test_register_and_list(self) -> None:
        registry = ToolRegistry()
        defn = ToolDefinition(name="test", description="Test")
        registry.add(defn, handler=_noop_handler)
        assert len(registry.list_tools()) == 1

    def test_register_duplicate_raises(self) -> None:
        registry = ToolRegistry()
        defn = ToolDefinition(name="test", description="Test")
        registry.add(defn, handler=_noop_handler)
        with pytest.raises(ToolError):
            registry.add(defn, handler=_noop_handler)

    def test_remove(self) -> None:
        registry = ToolRegistry()
        defn = ToolDefinition(name="test", description="Test")
        registry.add(defn, handler=_noop_handler)
        registry.remove("test")
        assert len(registry.list_tools()) == 0

    def test_get_schemas(self) -> None:
        registry = ToolRegistry()
        defn = ToolDefinition(name="test", description="Test")
        registry.add(defn, handler=_noop_handler)
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "test"

    def test_contains(self) -> None:
        registry = ToolRegistry()
        defn = ToolDefinition(name="test", description="Test")
        registry.add(defn, handler=_noop_handler)
        assert "test" in registry
        assert "missing" not in registry

    def test_get_handler(self) -> None:
        registry = ToolRegistry()
        defn = ToolDefinition(name="test", description="Test")
        registry.add(defn, handler=_noop_handler)
        result = registry.get_handler("test")
        assert result is _noop_handler

    def test_get_handler_missing_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolError):
            registry.get_handler("missing")


class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_execute_change_emotion(self) -> None:
        bus = InMemoryEventBus()
        registry = ToolRegistry()
        registry.add(BUILTIN_TOOLS["change_emotion"], handler=_noop_handler)
        executor = ToolExecutor(registry=registry, bus=bus)

        result = await executor.execute_tool_call(
            "change_emotion", {"emotion": "happy", "intensity": 0.8}
        )
        assert result["status"] == "ok"
        assert result["emotion"] == "happy"

    @pytest.mark.asyncio
    async def test_execute_play_sound(self) -> None:
        bus = InMemoryEventBus()
        registry = ToolRegistry()
        registry.add(BUILTIN_TOOLS["play_sound"], handler=_noop_handler)
        executor = ToolExecutor(registry=registry, bus=bus)

        result = await executor.execute_tool_call("play_sound", {"name": "greet"})
        assert result["status"] == "ok"
        assert result["sound"] == "greet"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_raises(self) -> None:
        bus = InMemoryEventBus()
        registry = ToolRegistry()
        executor = ToolExecutor(registry=registry, bus=bus)

        with pytest.raises(ToolExecutionError):
            await executor.execute_tool_call("nonexistent", {})

    def test_coerce_type_string_to_int(self) -> None:
        param = ToolParameter(name="count", type=ToolParameterType.INTEGER, description="Count")
        result = ToolExecutor._coerce_type(param, "42")
        assert result == 42

    def test_coerce_type_string_to_float(self) -> None:
        param = ToolParameter(name="amount", type=ToolParameterType.NUMBER, description="Amount")
        result = ToolExecutor._coerce_type(param, "3.14")
        assert abs(result - 3.14) < 0.001

    def test_coerce_type_string_to_bool(self) -> None:
        param = ToolParameter(name="flag", type=ToolParameterType.BOOLEAN, description="Flag")
        assert ToolExecutor._coerce_type(param, "true") is True
        assert ToolExecutor._coerce_type(param, "false") is False

    def test_coerce_type_int_to_string(self) -> None:
        param = ToolParameter(name="text", type=ToolParameterType.STRING, description="Text")
        result = ToolExecutor._coerce_type(param, 42)
        assert result == "42"

    def test_coerce_type_float_to_int(self) -> None:
        param = ToolParameter(name="count", type=ToolParameterType.INTEGER, description="Count")
        result = ToolExecutor._coerce_type(param, 3.7)
        assert result == 3


class TestToolExecutorExtended:
    """Extended tests for :class:`ToolExecutor` built-in tool handlers."""

    @pytest.mark.asyncio
    async def test_execute_set_state(self) -> None:
        bus = InMemoryEventBus()
        registry = ToolRegistry()
        registry.add(BUILTIN_TOOLS["set_state"], handler=_noop_handler)
        executor = ToolExecutor(registry=registry, bus=bus)

        result = await executor.execute_tool_call("set_state", {"state": "curious"})
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_execute_change_emotion_with_coercion(self) -> None:
        bus = InMemoryEventBus()
        registry = ToolRegistry()
        registry.add(BUILTIN_TOOLS["change_emotion"], handler=_noop_handler)
        executor = ToolExecutor(registry=registry, bus=bus)

        result = await executor.execute_tool_call(
            "change_emotion", {"emotion": "happy", "intensity": "0.8"}
        )
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_execute_move_servo_no_controller(self) -> None:
        """Move servo without a controller should still return a result."""
        bus = InMemoryEventBus()
        registry = ToolRegistry()
        registry.add(BUILTIN_TOOLS["move_servo"], handler=_noop_handler)
        executor = ToolExecutor(registry=registry, bus=bus, servo_controller=None)

        result = await executor.execute_tool_call("move_servo", {"servo": "pan", "angle": 45.0})
        # Without a servo controller, it should still respond
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_execute_speak(self) -> None:
        bus = InMemoryEventBus()
        registry = ToolRegistry()
        registry.add(BUILTIN_TOOLS["speak"], handler=_noop_handler)
        executor = ToolExecutor(registry=registry, bus=bus, tts=None)

        result = await executor.execute_tool_call("speak", {"text": "Hello there"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_raises(self) -> None:
        bus = InMemoryEventBus()
        registry = ToolRegistry()
        executor = ToolExecutor(registry=registry, bus=bus)

        with pytest.raises(ToolExecutionError):
            await executor.execute_tool_call("nonexistent", {})


class TestToolRegistryExtended:
    """Extended tests for :class:`ToolRegistry`."""

    def test_tool_count(self) -> None:
        registry = ToolRegistry()
        defn = ToolDefinition(name="test1", description="Test 1")
        registry.add(defn, handler=_noop_handler)
        assert registry.tool_count == 1

        defn2 = ToolDefinition(name="test2", description="Test 2")
        registry.add(defn2, handler=_noop_handler)
        assert registry.tool_count == 2

    def test_remove_nonexistent_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolError):
            registry.remove("nonexistent")

    def test_get(self) -> None:
        registry = ToolRegistry()
        defn = ToolDefinition(name="test", description="Test")
        registry.add(defn, handler=_noop_handler)
        retrieved = registry.get("test")
        assert retrieved.name == "test"

    def test_get_missing_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolError):
            registry.get("missing")

    @pytest.mark.asyncio
    async def test_execute_via_registry(self) -> None:
        registry = ToolRegistry()

        async def handler(**kwargs: object) -> dict[str, str]:
            return {"status": "handled"}

        defn = ToolDefinition(name="test", description="Test")
        registry.add(defn, handler=handler)
        result = await registry.execute("test", {"key": "value"})
        assert result["status"] == "handled"
