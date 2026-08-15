"""Tool registry - manages available LLM-callable tools.

The registry holds :class:`ToolDefinition` instances and their
corresponding handler functions. When the LLM produces a tool call,
the :class:`ToolExecutor` looks up the tool by name and invokes the
handler.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from robot.ai.tools.schema import ToolDefinition, ToolParameter, ToolParameterType
from robot.errors import DeskBotError
from robot.logging import get_logger

_log = get_logger("ai.tools.registry")


class ToolError(DeskBotError):
    """Error during tool registration or execution."""


# Tool handler type: receives a dict of arguments, returns a result dict.
ToolHandler = Callable[..., Awaitable[dict[str, Any]]]


class ToolRegistry:
    """Registry of available LLM tools and their handlers.

    Usage::

        registry = ToolRegistry()


        @registry.register_tool
        async def change_emotion(emotion: str, intensity: float = 1.0) -> dict:
            ...
            return {"status": "ok", "emotion": emotion}


        # Get schemas for the LLM prompt
        schemas = registry.get_schemas()

        # Execute a tool call from the LLM
        result = await registry.execute("change_emotion", {"emotion": "happy"})
    """

    def __init__(self) -> None:
        self._tools: OrderedDict[str, ToolDefinition] = OrderedDict()
        self._handlers: dict[str, ToolHandler] = {}

    def add(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
    ) -> None:
        """Register a tool and its handler."""
        name = definition.name
        if name in self._tools:
            raise ToolError(f"tool {name!r} is already registered")
        self._tools[name] = definition
        self._handlers[name] = handler
        _log.info("tool.registered", name=name)

    def remove(self, name: str) -> None:
        """Remove a tool by name."""
        if name not in self._tools:
            raise ToolError(f"tool {name!r} is not registered")
        del self._tools[name]
        del self._handlers[name]

    def get(self, name: str) -> ToolDefinition:
        """Return a tool definition by name."""
        if name not in self._tools:
            raise ToolError(f"tool {name!r} is not registered")
        return self._tools[name]

    def get_handler(self, name: str) -> ToolHandler:
        """Return a tool handler by name."""
        if name not in self._handlers:
            raise ToolError(f"no handler for tool {name!r}")
        return self._handlers[name]

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name with the given arguments.

        Returns the handler's result dict. Raises :class:`ToolError`
        if the tool is not found, and propagates any handler exceptions.
        """
        if name not in self._handlers:
            raise ToolError(f"unknown tool {name!r}")
        handler = self._handlers[name]
        try:
            result = await handler(**arguments)
            _log.info("tool.executed", name=name, args_count=len(arguments))
            return result
        except Exception as exc:
            _log.exception("tool.execution_failed", name=name)
            return {"error": str(exc), "tool": name}

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# Built-in tool definitions
# ---------------------------------------------------------------------------
BUILTIN_TOOLS: dict[str, ToolDefinition] = {
    "change_emotion": ToolDefinition(
        name="change_emotion",
        description="Change the robot's emotional expression.",
        parameters=(
            ToolParameter(
                name="emotion",
                type=ToolParameterType.STRING,
                description="The emotion to express.",
                enum=(
                    "neutral",
                    "happy",
                    "curious",
                    "thinking",
                    "sleepy",
                    "embarrassed",
                    "excited",
                    "sad",
                    "surprised",
                    "angry",
                ),
            ),
            ToolParameter(
                name="intensity",
                type=ToolParameterType.NUMBER,
                description="Intensity of the emotion (0.0 to 1.0).",
                required=False,
                default=1.0,
            ),
        ),
    ),
    "play_sound": ToolDefinition(
        name="play_sound",
        description="Play a sound effect through the robot's speaker.",
        parameters=(
            ToolParameter(
                name="name",
                type=ToolParameterType.STRING,
                description="The sound effect name (e.g. 'talk', 'greet', 'notify').",
            ),
        ),
    ),
    "set_state": ToolDefinition(
        name="set_state",
        description="Transition the robot to a new state.",
        parameters=(
            ToolParameter(
                name="state",
                type=ToolParameterType.STRING,
                description="The target state.",
                enum=(
                    "idle",
                    "curious",
                    "listening",
                    "thinking",
                    "speaking",
                    "sleeping",
                ),
            ),
        ),
    ),
    "move_servo": ToolDefinition(
        name="move_servo",
        description="Move a specific servo to a target angle.",
        parameters=(
            ToolParameter(
                name="servo",
                type=ToolParameterType.STRING,
                description="Servo name: 'pan', 'tilt', 'left_arm', or 'right_arm'.",
                enum=("pan", "tilt", "left_arm", "right_arm"),
            ),
            ToolParameter(
                name="angle",
                type=ToolParameterType.NUMBER,
                description="Target angle in degrees (0-180).",
            ),
            ToolParameter(
                name="duration_s",
                type=ToolParameterType.NUMBER,
                description="Movement duration in seconds.",
                required=False,
                default=0.4,
            ),
        ),
    ),
    "speak": ToolDefinition(
        name="speak",
        description="Make the robot say something out loud using TTS.",
        parameters=(
            ToolParameter(
                name="text",
                type=ToolParameterType.STRING,
                description="The text for the robot to speak.",
            ),
        ),
    ),
}


__all__ = ["BUILTIN_TOOLS", "ToolHandler", "ToolRegistry"]
