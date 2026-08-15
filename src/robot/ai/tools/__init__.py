"""Function calling / tool use system.

Allows the LLM to invoke robot actions (change emotion, move servos,
play sounds, etc.) through a structured tool-calling interface. The
:class:`ToolRegistry` manages available tools, and :class:`ToolExecutor`
dispatches tool calls from LLM responses.
"""

from robot.ai.tools.executor import ToolExecutor
from robot.ai.tools.registry import ToolRegistry
from robot.ai.tools.schema import ToolDefinition, ToolParameter, ToolParameterType

__all__ = [
    "ToolDefinition",
    "ToolExecutor",
    "ToolParameter",
    "ToolParameterType",
    "ToolRegistry",
]
