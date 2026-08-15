"""Tool definition schema for LLM function calling.

Follows the OpenAI function-calling format so tools can be passed
directly to compatible LLMs (OpenAI, Ollama, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolParameterType(str, Enum):
    """JSON Schema types for tool parameters."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass(slots=True, frozen=True)
class ToolParameter:
    """A single parameter in a tool definition."""

    name: str
    type: ToolParameterType = ToolParameterType.STRING
    description: str = ""
    required: bool = True
    enum: tuple[str, ...] = ()
    default: Any = None


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    """A complete tool definition that can be serialised for an LLM.

    Attributes
    ----------
    name:
        The tool name (e.g. ``"change_emotion"``).
    description:
        A human-readable description of what the tool does.
    parameters:
        The list of parameters the tool accepts.
    """

    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()

    def to_openai_schema(self) -> dict[str, Any]:
        """Serialise to the OpenAI function-calling schema format."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type.value,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = list(param.enum)
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


__all__ = ["ToolDefinition", "ToolParameter", "ToolParameterType"]
