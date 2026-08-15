"""LLM interface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Role(str, Enum):
    """Speaker role in a chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True, frozen=True)
class Message:
    """A single message in a chat conversation."""

    role: Role
    content: str


@dataclass(slots=True, frozen=True)
class ToolCall:
    """A single tool/function call from an LLM response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """Structured response from an LLM that may include tool calls.

    Attributes
    ----------
    text:
        The text content of the response (may be empty when the LLM
        only makes tool calls).
    tool_calls:
        A tuple of :class:`ToolCall` objects requested by the LLM.
        Empty when the LLM responds with plain text.
    done:
        Whether this is the final response in a streaming sequence.
        Always ``True`` for non-streaming calls.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    done: bool = True


def text_response(text: str) -> LLMResponse:
    """Convenience factory for a plain-text LLM response."""
    return LLMResponse(text=text, tool_calls=(), done=True)


def tool_response(
    tool_calls: Sequence[ToolCall],
    text: str = "",
) -> LLMResponse:
    """Convenience factory for a tool-call LLM response."""
    return LLMResponse(text=text, tool_calls=tuple(tool_calls), done=True)


@runtime_checkable
class LLM(Protocol):
    """Async language model interface."""

    @property
    def name(self) -> str:
        """Human-readable model name."""

    async def complete(self, messages: Sequence[Message]) -> str:
        """Return the assistant's reply for the given conversation."""

    async def complete_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Return a structured response that may include tool calls.

        When ``tools`` is ``None`` or empty, behaves like
        :meth:`complete` but returns an :class:`LLMResponse` with
        ``tool_calls=()``.

        The default implementation calls :meth:`complete` and wraps
        the result in a plain-text :class:`LLMResponse`.  Providers
        that support function calling override this method.
        """

    async def close(self) -> None: ...


__all__ = [
    "LLM",
    "LLMResponse",
    "Message",
    "Role",
    "ToolCall",
    "text_response",
    "tool_response",
]
