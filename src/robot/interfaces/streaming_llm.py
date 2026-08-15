"""Streaming LLM interface for token-by-token response generation.

The :class:`StreamingLLM` protocol extends :class:`LLM` with a
:meth:`stream_complete` method that yields response tokens one at a
time. This enables real-time face animation (thinking -> speaking)
as the LLM generates its reply.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
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
class TokenChunk:
    """A single token from a streaming LLM response.

    Attributes
    ----------
    token:
        The text of this token.
    done:
        Whether this is the final token in the response.
    tool_calls:
        Tool calls accumulated during streaming. Only present on the
        final chunk (``done=True``) when the LLM makes function calls.
    """

    token: str
    done: bool = False
    tool_calls: tuple[Any, ...] = ()


@runtime_checkable
class StreamingLLM(Protocol):
    """Async streaming language model interface.

    Like :class:`LLM`, but with an additional :meth:`stream_complete`
    method that yields tokens one at a time for real-time animation.
    """

    @property
    def name(self) -> str:
        """Human-readable model name."""

    async def complete(self, messages: Sequence[Message]) -> str:
        """Return the assistant's full reply for the given conversation."""

    async def stream_complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[TokenChunk]:
        """Yield :class:`TokenChunk` objects as the LLM generates them.

        When ``tools`` is provided and the LLM supports function calling,
        the final chunk (``done=True``) may contain ``tool_calls``.

        The final chunk will have ``done=True`` and may contain an empty
        token string.
        """

    async def close(self) -> None: ...


__all__ = ["Message", "Role", "StreamingLLM", "TokenChunk"]
