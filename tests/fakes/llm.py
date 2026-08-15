"""Fake LLM for tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from robot.interfaces.llm import LLMResponse, Message, Role, ToolCall, text_response, tool_response
from robot.interfaces.streaming_llm import TokenChunk


class FakeLLM:
    """Deterministic LLM for unit tests.

    Supports both plain text and tool-call responses.
    Register text rules with :meth:`register` and tool-call
    rules with :meth:`register_tool_call`.
    """

    def __init__(self, name: str = "fake") -> None:
        self._name = name
        self.calls: list[list[Message]] = []
        self._rules: list[tuple[str, str]] = []
        self._tool_rules: list[tuple[str, list[ToolCall]]] = []
        self._default = "ok"

    def register(self, substring: str, reply: str) -> None:
        """Register a text reply rule."""
        self._rules.append((substring.lower(), reply))

    def register_tool_call(self, substring: str, tool_calls: list[ToolCall]) -> None:
        """Register a tool-call reply rule."""
        self._tool_rules.append((substring.lower(), tool_calls))

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, messages: Sequence[Message]) -> str:
        snapshot = list(messages)
        self.calls.append(snapshot)
        response = await self.complete_with_tools(messages)
        return response.text

    async def complete_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        snapshot = list(messages)
        self.calls.append(snapshot)
        last_user = next(
            (m.content for m in reversed(messages) if m.role is Role.USER),
            "",
        )
        lowered = last_user.lower()

        # Check tool-call rules first.
        for needle, calls in self._tool_rules:
            if needle in lowered:
                return tool_response(calls)

        # Then text rules.
        for needle, reply in self._rules:
            if needle in lowered:
                return text_response(reply)

        return text_response(self._default)

    async def stream_complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[TokenChunk]:
        """Stream the response token by token.

        Yields each word as a separate token, then a final done chunk.
        If tool calls are present, yields them on the final chunk.
        """
        response = await self.complete_with_tools(messages, tools)
        if response.tool_calls:
            # Yield tool calls on the done chunk.
            yield TokenChunk(token="", done=True, tool_calls=response.tool_calls)
        else:
            # Yield text token by token.
            words = response.text.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else f" {word}"
                yield TokenChunk(token=token, done=False)
            yield TokenChunk(token="", done=True)

    async def close(self) -> None:
        return None
