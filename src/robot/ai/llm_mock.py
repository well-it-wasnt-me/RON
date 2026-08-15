"""Deterministic LLM for tests and headless development."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from robot.interfaces.llm import LLMResponse, Message, Role, ToolCall, text_response, tool_response
from robot.logging import get_logger

_log = get_logger("ai.llm.mock")


class MockLLM:
    """Returns canned responses based on the last user message.

    Add new rules with :meth:`register`. If no rule matches, the default
    reply is returned.  Tool calls can be registered with
    :meth:`register_tool_call`.
    """

    def __init__(self, name: str = "mock-llm", default_reply: str = "Beep boop!") -> None:
        self._name = name
        self._default = default_reply
        self._rules: list[tuple[str, str]] = []
        self._tool_call_rules: list[tuple[str, list[ToolCall]]] = []
        self._history: list[list[Message]] = []

    @property
    def name(self) -> str:
        return self._name

    def register(self, substring: str, reply: str) -> None:
        """Register a text reply rule.  When the last user message
        contains ``substring`` (case-insensitive), return ``reply``.
        """
        self._rules.append((substring.lower(), reply))

    def register_tool_call(self, substring: str, tool_calls: list[ToolCall]) -> None:
        """Register a tool-call rule.  When the last user message
        contains ``substring`` (case-insensitive), the LLM responds
        with the given tool calls instead of plain text.
        """
        self._tool_call_rules.append((substring.lower(), tool_calls))

    @property
    def history(self) -> list[list[Message]]:
        return self._history

    async def complete(self, messages: Sequence[Message]) -> str:
        snapshot = list(messages)
        self._history.append(snapshot)
        response = await self.complete_with_tools(messages)
        return response.text

    async def complete_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        snapshot = list(messages)
        self._history.append(snapshot)
        last_user = next(
            (m.content for m in reversed(messages) if m.role is Role.USER),
            "",
        )
        lowered = last_user.lower()

        # Check tool-call rules first.
        for needle, calls in self._tool_call_rules:
            if needle in lowered:
                _log.debug("mock_llm.tool_call_rule", needle=needle, calls=len(calls))
                return tool_response(calls)

        # Then text rules.
        for needle, reply in self._rules:
            if needle in lowered:
                _log.debug("mock_llm.text_rule", needle=needle)
                return text_response(reply)

        return text_response(self._default)

    async def close(self) -> None:
        return None


def make_tool_call(
    name: str, arguments: dict[str, Any] | None = None, id: str | None = None
) -> ToolCall:
    """Convenience factory for creating :class:`ToolCall` instances in tests."""
    return ToolCall(
        id=id or f"call_{uuid.uuid4().hex[:8]}",
        name=name,
        arguments=arguments or {},
    )


__all__ = ["MockLLM", "make_tool_call"]
