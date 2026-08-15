"""OpenAI-compatible LLM driver.

Talks to OpenAI's public API or any local server speaking the
``/v1/chat/completions`` protocol (LMStudio, llama.cpp, vLLM,
text-generation-webui, etc.) via the ``base_url`` setting.

Supports both plain text completion and function/tool calling.

Install with::

    uv pip install openai
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from robot.interfaces.llm import LLMResponse, Message, ToolCall, text_response, tool_response
from robot.logging import get_logger

_log = get_logger("ai.llm.openai")


class OpenAILLM:
    """OpenAI / OpenAI-compatible LLM.

    Parameters
    ----------
    api_key:
        Bearer token. Required for OpenAI's hosted API; can be empty for
        local servers.
    base_url:
        API base URL. Default: OpenAI's hosted API. For LMStudio use
        ``"http://localhost:1234/v1"``, for Ollama's OpenAI bridge
        use ``"http://localhost:11434/v1"``, etc.
    model:
        Model name (e.g. ``"gpt-4o-mini"``, ``"llama3"``).
    timeout_s:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_s: float = 30.0,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def _messages_to_dicts(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Convert internal Message objects to OpenAI wire format."""
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def complete(self, messages: Sequence[Message]) -> str:
        response = await self.complete_with_tools(messages, tools=None)
        return response.text

    async def complete_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send messages to the OpenAI-compatible API, optionally with tool calling.

        When ``tools`` is provided, the request includes function definitions
        and the response may contain ``tool_calls`` in the assistant message.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": self._messages_to_dicts(messages),
        }
        if tools:
            payload["tools"] = list(tools)

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = f"{self._base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            _log.debug(
                "openai.request", model=self._model, messages=len(messages), tools=bool(tools)
            )
            response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            _log.warning("openai.empty_choices", response=data)
            return text_response("")

        choice = choices[0]
        message = choice.get("message", {})
        text = str(message.get("content", "") or "")
        raw_tool_calls = message.get("tool_calls")

        if raw_tool_calls:
            calls: list[ToolCall] = []
            for tc in raw_tool_calls:
                fn = tc.get("function", {})
                import json

                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        arguments=args,
                    )
                )
            _log.info("openai.tool_calls", count=len(calls), names=[c.name for c in calls])
            return tool_response(calls, text=text)

        return text_response(text)

    async def close(self) -> None:
        return None


__all__ = ["OpenAILLM"]
