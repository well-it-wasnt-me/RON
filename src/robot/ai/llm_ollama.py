"""Ollama LLM driver.

Talks to a local `Ollama <https://ollama.ai>`_ server via its native
``/api/chat`` endpoint.  Supports both single-shot and streaming
responses, plus function/tool calling.

Install Ollama on the Pi::

    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.2

Configure with::

    DESKBOT_LLM__PROVIDER=ollama
    DESKBOT_LLM__MODEL=llama3.2
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from robot.interfaces.llm import LLMResponse, Message, ToolCall, text_response, tool_response
from robot.interfaces.streaming_llm import TokenChunk
from robot.logging import get_logger

_log = get_logger("ai.llm.ollama")


def _messages_to_dicts(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert internal Message objects to Ollama wire format."""
    result: list[dict[str, Any]] = []
    for m in messages:
        result.append({"role": m.role.value, "content": m.content})
    return result


def _tools_to_ollama(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool schemas to Ollama's tool format.

    Ollama expects tools in the same format as OpenAI's function calling
    API (``type: "function"``, ``function: { name, description, parameters }``).
    This is a pass-through since we already store tools in that format.
    """
    return list(tools)


class OllamaLLM:
    """Ollama local LLM driver.

    Uses the ``/api/chat`` endpoint which returns a single response
    (or a streaming NDJSON response when ``stream=True``).

    Parameters
    ----------
    model:
        Ollama model name (e.g. ``"llama3.2"``, ``"mistral"``).
    base_url:
        Ollama server URL. Default: ``"http://localhost:11434"``.
    timeout_s:
        HTTP request timeout in seconds. Ollama can be slow on a Pi;
        120 s is a reasonable default for a local model.
    temperature:
        Sampling temperature (0 = deterministic, 1 = creative).
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
        temperature: float = 0.7,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._temperature = temperature

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    async def complete(self, messages: Sequence[Message]) -> str:
        response = await self.complete_with_tools(messages, tools=None)
        return response.text

    async def complete_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send messages to Ollama and return the full response.

        When ``tools`` is provided, the request includes tool definitions
        and the response may contain ``tool_calls``.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _messages_to_dicts(messages),
            "stream": False,
            "options": {
                "temperature": self._temperature,
            },
        }
        if tools:
            payload["tools"] = _tools_to_ollama(tools)

        url = f"{self._base_url}/api/chat"
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            _log.debug(
                "ollama.request", model=self._model, messages=len(messages), tools=bool(tools)
            )
            response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        content = str(data.get("message", {}).get("content", ""))

        # Check for tool calls in the response.
        raw_tool_calls = data.get("message", {}).get("tool_calls")
        if raw_tool_calls:
            calls: list[ToolCall] = []
            for tc in raw_tool_calls:
                fn = tc.get("function", {})
                try:
                    args = (
                        json.loads(fn.get("arguments", "{}"))
                        if isinstance(fn.get("arguments"), str)
                        else fn.get("arguments", {})
                    )
                except (json.JSONDecodeError, TypeError):
                    args = {}
                calls.append(
                    ToolCall(
                        id=tc.get("id", fn.get("name", "")),
                        name=fn.get("name", ""),
                        arguments=args,
                    )
                )
            _log.info("ollama.tool_calls", count=len(calls), names=[c.name for c in calls])
            return tool_response(calls, text=content)

        if not content:
            _log.warning("ollama.empty_response", response=data)
        _log.debug("ollama.response", model=self._model, content_length=len(content))
        return text_response(content)

    async def stream_complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[TokenChunk]:
        """Stream tokens from Ollama using NDJSON ``stream=True``.

        Yields :class:`TokenChunk` objects as the model generates each
        token. The final chunk has ``done=True``.

        When ``tools`` is provided and the model makes tool calls, the
        final chunk will contain ``tool_calls``.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _messages_to_dicts(messages),
            "stream": True,
            "options": {
                "temperature": self._temperature,
            },
        }
        if tools:
            payload["tools"] = _tools_to_ollama(tools)

        url = f"{self._base_url}/api/chat"
        _log.debug(
            "ollama.stream_request", model=self._model, messages=len(messages), tools=bool(tools)
        )

        accumulated_tool_calls: list[ToolCall] = []
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:  # noqa: SIM117
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk_data = json.loads(line)
                    except Exception:
                        continue
                    if chunk_data.get("done", False):
                        # Final chunk - yield with tool calls if any.
                        yield TokenChunk(
                            token="", done=True, tool_calls=tuple(accumulated_tool_calls)
                        )
                        break

                    # Check for tool calls in streaming response.
                    msg = chunk_data.get("message", {})
                    raw_tool_calls = msg.get("tool_calls")
                    if raw_tool_calls:
                        for tc in raw_tool_calls:
                            fn = tc.get("function", {})
                            name = fn.get("name", "")
                            args_str = fn.get("arguments", "{}")
                            try:
                                args = (
                                    json.loads(args_str) if isinstance(args_str, str) else args_str
                                )
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                            accumulated_tool_calls.append(
                                ToolCall(
                                    id=tc.get("id", name),
                                    name=name,
                                    arguments=args,
                                )
                            )
                        continue

                    content = msg.get("content", "")
                    if content:
                        yield TokenChunk(token=content, done=False)

    async def close(self) -> None:
        return None


__all__ = ["OllamaLLM"]
