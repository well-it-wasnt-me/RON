"""Tests for the Ollama LLM driver."""

from __future__ import annotations

import pytest

from robot.ai.llm_ollama import OllamaLLM
from robot.interfaces.llm import LLM, Message, Role


def test_ollama_llm_name() -> None:
    llm = OllamaLLM(model="llama3.2")
    assert llm.name == "ollama:llama3.2"


def test_ollama_llm_default_name() -> None:
    llm = OllamaLLM()
    assert llm.name == "ollama:llama3.2"


def test_ollama_llm_is_llm_protocol() -> None:
    llm = OllamaLLM()
    assert isinstance(llm, LLM)


def test_ollama_llm_custom_base_url() -> None:
    llm = OllamaLLM(model="mistral", base_url="http://192.168.1.100:11434")
    assert llm._base_url == "http://192.168.1.100:11434"


def test_ollama_llm_strips_trailing_slash() -> None:
    llm = OllamaLLM(base_url="http://localhost:11434/")
    assert llm._base_url == "http://localhost:11434"


@pytest.mark.anyio
async def test_ollama_llm_complete_builds_correct_payload() -> None:
    """Verify that OllamaLLM.complete sends the right payload to /api/chat."""
    captured: dict[str, object] = {}
    response_payload: dict[str, object] = {
        "message": {"role": "assistant", "content": "Hello from Ollama!"}
    }

    class MockResponse:
        def __init__(self, data: dict[str, object]) -> None:
            self._data = data

        def json(self) -> dict[str, object]:
            return self._data

        def raise_for_status(self) -> None:
            pass

    class MockAsyncClient:
        _base_url: str = ""

        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def post(
            self, url: str, json: dict[str, object], headers: dict[str, str] | None = None
        ) -> MockResponse:
            captured["url"] = url
            captured["payload"] = json
            captured["headers"] = headers
            return MockResponse(response_payload)

    import robot.ai.llm_ollama as ollama_module

    original_client = ollama_module.httpx.AsyncClient  # type: ignore[attr-defined]
    ollama_module.httpx.AsyncClient = MockAsyncClient  # type: ignore[attr-defined]
    try:
        llm = OllamaLLM(model="llama3.2", base_url="http://testhost:11434")
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful robot."),
            Message(role=Role.USER, content="hello"),
        ]
        result = await llm.complete(messages)
        assert result == "Hello from Ollama!"
        assert captured["url"] == "http://testhost:11434/api/chat"
        payload = captured["payload"]
        assert isinstance(payload, dict)
        assert payload["model"] == "llama3.2"
        assert payload["stream"] is False
        assert isinstance(payload["messages"], list)
        assert len(payload["messages"]) == 2
        msg0 = payload["messages"][0]
        msg1 = payload["messages"][1]
        assert isinstance(msg0, dict)
        assert isinstance(msg1, dict)
        assert msg0["role"] == "system"
        assert msg1["role"] == "user"
        options = payload.get("options")
        assert isinstance(options, dict)
        assert options["temperature"] == 0.7
    finally:
        ollama_module.httpx.AsyncClient = original_client  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_ollama_llm_complete_handles_empty_response() -> None:
    """Verify that empty Ollama responses return empty string."""
    response_payload: dict[str, object] = {"message": {"role": "assistant", "content": ""}}

    class MockResponse:
        def __init__(self, data: dict[str, object]) -> None:
            self._data = data

        def json(self) -> dict[str, object]:
            return self._data

        def raise_for_status(self) -> None:
            pass

    class MockAsyncClient:
        _base_url: str = ""

        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def post(
            self, url: str, json: dict[str, object], headers: dict[str, str] | None = None
        ) -> MockResponse:
            return MockResponse(response_payload)

    import robot.ai.llm_ollama as ollama_module

    original_client = ollama_module.httpx.AsyncClient  # type: ignore[attr-defined]
    ollama_module.httpx.AsyncClient = MockAsyncClient  # type: ignore[attr-defined]
    try:
        llm = OllamaLLM(model="llama3.2", base_url="http://testhost:11434")
        messages = [Message(role=Role.USER, content="hello")]
        result = await llm.complete(messages)
        assert result == ""
    finally:
        ollama_module.httpx.AsyncClient = original_client  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_ollama_llm_close() -> None:
    """close() should return None without error."""
    llm = OllamaLLM()
    await llm.close()
