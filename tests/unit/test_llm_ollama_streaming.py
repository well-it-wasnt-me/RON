"""Tests for OllamaLLM streaming support."""

from __future__ import annotations

import pytest

from robot.ai.llm_ollama import OllamaLLM
from robot.interfaces.llm import Message, Role
from robot.interfaces.streaming_llm import TokenChunk


def test_ollama_stream_complete_is_method() -> None:
    """OllamaLLM should have a stream_complete method."""
    llm = OllamaLLM(model="llama3.2")
    assert hasattr(llm, "stream_complete")
    assert callable(llm.stream_complete)


@pytest.mark.anyio
async def test_ollama_stream_complete_yields_tokens() -> None:
    """Verify that stream_complete yields TokenChunk objects."""
    collected_tokens: list[str] = []
    response_lines: list[str] = [
        '{"message": {"role": "assistant", "content": "Hello"}, "done": false}',
        '{"message": {"role": "assistant", "content": " there"}, "done": false}',
        '{"message": {"role": "assistant", "content": ""}, "done": true}',
    ]

    class MockStreamResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        async def aiter_lines(self):
            for line in response_lines:
                yield line

    class MockStreamContext:
        async def __aenter__(self) -> MockStreamResponse:
            return MockStreamResponse()

        async def __aexit__(self, *args: object) -> None:
            pass

    class MockAsyncClient:
        _base_url: str = ""

        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        def stream(
            self, method: str, url: str, json: dict[str, object], **kwargs: object
        ) -> MockStreamContext:
            return MockStreamContext()

    import robot.ai.llm_ollama as ollama_module

    original_client_class = ollama_module.httpx.AsyncClient  # type: ignore[attr-defined]
    ollama_module.httpx.AsyncClient = MockAsyncClient  # type: ignore[attr-defined]
    try:
        llm = OllamaLLM(model="llama3.2", base_url="http://testhost:11434")
        messages = [Message(role=Role.USER, content="hello")]
        async for chunk in llm.stream_complete(messages):
            assert isinstance(chunk, TokenChunk)
            if not chunk.done:
                collected_tokens.append(chunk.token)

        assert collected_tokens == ["Hello", " there"]
    finally:
        ollama_module.httpx.AsyncClient = original_client_class  # type: ignore[attr-defined]
