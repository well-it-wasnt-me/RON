"""Tests for the text chat interface.

Verifies that typed input enters the same conversation pipeline used
by speech input, that LLM responses are printed, that TTS is invoked,
that TTS failure doesn't destroy text interaction, and that the
interface works without a microphone.
"""

from __future__ import annotations

from typing import Protocol

import pytest
from tests.fakes.llm import FakeLLM

from robot.ai.conversation import ConversationManager
from robot.ai.prompts import system_prompt
from robot.ai.vector_memory import VectorMemory, VectorMemoryEntry
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import BotReply, LLMTokenReceived
from robot.interfaces.audio import AudioBuffer, AudioOutput
from robot.services.conversation_service import ConversationService
from robot.speech.stt import MockSTT
from robot.speech.tts import MockTTS


class _TTSLike(Protocol):
    """Minimal TTS interface required by ConversationService."""

    async def speak(self, text: str) -> AudioBuffer: ...

    async def close(self) -> None: ...


class _RecordingTTS:
    """TTS that records spoken text and returns a non-empty buffer."""

    def __init__(self, *, fail: bool = False) -> None:
        self.spoken: list[str] = []
        self.fail = fail

    async def speak(self, text: str) -> AudioBuffer:
        self.spoken.append(text)
        if self.fail:
            raise RuntimeError("TTS engine crashed")
        return AudioBuffer(
            pcm=b"\x01\x02\x03\x04",
            sample_rate=24000,
            channels=1,
        )

    async def close(self) -> None:
        return None


class _RecordingAudio(AudioOutput):
    """Audio output that records buffers played."""

    def __init__(self, *, fail: bool = False) -> None:
        self.played: list[AudioBuffer] = []
        self.fail = fail

    @property
    def sample_rate(self) -> int:
        return 48000

    @property
    def channels(self) -> int:
        return 1

    async def play(self, buffer: AudioBuffer) -> None:
        self.played.append(buffer)
        if self.fail:
            raise RuntimeError("playback failed")

    async def stop(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _build_service(
    *,
    tts: _TTSLike | None = None,
    audio: AudioOutput | None = None,
) -> ConversationService:
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    llm = FakeLLM(name="fake")

    llm.register("hello", "Hi there!")
    llm.register("my name is", "Nice to meet you!")
    llm.register("what is my name", "I don't know your name yet.")

    conv = ConversationManager(
        llm=llm,
        system_prompt=system_prompt(),
    )

    service = ConversationService(
        bus=bus,
        state_machine=sm,
        stt=MockSTT(),
        tts=tts if tts is not None else MockTTS(),
        llm=llm,
        conversation=conv,
        audio=audio,
    )

    service.attach()
    return service


# ---------------------------------------------------------------------------
# Text input reaches conversation service
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_input_becomes_user_message() -> None:
    """Typed text becomes exactly one user message in the conversation."""
    service = _build_service()

    await service.state_machine.transition(RobotState.IDLE)
    await service.handle_user_text("hello", source="text")

    messages = service.conversation.current.messages
    user_msgs = [m for m in messages if m.role.value == "user"]

    assert len(user_msgs) == 1
    assert user_msgs[0].content == "hello"

    service.detach()


@pytest.mark.asyncio
async def test_empty_input_is_ignored() -> None:
    """Blank input is not sent to the LLM."""
    service = _build_service()

    await service.state_machine.transition(RobotState.IDLE)
    await service.handle_user_text("   ")

    assert service.state_machine.state is RobotState.IDLE
    assert len(service.conversation.current.messages) == 0

    service.detach()


# ---------------------------------------------------------------------------
# LLM response is captured via BotReply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_response_published_as_bot_reply() -> None:
    """The LLM response is published as a BotReply event."""
    service = _build_service()

    await service.state_machine.transition(RobotState.IDLE)

    replies: list[BotReply] = []
    service.bus.subscribe(BotReply, replies.append)

    await service.handle_user_text("hello", source="text")

    assert len(replies) == 1
    assert replies[0].text == "Hi there!"
    assert replies[0].user_text == "hello"

    service.detach()


# ---------------------------------------------------------------------------
# TTS is invoked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tts_is_invoked_with_response_text() -> None:
    """The response text is passed to tts.speak()."""
    tts = _RecordingTTS()
    service = _build_service(
        tts=tts,
        audio=_RecordingAudio(),
    )

    await service.state_machine.transition(RobotState.IDLE)
    await service.handle_user_text("hello", source="text")

    assert tts.spoken == ["Hi there!"]

    service.detach()


# ---------------------------------------------------------------------------
# TTS failure does not destroy text interaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tts_failure_preserves_text_response() -> None:
    """TTS failure doesn't lose the LLM response from conversation history."""
    tts = _RecordingTTS(fail=True)
    service = _build_service(
        tts=tts,
        audio=_RecordingAudio(),
    )

    await service.state_machine.transition(RobotState.IDLE)

    replies: list[BotReply] = []
    service.bus.subscribe(BotReply, replies.append)

    await service.handle_user_text("hello", source="text")

    assert len(replies) == 1
    assert replies[0].text == "Hi there!"

    assert service.state_machine.state is RobotState.IDLE

    messages = service.conversation.current.messages
    assert any(m.role.value == "assistant" and m.content == "Hi there!" for m in messages)

    service.detach()


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_transitions_through_thinking_speaking_idle() -> None:
    """Text input triggers the same state transitions as speech."""
    service = _build_service(
        tts=_RecordingTTS(),
        audio=_RecordingAudio(),
    )

    await service.state_machine.transition(RobotState.IDLE)

    states: list[str] = []

    from robot.events.events import StateChanged

    def _track(event: StateChanged) -> None:
        states.append(event.current.value)

    service.bus.subscribe(StateChanged, _track)

    await service.handle_user_text("hello", source="text")

    assert "listening" in states
    assert "thinking" in states
    assert "speaking" in states
    assert service.state_machine.state is RobotState.IDLE

    service.detach()


# ---------------------------------------------------------------------------
# No microphone required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_input_works_without_microphone() -> None:
    """handle_user_text works when microphone is None."""
    service = _build_service(
        tts=_RecordingTTS(),
        audio=_RecordingAudio(),
    )

    assert service.microphone is None

    await service.state_machine.transition(RobotState.IDLE)
    await service.handle_user_text("hello", source="text")

    assert service.state_machine.state is RobotState.IDLE

    service.detach()


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typed_messages_share_conversation_history() -> None:
    """Multiple typed messages use the same conversation state."""
    service = _build_service(
        tts=_RecordingTTS(),
        audio=_RecordingAudio(),
    )

    await service.state_machine.transition(RobotState.IDLE)

    await service.handle_user_text("my name is Bob", source="text")
    await service.handle_user_text("what is my name", source="text")

    messages = service.conversation.current.messages

    user_msgs = [m for m in messages if m.role.value == "user"]
    assistant_msgs = [m for m in messages if m.role.value == "assistant"]

    assert len(user_msgs) == 2
    assert len(assistant_msgs) == 2

    assert user_msgs[0].content == "my name is Bob"
    assert user_msgs[1].content == "what is my name"

    service.detach()


# ---------------------------------------------------------------------------
# Exit commands
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quit_commands_recognised() -> None:
    """The chat module recognises /quit and /exit."""
    from robot.cli.chat import _QUIT_COMMANDS

    assert "/quit" in _QUIT_COMMANDS
    assert "/exit" in _QUIT_COMMANDS


# ---------------------------------------------------------------------------
# Audio failure doesn't lose text response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_failure_preserves_text_response() -> None:
    """Audio playback failure doesn't lose the LLM response."""
    audio = _RecordingAudio(fail=True)

    service = _build_service(
        tts=_RecordingTTS(),
        audio=audio,
    )

    await service.state_machine.transition(RobotState.IDLE)

    replies: list[BotReply] = []
    service.bus.subscribe(BotReply, replies.append)

    await service.handle_user_text("hello", source="text")

    assert len(replies) == 1
    assert replies[0].text == "Hi there!"
    assert service.state_machine.state is RobotState.IDLE

    service.detach()


# ---------------------------------------------------------------------------
# Streaming tokens are published
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_tokens_published() -> None:
    """Streaming LLM tokens are published for real-time display."""
    tts = _RecordingTTS()

    service = _build_service(
        tts=tts,
        audio=_RecordingAudio(),
    )

    await service.state_machine.transition(RobotState.IDLE)

    tokens: list[str] = []
    service.bus.subscribe(
        LLMTokenReceived,
        lambda event: tokens.append(event.token),
    )

    await service.handle_user_text("hello", source="text")

    assert len(tokens) > 0

    assembled = "".join(tokens)
    assert "Hi there!" in assembled
    assert tts.spoken == ["Hi there!"]

    service.detach()


# ---------------------------------------------------------------------------
# Mock TTS detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_tts_does_not_produce_audio() -> None:
    """When TTS is MockTTS, no audio buffer reaches the output."""
    audio = _RecordingAudio()

    service = _build_service(
        tts=MockTTS(),
        audio=audio,
    )

    await service.state_machine.transition(RobotState.IDLE)
    await service.handle_user_text("hello", source="text")

    assert service.tts.spoken == ["Hi there!"]  # type: ignore[attr-defined]
    assert len(audio.played) == 0

    service.detach()


# ---------------------------------------------------------------------------
# Tool calls work through text input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_calls_work_through_text_input() -> None:
    """Typed input can trigger the tool-calling pipeline."""
    from robot.ai.tools.executor import ToolExecutor
    from robot.ai.tools.registry import ToolRegistry
    from robot.interfaces.llm import ToolCall

    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)

    llm = FakeLLM(name="fake")
    llm.register_tool_call(
        "what time",
        [
            ToolCall(
                id="1",
                name="get_time",
                arguments={},
            )
        ],
    )
    llm.register("the time is", "It's 3pm.")

    registry = ToolRegistry()
    executor = ToolExecutor(
        registry=registry,
        bus=bus,
        servo_controller=None,
    )

    conv = ConversationManager(
        llm=llm,
        system_prompt=system_prompt(),
    )

    service = ConversationService(
        bus=bus,
        state_machine=sm,
        stt=MockSTT(),
        tts=_RecordingTTS(),
        llm=llm,
        conversation=conv,
        audio=_RecordingAudio(),
        tool_registry=registry,
        tool_executor=executor,
    )

    service.attach()

    await service.state_machine.transition(RobotState.IDLE)

    replies: list[BotReply] = []
    service.bus.subscribe(BotReply, replies.append)

    await service.handle_user_text("what time is it", source="text")

    assert len(llm.calls) >= 2
    assert service.state_machine.state is RobotState.IDLE

    service.detach()


# ---------------------------------------------------------------------------
# Memory failure doesn't crash conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_failure_does_not_crash_conversation() -> None:
    """When memory.add() raises, the conversation still completes."""
    service = _build_service(
        tts=_RecordingTTS(),
        audio=_RecordingAudio(),
    )

    await service.state_machine.transition(RobotState.IDLE)

    class _CrashingMemory:
        def add(
            self,
            _content: str,
            _importance: float = 0.5,
        ) -> None:
            raise RuntimeError("embedding model crashed")

        def search(
            self,
            _query: str,
            _limit: int = 5,
        ) -> list[object]:
            return []

        def recall(
            self,
            _query: str,
            _limit: int = 5,
        ) -> list[object]:
            return []

    object.__setattr__(
        service,
        "memory",
        _CrashingMemory(),
    )

    replies: list[BotReply] = []
    service.bus.subscribe(BotReply, replies.append)

    await service.handle_user_text("hello", source="text")

    assert len(replies) == 1
    assert replies[0].text == "Hi there!"
    assert service.state_machine.state is RobotState.IDLE

    service.detach()


# ---------------------------------------------------------------------------
# VectorMemory _dim fallback
# ---------------------------------------------------------------------------


def test_vector_memory_dim_none_fallback() -> None:
    """VectorMemory.add() handles _dim being None gracefully."""

    class _BrokenEmbedding:
        _dim: int | None = None

        def embed(self, _text: str) -> list[float]:
            raise RuntimeError("CUDA error")

    memory = VectorMemory(
        embedding_fn=_BrokenEmbedding(),
    )

    memory.add(
        "test content",
        importance=0.5,
    )

    assert len(memory.entries) == 1
    assert len(memory.entries[0].embedding) == 128


# ---------------------------------------------------------------------------
# Memory search failure doesn't crash conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_search_failure_does_not_crash_conversation() -> None:
    """When VectorMemory.search_similar() raises, the conversation still works."""
    service = _build_service(
        tts=_RecordingTTS(),
        audio=_RecordingAudio(),
    )

    await service.state_machine.transition(RobotState.IDLE)

    class _CrashingVectorMemory(VectorMemory):
        def search_similar(
            self,
            query: str,
            limit: int = 5,
            min_similarity: float = 0.0,
        ) -> list[tuple[VectorMemoryEntry, float]]:
            raise RuntimeError("CUDA error during search")

    object.__setattr__(
        service,
        "memory",
        _CrashingVectorMemory(),
    )

    replies: list[BotReply] = []
    service.bus.subscribe(BotReply, replies.append)

    await service.handle_user_text("hello", source="text")

    assert len(replies) == 1
    assert replies[0].text == "Hi there!"
    assert service.state_machine.state is RobotState.IDLE

    service.detach()


# ---------------------------------------------------------------------------
# VectorMemory search_similar _dim fallback
# ---------------------------------------------------------------------------


def test_vector_memory_search_similar_dim_none_fallback() -> None:
    """search_similar handles _dim being None gracefully."""

    class _BrokenEmbedding:
        _dim: int | None = None

        def embed(self, _text: str) -> list[float]:
            raise RuntimeError("CUDA error")

    memory = VectorMemory(
        embedding_fn=_BrokenEmbedding(),
    )

    memory.add("test entry")

    results = memory.search_similar("test query")

    assert len(results) == 1
    assert results[0][0].content == "test entry"
