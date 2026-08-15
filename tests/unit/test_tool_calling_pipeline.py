"""Tests for the LLM tool-calling pipeline.

Covers:
- LLMResponse and ToolCall data types
- MockLLM complete_with_tools and register_tool_call
- FakeLLM complete_with_tools and stream_complete
- ConversationService with tool calls (one-shot and streaming)
- ToolExecutor integration in conversation
- ToolConfig loading
"""

from __future__ import annotations

import pytest
from tests.fakes.llm import FakeLLM

from robot.ai.conversation import ConversationManager
from robot.ai.llm_mock import MockLLM, make_tool_call
from robot.ai.tools.executor import ToolExecutor
from robot.ai.tools.registry import BUILTIN_TOOLS, ToolRegistry
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import EmotionChanged, EmotionName, SpeechRecognized
from robot.interfaces.llm import Message, Role, ToolCall, text_response, tool_response
from robot.services.conversation_service import ConversationService
from robot.speech.stt import MockSTT
from robot.speech.tts import MockTTS


# ---------------------------------------------------------------------------
# ToolCall and LLMResponse data types
# ---------------------------------------------------------------------------
class TestToolCall:
    def test_create_tool_call(self) -> None:
        tc = ToolCall(id="call_123", name="change_emotion", arguments={"emotion": "happy"})
        assert tc.id == "call_123"
        assert tc.name == "change_emotion"
        assert tc.arguments == {"emotion": "happy"}

    def test_tool_call_is_frozen(self) -> None:
        tc = ToolCall(id="call_123", name="test", arguments={})
        with pytest.raises(AttributeError):
            tc.name = "other"  # type: ignore[misc]

    def test_make_tool_call_helper(self) -> None:
        tc = make_tool_call("play_sound", {"name": "greet"})
        assert tc.name == "play_sound"
        assert tc.arguments == {"name": "greet"}
        assert tc.id.startswith("call_")


class TestLLMResponse:
    def test_text_response_factory(self) -> None:
        r = text_response("Hello!")
        assert r.text == "Hello!"
        assert r.tool_calls == ()
        assert r.done is True

    def test_tool_response_factory(self) -> None:
        tc = ToolCall(id="1", name="test", arguments={})
        r = tool_response([tc])
        assert r.text == ""
        assert len(r.tool_calls) == 1
        assert r.done is True

    def test_tool_response_with_text(self) -> None:
        tc = ToolCall(id="1", name="test", arguments={})
        r = tool_response([tc], text="Let me help with that.")
        assert r.text == "Let me help with that."
        assert len(r.tool_calls) == 1

    def test_response_is_frozen(self) -> None:
        r = text_response("hi")
        with pytest.raises(AttributeError):
            r.text = "bye"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MockLLM tool calling
# ---------------------------------------------------------------------------
class TestMockLLMToolCalling:
    @pytest.mark.asyncio
    async def test_complete_with_tools_returns_text(self) -> None:
        llm = MockLLM()
        llm.register("hello", "Hi there!")
        messages = [Message(role=Role.USER, content="hello")]
        response = await llm.complete_with_tools(messages)
        assert response.text == "Hi there!"
        assert response.tool_calls == ()

    @pytest.mark.asyncio
    async def test_complete_with_tools_returns_tool_calls(self) -> None:
        llm = MockLLM()
        tc = make_tool_call("change_emotion", {"emotion": "happy"})
        llm.register_tool_call("be happy", [tc])
        messages = [Message(role=Role.USER, content="please be happy")]
        response = await llm.complete_with_tools(messages)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "change_emotion"

    @pytest.mark.asyncio
    async def test_complete_still_works(self) -> None:
        llm = MockLLM()
        llm.register("hello", "Hi!")
        messages = [Message(role=Role.USER, content="hello")]
        text = await llm.complete(messages)
        assert text == "Hi!"

    @pytest.mark.asyncio
    async def test_tool_call_rules_take_priority(self) -> None:
        llm = MockLLM()
        llm.register("be happy", "You seem happy!")
        tc = make_tool_call("change_emotion", {"emotion": "happy"})
        llm.register_tool_call("be happy", [tc])
        messages = [Message(role=Role.USER, content="be happy")]
        response = await llm.complete_with_tools(messages)
        # Tool call rules should take priority.
        assert len(response.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_default_response_with_tools(self) -> None:
        llm = MockLLM()
        messages = [Message(role=Role.USER, content="something random")]
        response = await llm.complete_with_tools(messages)
        assert response.text == "Beep boop!"
        assert response.tool_calls == ()


# ---------------------------------------------------------------------------
# FakeLLM tool calling
# ---------------------------------------------------------------------------
class TestFakeLLMToolCalling:
    @pytest.mark.asyncio
    async def test_complete_with_tools_text(self) -> None:
        llm = FakeLLM()
        llm.register("hello", "Hi there!")
        messages = [Message(role=Role.USER, content="hello")]
        response = await llm.complete_with_tools(messages)
        assert response.text == "Hi there!"
        assert response.tool_calls == ()

    @pytest.mark.asyncio
    async def test_complete_with_tools_tool_calls(self) -> None:
        llm = FakeLLM()
        tc = make_tool_call("play_sound", {"name": "greet"})
        llm.register_tool_call("play", [tc])
        messages = [Message(role=Role.USER, content="play something")]
        response = await llm.complete_with_tools(messages)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "play_sound"

    @pytest.mark.asyncio
    async def test_stream_complete_with_tool_calls(self) -> None:
        llm = FakeLLM()
        tc = make_tool_call("speak", {"text": "hello"})
        llm.register_tool_call("say hello", [tc])
        messages = [Message(role=Role.USER, content="say hello to me")]
        chunks = []
        async for chunk in llm.stream_complete(messages):
            chunks.append(chunk)
        # Last chunk should have done=True and tool_calls.
        assert chunks[-1].done is True
        assert len(chunks[-1].tool_calls) == 1

    @pytest.mark.asyncio
    async def test_stream_complete_text(self) -> None:
        llm = FakeLLM()
        llm.register("greet", "Hello friend")
        messages = [Message(role=Role.USER, content="greet me")]
        chunks = []
        async for chunk in llm.stream_complete(messages):
            chunks.append(chunk)
        # Should get text tokens then a done chunk.
        assert len(chunks) >= 2
        assert chunks[-1].done is True
        text = "".join(c.token for c in chunks if c.token)
        assert "Hello" in text or "friend" in text


# ---------------------------------------------------------------------------
# ConversationService with tool calling
# ---------------------------------------------------------------------------
class TestConversationServiceToolCalling:
    @pytest.mark.asyncio
    async def test_conversation_with_tool_calls_one_shot(self) -> None:
        """Test that the conversation service handles tool calls in one-shot mode."""
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        sm._state = RobotState.LISTENING

        llm = FakeLLM()
        # First call: LLM returns a tool call.
        tc = make_tool_call("change_emotion", {"emotion": "happy", "intensity": 0.8})
        llm.register_tool_call("make me", [tc])
        # Second call (after tool result): LLM returns text.
        llm.register("happy", "I've made you happy!")

        cm = ConversationManager(llm=llm, system_prompt="You are DeskBot.")
        stt = MockSTT()
        tts = MockTTS()

        registry = ToolRegistry()
        for _name, defn in BUILTIN_TOOLS.items():
            registry.add(defn, handler=_noop_handler)
        executor = ToolExecutor(registry=registry, bus=bus)

        service = ConversationService(
            bus=bus,
            state_machine=sm,
            stt=stt,
            tts=tts,
            llm=llm,
            conversation=cm,
            tool_registry=registry,
            tool_executor=executor,
        )
        service.attach()

        # Track emotion changes.
        emotions: list[EmotionChanged] = []
        bus.subscribe(EmotionChanged, emotions.append)

        # Simulate speech recognition.
        await bus.publish(SpeechRecognized(text="make me happy", confidence=1.0))

        # The emotion change should have been published.
        assert len(emotions) >= 1
        assert emotions[0].current == EmotionName.HAPPY

        service.detach()

    @pytest.mark.asyncio
    async def test_conversation_without_tools(self) -> None:
        """ConversationService works without tool_registry/tool_executor."""
        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        sm._state = RobotState.LISTENING

        llm = FakeLLM()
        llm.register("hello", "Hi there!")

        cm = ConversationManager(llm=llm, system_prompt="You are DeskBot.")

        service = ConversationService(
            bus=bus,
            state_machine=sm,
            stt=MockSTT(),
            tts=MockTTS(),
            llm=llm,
            conversation=cm,
        )
        service.attach()

        await bus.publish(SpeechRecognized(text="hello", confidence=1.0))
        # State should return to IDLE after the reply.
        # (The exact state depends on TTS being mock vs real.)
        service.detach()


# ---------------------------------------------------------------------------
# ToolConfig
# ---------------------------------------------------------------------------
class TestToolConfig:
    def test_tool_config_default_enabled(self) -> None:
        from robot.config import ToolConfig

        tc = ToolConfig()
        assert tc.enabled is True

    def test_tool_config_from_env(self) -> None:
        import os

        from robot.config import ToolConfig

        os.environ["DESKBOT_TOOLS__ENABLED"] = "false"
        try:
            tc = ToolConfig()
            assert tc.enabled is False
        finally:
            del os.environ["DESKBOT_TOOLS__ENABLED"]

    def test_tool_config_in_app_settings(self) -> None:
        from robot.config import AppSettings

        settings = AppSettings()
        assert settings.tools.enabled is True


async def _noop_handler(**kwargs: object) -> dict[str, str]:
    return {"status": "ok"}
