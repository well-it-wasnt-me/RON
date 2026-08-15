"""Tests for memory-aware conversation processing."""

from __future__ import annotations

import pytest

from robot.ai.conversation import ConversationManager
from robot.ai.llm_mock import MockLLM
from robot.ai.memory import Memory
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import SpeechRecognized
from robot.services.conversation_service import ConversationService
from robot.speech.stt import MockSTT
from robot.speech.tts import MockTTS


@pytest.mark.anyio
async def test_memory_is_saved_and_injected_into_next_prompt() -> None:
    bus = InMemoryEventBus()
    state_machine = StateMachine(bus=bus)
    await state_machine.transition(RobotState.IDLE)
    llm = MockLLM()
    llm.register("my name is Ada", "Nice to meet you, Ada.")
    llm.register("what is my name?", "Your name is Ada.")
    memory = Memory()
    conversation = ConversationManager(llm=llm, system_prompt="Be helpful.")
    service = ConversationService(
        bus=bus,
        state_machine=state_machine,
        stt=MockSTT(),
        tts=MockTTS(),
        llm=llm,
        conversation=conversation,
        memory=memory,
    )
    service.attach()

    await state_machine.transition(RobotState.LISTENING)
    await bus.publish(SpeechRecognized(text="my name is Ada"))
    assert any("Ada" in entry.content for entry in memory.entries)

    await state_machine.transition(RobotState.LISTENING)
    await bus.publish(SpeechRecognized(text="what is my name?"))
    assert "Relevant remembered context:" in llm.history[-1][0].content


def test_memory_can_be_disabled() -> None:
    service = ConversationService(
        bus=InMemoryEventBus(),
        state_machine=StateMachine(bus=InMemoryEventBus()),
        stt=MockSTT(),
        tts=MockTTS(),
        llm=MockLLM(),
        conversation=ConversationManager(llm=MockLLM(), system_prompt="Be helpful."),
    )
    assert service._memory_context("anything") == ""
