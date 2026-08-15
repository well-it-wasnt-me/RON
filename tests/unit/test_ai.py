"""Tests for the AI layer."""

from __future__ import annotations

from tests.fakes.llm import FakeLLM

from robot.ai.conversation import MAX_HISTORY, ConversationManager
from robot.ai.memory import Memory
from robot.ai.prompts import system_prompt


def test_system_prompt_default() -> None:
    prompt = system_prompt()
    assert "DeskBot" in prompt


def test_system_prompt_with_personality() -> None:
    prompt = system_prompt(name="Pip", personality_summary="mischievous")
    assert "Pip" in prompt
    assert "mischievous" in prompt


async def test_conversation_reply() -> None:
    llm = FakeLLM()
    llm.register("hello", "hi there")
    cm = ConversationManager(llm=llm, system_prompt="be nice")
    reply = await cm.reply("hello!")
    assert reply == "hi there"
    assert len(cm.current.messages) == 2


async def test_conversation_truncates_history() -> None:
    llm = FakeLLM()
    cm = ConversationManager(llm=llm, system_prompt="be nice")
    for i in range(MAX_HISTORY + 5):
        await cm.reply(f"msg {i}")
    assert len(cm.current.messages) == MAX_HISTORY


def test_memory_add_and_recall() -> None:
    m = Memory()
    m.add("user said hi", importance=0.4)
    m.add("user said bye", importance=0.6)
    recent = m.recall(limit=2)
    assert len(recent) == 2
    assert recent[1].content == "user said bye"


def test_memory_search() -> None:
    m = Memory()
    m.add("likes apples")
    m.add("likes oranges")
    results = m.search("apples")
    assert len(results) == 1


def test_memory_caps_entries() -> None:
    m = Memory(capacity=3)
    for i in range(5):
        m.add(f"e{i}")
    assert len(m.entries) == 3
