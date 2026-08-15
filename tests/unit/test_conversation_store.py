"""Tests for conversation persistence stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot.ai.conversation import Conversation, ConversationManager
from robot.ai.conversation_store import InMemoryStore


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------
class TestInMemoryStore:
    @pytest.mark.anyio
    async def test_save_and_load(self) -> None:
        store = InMemoryStore()
        await store.save("conv1", "be helpful", [("user", "hello"), ("assistant", "hi")])
        result = await store.load("conv1")
        assert result is not None
        assert len(result) == 2
        assert result[0] == ("user", "hello")
        assert result[1] == ("assistant", "hi")

    @pytest.mark.anyio
    async def test_load_nonexistent_returns_none(self) -> None:
        store = InMemoryStore()
        result = await store.load("nonexistent")
        assert result is None

    @pytest.mark.anyio
    async def test_save_overwrites(self) -> None:
        store = InMemoryStore()
        await store.save("conv1", "v1", [("user", "hello")])
        await store.save("conv1", "v2", [("user", "bye")])
        result = await store.load("conv1")
        assert result is not None
        assert len(result) == 1
        assert result[0] == ("user", "bye")

    @pytest.mark.anyio
    async def test_list_conversations(self) -> None:
        store = InMemoryStore()
        await store.save("conv1", "prompt1", [("user", "a")])
        await store.save("conv2", "prompt2", [("user", "b")])
        convs = await store.list_conversations()
        assert len(convs) == 2
        ids = {c.id for c in convs}
        assert ids == {"conv1", "conv2"}

    @pytest.mark.anyio
    async def test_delete(self) -> None:
        store = InMemoryStore()
        await store.save("conv1", "p", [("user", "hello")])
        assert await store.delete("conv1") is True
        assert await store.load("conv1") is None
        assert await store.delete("conv1") is False

    @pytest.mark.anyio
    async def test_delete_nonexistent(self) -> None:
        store = InMemoryStore()
        assert await store.delete("nonexistent") is False

    @pytest.mark.anyio
    async def test_meta_fields(self) -> None:
        store = InMemoryStore()
        await store.save("conv1", "be nice", [("user", "hi"), ("assistant", "hello")])
        convs = await store.list_conversations()
        assert len(convs) == 1
        assert convs[0].id == "conv1"
        assert convs[0].system_prompt == "be nice"
        assert convs[0].message_count == 2


# ---------------------------------------------------------------------------
# Conversation as_tuples / from_tuples
# ---------------------------------------------------------------------------
class TestConversationPersistence:
    def test_as_tuples(self) -> None:
        conv = Conversation(system_prompt="be helpful")
        conv.add_user("hello")
        conv.add_assistant("hi there")
        tuples = conv.as_tuples()
        assert tuples == [("user", "hello"), ("assistant", "hi there")]

    def test_from_tuples(self) -> None:
        conv = Conversation.from_tuples(
            system_prompt="be helpful",
            tuples=[("user", "hello"), ("assistant", "hi there")],
        )
        assert conv.system_prompt == "be helpful"
        assert len(conv.messages) == 2
        assert conv.messages[0].content == "hello"
        assert conv.messages[1].content == "hi there"

    def test_round_trip(self) -> None:
        conv = Conversation(system_prompt="be helpful")
        conv.add_user("hello")
        conv.add_assistant("hi there")
        conv.add_user("how are you")
        conv.add_assistant("fine thanks")
        tuples = conv.as_tuples()
        restored = Conversation.from_tuples(system_prompt="be helpful", tuples=tuples)
        assert len(restored.messages) == len(conv.messages)
        for orig, rest in zip(conv.messages, restored.messages, strict=True):
            assert orig.role == rest.role
            assert orig.content == rest.content


# ---------------------------------------------------------------------------
# ConversationManager with store
# ---------------------------------------------------------------------------
class TestConversationManagerWithStore:
    @pytest.mark.anyio
    async def test_manager_saves_after_reply(self) -> None:
        from tests.fakes.llm import FakeLLM

        llm = FakeLLM()
        llm.register("hello", "hi there")
        store = InMemoryStore()
        manager = ConversationManager(llm=llm, system_prompt="be nice", store=store)
        reply = await manager.reply("hello")
        assert reply == "hi there"
        # Check that the conversation was saved.
        result = await store.load("default")
        assert result is not None
        assert len(result) == 2  # user + assistant

    @pytest.mark.anyio
    async def test_manager_load_restores_conversation(self) -> None:
        from tests.fakes.llm import FakeLLM

        llm = FakeLLM()
        llm.register("hello", "hi there")
        store = InMemoryStore()
        manager1 = ConversationManager(llm=llm, system_prompt="be nice", store=store)
        await manager1.reply("hello")
        # Create a new manager with the same store and load.
        llm2 = FakeLLM()
        llm2.register("how are you", "fine")
        manager2 = ConversationManager(llm=llm2, system_prompt="be nice", store=store)
        await manager2.load()
        assert len(manager2.current.messages) == 2
        assert manager2.current.messages[0].content == "hello"

    @pytest.mark.anyio
    async def test_manager_without_store(self) -> None:
        from tests.fakes.llm import FakeLLM

        llm = FakeLLM()
        llm.register("hello", "hi there")
        manager = ConversationManager(llm=llm, system_prompt="be nice", store=None)
        reply = await manager.reply("hello")
        assert reply == "hi there"
        # No store, so load is a no-op.
        await manager.load()
        assert len(manager.current.messages) == 2  # Still the same messages in memory


# ---------------------------------------------------------------------------
# SqliteConversationStore (requires aiosqlite)
# ---------------------------------------------------------------------------
class TestSqliteConversationStore:
    @pytest.mark.anyio
    async def test_sqlite_save_and_load(self, tmp_path: Path) -> None:
        from robot.ai.conversation_sqlite import SqliteConversationStore

        store = SqliteConversationStore(db_path=str(tmp_path / "test.db"))
        await store.save("conv1", "be helpful", [("user", "hello"), ("assistant", "hi")])
        result = await store.load("conv1")
        assert result is not None
        assert len(result) == 2
        assert result[0] == ("user", "hello")
        assert result[1] == ("assistant", "hi")
        await store.close()

    @pytest.mark.anyio
    async def test_sqlite_load_nonexistent(self, tmp_path: Path) -> None:
        from robot.ai.conversation_sqlite import SqliteConversationStore

        store = SqliteConversationStore(db_path=str(tmp_path / "test.db"))
        result = await store.load("nonexistent")
        assert result is None
        await store.close()

    @pytest.mark.anyio
    async def test_sqlite_list_conversations(self, tmp_path: Path) -> None:
        from robot.ai.conversation_sqlite import SqliteConversationStore

        store = SqliteConversationStore(db_path=str(tmp_path / "test.db"))
        await store.save("conv1", "prompt1", [("user", "a")])
        await store.save("conv2", "prompt2", [("user", "b")])
        convs = await store.list_conversations()
        assert len(convs) == 2
        ids = {c.id for c in convs}
        assert ids == {"conv1", "conv2"}
        await store.close()

    @pytest.mark.anyio
    async def test_sqlite_delete(self, tmp_path: Path) -> None:
        from robot.ai.conversation_sqlite import SqliteConversationStore

        store = SqliteConversationStore(db_path=str(tmp_path / "test.db"))
        await store.save("conv1", "p", [("user", "hello")])
        assert await store.delete("conv1") is True
        assert await store.load("conv1") is None
        assert await store.delete("conv1") is False
        await store.close()

    @pytest.mark.anyio
    async def test_sqlite_overwrite(self, tmp_path: Path) -> None:
        from robot.ai.conversation_sqlite import SqliteConversationStore

        store = SqliteConversationStore(db_path=str(tmp_path / "test.db"))
        await store.save("conv1", "v1", [("user", "hello")])
        await store.save("conv1", "v2", [("user", "bye")])
        result = await store.load("conv1")
        assert result is not None
        assert len(result) == 1
        assert result[0] == ("user", "bye")
        await store.close()
