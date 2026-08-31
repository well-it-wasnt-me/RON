"""Multi-turn conversation state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from robot.interfaces.llm import LLM, Message, Role
from robot.logging import get_logger as _get_logger

if TYPE_CHECKING:
    from robot.ai.conversation_store import ConversationStore

MAX_HISTORY: int = 20

_log = _get_logger("ai.conversation")


@dataclass(slots=True)
class Conversation:
    """A bounded history of messages in one conversation."""

    system_prompt: str
    messages: deque[Message] = field(default_factory=deque)

    def add_user(self, text: str) -> None:
        self._append(Message(role=Role.USER, content=text))

    def add_assistant(self, text: str) -> None:
        self._append(Message(role=Role.ASSISTANT, content=text))

    def as_list(self) -> list[Message]:
        return [Message(role=Role.SYSTEM, content=self.system_prompt), *self.messages]

    def as_list_with_context(self, context: str = "") -> list[Message]:
        """Return model messages with optional trusted memory context."""
        prompt = self.system_prompt
        if context:
            prompt = f"{prompt}\n\nRelevant remembered context:\n{context}"
        return [Message(role=Role.SYSTEM, content=prompt), *self.messages]

    def _append(self, message: Message) -> None:
        self.messages.append(message)
        while len(self.messages) > MAX_HISTORY:
            self.messages.popleft()

    def as_tuples(self) -> list[tuple[str, str]]:
        """Return messages as ``(role, content)`` tuples for persistence."""
        return [(m.role.value, m.content) for m in self.messages]

    @classmethod
    def from_tuples(cls, system_prompt: str, tuples: list[tuple[str, str]]) -> Conversation:
        """Reconstruct a Conversation from persisted ``(role, content)`` tuples."""
        conv = cls(system_prompt=system_prompt)
        for role_str, content in tuples:
            role = Role(role_str)
            conv.messages.append(Message(role=role, content=content))
        return conv


class ConversationManager:
    """Tracks the current :class:`Conversation` and forwards it to an LLM.

    When a ``store`` is provided, conversations are automatically
    persisted after each reply and the last conversation is loaded
    on startup.
    """

    def __init__(
        self,
        llm: LLM,
        system_prompt: str,
        store: ConversationStore | None = None,
        conversation_id: str = "default",
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._store = store
        self._conversation_id = conversation_id
        self._current = Conversation(system_prompt=system_prompt)

    @property
    def current(self) -> Conversation:
        return self._current

    @property
    def store(self) -> ConversationStore | None:
        """The optional persistent store backing this manager."""
        return self._store

    @property
    def conversation_id(self) -> str:
        """Identifier of the active conversation."""
        return self._conversation_id

    def reset(self) -> None:
        self._current = Conversation(system_prompt=self._current.system_prompt)

    def messages_for(self, context: str = "") -> list[Message]:
        """Build model messages for the active conversation and memory context."""
        return self._current.as_list_with_context(context)

    async def reply(self, user_text: str, context: str = "") -> str:
        self._current.add_user(user_text)
        answer = await self._llm.complete(self.messages_for(context))
        self._current.add_assistant(answer)
        await self._save()
        return answer

    async def summarise(self) -> str:
        messages = self._current.as_list()
        return await self._llm.complete(messages)

    async def load(self) -> None:
        """Load the last conversation from the store, if available.

        Note: this loads the conversation identified by ``conversation_id``
        (the configured default, typically ``"default"``). If the user
        previously switched to a different conversation via the API, that
        history is not automatically resumed on restart. A future version
        should persist the last-active conversation id.
        """
        if self._store is None:
            return
        tuples = await self._store.load(self._conversation_id)
        if tuples is not None:
            self._current = Conversation.from_tuples(
                system_prompt=self._system_prompt, tuples=tuples
            )
            _log.info(
                "conversation.loaded",
                id=self._conversation_id,
                messages=len(self._current.messages),
            )

    async def _save(self) -> None:
        """Persist the current conversation to the store."""
        if self._store is None:
            return
        await self._store.save(
            conversation_id=self._conversation_id,
            system_prompt=self._current.system_prompt,
            messages=self._current.as_tuples(),
        )

    async def save(self) -> None:
        """Persist the active conversation after externally managed updates."""
        await self._save()

    async def close(self) -> None:
        """Close the configured store, if the manager owns one."""
        if self._store is not None:
            await self._store.close()


__all__ = ["Conversation", "ConversationManager"]
