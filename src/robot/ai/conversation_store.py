"""Conversation persistence store protocol and in-memory implementation.

The :class:`ConversationStore` protocol defines the interface for
saving and loading conversations. The default :class:`InMemoryStore`
keeps everything in memory (no persistence). Use
:class:`SqliteConversationStore` for durable storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol


@dataclass(slots=True, frozen=True)
class ConversationMeta:
    """Metadata for a saved conversation (no messages)."""

    id: str
    created_at: datetime
    system_prompt: str
    message_count: int


class ConversationStore(Protocol):
    """Persistence operations used by :class:`ConversationManager`."""

    async def save(
        self, conversation_id: str, system_prompt: str, messages: list[tuple[str, str]]
    ) -> None: ...

    async def load(self, conversation_id: str) -> list[tuple[str, str]] | None: ...

    async def list_conversations(self) -> list[ConversationMeta]: ...

    async def delete(self, conversation_id: str) -> bool: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class InMemoryStore:
    """Simple in-memory store for testing. Not persistent."""

    _conversations: dict[str, list[tuple[str, str]]] = field(default_factory=dict, init=False)
    _meta: dict[str, ConversationMeta] = field(default_factory=dict, init=False)

    async def save(
        self,
        conversation_id: str,
        system_prompt: str,
        messages: list[tuple[str, str]],
    ) -> None:
        """Save (or overwrite) a conversation."""
        self._conversations[conversation_id] = list(messages)
        self._meta[conversation_id] = ConversationMeta(
            id=conversation_id,
            created_at=datetime.now(tz=UTC),
            system_prompt=system_prompt,
            message_count=len(messages),
        )

    async def load(self, conversation_id: str) -> list[tuple[str, str]] | None:
        """Load messages for a conversation. Returns None if not found."""
        return self._conversations.get(conversation_id)

    async def list_conversations(self) -> list[ConversationMeta]:
        """List all saved conversations (newest first)."""
        return sorted(self._meta.values(), key=lambda m: m.created_at, reverse=True)

    async def delete(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if found and deleted."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            self._meta.pop(conversation_id, None)
            return True
        return False

    async def close(self) -> None:
        """Release resources (no-op for the in-memory store)."""


__all__ = ["ConversationMeta", "ConversationStore", "InMemoryStore"]
