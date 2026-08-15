"""Long-term memory backed by an in-memory list.

Future versions will persist this to SQLite or vector storage.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

MAX_ENTRIES: Final[int] = 1024


@dataclass(slots=True, frozen=True)
class MemoryEntry:
    timestamp: datetime
    content: str
    importance: float = 0.5
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class Memory:
    """Bounded ring buffer of :class:`MemoryEntry`."""

    entries: deque[MemoryEntry] = field(default_factory=deque)
    capacity: int = MAX_ENTRIES

    def add(self, content: str, importance: float = 0.5, tags: Iterable[str] = ()) -> None:
        entry = MemoryEntry(
            timestamp=datetime.now(tz=UTC),
            content=content,
            importance=max(0.0, min(1.0, importance)),
            tags=tuple(tags),
        )
        self.entries.append(entry)
        while len(self.entries) > self.capacity:
            self.entries.popleft()

    def recall(self, limit: int = 10) -> list[MemoryEntry]:
        return list(self.entries)[-limit:]

    def search(self, query: str) -> list[MemoryEntry]:
        q = query.lower()
        return [e for e in self.entries if q in e.content.lower()]

    def clear(self) -> None:
        self.entries.clear()


__all__ = ["Memory", "MemoryEntry"]
