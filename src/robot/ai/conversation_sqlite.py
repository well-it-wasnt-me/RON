"""SQLite-backed conversation persistence store.

Stores conversations and their messages in a local SQLite database.
Uses ``aiosqlite`` for async access so the event loop is never blocked
by disk I/O.

Configuration::

    DESKBOT_CONVERSATION__STORE=sqlite
    DESKBOT_CONVERSATION__DB_PATH=~/.deskbot/conversations.db
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot.ai.conversation_store import ConversationMeta
from robot.logging import get_logger

_log = get_logger("ai.conversation_sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    system_prompt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
    ON messages(conversation_id);
"""


class SqliteConversationStore:
    """Persist conversations to a local SQLite database.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Parent directories are created
        automatically. Defaults to ``~/.deskbot/conversations.db``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".deskbot" / "conversations.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Any = None  # aiosqlite connection, set on first use

    async def _ensure_conn(self) -> Any:
        """Lazy-open the database connection and create tables."""
        if self._conn is not None:
            return self._conn
        try:
            import aiosqlite
        except ImportError as exc:
            raise ImportError(
                "aiosqlite is required for SqliteConversationStore. "
                "Install it with: uv pip install aiosqlite"
            ) from exc
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        _log.info("conversation_sqlite.opened", db=str(self._db_path))
        return self._conn

    async def save(
        self,
        conversation_id: str,
        system_prompt: str,
        messages: list[tuple[str, str]],
    ) -> None:
        """Save (or overwrite) a conversation and its messages."""
        conn = await self._ensure_conn()
        now = datetime.now(tz=UTC).isoformat()

        # Upsert the conversation record.
        await conn.execute(
            "INSERT OR REPLACE INTO conversations (id, created_at, system_prompt) VALUES (?, ?, ?)",
            (conversation_id, now, system_prompt),
        )
        # Delete old messages for this conversation (if overwriting).
        await conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        # Insert all messages.
        await conn.executemany(
            "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            [(conversation_id, role, content, now) for role, content in messages],
        )
        await conn.commit()
        _log.debug("conversation_sqlite.saved", id=conversation_id, messages=len(messages))

    async def load(self, conversation_id: str) -> list[tuple[str, str]] | None:
        """Load messages for a conversation. Returns None if not found."""
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        if not rows:
            # Check if the conversation exists at all.
            cursor2 = await conn.execute(
                "SELECT id FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            exists = await cursor2.fetchone()
            if exists is None:
                return None
            # Conversation exists but has no messages (e.g., only a system prompt).
            return []
        return [(row["role"], row["content"]) for row in rows]

    async def list_conversations(self) -> list[ConversationMeta]:
        """List all saved conversations (newest first)."""
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            "SELECT c.id, c.created_at, c.system_prompt, "
            "  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
            "FROM conversations c ORDER BY c.created_at DESC"
        )
        rows = await cursor.fetchall()
        return [
            ConversationMeta(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                system_prompt=row["system_prompt"],
                message_count=row["message_count"],
            )
            for row in rows
        ]

    async def delete(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if found and deleted."""
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        await conn.commit()
        deleted = bool(cursor.rowcount > 0)
        _log.debug("conversation_sqlite.deleted", id=conversation_id, found=deleted)
        return deleted

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            _log.info("conversation_sqlite.closed")


__all__ = ["SqliteConversationStore"]
