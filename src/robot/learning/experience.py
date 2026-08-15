"""Experience data structures for DeskBot learning.

An :class:`Experience` records a single observation-action-outcome tuple.
Experiences flow through three memory layers:

* :class:`WorkingMemory` - recent observations (ring buffer).
* :class:`ReplayBuffer` - uniform-random sampling for training.
* :class:`EpisodicMemory` - persistent SQLite-backed storage.

The :class:`ExperienceRecorder` subscribes to the event bus and
automatically records experiences from robot events.
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from robot.learning.tensor import Tensor
from robot.logging import get_logger

_log = get_logger("learning.experience")


# ---------------------------------------------------------------------------
# Experience dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Experience:
    """A single observation-action-outcome tuple.

    Attributes
    ----------
    timestamp:
        When the experience was recorded (UTC).
    state:
        The observation before acting.  A flat float vector describing
        the robot's state (emotions, servo positions, perception, …).
    action:
        The action taken.  A flat float vector describing what the
        robot did (servo targets, expression changes, …).
    reward:
        Scalar feedback signal.  Positive = good outcome, negative =
        bad outcome, 0 = neutral.
    next_state:
        The observation after acting.  Same shape as ``state``.
    metadata:
        Optional key-value pairs with extra context (event type,
        source, tags, …).
    """

    timestamp: datetime
    state: list[float]
    action: list[float]
    reward: float
    next_state: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def state_tensor(self) -> Tensor:
        """Return state as a Tensor (1-D)."""
        return Tensor(self.state)

    def action_tensor(self) -> Tensor:
        """Return action as a Tensor (1-D)."""
        return Tensor(self.action)

    def next_state_tensor(self) -> Tensor:
        """Return next_state as a Tensor (1-D)."""
        return Tensor(self.next_state)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experience:
        """Deserialise from a dict (as produced by :meth:`to_dict`)."""
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            timestamp=ts,
            state=list(data["state"]),
            action=list(data["action"]),
            reward=float(data["reward"]),
            next_state=list(data["next_state"]),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Working memory (recent observations ring buffer)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorkingMemory:
    """Bounded ring buffer of recent experiences.

    Working memory holds the most recent experiences for quick recall.
    It is **not** persisted - it represents short-term, in-memory
    awareness of what just happened.

    Parameters
    ----------
    capacity:
        Maximum number of experiences to keep.  Oldest entries are
        evicted when capacity is exceeded.
    """

    capacity: int = 256
    _buffer: deque[Experience] = field(default_factory=deque, init=False, repr=False)

    def add(self, experience: Experience) -> None:
        """Add an experience, evicting the oldest if at capacity."""
        self._buffer.append(experience)
        while len(self._buffer) > self.capacity:
            self._buffer.popleft()

    def recent(self, limit: int = 10) -> list[Experience]:
        """Return the ``limit`` most recent experiences."""
        return list(self._buffer)[-limit:]

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def __iter__(self) -> Iterator[Experience]:
        return iter(self._buffer)


# ---------------------------------------------------------------------------
# Replay buffer (random sampling for training)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReplayBuffer:
    """Experience replay buffer with uniform-random sampling.

    Used during training to break temporal correlations.  The buffer
    stores experiences in a ring buffer and supports sampling random
    mini-batches.

    Parameters
    ----------
    capacity:
        Maximum number of experiences to keep.
    seed:
        Random seed for reproducible sampling.
    """

    capacity: int = 10_000
    seed: int = 42
    _buffer: deque[Experience] = field(default_factory=deque, init=False, repr=False)
    _rng: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import numpy as np

        self._rng = np.random.default_rng(self.seed)

    def add(self, experience: Experience) -> None:
        """Add an experience, evicting the oldest if at capacity."""
        self._buffer.append(experience)
        while len(self._buffer) > self.capacity:
            self._buffer.popleft()

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample ``batch_size`` experiences uniformly at random.

        If the buffer contains fewer experiences than ``batch_size``,
        returns all available experiences.
        """
        n = min(batch_size, len(self._buffer))
        if n == 0:
            return []
        indices = self._rng.choice(len(self._buffer), size=n, replace=False)
        return [self._buffer[int(i)] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def __iter__(self) -> Iterator[Experience]:
        return iter(self._buffer)


# ---------------------------------------------------------------------------
# Persistence protocol and SQLite store
# ---------------------------------------------------------------------------


@runtime_checkable
class ExperienceStore(Protocol):
    """Storage backend for episodic experiences."""

    def save(self, experience: Experience) -> int:
        """Persist an experience. Returns the row id."""
        ...

    def save_batch(self, experiences: list[Experience]) -> list[int]:
        """Persist a batch of experiences. Returns row ids."""
        ...

    def load_recent(self, limit: int = 100) -> list[Experience]:
        """Load the most recent experiences."""
        ...

    def load_all(self) -> list[Experience]:
        """Load all stored experiences."""
        ...

    def delete_before(self, timestamp: datetime) -> int:
        """Delete experiences older than ``timestamp``. Returns count."""
        ...

    def count(self) -> int:
        """Return total number of stored experiences."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    state TEXT NOT NULL,
    action TEXT NOT NULL,
    reward REAL NOT NULL,
    next_state TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_experiences_timestamp
    ON experiences(timestamp);
"""


class SqliteExperienceStore:
    """SQLite-backed experience persistence.

    Stores experiences in a local SQLite database, matching the
    pattern used by :class:`SqliteConversationStore` and
    :class:`SqlitePreferenceStore`.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for
        testing.  Parent directories are created automatically for
        file paths.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".deskbot" / "experiences.db"
        self._db_path = Path(db_path)
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            _log.info("experience_sqlite.opened", db=str(self._db_path))
        return self._conn

    def save(self, experience: Experience) -> int:
        """Persist an experience and return its row id."""
        conn = self._ensure_conn()
        cursor = conn.execute(
            """INSERT INTO experiences (timestamp, state, action, reward, next_state, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                experience.timestamp.isoformat(),
                json.dumps(experience.state),
                json.dumps(experience.action),
                experience.reward,
                json.dumps(experience.next_state),
                json.dumps(experience.metadata),
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def save_batch(self, experiences: list[Experience]) -> list[int]:
        """Persist a batch of experiences. Returns row ids."""
        conn = self._ensure_conn()
        ids: list[int] = []
        for exp in experiences:
            cursor = conn.execute(
                """INSERT INTO experiences (timestamp, state, action, reward, next_state, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    exp.timestamp.isoformat(),
                    json.dumps(exp.state),
                    json.dumps(exp.action),
                    exp.reward,
                    json.dumps(exp.next_state),
                    json.dumps(exp.metadata),
                ),
            )
            assert cursor.lastrowid is not None
            ids.append(cursor.lastrowid)
        conn.commit()
        return ids

    def load_recent(self, limit: int = 100) -> list[Experience]:
        """Load the most recent ``limit`` experiences."""
        conn = self._ensure_conn()
        rows = conn.execute(
            """SELECT timestamp, state, action, reward, next_state, metadata
               FROM experiences ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_experience(row) for row in rows]

    def load_all(self) -> list[Experience]:
        """Load all stored experiences."""
        conn = self._ensure_conn()
        rows = conn.execute(
            """SELECT timestamp, state, action, reward, next_state, metadata
               FROM experiences ORDER BY id ASC"""
        ).fetchall()
        return [self._row_to_experience(row) for row in rows]

    def delete_before(self, timestamp: datetime) -> int:
        """Delete experiences older than ``timestamp``. Returns count."""
        conn = self._ensure_conn()
        cursor = conn.execute(
            "DELETE FROM experiences WHERE timestamp < ?",
            (timestamp.isoformat(),),
        )
        conn.commit()
        return cursor.rowcount

    def count(self) -> int:
        """Return total number of stored experiences."""
        conn = self._ensure_conn()
        row = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()
        return int(row[0])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            _log.info("experience_sqlite.closed")

    @staticmethod
    def _row_to_experience(row: tuple[Any, ...]) -> Experience:
        ts = row[0]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return Experience(
            timestamp=ts,
            state=json.loads(row[1]),
            action=json.loads(row[2]),
            reward=float(row[3]),
            next_state=json.loads(row[4]),
            metadata=json.loads(row[5]),
        )


class InMemoryExperienceStore:
    """Simple list-backed store for testing. Not persistent."""

    def __init__(self) -> None:
        self._data: list[Experience] = []

    def save(self, experience: Experience) -> int:
        self._data.append(experience)
        return len(self._data)

    def save_batch(self, experiences: list[Experience]) -> list[int]:
        ids = []
        for exp in experiences:
            self._data.append(exp)
            ids.append(len(self._data))
        return ids

    def load_recent(self, limit: int = 100) -> list[Experience]:
        return list(self._data[-limit:])

    def load_all(self) -> list[Experience]:
        return list(self._data)

    def delete_before(self, timestamp: datetime) -> int:
        before = [e for e in self._data if e.timestamp < timestamp]
        self._data = [e for e in self._data if e.timestamp >= timestamp]
        return len(before)

    def count(self) -> int:
        return len(self._data)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Episodic memory (combines working memory + persistent store)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EpisodicMemory:
    """Persistent experience memory that survives restarts.

    On creation, loads past experiences from the store (up to
    ``max_load``).  New experiences are persisted immediately and
    also kept in a ring buffer for fast recent access.

    Parameters
    ----------
    store:
        Persistence backend.  Use :class:`SqliteExperienceStore` for
        production or :class:`InMemoryExperienceStore` for testing.
    capacity:
        Maximum number of experiences in the in-memory ring buffer.
    max_load:
        Maximum number of past experiences to load from the store
        on initialisation.
    """

    store: ExperienceStore
    capacity: int = 10_000
    max_load: int = 1_000
    _buffer: deque[Experience] = field(default_factory=deque, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    def load_from_store(self) -> None:
        """Load past experiences from the persistent store.

        Called automatically on first add if not called explicitly.
        """
        if self._loaded:
            return
        past = self.store.load_recent(limit=self.max_load)
        for exp in reversed(past):
            self._buffer.append(exp)
            while len(self._buffer) > self.capacity:
                self._buffer.popleft()
        self._loaded = True
        _log.info("episodic_memory.loaded", count=len(self._buffer))

    def add(self, experience: Experience) -> None:
        """Add an experience, persisting it and updating the ring buffer."""
        if not self._loaded:
            self.load_from_store()
        self.store.save(experience)
        self._buffer.append(experience)
        while len(self._buffer) > self.capacity:
            self._buffer.popleft()

    def recent(self, limit: int = 10) -> list[Experience]:
        """Return the ``limit`` most recent experiences."""
        if not self._loaded:
            self.load_from_store()
        return list(self._buffer)[-limit:]

    def sample(self, batch_size: int, seed: int | None = None) -> list[Experience]:
        """Sample ``batch_size`` experiences uniformly at random.

        If the buffer is smaller than ``batch_size``, returns all
        available experiences.
        """
        if not self._loaded:
            self.load_from_store()
        n = min(batch_size, len(self._buffer))
        if n == 0:
            return []
        import numpy as np

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self._buffer), size=n, replace=False)
        return [self._buffer[int(i)] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        """Clear the in-memory buffer. Does not delete the persistent store."""
        self._buffer.clear()


__all__ = [
    "EpisodicMemory",
    "Experience",
    "ExperienceStore",
    "InMemoryExperienceStore",
    "ReplayBuffer",
    "SqliteExperienceStore",
    "WorkingMemory",
]
