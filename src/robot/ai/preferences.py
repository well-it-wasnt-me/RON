"""User preference learning.

The :class:`PreferenceTracker` observes user interactions (speech
events, emotion reactions, etc.) and builds a profile of the user's
preferences. Preferences are stored as key-value pairs with
confidence scores and can be recalled to personalise the robot's
behaviour.

Preferences are persisted via :class:`PreferenceStore` (in-memory by
default, SQLite-backed available).
"""

from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from robot.logging import get_logger

_log = get_logger("ai.preferences")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Preference:
    """A single learned preference.

    Attributes
    ----------
    key:
        The preference category (e.g. ``"humour"``, ``"volume"``).
    value:
        The learned value (e.g. ``"dry"``, ``0.7``).
    confidence:
        How confident the tracker is, in ``[0, 1]``.
    source:
        How this preference was learned (``"explicit"`` for direct
        statements, ``"inferred"`` for deduced preferences).
    updated_at:
        When the preference was last updated.
    observation_count:
        Number of observations contributing to this preference.
    total_reward:
        Cumulative reward associated with this preference.
    first_observed / last_observed:
        Timestamps defining the observation history.
    """

    key: str
    value: str
    confidence: float = 0.5
    source: str = "inferred"
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    observation_count: int = 1
    total_reward: float = 0.0
    first_observed: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    last_observed: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


# ---------------------------------------------------------------------------
# Preference store protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class PreferenceStore(Protocol):
    """Storage backend for preferences."""

    def save(self, preference: Preference) -> None:
        """Persist a preference (upsert)."""
        ...

    def load(self, key: str) -> Preference | None:
        """Load a preference by key, or ``None`` if not found."""
        ...

    def load_all(self) -> list[Preference]:
        """Load all stored preferences."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a preference. Returns ``True`` if it existed."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------
class InMemoryPreferenceStore:
    """Simple dict-backed store for testing."""

    def __init__(self) -> None:
        self._data: dict[str, Preference] = {}

    def save(self, preference: Preference) -> None:
        self._data[preference.key] = preference

    def load(self, key: str) -> Preference | None:
        return self._data.get(key)

    def load_all(self) -> list[Preference]:
        return list(self._data.values())

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------
class SqlitePreferenceStore:
    """SQLite-backed preference store.

    Uses a single table ``preferences`` containing both the normal
    preference state and learner metadata.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """Create a SQLite-backed preference store.

        ``:memory:`` is kept as SQLite's special in-memory database name.
        File-backed databases are expanded and their parent directory is
        created automatically.
        """
        if db_path == ":memory:":
            self._db_path: Path | None = None
        else:
            self._db_path = Path(db_path).expanduser()
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def _ensure_conn(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is not None:
                return self._conn
            db_path = ":memory:" if self._db_path is None else str(self._db_path)

            self._conn = sqlite3.connect(
                db_path,
                check_same_thread=False,
            )

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                                                           key TEXT PRIMARY KEY,
                                                           value TEXT NOT NULL,
                                                           confidence REAL NOT NULL DEFAULT 0.5,
                                                           source TEXT NOT NULL DEFAULT 'inferred',
                                                           updated_at TEXT NOT NULL,
                                                           observation_count INTEGER NOT NULL DEFAULT 1,
                                                           total_reward REAL NOT NULL DEFAULT 0.0,
                                                           first_observed TEXT NOT NULL,
                                                           last_observed TEXT NOT NULL
                )
                """
            )

            self._migrate_v1_to_v2()
            self._conn.commit()

        return self._conn

    def _migrate_v1_to_v2(self) -> None:
        """Migrate databases created before learner state was added."""
        assert self._conn is not None

        cursor = self._conn.execute("PRAGMA table_info(preferences)")
        columns = {row[1] for row in cursor.fetchall()}

        if "observation_count" not in columns:
            self._conn.execute(
                "ALTER TABLE preferences ADD COLUMN observation_count INTEGER NOT NULL DEFAULT 1"
            )

        if "total_reward" not in columns:
            self._conn.execute(
                "ALTER TABLE preferences ADD COLUMN total_reward REAL NOT NULL DEFAULT 0.0"
            )

        if "first_observed" not in columns:
            self._conn.execute("ALTER TABLE preferences ADD COLUMN first_observed TEXT")

            self._conn.execute(
                """
                UPDATE preferences
                SET first_observed = updated_at
                WHERE first_observed IS NULL
                """
            )

        if "last_observed" not in columns:
            self._conn.execute("ALTER TABLE preferences ADD COLUMN last_observed TEXT")

            self._conn.execute(
                """
                UPDATE preferences
                SET last_observed = updated_at
                WHERE last_observed IS NULL
                """
            )

    def save(self, preference: Preference) -> None:
        """Persist the complete preference state."""
        with self._lock:
            conn = self._ensure_conn()

        updated_at = preference.updated_at
        first_observed = preference.first_observed or updated_at
        last_observed = preference.last_observed or updated_at

        last_observed = max(last_observed, first_observed)

        conn.execute(
            """
            INSERT INTO preferences (
                key,
                value,
                confidence,
                source,
                updated_at,
                observation_count,
                total_reward,
                first_observed,
                last_observed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                                        value = excluded.value,
                                        confidence = excluded.confidence,
                                        source = excluded.source,
                                        updated_at = excluded.updated_at,
                                        observation_count = excluded.observation_count,
                                        total_reward = excluded.total_reward,
                                        first_observed = excluded.first_observed,
                                        last_observed = excluded.last_observed
            """,
            (
                preference.key,
                preference.value,
                preference.confidence,
                preference.source,
                updated_at.isoformat(),
                max(1, preference.observation_count),
                preference.total_reward,
                first_observed.isoformat(),
                last_observed.isoformat(),
            ),
        )

        conn.commit()

    def load(self, key: str) -> Preference | None:
        with self._lock:
            conn = self._ensure_conn()
        row = conn.execute(
            "SELECT key, value, confidence, source, updated_at FROM preferences WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return Preference(
            key=row[0],
            value=row[1],
            confidence=row[2],
            source=row[3],
            updated_at=datetime.fromisoformat(row[4]),
        )

    def load_all(self) -> list[Preference]:
        with self._lock:
            conn = self._ensure_conn()
        rows = conn.execute(
            "SELECT key, value, confidence, source, updated_at FROM preferences"
        ).fetchall()
        return [
            Preference(
                key=row[0],
                value=row[1],
                confidence=row[2],
                source=row[3],
                updated_at=datetime.fromisoformat(row[4]),
            )
            for row in rows
        ]

    def delete(self, key: str) -> bool:
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute("DELETE FROM preferences WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ learned preference persistence

    def save_learned(
        self,
        key: str,
        value: str,
        confidence: float,
        source: str,
        updated_at: datetime,
        observation_count: int,
        total_reward: float,
        first_observed: datetime,
        last_observed: datetime,
    ) -> None:
        """Persist the full learner state for a preference.

        This is the canonical persistence path for
        :class:`PreferenceLearner` - it stores all fields needed to
        reconstruct the learner's internal state exactly.
        """
        with self._lock:
            conn = self._ensure_conn()

        conn.execute(
            """
            INSERT INTO preferences
                (key, value, confidence, source, updated_at,
                 observation_count, total_reward, first_observed, last_observed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                confidence = excluded.confidence,
                source = excluded.source,
                updated_at = excluded.updated_at,
                observation_count = excluded.observation_count,
                total_reward = excluded.total_reward,
                first_observed = excluded.first_observed,
                last_observed = excluded.last_observed
            """,
            (
                key,
                value,
                confidence,
                source,
                updated_at.isoformat(),
                observation_count,
                total_reward,
                first_observed.isoformat(),
                last_observed.isoformat(),
            ),
        )
        conn.commit()

    def load_learned_all(self) -> list[dict[str, object]]:
        """Load all preferences with full learner state.

        Returns a list of dicts with all persisted fields,
        suitable for reconstructing :class:`LearnedPreference` objects.
        """
        with self._lock:
            conn = self._ensure_conn()

        rows = conn.execute(
            """
            SELECT
                key,
                value,
                confidence,
                source,
                updated_at,
                observation_count,
                total_reward,
                first_observed,
                last_observed
            FROM preferences
            """
        ).fetchall()

        result: list[dict[str, object]] = []

        for row in rows:
            updated_at = datetime.fromisoformat(row[4])

            result.append(
                {
                    "key": row[0],
                    "value": row[1],
                    "confidence": row[2],
                    "source": row[3],
                    "updated_at": updated_at,
                    "observation_count": max(1, row[5] or 1),
                    "total_reward": row[6] or 0.0,
                    "first_observed": (datetime.fromisoformat(row[7]) if row[7] else updated_at),
                    "last_observed": (datetime.fromisoformat(row[8]) if row[8] else updated_at),
                }
            )

        return result

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# ---------------------------------------------------------------------------
# Preference tracker
# ---------------------------------------------------------------------------

# Negation patterns that flip the meaning of what follows.
_NEGATION_PATTERNS: tuple[str, ...] = (
    "don't ",
    "do not ",
    "dont ",
    "not ",
    "never ",
    "no ",
    "stop ",
    "i don't want ",
    "i don't like ",
    "i hate ",
    "i dislike ",
)


def _is_negated(text: str, pattern: str) -> bool:
    """Check whether a matched pattern is negated by a preceding phrase.

    For example, in "don\'t be funny", the negation "don\'t"
    appears immediately before "be funny".
    """
    idx = text.find(pattern)
    if idx <= 0:
        return False
    prefix = text[:idx].rstrip().lower()
    for neg in _NEGATION_PATTERNS:
        neg_lower = neg.lower().rstrip()
        if prefix.endswith(neg_lower):
            return True
    # Also check if the negation is the entire prefix
    return prefix in ("don't", "dont", "not", "never", "no", "stop")


# Simple keyword patterns for explicit preference detection.
_EXPLICIT_PATTERNS: dict[str, dict[str, str]] = {
    "name": {
        "my name is": "extract_after",
        "i'm called": "extract_after",
        "call me": "extract_after",
        "i go by": "extract_after",
    },
    "nickname": {
        "my nickname is": "extract_after",
    },
    "volume": {
        "louder": "high",
        "softer": "low",
        "quieter": "low",
        "turn it up": "high",
        "turn it down": "low",
        "too loud": "low",
        "too quiet": "high",
    },
    "pace": {
        "slower": "slow",
        "faster": "fast",
        "slow down": "slow",
        "speed up": "fast",
        "too fast": "slow",
        "too slow": "fast",
    },
    "formality": {
        "be formal": "formal",
        "be casual": "casual",
        "be informal": "casual",
        "use formal language": "formal",
        "talk casually": "casual",
    },
    "humour": {
        "be funny": "humorous",
        "be serious": "serious",
        "make me laugh": "humorous",
        "stop joking": "serious",
        "more jokes": "humorous",
    },
    "verbosity": {
        "be brief": "brief",
        "be concise": "brief",
        "short answer": "brief",
        "more detail": "detailed",
        "explain more": "detailed",
        "be verbose": "detailed",
    },
    "language": {
        "speak english": "en",
        "speak spanish": "es",
        "speak french": "fr",
        "speak german": "de",
        "in english": "en",
        "en español": "es",
    },
}


def _extract_after(prefix: str, text: str) -> str:
    """Extract the text after a given prefix, stripped."""
    idx = text.find(prefix)
    if idx == -1:
        return ""
    return text[idx + len(prefix) :].strip().rstrip(".!?,")


@dataclass(slots=True)
class PreferenceTracker:
    """Learns user preferences from conversation events.

    The tracker analyses each user utterance for explicit preference
    statements (e.g. "my name is Alice", "be funny") and infers
    preferences from repeated patterns. Preferences are stored with
    confidence scores that increase with repeated observations.

    Negation handling:
        Phrases like "don\'t be funny" are detected and skipped
        rather than incorrectly learning the opposite of what was
        intended.  This prevents a user saying "I don\'t want you
        to be funny" from being recorded as preferring humour.
    """

    store: PreferenceStore = field(default_factory=InMemoryPreferenceStore)
    _interaction_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def process_user_text(self, text: str) -> list[Preference]:
        """Analyse a user utterance for preferences.

        Returns any newly created or updated preferences.

        Handles negation: "don\'t be funny" is skipped rather than
        recording a preference for humour.
        """
        text_lower = text.lower().strip()
        updated: list[Preference] = []

        for key, patterns in _EXPLICIT_PATTERNS.items():
            for pattern, extraction in patterns.items():
                if pattern not in text_lower:
                    continue

                # Check for negation preceding this pattern
                if _is_negated(text_lower, pattern):
                    _log.debug(
                        "preferences.negation_detected",
                        key=key,
                        pattern=pattern,
                        text=text_lower[:80],
                    )
                    continue

                if extraction == "extract_after":
                    value = _extract_after(pattern, text_lower)
                    if not value:
                        continue
                else:
                    value = extraction

                pref = self._upsert(key, value, source="explicit")
                if pref is not None:
                    updated.append(pref)

        self._interaction_counts["total"] += 1

        return updated

    def get(self, key: str) -> Preference | None:
        """Return the current preference for ``key``, or ``None``."""
        return self.store.load(key)

    def get_all(self) -> list[Preference]:
        """Return all stored preferences."""
        return self.store.load_all()

    def format_for_prompt(self, limit: int = 10) -> str:
        """Format preferences as a human-readable string for system prompts.

        Only includes preferences above a minimum confidence threshold.
        """
        prefs = self.store.load_all()
        # Sort by confidence, descending.
        prefs.sort(key=lambda p: p.confidence, reverse=True)
        prefs = prefs[:limit]
        if not prefs:
            return ""
        lines = ["User preferences:"]
        for p in prefs:
            lines.append(
                f"- {p.key}: {p.value} (confidence: {p.confidence:.0%}, source: {p.source})"
            )
        return "\n".join(lines)

    def _upsert(self, key: str, value: str, source: str = "inferred") -> Preference | None:
        """Insert or update a preference, increasing confidence on repeats."""
        now = datetime.now(tz=UTC)
        existing = self.store.load(key)

        if existing is not None:
            boost = 0.3 if source == "explicit" else 0.1
            new_confidence = min(1.0, existing.confidence + boost)

            pref = Preference(
                key=key,
                value=value,
                confidence=new_confidence,
                source=source,
                updated_at=now,
                observation_count=existing.observation_count + 1,
                total_reward=existing.total_reward,
                first_observed=existing.first_observed or existing.updated_at,
                last_observed=now,
            )
        else:
            pref = Preference(
                key=key,
                value=value,
                confidence=0.5 if source == "explicit" else 0.2,
                source=source,
                updated_at=now,
                observation_count=1,
                total_reward=0.0,
                first_observed=now,
                last_observed=now,
            )

        self.store.save(pref)
        return pref


__all__ = [
    "InMemoryPreferenceStore",
    "Preference",
    "PreferenceStore",
    "PreferenceTracker",
    "SqlitePreferenceStore",
]
