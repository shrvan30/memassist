"""SQLite storage for core memory (editable blocks) and recall memory (messages).

Schema matches PROJECT_SPEC.md §5. These functions are deterministic and contain
no LLM calls, so they are unit-tested directly (see tests/).
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable

VALID_BLOCKS: tuple[str, ...] = ("persona", "human")

# event_type vocabulary used across the recall log.
EVENT_MESSAGE = "message"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_PRESSURE_WARNING = "pressure_warning"


def _normalize_date_bound(value: str, *, is_end: bool) -> str:
    """Validate and normalize a date/datetime search bound.

    Accepts ``YYYY-MM-DD`` (expanded to the start or end of that day) or a full
    ``YYYY-MM-DD HH:MM:SS`` timestamp, returning a normalized
    ``YYYY-MM-DD HH:MM:SS`` string. Raises ``ValueError`` on anything else so the
    caller — and the model driving conversation_search_date — gets a clear signal
    to correct the input instead of a silent empty result.
    """
    text = (value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    try:
        day = datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Invalid date {value!r}: use 'YYYY-MM-DD' (e.g. 2026-07-24) "
            "or 'YYYY-MM-DD HH:MM:SS'."
        ) from None
    return day.strftime("%Y-%m-%d") + (" 23:59:59" if is_end else " 00:00:00")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS core_blocks (
    id          INTEGER PRIMARY KEY,
    name        TEXT UNIQUE CHECK(name IN ('persona','human')),
    content     TEXT NOT NULL,
    char_limit  INTEGER DEFAULT 2000,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,          -- user | assistant | tool | system_event
    event_type  TEXT,                   -- message | tool_call | tool_result | pressure_warning
    served_by   TEXT,                   -- gemini | groq | openrouter | mistral | NULL
    content     TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_event ON messages(event_type);
"""


class SQLiteStore:
    """Thread-safe wrapper around a single SQLite connection.

    Streamlit may re-run the script on different threads within one session, so
    the connection is opened with ``check_same_thread=False`` and every write is
    serialized behind a lock.
    """

    def __init__(
        self,
        db_path: str,
        default_persona: str,
        default_human: str,
        core_block_char_limit: int = 2000,
    ) -> None:
        self.db_path = db_path
        self.core_block_char_limit = core_block_char_limit
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()
        self._seed_core_blocks(default_persona, default_human)

    # -- setup -------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # Migrate older DBs created before the served_by column existed.
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(messages)")}
            if "served_by" not in cols:
                self._conn.execute("ALTER TABLE messages ADD COLUMN served_by TEXT")
            self._conn.commit()

    def _seed_core_blocks(self, persona: str, human: str) -> None:
        seeds = {"persona": persona, "human": human}
        with self._lock:
            for name, content in seeds.items():
                self._conn.execute(
                    "INSERT OR IGNORE INTO core_blocks (name, content, char_limit) "
                    "VALUES (?, ?, ?)",
                    (name, content, self.core_block_char_limit),
                )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- core memory -------------------------------------------------------
    @staticmethod
    def _check_block(name: str) -> None:
        if name not in VALID_BLOCKS:
            raise ValueError(
                f"Unknown core memory block '{name}'. Valid blocks: {', '.join(VALID_BLOCKS)}."
            )

    def get_block(self, name: str) -> str:
        self._check_block(name)
        row = self._conn.execute(
            "SELECT content FROM core_blocks WHERE name = ?", (name,)
        ).fetchone()
        return row["content"] if row else ""

    def get_block_limit(self, name: str) -> int:
        self._check_block(name)
        row = self._conn.execute(
            "SELECT char_limit FROM core_blocks WHERE name = ?", (name,)
        ).fetchone()
        return row["char_limit"] if row else self.core_block_char_limit

    def get_core_blocks(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT name, content FROM core_blocks").fetchall()
        blocks = {r["name"]: r["content"] for r in rows}
        # Guarantee both keys are always present.
        for name in VALID_BLOCKS:
            blocks.setdefault(name, "")
        return blocks

    def _set_block(self, name: str, content: str) -> None:
        limit = self.get_block_limit(name)
        if len(content) > limit:
            raise ValueError(
                f"Update would exceed the {limit}-character limit for the "
                f"'{name}' block (result would be {len(content)} chars). "
                "Summarize or move detail into archival memory instead."
            )
        with self._lock:
            self._conn.execute(
                "UPDATE core_blocks SET content = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE name = ?",
                (content, name),
            )
            self._conn.commit()

    def append_block(self, name: str, content: str) -> str:
        """Append a line to a core block. Returns the new block content."""
        self._check_block(name)
        current = self.get_block(name)
        new_content = f"{current}\n{content}".strip() if current.strip() else content.strip()
        self._set_block(name, new_content)
        return new_content

    def replace_block(self, name: str, old_text: str, new_text: str) -> str:
        """Replace the first occurrence of ``old_text``. Returns the new content.

        Raises ValueError if ``old_text`` is not found (verbatim).
        """
        self._check_block(name)
        current = self.get_block(name)
        if old_text not in current:
            raise ValueError(
                f"Text not found in the '{name}' block, so nothing was changed. "
                "The old_text must match exactly. Read the block again and retry."
            )
        new_content = current.replace(old_text, new_text, 1)
        self._set_block(name, new_content)
        return new_content

    # -- recall memory (message log) --------------------------------------
    def add_message(
        self,
        session_id: str,
        role: str,
        event_type: str,
        content: str,
        served_by: str | None = None,
        created_at: str | None = None,
    ) -> int:
        with self._lock:
            if created_at is None:
                cur = self._conn.execute(
                    "INSERT INTO messages (session_id, role, event_type, served_by, content) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, event_type, served_by, content),
                )
            else:
                cur = self._conn.execute(
                    "INSERT INTO messages "
                    "(session_id, role, event_type, served_by, content, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, role, event_type, served_by, content, created_at),
                )
            self._conn.commit()
            return int(cur.lastrowid)

    def count_messages(self, event_types: Iterable[str] | None = None) -> int:
        if event_types is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()
            return int(row["n"])
        types = tuple(event_types)
        placeholders = ",".join("?" for _ in types)
        row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM messages WHERE event_type IN ({placeholders})",
            types,
        ).fetchone()
        return int(row["n"])

    def search_messages(
        self,
        query: str,
        page: int = 0,
        page_size: int = 5,
        event_types: Iterable[str] = (EVENT_MESSAGE,),
    ) -> tuple[list[dict], int]:
        """Keyword (AND-of-terms, case-insensitive substring) search over recall.

        Returns ``(rows, total_matches)`` where rows are newest-first for the page.
        """
        terms = [t for t in query.split() if t]
        conditions = []
        params: list[object] = []
        for term in terms:
            conditions.append("LOWER(content) LIKE ?")
            params.append(f"%{term.lower()}%")

        types = tuple(event_types)
        if types:
            placeholders = ",".join("?" for _ in types)
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(types)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        total = int(
            self._conn.execute(
                f"SELECT COUNT(*) AS n FROM messages{where}", params
            ).fetchone()["n"]
        )
        rows = self._conn.execute(
            f"SELECT id, session_id, role, event_type, served_by, content, created_at "
            f"FROM messages{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, page_size, page * page_size),
        ).fetchall()
        return [dict(r) for r in rows], total

    def search_messages_by_date(
        self,
        start: str,
        end: str,
        page: int = 0,
        page_size: int = 5,
        event_types: Iterable[str] = (EVENT_MESSAGE,),
    ) -> tuple[list[dict], int]:
        """Search recall memory within an inclusive date range.

        ``start``/``end`` are ``YYYY-MM-DD`` (or full ``YYYY-MM-DD HH:MM:SS``
        timestamps); bare dates expand to cover the whole day. Raises
        ``ValueError`` on a malformed date so the caller can surface a
        correctable error instead of a silently empty result.
        """
        start_ts = _normalize_date_bound(start, is_end=False)
        end_ts = _normalize_date_bound(end, is_end=True)

        conditions = ["created_at >= ?", "created_at <= ?"]
        params: list[object] = [start_ts, end_ts]
        types = tuple(event_types)
        if types:
            placeholders = ",".join("?" for _ in types)
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(types)

        where = " WHERE " + " AND ".join(conditions)
        total = int(
            self._conn.execute(
                f"SELECT COUNT(*) AS n FROM messages{where}", params
            ).fetchone()["n"]
        )
        rows = self._conn.execute(
            f"SELECT id, session_id, role, event_type, served_by, content, created_at "
            f"FROM messages{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, page_size, page * page_size),
        ).fetchall()
        return [dict(r) for r in rows], total
