"""Postgres storage for core memory and recall memory (spec §8, Phase 4).

A sibling of :mod:`sqlite`, not a replacement: both implement the same surface
and ``assembly`` picks one from config, so nothing above the storage layer knows
which is underneath. ``tests/test_storage_backends.py`` runs the same suite
against both, which is what keeps them honest.

Differences from the SQLite version are all dialect, never behaviour:
``?`` → ``%s``, ``INTEGER PRIMARY KEY`` → ``BIGSERIAL``, ``INSERT OR IGNORE`` →
``ON CONFLICT DO NOTHING``, and ``created_at`` is a real ``timestamptz`` rather
than a string, so it is formatted back to ``YYYY-MM-DD HH:MM:SS`` on read to
keep the two backends' rows identical.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable

from .sqlite import VALID_BLOCKS, _normalize_date_bound

_SCHEMA = """
CREATE TABLE IF NOT EXISTS core_blocks (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT UNIQUE CHECK(name IN ('persona','human')),
    content     TEXT NOT NULL,
    char_limit  INTEGER DEFAULT 2000,
    updated_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    event_type  TEXT,
    served_by   TEXT,
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_event ON messages(event_type);
"""

_TS_FMT = "YYYY-MM-DD HH24:MI:SS"
# Selected everywhere rows are returned, so both backends hand callers the same
# string shape for created_at (SQLite stores text; Postgres stores timestamptz).
_ROW_COLUMNS = (
    "id, session_id, role, event_type, served_by, content, "
    f"to_char(created_at, '{_TS_FMT}') AS created_at"
)


def connect(dsn: str):
    """Open a psycopg connection with the vector type registered."""
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    return conn


class PostgresStore:
    """Core + recall memory on Postgres. Same surface as :class:`SQLiteStore`."""

    def __init__(
        self,
        dsn: str,
        default_persona: str,
        default_human: str,
        core_block_char_limit: int = 2000,
    ) -> None:
        self.dsn = dsn
        self.core_block_char_limit = core_block_char_limit
        self._conn = connect(dsn)
        self._lock = threading.Lock()
        self._init_schema()
        self._seed_core_blocks(default_persona, default_human)

    # -- setup -------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(_SCHEMA)

    def _seed_core_blocks(self, persona: str, human: str) -> None:
        with self._lock:
            for name, content in (("persona", persona), ("human", human)):
                self._conn.execute(
                    "INSERT INTO core_blocks (name, content, char_limit) "
                    "VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                    (name, content, self.core_block_char_limit),
                )

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

    def _one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        return self._conn.execute(sql, params).fetchone()

    def get_block(self, name: str) -> str:
        self._check_block(name)
        row = self._one("SELECT content FROM core_blocks WHERE name = %s", (name,))
        return row["content"] if row else ""

    def get_block_limit(self, name: str) -> int:
        self._check_block(name)
        row = self._one("SELECT char_limit FROM core_blocks WHERE name = %s", (name,))
        return row["char_limit"] if row else self.core_block_char_limit

    def get_core_blocks(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT name, content FROM core_blocks").fetchall()
        blocks = {r["name"]: r["content"] for r in rows}
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
                "UPDATE core_blocks SET content = %s, "
                "updated_at = CURRENT_TIMESTAMP WHERE name = %s",
                (content, name),
            )

    def append_block(self, name: str, content: str) -> str:
        self._check_block(name)
        current = self.get_block(name)
        new_content = f"{current}\n{content}".strip() if current.strip() else content.strip()
        self._set_block(name, new_content)
        return new_content

    def replace_block(self, name: str, old_text: str, new_text: str) -> str:
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
                row = self._conn.execute(
                    "INSERT INTO messages (session_id, role, event_type, served_by, content) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (session_id, role, event_type, served_by, content),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "INSERT INTO messages "
                    "(session_id, role, event_type, served_by, content, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                    (session_id, role, event_type, served_by, content, created_at),
                ).fetchone()
            return int(row["id"])

    def count_messages(self, event_types: Iterable[str] | None = None) -> int:
        if event_types is None:
            return int(self._one("SELECT COUNT(*) AS n FROM messages")["n"])
        types = tuple(event_types)
        placeholders = ",".join("%s" for _ in types)
        return int(
            self._one(
                f"SELECT COUNT(*) AS n FROM messages WHERE event_type IN ({placeholders})",
                types,
            )["n"]
        )

    def _paged(self, where: str, params: list, page: int, page_size: int):
        total = int(self._one(f"SELECT COUNT(*) AS n FROM messages{where}", tuple(params))["n"])
        rows = self._conn.execute(
            f"SELECT {_ROW_COLUMNS} FROM messages{where} "
            "ORDER BY id DESC LIMIT %s OFFSET %s",
            (*params, page_size, page * page_size),
        ).fetchall()
        return [dict(r) for r in rows], total

    def search_messages(
        self,
        query: str,
        page: int = 0,
        page_size: int = 5,
        event_types: Iterable[str] = ("message",),
    ) -> tuple[list[dict], int]:
        conditions: list[str] = []
        params: list[object] = []
        for term in (t for t in query.split() if t):
            conditions.append("LOWER(content) LIKE %s")
            params.append(f"%{term.lower()}%")

        types = tuple(event_types)
        if types:
            conditions.append(f"event_type IN ({','.join('%s' for _ in types)})")
            params.extend(types)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        return self._paged(where, params, page, page_size)

    def search_messages_by_date(
        self,
        start: str,
        end: str,
        page: int = 0,
        page_size: int = 5,
        event_types: Iterable[str] = ("message",),
    ) -> tuple[list[dict], int]:
        conditions = ["created_at >= %s::timestamptz", "created_at <= %s::timestamptz"]
        params: list[object] = [
            _normalize_date_bound(start, is_end=False),
            _normalize_date_bound(end, is_end=True),
        ]
        types = tuple(event_types)
        if types:
            conditions.append(f"event_type IN ({','.join('%s' for _ in types)})")
            params.extend(types)

        return self._paged(" WHERE " + " AND ".join(conditions), params, page, page_size)
