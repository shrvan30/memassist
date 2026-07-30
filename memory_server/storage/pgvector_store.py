"""Archival memory on Postgres + pgvector (spec §8, Phase 4).

Same surface as :class:`chroma.ArchivalStore`, so ``MemoryTools`` cannot tell
them apart. Vectors are ``vector(384)`` to match bge-small-en-v1.5 — the
dimension is fixed in the schema, exactly as a Chroma collection fixes it at
creation, so changing embedder still means a re-embed migration.

Distance is cosine (``<=>``), matching Chroma's ``hnsw:space: cosine``, so a
passage that ranked first on one backend ranks first on the other.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Callable

from security.sensitivity import is_sensitive

from . import embedder
from .postgres import connect

EMBED_DIM = 384

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS archival_passages (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL,
    source      TEXT NOT NULL DEFAULT 'agent',
    sensitive   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TEXT NOT NULL
);

-- Cosine index. Only pays off past a few thousand rows; harmless before that.
CREATE INDEX IF NOT EXISTS idx_archival_embedding
    ON archival_passages USING hnsw (embedding vector_cosine_ops);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class PgVectorStore:
    """Archival memory backed by pgvector."""

    def __init__(
        self,
        dsn: str,
        embed_fn: Callable[[str], list[float]] | None = None,
        table: str = "archival_passages",
    ) -> None:
        self._embed = embed_fn or embedder.embed
        self._table = table
        self._conn = connect(dsn)
        self._lock = threading.Lock()
        from pgvector.psycopg import register_vector

        with self._lock:
            self._conn.execute(_SCHEMA)
        register_vector(self._conn)

    def insert(
        self,
        content: str,
        source: str = "agent",
        created_at: str | None = None,
        sensitive: bool | None = None,
    ) -> str:
        """See ``ArchivalStore.insert`` — ``sensitive=None`` detects, and is the
        default so a caller that forgets the flag fails closed."""
        passage_id = uuid.uuid4().hex
        if sensitive is None:
            sensitive = is_sensitive(content)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {self._table} "
                "(id, content, embedding, source, sensitive, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    passage_id,
                    content,
                    self._embed(content),
                    source,
                    bool(sensitive),
                    created_at or _utc_now(),
                ),
            )
        return passage_id

    def search(self, query: str, top_k: int = 5, page: int = 0) -> tuple[list[dict], int]:
        """Semantic search. Returns ``(page_items, total_passages)``.

        Paged in SQL rather than sliced in Python (Chroma has no offset), which
        is the one place the pgvector backend is genuinely better.
        """
        total = self.count()
        if total == 0:
            return [], 0
        rows = self._conn.execute(
            f"SELECT id, content, source, sensitive, created_at, "
            f"embedding <=> %s::vector AS distance "
            f"FROM {self._table} ORDER BY distance, id LIMIT %s OFFSET %s",
            (self._embed(query), top_k, page * top_k),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "content": r["content"],
                "created_at": r["created_at"],
                "source": r["source"],
                "sensitive": bool(r["sensitive"]),
                "distance": float(r["distance"]),
            }
            for r in rows
        ], total

    def count(self) -> int:
        return int(self._conn.execute(f"SELECT COUNT(*) AS n FROM {self._table}").fetchone()["n"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
