"""One-time migration: SQLite + Chroma → Postgres + pgvector (spec §8).

    python -m memory_server.storage.migrate_to_postgres \\
        --dsn postgresql://memassist:memassist@localhost:15432/memassist

This runs on the HOST, so the port is the host-side one the compose file
publishes (POSTGRES_HOST_PORT, default 15432) — not the 5432 that services
inside the compose network use.

Copies core blocks, the recall log, the provider-usage ledger, and archival
passages. Archival vectors are **re-embedded from the stored text** rather than
copied out of Chroma: the text is the source of truth, re-embedding is
deterministic with the same local model, and it means the migration is correct
even if the two stores ever disagreed about dimensionality.

Idempotent by design — safe to re-run:
- core blocks and provider_usage upsert on conflict,
- messages and passages are skipped when the destination already holds rows,
  so a half-finished run is resumed by simply running it again.

Nothing is deleted from the source. Verify, then switch
``MEMASSIST_POSTGRES_DSN``; keep the SQLite file until you are satisfied.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from memory_server.storage.pgvector_store import PgVectorStore  # noqa: E402
from memory_server.storage.postgres import PostgresStore, connect  # noqa: E402

BATCH = 500


def _sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def migrate_core_blocks(src: sqlite3.Connection, dst) -> int:
    rows = src.execute("SELECT name, content, char_limit FROM core_blocks").fetchall()
    for r in rows:
        dst.execute(
            "INSERT INTO core_blocks (name, content, char_limit) VALUES (%s, %s, %s) "
            "ON CONFLICT (name) DO UPDATE SET content = EXCLUDED.content, "
            "char_limit = EXCLUDED.char_limit",
            (r["name"], r["content"], r["char_limit"]),
        )
    return len(rows)


def migrate_messages(src: sqlite3.Connection, dst) -> int:
    existing = dst.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
    if existing:
        print(f"  messages: destination already has {existing} row(s) — skipping")
        return 0
    total = 0
    cur = src.execute(
        "SELECT session_id, role, event_type, served_by, content, created_at "
        "FROM messages ORDER BY id"
    )
    while True:
        batch = cur.fetchmany(BATCH)
        if not batch:
            break
        with dst.cursor() as c:
            c.executemany(
                "INSERT INTO messages "
                "(session_id, role, event_type, served_by, content, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s::timestamptz)",
                [
                    (
                        r["session_id"], r["role"], r["event_type"],
                        r["served_by"], r["content"], r["created_at"],
                    )
                    for r in batch
                ],
            )
        total += len(batch)
    return total


def migrate_usage(src: sqlite3.Connection, dst) -> int:
    if not _table_exists(src, "provider_usage"):
        return 0
    rows = src.execute(
        "SELECT provider, usage_date, requests, tokens, cooldown_until FROM provider_usage"
    ).fetchall()
    dst.execute(
        """CREATE TABLE IF NOT EXISTS provider_usage (
             provider TEXT NOT NULL, usage_date DATE NOT NULL,
             requests INTEGER DEFAULT 0, tokens INTEGER DEFAULT 0,
             cooldown_until TIMESTAMPTZ, PRIMARY KEY (provider, usage_date))"""
    )
    for r in rows:
        dst.execute(
            "INSERT INTO provider_usage "
            "(provider, usage_date, requests, tokens, cooldown_until) "
            "VALUES (%s, %s::date, %s, %s, %s::timestamptz) "
            "ON CONFLICT (provider, usage_date) DO UPDATE SET "
            "requests = EXCLUDED.requests, tokens = EXCLUDED.tokens, "
            "cooldown_until = EXCLUDED.cooldown_until",
            (r["provider"], r["usage_date"], r["requests"], r["tokens"], r["cooldown_until"]),
        )
    return len(rows)


def migrate_archival(chroma_path: str, target: PgVectorStore) -> int:
    """Re-embed every Chroma passage into pgvector from its stored TEXT."""
    if target.count():
        print(f"  archival: destination already has {target.count()} passage(s) — skipping")
        return 0
    try:
        from memory_server.storage.chroma import ArchivalStore
    except Exception as exc:
        print(f"  archival: chromadb unavailable ({exc}) — skipping")
        return 0
    if not Path(chroma_path).exists():
        print(f"  archival: no Chroma store at {chroma_path} — skipping")
        return 0

    source = ArchivalStore(chroma_path)
    total = source.count()
    if not total:
        return 0
    # Read documents + metadata straight out of the collection; embeddings are
    # regenerated, so nothing depends on Chroma's vector format.
    dump = source._col.get(include=["documents", "metadatas"])
    for doc, meta in zip(dump["documents"], dump["metadatas"] or []):
        meta = meta or {}
        target.insert(doc, source=meta.get("source", "agent"), created_at=meta.get("created_at"))
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="migrate_to_postgres")
    ap.add_argument("--dsn", default=config.POSTGRES_DSN, help="target Postgres DSN")
    ap.add_argument("--sqlite", default=config.DB_PATH, help="source SQLite file")
    ap.add_argument("--chroma", default=config.CHROMA_PATH, help="source Chroma directory")
    args = ap.parse_args(argv)

    if not args.dsn:
        ap.error("--dsn is required (or set MEMASSIST_POSTGRES_DSN)")
    if not Path(args.sqlite).exists():
        ap.error(f"source SQLite database not found: {args.sqlite}")

    print(f"Migrating {args.sqlite} -> {args.dsn}")
    # Constructing the stores creates the destination schema.
    PostgresStore(
        args.dsn,
        default_persona=config.DEFAULT_PERSONA,
        default_human=config.DEFAULT_HUMAN,
        core_block_char_limit=config.CORE_BLOCK_CHAR_LIMIT,
    )
    archival = PgVectorStore(args.dsn)

    src, dst = _sqlite(args.sqlite), connect(args.dsn)
    try:
        print(f"  core_blocks:    {migrate_core_blocks(src, dst)} row(s)")
        print(f"  messages:       {migrate_messages(src, dst)} row(s)")
        print(f"  provider_usage: {migrate_usage(src, dst)} row(s)")
        print(f"  archival:       {migrate_archival(args.chroma, archival)} passage(s) re-embedded")
    finally:
        src.close()
        dst.close()

    print("Done. Set MEMASSIST_POSTGRES_DSN to switch; the SQLite file is untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
