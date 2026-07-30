"""One storage suite, run against BOTH backends (spec §8, Phase 4).

Two implementations behind one surface only stay equivalent if something checks.
Every test here is parametrized over SQLite+Chroma and Postgres+pgvector, so a
behavioural drift fails rather than waiting to be discovered in production.

Postgres tests are skipped — never silently passed — when
``MEMASSIST_TEST_POSTGRES_DSN`` is unset, so a laptop without a database still
gets the SQLite half. CI sets it and runs both.
"""

from __future__ import annotations

import os
import uuid

import pytest

import config

PG_DSN = os.getenv("MEMASSIST_TEST_POSTGRES_DSN")
pg_only = pytest.mark.skipif(not PG_DSN, reason="MEMASSIST_TEST_POSTGRES_DSN not set")

TEST_CORE_LIMIT = 200


# --- backend fixtures -----------------------------------------------------
def _sqlite_backend(tmp_path):
    from memory_server.storage.chroma import ArchivalStore, hashing_embedding
    from memory_server.storage.sqlite import SQLiteStore

    store = SQLiteStore(
        str(tmp_path / "backend.db"),
        default_persona=config.DEFAULT_PERSONA,
        default_human=config.DEFAULT_HUMAN,
        core_block_char_limit=TEST_CORE_LIMIT,
    )
    archival = ArchivalStore(str(tmp_path / "chroma"), embed_fn=hashing_embedding)
    return store, archival


def _postgres_backend(tmp_path):
    from memory_server.storage.chroma import hashing_embedding
    from memory_server.storage.pgvector_store import PgVectorStore
    from memory_server.storage.postgres import PostgresStore, connect

    # A private schema per test: real isolation without a database per test.
    schema = f"t_{uuid.uuid4().hex[:12]}"
    admin = connect(PG_DSN)
    admin.execute(f'CREATE SCHEMA "{schema}"')
    admin.close()
    # public stays on the path: CREATE EXTENSION installs the `vector` TYPE
    # there, and a schema-only search_path cannot see it.
    dsn = f"{PG_DSN}?options=-csearch_path%3D{schema},public"

    store = PostgresStore(
        dsn,
        default_persona=config.DEFAULT_PERSONA,
        default_human=config.DEFAULT_HUMAN,
        core_block_char_limit=TEST_CORE_LIMIT,
    )
    # The hashing embedder is 512-d; pgvector fixes the column at 384, so the
    # test double is truncated to fit. Retrieval ORDER is what matters here —
    # semantic quality is the benchmark's job (T5), with the real model.
    archival = PgVectorStore(dsn, embed_fn=lambda t: hashing_embedding(t)[:384])
    store._schema_name = schema
    return store, archival


BACKENDS = [
    pytest.param(_sqlite_backend, id="sqlite+chroma"),
    pytest.param(_postgres_backend, id="postgres+pgvector", marks=pg_only),
]


@pytest.fixture(params=BACKENDS)
def backend(request, tmp_path):
    store, archival = request.param(tmp_path)
    yield store, archival
    try:
        store.close()
    except Exception:
        pass


@pytest.fixture
def store(backend):
    return backend[0]


@pytest.fixture
def archival(backend):
    return backend[1]


# --- core memory ----------------------------------------------------------
def test_core_blocks_seed_on_first_open(store):
    blocks = store.get_core_blocks()
    assert set(blocks) == {"persona", "human"}
    assert blocks["persona"].strip()


def test_append_and_replace(store):
    store.append_block("human", "Name: Ada.")
    assert "Ada" in store.get_block("human")
    store.replace_block("human", "Ada", "Alicia")
    assert "Alicia" in store.get_block("human")
    assert "Ada" not in store.get_block("human")


def test_char_limit_raises(store):
    with pytest.raises(ValueError, match="character limit"):
        store.append_block("human", "x" * (TEST_CORE_LIMIT + 1))


def test_replace_miss_raises(store):
    with pytest.raises(ValueError, match="not found"):
        store.replace_block("human", "nothing like this", "x")


def test_unknown_block_raises(store):
    with pytest.raises(ValueError, match="Unknown core memory block"):
        store.get_block("nonexistent")


# --- recall memory --------------------------------------------------------
def test_add_and_count_messages(store):
    store.add_message("s", "user", "message", "hello world")
    store.add_message("s", "assistant", "message", "hi there", served_by="groq")
    assert store.count_messages(event_types=("message",)) == 2


def test_keyword_search_is_and_of_terms(store):
    store.add_message("s", "user", "message", "coffee in the morning")
    store.add_message("s", "user", "message", "tea in the morning")
    rows, total = store.search_messages("coffee morning")
    assert total == 1
    assert "coffee" in rows[0]["content"]


def test_search_returns_served_by_and_iso_created_at(store):
    store.add_message("s", "assistant", "message", "Lisbon flight booked", served_by="groq")
    rows, _ = store.search_messages("Lisbon")
    assert rows[0]["served_by"] == "groq"
    # Both backends must hand back the SAME timestamp shape.
    assert len(rows[0]["created_at"]) >= 19
    assert rows[0]["created_at"][4] == "-" and rows[0]["created_at"][10] == " "


def test_search_pagination_is_newest_first(store):
    for i in range(7):
        store.add_message("s", "user", "message", f"note number {i}")
    p0, total = store.search_messages("note", page=0, page_size=3)
    p1, _ = store.search_messages("note", page=1, page_size=3)
    assert total == 7
    assert len(p0) == 3 and len(p1) == 3
    assert p0[0]["content"] == "note number 6"
    assert {r["content"] for r in p0}.isdisjoint({r["content"] for r in p1})


def test_date_search_bounds_and_validation(store):
    store.add_message("s", "user", "message", "dated entry", created_at="2026-05-05 12:00:00")
    rows, total = store.search_messages_by_date("2026-05-05", "2026-05-05")
    assert total == 1 and "dated entry" in rows[0]["content"]

    _, outside = store.search_messages_by_date("2026-05-06", "2026-05-07")
    assert outside == 0

    with pytest.raises(ValueError, match="Invalid date"):
        store.search_messages_by_date("not-a-date", "2026-05-05")


def test_event_type_filter_excludes_tool_traffic(store):
    store.add_message("s", "user", "message", "findme in a message")
    store.add_message("s", "tool", "tool_result", "findme in a tool result")
    _, total = store.search_messages("findme")
    assert total == 1, "recall search must default to user-visible messages"


# --- archival memory ------------------------------------------------------
def test_archival_insert_search_count(archival):
    assert archival.count() == 0
    archival.insert("The user signed the lease on 3 March.", source="stated")
    archival.insert("The user dislikes herbal tea.", source="inferred")
    assert archival.count() == 2

    items, total = archival.search("lease", top_k=1)
    assert total == 2 and len(items) == 1
    assert "lease" in items[0]["content"]


def test_archival_records_provenance(archival):
    archival.insert("From a web page.", source="external")
    items, _ = archival.search("web page", top_k=1)
    assert items[0]["source"] == "external"
    assert items[0]["created_at"]


def test_archival_search_on_empty_store(archival):
    assert archival.search("anything") == ([], 0)


def test_archival_paging(archival):
    for i in range(5):
        archival.insert(f"passage number {i}")
    page0, total = archival.search("passage", top_k=2, page=0)
    page1, _ = archival.search("passage", top_k=2, page=1)
    assert total == 5
    assert len(page0) == 2 and len(page1) == 2
    assert {i["content"] for i in page0}.isdisjoint({i["content"] for i in page1})


# --- the budget ledger ----------------------------------------------------
@pytest.mark.parametrize(
    "dsn_factory",
    [
        pytest.param(lambda tmp: str(tmp / "ledger.db"), id="sqlite"),
        pytest.param(lambda tmp: PG_DSN, id="postgres", marks=pg_only),
    ],
)
def test_ledger_records_and_cools_on_both_backends(dsn_factory, tmp_path):
    from datetime import datetime, timezone

    from llm.budgets import BudgetLedger

    provider = f"p_{uuid.uuid4().hex[:8]}"  # unique: Postgres is shared here
    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    ledger = BudgetLedger(dsn_factory(tmp_path), now_fn=lambda: now)

    assert ledger.get_usage(provider) == (0, 0)
    ledger.record_request(provider, tokens=120)
    ledger.record_request(provider, tokens=80)
    assert ledger.get_usage(provider) == (2, 200)

    assert not ledger.is_cooling(provider)
    ledger.cooldown_for_seconds(provider, 60)
    assert ledger.is_cooling(provider)
    assert 0 < ledger.cooldown_remaining(provider) <= 60

    # A new UTC day is a new row: usage resets and the cooldown is gone.
    ledger._now = lambda: datetime(2026, 7, 31, 0, 30, tzinfo=timezone.utc)
    assert ledger.get_usage(provider) == (0, 0)
    assert not ledger.is_cooling(provider)
    ledger.close()
