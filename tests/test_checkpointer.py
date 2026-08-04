"""Durable turn state — the checkpointer paired with the storage backend.

The §9 gap: core, recall and archival memory were already in a database, so an
API restart lost nothing *saved* — but turn state lived in an ``InMemorySaver``,
so it lost every turn still in flight. The turn most likely to be in flight
across a restart is exactly the one suspended on a human approval, which can sit
there for as long as the human takes.

The interesting test is therefore not "does state persist" but "does a SUSPENDED
turn resume in a process that never saw it suspend". Process death is simulated
honestly: the saver cache is dropped and a brand-new saver and loop are built,
so nothing in memory carries over — only the row in Postgres does.

Postgres tests skip (never silently pass) without ``MEMASSIST_TEST_POSTGRES_DSN``.
"""

from __future__ import annotations

import os
import uuid

import pytest

import assembly
import config
from agent.loop import AgentLoop
from tests.test_loop import FakeRouter, result, tool

PG_DSN = os.getenv("MEMASSIST_TEST_POSTGRES_DSN")
pg_only = pytest.mark.skipif(not PG_DSN, reason="MEMASSIST_TEST_POSTGRES_DSN not set")

WRITE_CALL = '{"path":"notes.txt","content":"hello"}'


@pytest.fixture(autouse=True)
def _close_pools():
    """A ConnectionPool keeps worker threads alive, so an unclosed one makes the
    interpreter hang on exit instead of exiting. Same call the API makes."""
    yield
    assembly.close_checkpointers()


def build(mem, scripted, external, **kw):
    return AgentLoop(
        FakeRouter(scripted),
        mem,
        tools=[],
        planning_context_limit=100_000,
        pressure_threshold=0.7,
        max_heartbeats=5,
        external=external,
        **kw,
    )


def a_write():
    return [result(tool_calls=[tool("write_file", WRITE_CALL)])]


def a_reply():
    return [result(tool_calls=[tool("send_message", '{"text":"all done"}', id="call_2")])]


# --- backend pairing ------------------------------------------------------
def test_sqlite_backend_gets_an_in_memory_saver():
    from langgraph.checkpoint.memory import InMemorySaver

    assert isinstance(assembly.build_checkpointer("sqlite"), InMemorySaver)


def test_postgres_backend_without_a_dsn_is_an_error_not_a_silent_downgrade(monkeypatch):
    """Falling back to InMemorySaver here would be the §9 gap, reintroduced
    silently: durability configured, durability not delivered, nothing said.

    Absence has to be constructed, not assumed. The variable alone is not
    enough: ``config`` reads it once at import into a module-level constant,
    and ``build_checkpointer`` falls back to *that*, so a developer with a DSN
    in their .env — or a container that sets one — failed this test for having
    a working configuration. Same fix as the router's resolver tests: make the
    condition under test true here rather than inheriting it from the shell.
    """
    monkeypatch.delenv("MEMASSIST_POSTGRES_DSN", raising=False)
    monkeypatch.setattr(config, "POSTGRES_DSN", None)

    with pytest.raises(RuntimeError, match="MEMASSIST_POSTGRES_DSN"):
        assembly.build_checkpointer("postgres", dsn=None)


@pg_only
def test_postgres_backend_gets_a_durable_saver_and_reuses_the_pool():
    from langgraph.checkpoint.postgres import PostgresSaver

    a = assembly.build_checkpointer("postgres", PG_DSN)
    b = assembly.build_checkpointer("postgres", PG_DSN)
    assert isinstance(a, PostgresSaver)
    assert a is b, "one pool per DSN per process, shared by every session"


# --- the thread id is the address ----------------------------------------
def test_thread_id_defaults_to_a_fresh_one_but_is_taken_when_given(mem):
    anon = build(mem, [], None)
    named = build(mem, [], None, thread_id="session-42")
    assert named.thread_id == "session-42"
    assert anon.thread_id != named.thread_id


def test_reset_clears_the_window_and_any_pending_approval(mem, make_external):
    ext = make_external(tools=("write_file",), gated=("write_file",), jail="./workspace")
    loop = build(mem, a_write() + a_reply(), ext)
    loop.step("save a note")
    assert loop.pending_approval is not None

    loop.reset()

    assert loop.messages == []
    assert loop.pending_approval is None, "a deleted thread holds no interrupt"


# --- the actual §9 fix ----------------------------------------------------
@pg_only
def test_a_suspended_turn_resumes_after_the_process_that_suspended_it_dies(
    mem, make_external
):
    thread = f"durability-{uuid.uuid4().hex[:12]}"
    ext = make_external(
        tools=("write_file",), gated=("write_file",), jail="./workspace", result="written"
    )

    # --- process 1: ask for a gated write, then die while suspended --------
    loop = build(
        mem, a_write(), ext, checkpointer=assembly.build_checkpointer("postgres", PG_DSN),
        thread_id=thread,
    )
    assert loop.step("save a note") == [], "nothing delivered while suspended"
    assert loop.pending_approval is not None
    assert ext.calls == [], "the write must not run before a human answers"

    del loop
    assembly.close_checkpointers()  # nothing in memory survives the "restart"

    # --- process 2: a new saver, a new loop, the same thread ---------------
    revived = build(
        mem, a_reply(), ext, checkpointer=assembly.build_checkpointer("postgres", PG_DSN),
        thread_id=thread,
    )

    pending = revived.pending_approval
    assert pending is not None, "the suspended turn did not survive the restart"
    assert pending["actions"][0]["arguments"]["path"] == "notes.txt"

    out = revived.resume(approved=True)

    assert out == ["all done"], "the turn must finish, not restart"
    assert ext.calls == [("write_file", {"path": "notes.txt", "content": "hello"})]
    assert revived.pending_approval is None
    revived.reset()  # leave no rows behind for the next run


@pg_only
def test_the_in_context_window_survives_a_restart_too(mem):
    thread = f"durability-{uuid.uuid4().hex[:12]}"
    saver = assembly.build_checkpointer("postgres", PG_DSN)

    loop = build(mem, a_reply(), None, checkpointer=saver, thread_id=thread)
    loop.step("hello")
    before = len(loop.messages)
    assert before > 0

    del loop
    assembly.close_checkpointers()

    revived = build(
        mem, [], None,
        checkpointer=assembly.build_checkpointer("postgres", PG_DSN),
        thread_id=thread,
    )
    assert len(revived.messages) == before, "the FIFO must not restart empty"
    revived.reset()


@pg_only
def test_two_sessions_on_one_saver_do_not_see_each_other(mem):
    saver = assembly.build_checkpointer("postgres", PG_DSN)
    a = build(mem, a_reply(), None, checkpointer=saver, thread_id=f"a-{uuid.uuid4().hex[:8]}")
    b = build(mem, [], None, checkpointer=saver, thread_id=f"b-{uuid.uuid4().hex[:8]}")

    a.step("hello")

    assert a.messages and b.messages == [], "thread id is what keeps sessions apart"
    a.reset()
