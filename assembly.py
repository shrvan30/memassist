"""Composition root: wire storage + memory tools + router + loop from config.

Kept separate from the Streamlit app so the same wiring is reusable (a smoke
test, the future MCP server, etc.). Archival memory is optional — if Chroma fails
to initialize, the assistant still runs with core + recall memory.
"""

from __future__ import annotations

from typing import Any

import config
import llm
from agent.loop import AgentLoop
from mcp_client import ExternalTools, build_external_tools
from memory_server.memory_tools import MemoryTools
from memory_server.schemas import ALL_TOOLS
from memory_server.storage.chroma import ArchivalStore
from memory_server.storage.sqlite import SQLiteStore


def build_stores(
    db_path: str | None = None,
    chroma_path: str | None = None,
    backend: str | None = None,
    dsn: str | None = None,
):
    """Open the core/recall store and the archival store for the active backend.

    The two backends are siblings behind the same surface (spec §8), so this is
    the only place in the codebase that knows which one is running.
    """
    backend = (backend or config.STORAGE_BACKEND).lower()
    if backend == "postgres":
        from memory_server.storage.pgvector_store import PgVectorStore
        from memory_server.storage.postgres import PostgresStore

        dsn = dsn or config.POSTGRES_DSN
        if not dsn:
            raise RuntimeError(
                "MEMASSIST_STORAGE_BACKEND=postgres but MEMASSIST_POSTGRES_DSN is unset."
            )
        store = PostgresStore(
            dsn,
            default_persona=config.DEFAULT_PERSONA,
            default_human=config.DEFAULT_HUMAN,
            core_block_char_limit=config.CORE_BLOCK_CHAR_LIMIT,
        )
        try:
            archival = PgVectorStore(dsn)
        except Exception:  # archival is optional; core + recall still work
            archival = None
        return store, archival

    store = SQLiteStore(
        db_path or config.DB_PATH,
        default_persona=config.DEFAULT_PERSONA,
        default_human=config.DEFAULT_HUMAN,
        core_block_char_limit=config.CORE_BLOCK_CHAR_LIMIT,
    )
    try:
        archival = ArchivalStore(chroma_path or config.CHROMA_PATH)
    except Exception:  # pragma: no cover - archival is optional; degrade gracefully
        archival = None
    return store, archival


def build_memory(
    db_path: str | None = None,
    chroma_path: str | None = None,
    session_id: str = "default",
    backend: str | None = None,
    dsn: str | None = None,
) -> MemoryTools:
    store, archival = build_stores(db_path, chroma_path, backend, dsn)
    return MemoryTools(
        store,
        archival,
        session_id=session_id,
        page_size=config.SEARCH_PAGE_SIZE,
        archival_top_k=config.ARCHIVAL_TOP_K,
        result_char_cap=config.TOOL_RESULT_CHAR_CAP,
    )


# One saver per DSN per process. Every session's loop shares it: the pool is
# the expensive part, and the thread id is what keeps sessions apart.
_checkpointers: dict[str, Any] = {}


def build_checkpointer(backend: str | None = None, dsn: str | None = None):
    """Open the checkpointer for the active backend — the stores' sibling.

    Turn state was the last thing still living only in process memory. Core,
    recall and archival memory were already in a database, so an API restart
    lost nothing durable — but it lost every *suspended* turn, and a turn
    suspended on a human approval is precisely the one that is likely to
    outlive the process it started in.

    Pairing the checkpointer with the storage backend keeps that a single
    decision: choose Postgres and turn state persists with everything else.
    """
    backend = (backend or config.STORAGE_BACKEND).lower()
    if backend != "postgres":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    dsn = dsn or config.POSTGRES_DSN
    if not dsn:
        raise RuntimeError(
            "MEMASSIST_STORAGE_BACKEND=postgres but MEMASSIST_POSTGRES_DSN is unset."
        )
    saver = _checkpointers.get(dsn)
    if saver is None:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        # PostgresSaver requires autocommit (it runs its own migrations) and
        # dict rows. A pool rather than one connection because a turn runs on
        # an SSE worker thread while the request thread reads state.
        pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=int(config.CHECKPOINTER_POOL_MAX),
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=True,
        )
        saver = PostgresSaver(pool)
        saver.setup()  # idempotent DDL, same contract as the storage migration
        _checkpointers[dsn] = saver
    return saver


def close_checkpointers() -> None:
    """Close the checkpointer pools. Called on API shutdown.

    A pool keeps worker threads alive, so leaving it open makes a clean process
    exit hang rather than exit.
    """
    for saver in _checkpointers.values():
        conn = getattr(saver, "conn", None)
        if hasattr(conn, "close"):
            conn.close()
    _checkpointers.clear()


def ledger_dsn(db_path: str | None = None) -> str:
    """Where the provider-usage ledger lives — same store as everything else.

    In a container the SQLite file is ephemeral, so a Postgres deployment that
    left the ledger on disk would hand the router a fresh, empty budget on every
    restart and cheerfully re-spend a free tier it had already exhausted.
    """
    if db_path:
        return db_path
    if config.STORAGE_BACKEND == "postgres" and config.POSTGRES_DSN:
        return config.POSTGRES_DSN
    return config.DB_PATH


def build_router(db_path: str | None = None) -> llm.Router:
    return llm.build_default_router(
        db_path=ledger_dsn(db_path),
        providers_path=config.PROVIDERS_YAML,
        temperature=config.TEMPERATURE,
    )


def build_tools(enabled: bool | None = None) -> ExternalTools:
    """Start the external MCP servers listed in ``mcp_servers.yaml`` (spec §5.1).

    Never fatal: a missing ``uvx`` or a server that will not boot costs the
    assistant a capability, not a startup.
    """
    if not (config.EXTERNAL_TOOLS if enabled is None else enabled):
        return ExternalTools({})
    return build_external_tools(config.MCP_SERVERS_YAML)


def build_loop(
    memory: MemoryTools | None = None,
    router: "llm.Router | None" = None,
    external: ExternalTools | None = None,
    session_id: str | None = None,
) -> AgentLoop:
    external = build_tools() if external is None else external
    return AgentLoop(
        router or build_router(),
        memory or build_memory(),
        [*ALL_TOOLS, *external.tool_definitions()],
        planning_context_limit=config.CONTEXT_LIMIT,
        pressure_threshold=config.PRESSURE_THRESHOLD,
        max_heartbeats=config.MAX_HEARTBEATS,
        external=external,
        checkpointer=build_checkpointer(),
        # The session id IS the thread id, so a restarted process addresses the
        # same thread. Without it a rebuilt loop would get a fresh uuid and the
        # persisted checkpoint would be unreachable — durable but unaddressable.
        thread_id=session_id,
    )
