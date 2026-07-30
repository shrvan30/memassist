"""Composition root: wire storage + memory tools + router + loop from config.

Kept separate from the Streamlit app so the same wiring is reusable (a smoke
test, the future MCP server, etc.). Archival memory is optional — if Chroma fails
to initialize, the assistant still runs with core + recall memory.
"""

from __future__ import annotations

import config
import llm
from agent.loop import AgentLoop
from mcp_client import ExternalTools, build_external_tools
from memory_server.memory_tools import MemoryTools
from memory_server.schemas import ALL_TOOLS
from memory_server.storage.chroma import ArchivalStore
from memory_server.storage.sqlite import SQLiteStore


def build_memory(
    db_path: str | None = None,
    chroma_path: str | None = None,
    session_id: str = "default",
) -> MemoryTools:
    store = SQLiteStore(
        db_path or config.DB_PATH,
        default_persona=config.DEFAULT_PERSONA,
        default_human=config.DEFAULT_HUMAN,
        core_block_char_limit=config.CORE_BLOCK_CHAR_LIMIT,
    )
    try:
        archival: ArchivalStore | None = ArchivalStore(chroma_path or config.CHROMA_PATH)
    except Exception:  # pragma: no cover - archival is optional; degrade gracefully
        archival = None
    return MemoryTools(
        store,
        archival,
        session_id=session_id,
        page_size=config.SEARCH_PAGE_SIZE,
        archival_top_k=config.ARCHIVAL_TOP_K,
        result_char_cap=config.TOOL_RESULT_CHAR_CAP,
    )


def build_router(db_path: str | None = None) -> llm.Router:
    # The provider-usage ledger shares the main SQLite DB (provider_usage table).
    return llm.build_default_router(
        db_path=db_path or config.DB_PATH,
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
    )
