"""FastMCP server exposing the six memory tools (spec §7).

    python -m memory_server                     stdio  (Claude Code/Desktop)
    python -m memory_server --http              Streamable HTTP on :8090

stdio is right for a desktop client that owns the subprocess; HTTP is right
for docker-compose, where the memory server is its own service with its own
lifecycle and healthcheck and several clients may reach it. Same six tools,
same handlers — only the transport differs, which is exactly what the MCP
transport split is for.

``send_message`` is deliberately absent: it is the agent's channel to its own
user, not a memory operation, and it means nothing to an external client.

The handlers delegate to the same :class:`MemoryTools` the agent dispatches
in-process, so there is one implementation of each tool and no second copy to
drift.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Importable when launched as `python -m memory_server` from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

import config  # noqa: E402
from memory_server.memory_tools import MemoryTools  # noqa: E402
from memory_server.storage.chroma import ArchivalStore  # noqa: E402
from memory_server.storage.sqlite import SQLiteStore  # noqa: E402

mcp = FastMCP("memgpt-memory")

_memory: MemoryTools | None = None


def memory() -> MemoryTools:
    """Open storage on first use, so importing the module never touches the DB."""
    global _memory
    if _memory is None:
        store = SQLiteStore(
            config.DB_PATH,
            default_persona=config.DEFAULT_PERSONA,
            default_human=config.DEFAULT_HUMAN,
            core_block_char_limit=config.CORE_BLOCK_CHAR_LIMIT,
        )
        try:
            archival: ArchivalStore | None = ArchivalStore(config.CHROMA_PATH)
        except Exception:  # archival is optional; core + recall still work
            archival = None
        _memory = MemoryTools(
            store,
            archival,
            session_id="mcp",
            page_size=config.SEARCH_PAGE_SIZE,
            archival_top_k=config.ARCHIVAL_TOP_K,
            result_char_cap=config.TOOL_RESULT_CHAR_CAP,
        )
    return _memory


@mcp.tool()
def core_memory_append(block: str, content: str, source: str = "inferred") -> str:
    """Append a durable, high-value line to the 'persona' or 'human' core block."""
    return memory().dispatch(
        "core_memory_append", {"block": block, "content": content, "source": source}
    )


@mcp.tool()
def core_memory_replace(block: str, old_text: str, new_text: str) -> str:
    """Replace an exact substring in a core block (old_text must match verbatim)."""
    return memory().dispatch(
        "core_memory_replace",
        {"block": block, "old_text": old_text, "new_text": new_text},
    )


@mcp.tool()
def conversation_search(query: str, page: int = 0) -> str:
    """Keyword search over the full conversation history (recall memory)."""
    return memory().dispatch("conversation_search", {"query": query, "page": page})


@mcp.tool()
def conversation_search_date(start: str, end: str, page: int = 0) -> str:
    """Search recall memory within an inclusive 'YYYY-MM-DD' date range."""
    return memory().dispatch(
        "conversation_search_date", {"start": start, "end": end, "page": page}
    )


@mcp.tool()
def archival_memory_insert(content: str, source: str = "inferred") -> str:
    """Save a passage to long-term archival memory (a semantic vector store)."""
    return memory().dispatch(
        "archival_memory_insert", {"content": content, "source": source}
    )


@mcp.tool()
def archival_memory_search(query: str, top_k: int = 5, page: int = 0) -> str:
    """Semantic search over archival memory."""
    return memory().dispatch(
        "archival_memory_search", {"query": query, "top_k": top_k, "page": page}
    )


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request):
    """Liveness for docker-compose. Deliberately does NOT open storage —
    a health probe that touches the database turns a slow query into a
    restart loop."""
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok", "transport": "http"})


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="memory_server")
    ap.add_argument(
        "--http",
        action="store_true",
        help="serve Streamable HTTP instead of stdio (used by docker-compose)",
    )
    ap.add_argument("--host", default=os.getenv("MEMASSIST_MCP_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.getenv("MEMASSIST_MCP_PORT", "8090")))
    args = ap.parse_args(argv)

    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
