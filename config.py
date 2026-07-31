"""Central configuration for MemAssist.

All values can be overridden via environment variables (see ``.env.example``).
Kept dependency-light so every module can import it without pulling in heavy
packages.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # Load .env if python-dotenv is available; harmless if it isn't.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at import time
    pass

ROOT = Path(__file__).resolve().parent

# --- LLM router (free-tier failover; see llm/router.py) ------------------
# A fixed temperature across every provider keeps the assistant's voice stable
# when the serving model changes mid-conversation.
TEMPERATURE: float = float(os.getenv("MEMASSIST_TEMPERATURE", "0.3"))
PROVIDERS_YAML: str = os.getenv(
    "MEMASSIST_PROVIDERS_YAML", str(ROOT / "llm" / "providers.yaml")
)
# Provider keys are read by the router directly from these env vars; mirrored
# here for visibility/debug. At least one must be set for the app to run.
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
MISTRAL_API_KEY: str | None = os.getenv("MISTRAL_API_KEY")

# --- Context / memory-pressure policy ------------------------------------
# Pressure is measured against the ACTIVE provider's context window, capped by
# this planning limit (the smallest window in the chain is the safe default).
# Set to 0 to disable the cap and use the real provider window.
CONTEXT_LIMIT: int = int(os.getenv("MEMASSIST_CONTEXT_LIMIT", "32000"))
PRESSURE_THRESHOLD: float = float(os.getenv("MEMASSIST_PRESSURE_THRESHOLD", "0.7"))

# Max number of chained tool-call rounds ("heartbeats") per user turn. Each
# heartbeat is a real API request and counts against provider budgets.
MAX_HEARTBEATS: int = int(os.getenv("MEMASSIST_MAX_HEARTBEATS", "5"))

# --- Memory limits -------------------------------------------------------
CORE_BLOCK_CHAR_LIMIT: int = int(os.getenv("MEMASSIST_CORE_BLOCK_CHAR_LIMIT", "2000"))
ARCHIVAL_TOP_K: int = int(os.getenv("MEMASSIST_ARCHIVAL_TOP_K", "5"))
SEARCH_PAGE_SIZE: int = int(os.getenv("MEMASSIST_SEARCH_PAGE_SIZE", "5"))
# All tool results are truncated to this many characters before being handed
# back to the model (defensive: results are treated as untrusted text).
TOOL_RESULT_CHAR_CAP: int = int(os.getenv("MEMASSIST_TOOL_RESULT_CHAR_CAP", "4000"))

# --- External MCP tools (spec §5) ----------------------------------------
# The registry of external servers. Set MEMASSIST_EXTERNAL_TOOLS=0 to run with
# memory tools only (no subprocesses) — the benchmark and tests do exactly that
# by never passing an ExternalTools instance.
MCP_SERVERS_YAML: str = os.getenv("MEMASSIST_MCP_SERVERS_YAML", str(ROOT / "mcp_servers.yaml"))
EXTERNAL_TOOLS: bool = os.getenv("MEMASSIST_EXTERNAL_TOOLS", "1") not in ("0", "false", "False")

# --- API service (Phase 4) -----------------------------------------------
API_DEFAULT_SESSION: str = os.getenv("MEMASSIST_DEFAULT_SESSION", "default")
# Browsers block cross-origin SSE without this; the Next.js dev server and the
# compose web service are the two origins that legitimately call the API.
API_CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv(
        "MEMASSIST_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]

# --- Storage backend (spec §8) -------------------------------------------
# "sqlite" -> SQLite + Chroma (zero-setup default, what the benchmark uses)
# "postgres" -> Postgres + pgvector, one DSN for all four tables
# Setting MEMASSIST_POSTGRES_DSN alone is enough; the backend follows it.
POSTGRES_DSN: str | None = os.getenv("MEMASSIST_POSTGRES_DSN") or None
STORAGE_BACKEND: str = os.getenv(
    "MEMASSIST_STORAGE_BACKEND", "postgres" if POSTGRES_DSN else "sqlite"
).lower()
# The graph checkpointer follows the storage backend (see assembly). Its pool
# is shared by every session, so it is sized by concurrent turns, not sessions.
CHECKPOINTER_POOL_MAX: int = int(os.getenv("MEMASSIST_CHECKPOINTER_POOL_MAX", "8"))

# --- Storage locations ---------------------------------------------------
DB_PATH: str = os.getenv("MEMASSIST_DB_PATH", str(ROOT / "data" / "memassist.db"))
CHROMA_PATH: str = os.getenv("MEMASSIST_CHROMA_PATH", str(ROOT / "data" / "chroma"))

# --- Default core-memory seeds -------------------------------------------
DEFAULT_PERSONA: str = (
    "I am MemAssist, a personal AI assistant with long-term memory modeled on "
    "the MemGPT architecture. I remember facts about my user across "
    "conversations by editing my own memory with tool calls. I am warm, "
    "concise, and proactive about saving important details the user shares."
)
DEFAULT_HUMAN: str = "(I do not know anything about the user yet.)"
