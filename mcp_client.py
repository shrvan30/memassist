"""External MCP tools, loaded from ``mcp_servers.yaml`` (spec §5).

``MultiServerMCPClient`` is async and every consumer here — the graph, the
benchmark, Streamlit — is sync, so the client runs on one background event loop
owned by :class:`ExternalTools`. One loop for the process, rather than
``asyncio.run`` per call, because each call would otherwise re-spawn the
server subprocess (``uvx`` costs seconds).

The registry's ``trust`` field is the whole point of this module: the graph asks
:meth:`ExternalTools.trust_of` which zone a result came from, and anything
``untrusted`` must go through ``security/sanitizer.py`` before it reaches the
model (spec §6.1).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY = Path(__file__).with_name("mcp_servers.yaml")

TRUST_INTERNAL = "internal"
TRUST_UNTRUSTED = "untrusted"

# Our own memory server. Dispatched in-process by MemoryTools rather than over
# stdio: it is the same code either way, and a subprocess hop would add latency
# and non-determinism to the benchmark for no isolation benefit (it is
# trust=internal by definition). The stdio entry point still exists — see
# memory_server/__main__.py and .mcp.json — for external clients.
INTERNAL_SERVER = "memgpt-memory"

_log = logging.getLogger(__name__)


class ExternalToolError(RuntimeError):
    """Registry or transport problem that left a server unusable."""


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, dict]:
    """Read ``mcp_servers.yaml``. Returns ``{name: spec}`` for enabled servers."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    servers = data.get("servers") or {}
    return {
        name: spec
        for name, spec in servers.items()
        if spec.get("enabled", True) and name != INTERNAL_SERVER
    }


class ExternalTools:
    """Sync facade over the external MCP servers.

    Degrades to empty rather than raising: a missing ``uvx`` or a server that
    will not start must not take the assistant down with it — it just loses a
    capability, exactly like archival memory does when Chroma fails to init.
    """

    def __init__(self, registry: dict[str, dict] | None = None) -> None:
        self._specs = registry if registry is not None else {}
        self._trust: dict[str, str] = {}      # tool name -> trust zone
        self._server_of: dict[str, str] = {}  # tool name -> server name
        self._defs: list[dict] = []           # OpenAI function schemas
        self._client: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self.errors: dict[str, str] = {}

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> ExternalTools:
        if not self._specs:
            return self
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:  # optional dependency
            self.errors["*"] = f"langchain-mcp-adapters unavailable: {exc}"
            _log.warning("External MCP tools disabled: %s", exc)
            return self

        self._start_loop()
        connections = {
            name: {
                "transport": spec.get("transport", "stdio"),
                "command": spec["command"],
                "args": list(spec.get("args", [])),
            }
            for name, spec in self._specs.items()
        }
        self._client = MultiServerMCPClient(connections)
        # Per server, not one aggregate call: a server that will not start then
        # costs only its own tools, and each tool's owner (and therefore its
        # trust zone) is known without having to guess.
        for name in self._specs:
            try:
                tools = self._run(self._client.get_tools(server_name=name))
            except Exception as exc:  # command not found, crash on boot, timeout…
                self.errors[name] = f"{type(exc).__name__}: {exc}"
                _log.warning("MCP server '%s' unavailable: %s", name, exc)
                continue
            for tool in tools:
                self._register(name, tool)
        _log.info("External MCP tools loaded: %s", ", ".join(sorted(self._trust)) or "none")
        return self

    def close(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._loop, self._thread, self._client = None, None, None

    # -- what the graph needs ---------------------------------------------
    def tool_definitions(self) -> list[dict]:
        """OpenAI function schemas to hand the router alongside the memory tools."""
        return list(self._defs)

    def names(self) -> frozenset[str]:
        return frozenset(self._trust)

    def trust_of(self, tool_name: str) -> str:
        """Trust zone of the server that owns ``tool_name`` (spec §6.1)."""
        return self._trust.get(tool_name, TRUST_UNTRUSTED)

    def server_of(self, tool_name: str) -> str | None:
        return self._server_of.get(tool_name)

    def is_gated(self, tool_name: str) -> bool:
        """True if this tool needs human approval before it runs (spec §6.3).

        A ``write_gate`` server with no explicit ``gated_tools`` gates
        everything it exports — failing closed, because guessing which of a
        stranger's fourteen tools mutate the disk is exactly the guess that
        ends with a file overwritten.
        """
        spec = self._specs.get(self._server_of.get(tool_name, ""), {})
        if not spec.get("write_gate"):
            return False
        gated = spec.get("gated_tools")
        return tool_name in gated if gated else True

    def jail_of(self, tool_name: str) -> str | None:
        """Directory this tool's server is confined to, if the registry sets one."""
        spec = self._specs.get(self._server_of.get(tool_name, ""), {})
        return spec.get("jail")

    def call(self, tool_name: str, arguments: dict) -> str:
        """Invoke an external tool. Errors become strings, never exceptions."""
        if self._client is None or tool_name not in self._server_of:
            return f"Error: external tool '{tool_name}' is not available."
        server = self._server_of[tool_name]
        try:
            result = self._run(self._call_async(server, tool_name, arguments))
        except Exception as exc:
            _log.warning("External tool %s failed: %s", tool_name, exc)
            return f"Error: external tool '{tool_name}' failed: {exc}"
        return _as_text(result)

    # -- internals ---------------------------------------------------------
    async def _call_async(self, server: str, tool_name: str, arguments: dict):
        async with self._client.session(server) as session:
            return await session.call_tool(tool_name, arguments)

    def _register(self, server_name: str, tool: Any) -> None:
        name = tool.name
        if name in self._server_of:
            # Two servers exporting the same tool name would make routing — and
            # therefore the trust decision — ambiguous. Refuse rather than guess.
            raise ExternalToolError(
                f"Tool name collision: '{name}' is exported by both "
                f"'{self._server_of[name]}' and '{server_name}'."
            )
        self._server_of[name] = server_name
        self._trust[name] = self._specs.get(server_name, {}).get("trust", TRUST_UNTRUSTED)
        self._defs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (tool.description or "").strip()[:1024],
                    "parameters": _flatten_schema(tool.args_schema),
                },
            }
        )

    def _start_loop(self) -> None:
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-client", daemon=True
        )
        self._thread.start()

    def _run(self, coro):
        if self._loop is None:
            raise ExternalToolError("MCP event loop is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=120)


# --- helpers --------------------------------------------------------------
def _as_text(result: Any) -> str:
    """Flatten an MCP CallToolResult (or whatever the adapter returned) to text."""
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
            for block in content
        ]
        return "\n".join(p for p in parts if p)
    return str(content)


def _flatten_schema(schema: Any) -> dict:
    """Reduce an MCP tool schema to the flat object shape all four providers accept.

    Nested objects and $ref are the most reliable way to break tool calling on
    one provider but not another, so anything that is not a scalar property is
    dropped rather than passed through (spec §2, §3.4).
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    props = schema.get("properties") or {}
    flat: dict[str, dict] = {}
    for key, spec in props.items():
        if not isinstance(spec, dict):
            continue
        kind = spec.get("type")
        if kind is None and "anyOf" in spec:  # Optional[x] -> take the concrete arm
            kind = next(
                (a.get("type") for a in spec["anyOf"] if a.get("type") not in (None, "null")),
                None,
            )
        if kind not in ("string", "integer", "number", "boolean"):
            continue
        entry = {"type": kind}
        if spec.get("description"):
            entry["description"] = str(spec["description"])[:300]
        flat[key] = entry
    required = [r for r in (schema.get("required") or []) if r in flat]
    return {"type": "object", "properties": flat, "required": required}


def build_external_tools(path: str | Path = DEFAULT_REGISTRY) -> ExternalTools:
    try:
        registry = load_registry(path)
    except Exception as exc:
        _log.warning("Could not read MCP registry %s: %s", path, exc)
        return ExternalTools({})
    return ExternalTools(registry).start()
