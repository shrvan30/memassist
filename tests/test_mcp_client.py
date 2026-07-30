"""External MCP tool plumbing — registry, trust mapping, schema flattening.

No subprocesses: these cover the parts that decide *routing and trust*, which is
what the security layer depends on. The live handshake is exercised by
`make bench LIVE=1`.
"""

from __future__ import annotations

from types import SimpleNamespace

from mcp_client import (
    INTERNAL_SERVER,
    TRUST_UNTRUSTED,
    ExternalTools,
    _flatten_schema,
    load_registry,
)

REGISTRY = """
servers:
  memgpt-memory:
    transport: stdio
    command: python
    args: ["-m", "memory_server"]
    trust: internal
  ddg-search:
    transport: stdio
    command: uvx
    args: ["duckduckgo-mcp-server"]
    trust: untrusted
    enabled: true
  filesystem:
    transport: stdio
    command: npx
    args: ["-y", "server-filesystem", "./workspace"]
    trust: untrusted
    enabled: false
"""


def write_registry(tmp_path, text=REGISTRY):
    path = tmp_path / "mcp_servers.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def fake_tool(name, description="", schema=None):
    return SimpleNamespace(name=name, description=description, args_schema=schema or {})


# --- registry -------------------------------------------------------------
def test_registry_skips_disabled_and_the_internal_server(tmp_path):
    servers = load_registry(write_registry(tmp_path))
    # memgpt-memory is dispatched in-process; filesystem is disabled.
    assert set(servers) == {"ddg-search"}
    assert INTERNAL_SERVER not in servers


def test_real_registry_is_parseable():
    servers = load_registry()
    assert "ddg-search" in servers
    assert servers["ddg-search"]["trust"] == TRUST_UNTRUSTED


# --- trust mapping --------------------------------------------------------
def test_trust_comes_from_the_owning_server(tmp_path):
    ext = ExternalTools(load_registry(write_registry(tmp_path)))
    ext._register("ddg-search", fake_tool("search"))

    assert ext.trust_of("search") == TRUST_UNTRUSTED
    assert ext.server_of("search") == "ddg-search"
    assert ext.names() == frozenset({"search"})


def test_unknown_tool_is_untrusted_by_default():
    # Deny-by-default: never assume a tool we cannot place is safe.
    assert ExternalTools({}).trust_of("mystery") == TRUST_UNTRUSTED


def test_tool_name_collision_is_refused(tmp_path):
    import pytest

    from mcp_client import ExternalToolError

    ext = ExternalTools({"a": {"trust": "untrusted"}, "b": {"trust": "internal"}})
    ext._register("a", fake_tool("search"))
    with pytest.raises(ExternalToolError, match="collision"):
        ext._register("b", fake_tool("search"))


# --- degradation ----------------------------------------------------------
def test_no_servers_means_no_tools_and_no_crash():
    ext = ExternalTools({}).start()
    assert ext.tool_definitions() == []
    assert ext.call("search", {"query": "x"}).startswith("Error:")
    ext.close()


# --- schema flattening ----------------------------------------------------
def test_flatten_keeps_scalars_and_required():
    flat = _flatten_schema(
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "title": "Query"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        }
    )
    assert flat["properties"] == {"query": {"type": "string"}, "max_results": {"type": "integer"}}
    assert flat["required"] == ["query"]


def test_flatten_drops_nested_objects_and_arrays():
    # Nested schemas are the most reliable way to break tool calling on one
    # provider but not another, so they never reach the router.
    flat = _flatten_schema(
        {
            "type": "object",
            "properties": {
                "ok": {"type": "string"},
                "opts": {"type": "object", "properties": {"deep": {"type": "string"}}},
                "many": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ok", "opts"],
        }
    )
    assert set(flat["properties"]) == {"ok"}
    assert flat["required"] == ["ok"]  # a dropped property cannot stay required


def test_flatten_unwraps_optional_anyof():
    flat = _flatten_schema(
        {
            "type": "object",
            "properties": {
                "backend": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}
            },
        }
    )
    assert flat["properties"]["backend"] == {"type": "string"}
