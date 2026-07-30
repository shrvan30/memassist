"""FastAPI service: SSE streaming, session→thread mapping, interrupt surface.

Everything runs against a scripted router and a fake external toolset, so the
suite needs no keys, no network and no MCP subprocesses.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from api import sessions as sessions_mod  # noqa: E402
from api.main import app  # noqa: E402
from api.sessions import EventRecorder, Session, registry  # noqa: E402
from tests.test_loop import FakeRouter, result, tool  # noqa: E402


def build_session(mem, scripted, session_id="test", external=None, **kw):
    from agent.loop import AgentLoop

    recorder = EventRecorder(mem)
    loop = AgentLoop(
        FakeRouter(scripted),
        recorder,
        tools=[],
        planning_context_limit=kw.get("limit", 100_000),
        pressure_threshold=0.7,
        max_heartbeats=5,
        external=external,
    )
    return Session(session_id=session_id, loop=loop, memory=recorder)


@pytest.fixture
def client(monkeypatch, mem):
    """A TestClient whose registry hands out one pre-scripted session."""
    made: dict = {}

    def install(scripted, external=None):
        session = build_session(mem, scripted, external=external)
        made["session"] = session
        monkeypatch.setattr(registry, "_sessions", {"test": session})
        monkeypatch.setattr(registry, "_router", session.loop.router)
        monkeypatch.setattr(registry, "_external", external)
        monkeypatch.setattr(registry, "startup", lambda: None)
        return session

    with TestClient(app) as c:
        c.install = install
        c.made = made
        yield c


def read_sse(response) -> list[dict]:
    events = []
    for line in response.iter_lines():
        if line and line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def types(events) -> list[str]:
    return [e["type"] for e in events]


# --- chat / SSE -----------------------------------------------------------
def test_chat_streams_tokens_then_message_then_done(client):
    client.install([result(tool_calls=[tool("send_message", '{"text":"Hello there, Ada."}')])])

    with client.stream("POST", "/chat", json={"message": "hi", "session_id": "test"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = read_sse(r)

    kinds = types(events)
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "token" in kinds and "message" in kinds and "state" in kinds

    # The chunks must reassemble into exactly the reply.
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    message = next(e for e in events if e["type"] == "message")
    assert streamed == message["text"] == "Hello there, Ada."
    assert message["served_by"] == "gemini"


def test_tool_activity_streams_live_during_the_turn(client):
    client.install(
        [
            result(tool_calls=[tool("core_memory_append",
                                    '{"block":"human","content":"Name: Ada.","request_heartbeat":true}')]),
            result(tool_calls=[tool("send_message", '{"text":"saved"}', id="call_2")]),
        ]
    )

    with client.stream("POST", "/chat", json={"message": "I am Ada", "session_id": "test"}) as r:
        events = read_sse(r)

    tool_calls = [e for e in events if e.get("event") == "tool_call"]
    assert tool_calls, "tool activity must reach the client as it happens"
    assert "core_memory_append" in tool_calls[0]["content"]
    # Ordering matters: the tool event precedes the final reply.
    assert types(events).index("event") < types(events).index("message")


def test_state_event_carries_the_memory_inspector_snapshot(client):
    client.install([result(tool_calls=[tool("send_message", '{"text":"ok"}')], prompt_tokens=321)])

    with client.stream("POST", "/chat", json={"message": "hi", "session_id": "test"}) as r:
        events = read_sse(r)

    state = next(e for e in events if e["type"] == "state")
    assert set(state["core"]) == {"persona", "human"}
    assert state["tiers"]["recall_messages"] >= 1
    assert state["context"]["input_tokens"] == 321
    assert state["served_by"] == "gemini"


def test_router_failure_becomes_an_sse_error_not_a_500(client):
    from llm.errors import ProviderConfigError

    client.install([ProviderConfigError("no providers")])

    with client.stream("POST", "/chat", json={"message": "hi", "session_id": "test"}) as r:
        assert r.status_code == 200  # the stream already started
        events = read_sse(r)

    assert types(events)[-1] == "error"
    assert "ProviderConfigError" in events[-1]["message"]


# --- the interrupt surface ------------------------------------------------
def gated_script():
    return [
        result(tool_calls=[tool("write_file", '{"path":"notes.txt","content":"hi"}')]),
        result(tool_calls=[tool("send_message", '{"text":"written"}', id="call_2")]),
    ]


def test_gated_tool_suspends_the_turn_and_reports_it(client, make_external):
    ext = make_external(tools=("write_file",), gated=("write_file",), jail="./workspace")
    client.install(gated_script(), external=ext)

    with client.stream("POST", "/chat", json={"message": "save it", "session_id": "test"}) as r:
        events = read_sse(r)

    assert "approval_required" in types(events)
    request = next(e for e in events if e["type"] == "approval_required")["request"]
    assert request["actions"][0]["name"] == "write_file"
    assert ext.calls == [], "nothing may run before a human answers"

    # …and it is still pending on a fresh request (session == checkpointer thread).
    pending = client.get("/sessions/test/pending").json()["pending_approval"]
    assert pending["actions"][0]["arguments"]["path"] == "notes.txt"


def test_approve_resumes_the_turn_and_runs_the_tool(client, make_external):
    ext = make_external(
        tools=("write_file",), gated=("write_file",), jail="./workspace", result="written"
    )
    client.install(gated_script(), external=ext)
    with client.stream("POST", "/chat", json={"message": "save it", "session_id": "test"}) as r:
        read_sse(r)

    with client.stream("POST", "/sessions/test/approve", json={"approved": True}) as r:
        events = read_sse(r)

    assert "".join(e["text"] for e in events if e["type"] == "token") == "written"
    assert ext.calls == [("write_file", {"path": "notes.txt", "content": "hi"})]
    assert client.get("/sessions/test/pending").json()["pending_approval"] is None


def test_deny_resumes_without_running_the_tool(client, make_external):
    ext = make_external(tools=("write_file",), gated=("write_file",), jail="./workspace")
    client.install(gated_script(), external=ext)
    with client.stream("POST", "/chat", json={"message": "save it", "session_id": "test"}) as r:
        read_sse(r)

    with client.stream("POST", "/sessions/test/approve", json={"approved": False}) as r:
        events = read_sse(r)

    assert types(events)[-1] == "done"
    assert ext.calls == [], "a denied write must never reach the server"


def test_chat_is_refused_while_an_approval_is_outstanding(client, make_external):
    ext = make_external(tools=("write_file",), gated=("write_file",), jail="./workspace")
    client.install(gated_script(), external=ext)
    with client.stream("POST", "/chat", json={"message": "save it", "session_id": "test"}) as r:
        read_sse(r)

    # Sending another message would leave the graph paused mid-turn.
    r = client.post("/chat", json={"message": "never mind", "session_id": "test"})
    assert r.status_code == 409


def test_approve_with_nothing_pending_is_a_conflict(client):
    client.install([])
    assert client.post("/sessions/test/approve", json={"approved": True}).status_code == 409


# --- read endpoints -------------------------------------------------------
def test_session_snapshot_and_messages(client):
    client.install([result(tool_calls=[tool("send_message", '{"text":"ok"}')])])
    with client.stream("POST", "/chat", json={"message": "hi", "session_id": "test"}) as r:
        read_sse(r)

    snap = client.get("/sessions/test").json()
    assert snap["session_id"] == "test"
    assert snap["tiers"]["context_messages"] > 0
    assert snap["context"]["usage"]

    messages = client.get("/sessions/test/messages").json()
    assert messages["count"] == snap["tiers"]["context_messages"]
    assert messages["messages"][0]["role"] == "user"


def test_reset_clears_context_but_not_saved_memory(client, mem):
    client.install(
        [
            result(tool_calls=[tool("core_memory_append",
                                    '{"block":"human","content":"Name: Ada.","request_heartbeat":true}')]),
            result(tool_calls=[tool("send_message", '{"text":"ok"}', id="call_2")]),
        ]
    )
    with client.stream("POST", "/chat", json={"message": "I am Ada", "session_id": "test"}) as r:
        read_sse(r)

    snap = client.post("/sessions/test/reset").json()
    assert snap["tiers"]["context_messages"] == 0
    assert "Ada" in snap["core"]["human"], "core memory must survive a context reset"


def test_providers_and_tools_panels(client, make_external):
    ext = make_external(tools=("search",), gated=(), jail=None)
    client.install([], external=ext)

    tools_panel = client.get("/tools").json()
    assert tools_panel["tools"][0]["name"] == "search"
    assert tools_panel["tools"][0]["trust"] == "untrusted"

    assert client.get("/healthz").json()["status"] == "ok"
    assert "test" in client.get("/sessions").json()["sessions"]


def test_unknown_session_is_404_on_read_paths(client):
    client.install([])
    assert client.get("/sessions/nope/pending").status_code == 404
    assert client.post("/sessions/nope/approve", json={"approved": True}).status_code == 404


def test_default_session_id_comes_from_config():
    assert sessions_mod.DEFAULT_SESSION
