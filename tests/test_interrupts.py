"""Human-in-the-loop gating for destructive tools (spec §4.3, §6.3).

Every filesystem write suspends the turn until a person answers. The path jail
is checked BEFORE the gate, so a traversal is refused outright rather than
offered for approval — a user cannot be talked into waving one through.
"""

from __future__ import annotations

from agent.loop import AgentLoop
from security.guards import escapes_jail
from tests.test_loop import FakeRouter, result, tool

WRITE_CALL = '{"path":"notes.txt","content":"hello"}'


def build(mem, scripted, external):
    return AgentLoop(
        FakeRouter(scripted),
        mem,
        tools=[],
        planning_context_limit=100_000,
        pressure_threshold=0.7,
        max_heartbeats=5,
        external=external,
    )


def write_then_reply():
    return [
        result(tool_calls=[tool("write_file", WRITE_CALL)]),
        result(tool_calls=[tool("send_message", '{"text":"all done"}', id="call_2")]),
    ]


# --- the path jail --------------------------------------------------------
def test_paths_inside_the_jail_are_fine(tmp_path):
    assert escapes_jail({"path": "notes.txt"}, tmp_path) is None
    assert escapes_jail({"path": "sub/dir/notes.txt"}, tmp_path) is None
    assert escapes_jail({"path": str(tmp_path / "ok.txt")}, tmp_path) is None


def test_traversal_escapes_are_caught(tmp_path):
    assert escapes_jail({"path": "../../etc/passwd"}, tmp_path) == "../../etc/passwd"
    assert escapes_jail({"path": "/etc/passwd"}, tmp_path) == "/etc/passwd"


def test_every_path_argument_is_checked_not_just_the_first(tmp_path):
    # move_file takes source/destination; checking only `path` would miss both.
    assert escapes_jail({"source": "a.txt", "destination": "../out.txt"}, tmp_path) == "../out.txt"
    assert escapes_jail({"paths": ["ok.txt", "../bad.txt"]}, tmp_path) == "../bad.txt"


def test_jail_escape_is_refused_before_the_gate(mem, make_external):
    """A traversal is never offered for approval — it just does not happen."""
    ext = make_external(tools=("write_file",), gated=("write_file",), jail="./workspace")
    loop = build(
        mem,
        [
            result(tool_calls=[tool("write_file", '{"path":"../../secrets.env","content":"x"}')]),
            result(tool_calls=[tool("send_message", '{"text":"could not"}', id="call_2")]),
        ],
        ext,
    )

    loop.step("write to ../../secrets.env")

    assert loop.pending_approval is None, "must not ask a human about a traversal"
    assert ext.calls == [], "the tool must never have run"
    refusal = next(m for m in loop.messages if m.get("tool_call_id") == "call_1")
    assert "outside the workspace" in refusal["content"]


# --- approve / deny -------------------------------------------------------
def test_write_suspends_the_turn_for_approval(mem, make_external):
    ext = make_external(tools=("write_file",), gated=("write_file",), jail="./workspace")
    loop = build(mem, write_then_reply(), ext)

    out = loop.step("save a note")

    assert out == [], "nothing delivered while the turn is suspended"
    assert loop.pending_approval is not None
    assert loop.pending_approval["kind"] == "tool_approval"
    action = loop.pending_approval["actions"][0]
    assert action["name"] == "write_file"
    assert action["arguments"]["path"] == "notes.txt"
    assert ext.calls == [], "the write must not run before a human answers"


def test_approval_runs_the_write_and_finishes_the_turn(mem, make_external):
    ext = make_external(
        tools=("write_file",), gated=("write_file",), jail="./workspace", result="written"
    )
    loop = build(mem, write_then_reply(), ext)
    loop.step("save a note")

    out = loop.resume(approved=True)

    assert out == ["all done"]
    assert loop.pending_approval is None
    assert ext.calls == [("write_file", {"path": "notes.txt", "content": "hello"})]


def test_denial_blocks_the_write_and_tells_the_model_not_to_retry(mem, make_external):
    ext = make_external(tools=("write_file",), gated=("write_file",), jail="./workspace")
    loop = build(mem, write_then_reply(), ext)
    loop.step("save a note")

    out = loop.resume(approved=False)

    assert ext.calls == [], "a denied write must never reach the server"
    assert out == ["all done"]
    refusal = next(m for m in loop.messages if m.get("tool_call_id") == "call_1")
    assert "user denied this action" in refusal["content"]
    assert "Do not retry" in refusal["content"]
    assert mem.store.count_messages(event_types=("security",)) >= 1


def test_ungated_tools_from_the_same_server_do_not_interrupt(mem, make_external):
    # read_file is not in gated_tools, so it runs without asking.
    ext = make_external(
        tools=("read_file",), gated=("write_file",), jail="./workspace", result="file contents"
    )
    loop = build(
        mem,
        [
            result(tool_calls=[tool("read_file", '{"path":"notes.txt","request_heartbeat":true}')]),
            result(tool_calls=[tool("send_message", '{"text":"read it"}', id="call_2")]),
        ],
        ext,
    )

    out = loop.step("read my notes")

    assert loop.pending_approval is None
    assert out == ["read it"]
    assert ext.calls == [("read_file", {"path": "notes.txt"})]


def test_resume_without_a_pending_approval_is_a_no_op(mem, make_external):
    loop = build(mem, [], make_external())
    assert loop.resume(approved=True) == []
