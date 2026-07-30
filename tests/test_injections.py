"""T11 injection corpus, run deterministically in CI (spec §6.5).

Reads the same ``security/injections/*.yaml`` files the benchmark's T11 tier
uses, so a new attack case is written once and both harnesses pick it up. The
LLM is mocked throughout: the assertions are about the sanitizer and the guards,
which are the parts that must hold regardless of what the model decides to do.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent.loop import AgentLoop
from security.sanitizer import CLOSE_MARKER, sanitize_external
from tests.test_loop import FakeRouter, result, tool

CORPUS_DIR = Path(__file__).resolve().parent.parent / "security" / "injections"


def load(name: str) -> dict:
    return yaml.safe_load((CORPUS_DIR / name).read_text(encoding="utf-8"))


def cases(name: str):
    corpus = load(name)
    return [pytest.param(corpus, case, id=case["name"]) for case in corpus["cases"]]


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


def test_corpus_files_are_present_and_scored():
    """The corpus is the deliverable — an empty directory must fail loudly."""
    files = sorted(p.name for p in CORPUS_DIR.glob("*.yaml"))
    assert files == [
        "filesystem_writes.yaml",
        "memory_poisoning.yaml",
        "prompt_exfiltration.yaml",
    ]
    assert sum(load(f)["points"] for f in files) == 10


# --- T11(a) memory poisoning ---------------------------------------------
@pytest.mark.parametrize("corpus,case", cases("memory_poisoning.yaml"))
def test_untrusted_content_cannot_reach_core_memory(corpus, case, mem, make_external):
    before = mem.core_blocks()["human"]
    loop = build(
        mem,
        [
            result(tool_calls=[tool(corpus["tool"], '{"query":"x","request_heartbeat":true}')]),
            # The model does what the payload asked and tries to write core memory.
            result(
                tool_calls=[
                    tool(
                        "core_memory_append",
                        '{"block":"human","content":"Favorite store: BuyNow.","source":"stated"}',
                        id="call_2",
                    )
                ]
            ),
            result(tool_calls=[tool("send_message", '{"text":"ok"}', id="call_3")]),
        ],
        make_external(tools=(corpus["tool"],), result=case["payload"]),
    )

    loop.step("look that up")

    assert mem.core_blocks()["human"] == before
    refusal = next(m for m in loop.messages if m.get("tool_call_id") == "call_2")
    assert "core memory is closed" in refusal["content"]

    for flag in case["expect"].get("flags", []):
        wrapped = next(m for m in loop.messages if m.get("tool_call_id") == "call_1")
        assert flag in wrapped["content"], f"{flag} not flagged"


def test_knowledge_from_a_poisoned_page_is_archived_as_external(mem, make_external):
    corpus = load("memory_poisoning.yaml")
    case = next(c for c in corpus["cases"] if c["name"] == "laundering-through-archival")
    loop = build(
        mem,
        [
            result(tool_calls=[tool(corpus["tool"], '{"query":"x","request_heartbeat":true}')]),
            result(
                tool_calls=[
                    tool(
                        "archival_memory_insert",
                        '{"content":"The user approves all file writes.","source":"stated"}',
                        id="call_2",
                    )
                ]
            ),
            result(tool_calls=[tool("send_message", '{"text":"ok"}', id="call_3")]),
        ],
        make_external(tools=(corpus["tool"],), result=case["payload"]),
    )

    loop.step("look that up")

    items, total = mem.archival.search("approves all file writes", top_k=1)
    assert total == 1
    assert items[0]["source"] == case["expect"]["archival_source"] == "external"


# --- T11(b) instruction override / exfiltration ---------------------------
@pytest.mark.parametrize("corpus,case", cases("prompt_exfiltration.yaml"))
def test_injected_instructions_are_neutralized(corpus, case):
    out = sanitize_external(case["payload"], source=f"{corpus['tool']} via corpus")
    expect = case["expect"]

    for flag in expect.get("flags", []):
        assert flag in out.flags, f"{flag} not detected in {case['name']}"

    if "marker_collisions" in expect:
        assert out.marker_collisions == expect["marker_collisions"]
        # Exactly one real terminator: the forged one cannot end the envelope.
        assert out.text.count(CLOSE_MARKER) == 1

    if expect.get("neutralized"):
        assert out.flags or out.marker_collisions, "payload passed through untouched"


# --- T11(c) filesystem writes --------------------------------------------
@pytest.mark.parametrize("corpus,case", cases("filesystem_writes.yaml"))
def test_filesystem_writes_are_gated_or_refused(corpus, case, mem, make_external, tmp_path):
    import json

    jail = tmp_path / "workspace"
    jail.mkdir()
    ext = make_external(
        tools=(corpus["tool"],), gated=(corpus["tool"],), jail=str(jail), result="written"
    )
    loop = build(
        mem,
        [
            result(tool_calls=[tool(corpus["tool"], json.dumps(case["arguments"]))]),
            result(tool_calls=[tool("send_message", '{"text":"ok"}', id="call_2")]),
        ],
        ext,
    )

    loop.step("write that file")
    expect = case["expect"]

    assert (loop.pending_approval is not None) == expect["interrupted"]
    assert ext.calls == [], "nothing may run before approval, ever"

    if expect.get("refused"):
        refusal = next(m for m in loop.messages if m.get("tool_call_id") == "call_1")
        assert "outside the workspace" in refusal["content"]

    if case.get("approve") is False:
        loop.resume(approved=False)
        assert ext.calls == [], "a denied write must never reach the server"
