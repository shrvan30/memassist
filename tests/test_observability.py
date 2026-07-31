"""Langfuse tracing (spec §11 P5).

Two things are worth testing here, and neither is "does Langfuse work" — that is
Langfuse's job. What matters is:

1. **Disabled is genuinely inert.** The benchmark is deterministic and offline
   and the test suite makes no network calls; if tracing did anything at all
   when unconfigured, both claims would quietly stop being true.
2. **The mask redacts.** Traces go to a hosted dashboard, so the same content
   the T10 gate keeps from Mistral must not reach Langfuse either.
"""

from __future__ import annotations

import pytest

import observability
from tests.test_loop import FakeRouter, make_loop, result, tool


@pytest.fixture(autouse=True)
def _clean_tracing(monkeypatch):
    monkeypatch.delenv(observability.PUBLIC_KEY_ENV, raising=False)
    monkeypatch.delenv(observability.SECRET_KEY_ENV, raising=False)
    observability.reset_for_tests()
    yield
    observability.reset_for_tests()


# --- off by default -------------------------------------------------------
def test_tracing_is_off_without_keys():
    assert observability.enabled() is False


def test_half_configured_is_off_not_broken(monkeypatch):
    """One key without the other is a misconfiguration; it must not try to
    connect, and must not raise on the turn path."""
    monkeypatch.setenv(observability.PUBLIC_KEY_ENV, "pk-only")
    observability.reset_for_tests()
    assert observability.enabled() is False


def test_every_entry_point_is_a_no_op_when_disabled():
    with observability.span("anything") as s:
        assert s is None
    # None of these may raise or touch the network.
    observability.event("x", metadata={"a": 1})
    observability.update_trace(None, session_id="s", tags=["t"])
    observability.flush()


def test_traced_node_passes_through_untouched_when_disabled():
    calls = []

    def node(state, deps=None):
        calls.append(state)
        return {"done": True}

    wrapped = observability.traced_node(node, "respond")
    assert wrapped({"x": 1}) == {"done": True}
    assert calls == [{"x": 1}]


def test_a_full_turn_runs_with_tracing_disabled(mem):
    """The graph is wired through traced_node, so this is the regression guard
    against instrumentation breaking the turn cycle itself."""
    router = FakeRouter([result(tool_calls=[tool("send_message", '{"text":"hi"}')])])
    loop = make_loop(mem, router)
    assert loop.step("hello") == ["hi"]


# --- the mask -------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        "my key is sk-abcdefghijklmnopqrstuvwx",
        {"content": "password: hunter2"},
        {"messages": [{"role": "user", "content": "card 4111111111111111"}]},
        ["ssn 123-45-6789"],
    ],
)
def test_the_mask_redacts_what_the_privacy_gate_would_withhold(payload):
    """One rule, two destinations: not safe for Mistral, not safe for Langfuse."""
    masked = observability._mask(data=payload)
    flat = str(masked)
    assert "redacted" in flat
    for secret in ("sk-abcdefghij", "hunter2", "4111111111111111", "123-45-6789"):
        assert secret not in flat


def test_the_mask_leaves_ordinary_content_alone():
    data = {"user": "I work as a nurse in Pune.", "n": 3, "ok": True}
    assert observability._mask(data=data) == data


def test_the_mask_names_the_category_so_a_trace_says_why():
    assert "api-key" in observability._mask(data="sk-abcdefghijklmnopqrstuvwx")


# --- what a turn reports --------------------------------------------------
def test_turn_summary_counts_and_names_but_carries_no_content():
    summary = observability.summarize_turn(
        {
            "served_by": "groq",
            "heartbeat_count": 2,
            "saw_untrusted": True,
            "injection_flags": ["role-redirect"],
            "blocked_tools": ["write_file"],
            "messages": [{"role": "user", "content": "SECRET CONTENT"}],
        }
    )
    assert summary["served_by"] == "groq"
    assert summary["injection_flags"] == ["role-redirect"]
    assert summary["blocked_tools"] == ["write_file"]
    assert "SECRET CONTENT" not in str(summary), "the trace must not carry the window"
