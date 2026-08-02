"""Tests for system-prompt construction."""

from __future__ import annotations

from agent.prompts import (
    BASE_SYSTEM_PROMPT,
    memory_pressure_warning,
    render_system_prompt,
)


def test_render_system_prompt_includes_memory_and_stats():
    stats = {"recall_messages": 2, "recall_events": 5, "archival_passages": 1}
    prompt = render_system_prompt(
        "<persona>I am a test.</persona>", stats, "10 / 100 tokens (10%)"
    )
    assert "I am a test." in prompt
    assert "Recall memory: 2 messages" in prompt
    assert "Archival memory: 1 passages" in prompt
    assert "10 / 100 tokens (10%)" in prompt
    # Core instructions are present.
    assert "send_message" in prompt


def test_prompt_carries_todays_date():
    """Without a date the agent cannot tell whether a stored deadline has
    passed, so it guesses or asks the user what day it is."""
    stats = {"recall_messages": 0, "recall_events": 0, "archival_passages": 0}
    assert "Today's date: 2026-09-15" in render_system_prompt(
        "x", stats, "0 / 0 (0%)", today="2026-09-15"
    )


def test_date_defaults_to_today_when_not_injected():
    from datetime import datetime, timezone

    stats = {"recall_messages": 0, "recall_events": 0, "archival_passages": 0}
    expected = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert f"Today's date: {expected}" in render_system_prompt("x", stats, "0 / 0 (0%)")


def test_read_side_synthesis_rules_are_present():
    """The behaviour T12 grades has to be instructed somewhere. If these rules
    are edited away, T12 would start failing against a live provider with no
    obvious cause — this fails first, offline."""
    p = BASE_SYSTEM_PROMPT
    assert "USING WHAT YOU FIND" in p
    # Answer from what came back, and say where it came from.
    assert "attribute" in p.lower()
    # Do not report a miss when the retrieved content describes it differently.
    assert "could not find" in p.lower()
    # Do not ask the user to re-supply what is already stored.
    assert "repeat something already in your memory" in p.lower()
    # Reason relative to a date the user supplies.
    assert "mid-September" in p


def test_durable_facts_include_goals_deadlines_projects_commitments():
    p = BASE_SYSTEM_PROMPT.lower()
    for kind in ("goals", "deadlines", "projects", "commitments"):
        assert kind in p, f"{kind} missing from the durable-fact classes"
    # The worked example is what makes the class concrete.
    assert "lean bulk" in p


def test_memory_pressure_warning():
    w = memory_pressure_warning("28,000 / 32,000 tokens (88%)")
    assert "[MEMORY PRESSURE]" in w
    assert "28,000 / 32,000 tokens (88%)" in w
    assert "archival_memory_insert" in w
