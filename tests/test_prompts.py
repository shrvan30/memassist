"""Tests for system-prompt construction."""

from __future__ import annotations

from agent.prompts import memory_pressure_warning, render_system_prompt


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


def test_memory_pressure_warning():
    w = memory_pressure_warning("28,000 / 32,000 tokens (88%)")
    assert "[MEMORY PRESSURE]" in w
    assert "28,000 / 32,000 tokens (88%)" in w
    assert "archival_memory_insert" in w
