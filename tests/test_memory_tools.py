"""Tests for the MemoryTools dispatcher (the six tools + loop helpers)."""

from __future__ import annotations

from memory_server.storage.sqlite import EVENT_MESSAGE


def test_dispatch_core_append(mem):
    out = mem.dispatch(
        "core_memory_append",
        {"block": "human", "content": "Name: Cara", "request_heartbeat": True},
    )
    assert "Cara" in out
    assert "Cara" in mem.core_blocks()["human"]


def test_dispatch_rejects_invalid_block(mem):
    out = mem.dispatch("core_memory_append", {"block": "nope", "content": "x"})
    assert out.lower().startswith("error")


def test_dispatch_unknown_tool(mem):
    out = mem.dispatch("frobnicate", {})
    assert "not a memory tool" in out


def test_core_replace_not_found(mem):
    out = mem.dispatch(
        "core_memory_replace",
        {"block": "human", "old_text": "missing", "new_text": "x"},
    )
    assert out.lower().startswith("error")


def test_conversation_search_via_dispatch(mem):
    mem.record_event("user", EVENT_MESSAGE, "I love coffee")
    mem.record_event("assistant", EVENT_MESSAGE, "Noted your love of coffee")
    out = mem.dispatch("conversation_search", {"query": "coffee"})
    assert "coffee" in out.lower()
    assert "match" in out.lower()


def test_archival_insert_and_search_via_dispatch(mem):
    insert_out = mem.dispatch(
        "archival_memory_insert", {"content": "The user is a marine biologist."}
    )
    assert "archival memory" in insert_out.lower()
    search_out = mem.dispatch("archival_memory_search", {"query": "biologist"})
    assert "biologist" in search_out.lower()


def test_result_is_truncated(mem):
    # result_char_cap is 120 in the fixture; a long recall result must be capped.
    long_text = "coffee " * 60
    mem.record_event("user", EVENT_MESSAGE, long_text)
    out = mem.dispatch("conversation_search", {"query": "coffee"})
    assert "truncated" in out
    assert len(out) <= 120 + len("\n… [truncated to 120 characters]")


def test_memory_stats(mem):
    mem.record_event("user", EVENT_MESSAGE, "hello")
    mem.dispatch("archival_memory_insert", {"content": "a passage"})
    stats = mem.memory_stats()
    assert stats["recall_messages"] == 1
    assert stats["archival_passages"] == 1


def test_render_core_memory_shape(mem):
    rendered = mem.render_core_memory()
    assert "<persona>" in rendered and "<human>" in rendered


def test_conversation_search_date_valid_range(mem):
    mem.record_event("user", EVENT_MESSAGE, "hello there")
    out = mem.conversation_search_date("2000-01-01", "2100-01-01")
    assert "hello there" in out


def test_conversation_search_date_malformed_returns_error(mem):
    # Malformed dates must come back as a correctable "Error: ..." string, not a
    # silent "no matches" (the T3b gap) and not an exception.
    out = mem.conversation_search_date("not-a-date", "also-bad")
    assert out.startswith("Error:")
