"""Tests for the SQLite core + recall storage layer."""

from __future__ import annotations

import pytest

from memory_server.storage.sqlite import EVENT_MESSAGE


def test_seed_defaults(store):
    blocks = store.get_core_blocks()
    assert set(blocks) == {"persona", "human"}
    assert "MemAssist" in blocks["persona"]


def test_append_block(store):
    new = store.append_block("human", "Name: Bob")
    assert "Name: Bob" in new
    again = store.append_block("human", "Likes: hiking")
    assert "Name: Bob" in again and "Likes: hiking" in again
    # Appends are newline-separated.
    assert again.count("\n") >= 1


def test_append_exceeds_char_limit(store):
    with pytest.raises(ValueError):
        store.append_block("human", "x" * (300))  # limit is 200 in tests


def test_replace_found(store):
    store.append_block("human", "Name: Bob")
    new = store.replace_block("human", "Bob", "Robert")
    assert "Robert" in new and "Bob" not in new.replace("Robert", "")


def test_replace_not_found_raises(store):
    with pytest.raises(ValueError):
        store.replace_block("human", "does-not-exist", "whatever")


def test_invalid_block_raises(store):
    with pytest.raises(ValueError):
        store.get_block("not_a_block")


def test_search_messages_keyword_and_pagination(store):
    store.add_message("s", "user", EVENT_MESSAGE, "I love coffee in the morning")
    store.add_message("s", "assistant", EVENT_MESSAGE, "Coffee is great, noted!")
    store.add_message("s", "user", EVENT_MESSAGE, "I also enjoy tea sometimes")

    rows, total = store.search_messages("coffee", page=0, page_size=5)
    assert total == 2
    assert all("coffee" in r["content"].lower() for r in rows)

    # AND-of-terms: both words must appear.
    rows2, total2 = store.search_messages("coffee morning", page=0, page_size=5)
    assert total2 == 1

    # Pagination: page_size 1 -> two pages for the coffee query.
    p0, t0 = store.search_messages("coffee", page=0, page_size=1)
    p1, t1 = store.search_messages("coffee", page=1, page_size=1)
    assert t0 == t1 == 2 and len(p0) == 1 and len(p1) == 1
    assert p0[0]["id"] != p1[0]["id"]


def test_search_messages_by_date(store):
    store.add_message("s", "user", EVENT_MESSAGE, "early note", created_at="2026-07-20 10:00:00")
    store.add_message("s", "user", EVENT_MESSAGE, "later note", created_at="2026-07-23 09:00:00")

    rows, total = store.search_messages_by_date("2026-07-22", "2026-07-24")
    assert total == 1
    assert rows[0]["content"] == "later note"

    # Whole-range covers both.
    _, total_all = store.search_messages_by_date("2026-07-01", "2026-07-31")
    assert total_all == 2


def test_search_messages_by_date_rejects_malformed(store):
    # Malformed bounds must raise (so the handler surfaces a correctable error and
    # the model self-corrects) instead of silently returning no matches.
    for bad_start, bad_end in [
        ("not-a-date", "also-bad"),
        ("2026-13-45", "2026-07-31"),  # impossible month/day
        ("07/24/2026", "2026-07-31"),  # wrong separator/order
        ("2026-07-24", "yesterday"),
        ("", "2026-07-31"),
    ]:
        with pytest.raises(ValueError):
            store.search_messages_by_date(bad_start, bad_end)


def test_search_messages_by_date_accepts_full_timestamp(store):
    store.add_message("s", "user", EVENT_MESSAGE, "noon note", created_at="2026-07-24 12:00:00")
    rows, total = store.search_messages_by_date("2026-07-24 11:00:00", "2026-07-24 13:00:00")
    assert total == 1 and rows[0]["content"] == "noon note"


def test_count_messages(store):
    assert store.count_messages() == 0
    store.add_message("s", "user", EVENT_MESSAGE, "hi")
    store.add_message("s", "tool", "tool_result", "some result")
    assert store.count_messages() == 2
    assert store.count_messages(event_types=(EVENT_MESSAGE,)) == 1
