"""Tests for the archival (Chroma) storage layer and offline embedder."""

from __future__ import annotations

import math

import pytest

chromadb = pytest.importorskip("chromadb")

from memory_server.storage.chroma import (  # noqa: E402
    DEFAULT_DIM,
    ArchivalStore,
    hashing_embedding,
)


def test_embedding_is_deterministic_and_normalized():
    a = hashing_embedding("the user loves coffee")
    b = hashing_embedding("the user loves coffee")
    assert a == b
    assert len(a) == DEFAULT_DIM
    norm = math.sqrt(sum(x * x for x in a))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_embedding_empty_text():
    v = hashing_embedding("")
    assert v == [0.0] * DEFAULT_DIM  # no tokens -> zero vector, no divide-by-zero


def test_insert_search_and_count(archival):
    assert archival.count() == 0
    archival.insert("The user loves espresso and dark roast coffee.")
    archival.insert("The user dislikes herbal tea.")
    assert archival.count() == 2

    items, total = archival.search("coffee", top_k=1)
    assert total == 2
    assert len(items) == 1
    assert "coffee" in items[0]["content"].lower()


def test_search_on_empty_store(archival):
    items, total = archival.search("anything")
    assert items == [] and total == 0


def test_persistence_across_reopen(tmp_path):
    path = str(tmp_path / "chroma")
    store = ArchivalStore(path, embed_fn=hashing_embedding)
    store.insert("Persisted passage about quantum computing.")
    assert store.count() == 1

    reopened = ArchivalStore(path, embed_fn=hashing_embedding)
    assert reopened.count() == 1
    items, _ = reopened.search("quantum", top_k=1)
    assert items and "quantum" in items[0]["content"].lower()
