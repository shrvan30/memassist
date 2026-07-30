"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

import config
from memory_server.memory_tools import MemoryTools
from memory_server.storage.chroma import ArchivalStore, hashing_embedding
from memory_server.storage.sqlite import SQLiteStore

# Small char limit so the core-block overflow path is easy to exercise.
TEST_CORE_LIMIT = 200


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(
        str(tmp_path / "memassist_test.db"),
        default_persona=config.DEFAULT_PERSONA,
        default_human=config.DEFAULT_HUMAN,
        core_block_char_limit=TEST_CORE_LIMIT,
    )
    yield s
    s.close()


@pytest.fixture
def archival(tmp_path):
    # A unique on-disk path per test → real isolation. (Chroma's in-memory
    # EphemeralClient shares one global system across the process, so it would
    # leak data between tests; a per-test path avoids that entirely.)
    #
    # The hashing embedder is injected as a test double so the suite needs no
    # bge-small download: CI stays offline and fast. Semantic retrieval quality
    # is the benchmark's job (bench/ T5), and it uses the real model.
    return ArchivalStore(str(tmp_path / "chroma"), embed_fn=hashing_embedding)


@pytest.fixture
def mem(store, archival):
    return MemoryTools(
        store,
        archival,
        session_id="test",
        page_size=3,
        archival_top_k=3,
        result_char_cap=120,
    )
