"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

import config
from memory_server.memory_tools import MemoryTools
from memory_server.storage.chroma import ArchivalStore, hashing_embedding
from memory_server.storage.sqlite import SQLiteStore

# Small char limit so the core-block overflow path is easy to exercise.
TEST_CORE_LIMIT = 200


@pytest.fixture(autouse=True)
def _no_provider_calls(request, monkeypatch):
    """Fail fast if a test builds a real provider client.

    CI already runs `pytest -m "not llm"`, but nothing was ever marked `llm`, so
    that filter deselected nothing and protected nothing. A test that reached a
    provider would not be *excluded* — it would run, block on the network, and
    stall the job with no indication of which test was responsible.

    This makes the marker load-bearing: unmarked tests cannot construct a real
    client at all, and the failure names the test instead of hanging it. Opt in
    with @pytest.mark.llm when a test genuinely needs a live provider.
    """
    if request.node.get_closest_marker("llm"):
        return

    def _refuse(*_a, **_kw):
        raise RuntimeError(
            "This test tried to build a real provider client. The suite is "
            "offline: inject a fake, or mark the test @pytest.mark.llm."
        )

    monkeypatch.setattr("llm.router.OpenAIChatClient", _refuse)


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


class FakeExternal:
    """A stand-in external toolset implementing the full ExternalToolset protocol.

    One shared fake rather than one per test module: the protocol is what the
    security layer branches on, so a partial fake would let a test pass while
    the real wiring is broken.
    """

    def __init__(self, tools=("search",), result="", trust="untrusted", gated=(), jail=None):
        self._tools = frozenset(tools)
        self._result = result
        self._trust = trust
        self._gated = frozenset(gated)
        self._jail = jail
        self.calls: list[tuple[str, dict]] = []
        self.errors: dict[str, str] = {}

    def names(self):
        return self._tools

    def trust_of(self, name):
        return self._trust

    def server_of(self, name):
        return "fake-server"

    def is_gated(self, name):
        return name in self._gated

    def jail_of(self, name):
        return self._jail

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return self._result

    def close(self):
        pass


@pytest.fixture
def make_external():
    return FakeExternal
