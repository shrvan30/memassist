"""Archival memory: a Chroma vector store with semantic search.

Embeddings come from the local bge-small-en-v1.5 model (see ``embedder.py``); the
lexical ``hashing_embedding`` below stays available as a zero-dependency fallback
for offline unit tests. Embeddings are computed here and passed to Chroma
explicitly (``embedding_function=None``) to avoid Chroma's default embedder, which
would download an ONNX model.

A collection's vector dimensionality is fixed at creation, so changing
``embed_fn`` to a model with a different width requires re-embedding an existing
store — see ``migrate_embeddings.py``.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import chromadb
from chromadb.config import Settings

from . import embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_DIM = 512


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def hashing_embedding(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    """Deterministic, offline, L2-normalized signed-hashing embedding."""
    vec = [0.0] * dim
    for tok in _tokenize(text):
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [x / norm for x in vec]
    return vec


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class ArchivalStore:
    """Thin wrapper over a persistent Chroma collection.

    ``embed_fn`` maps text -> vector; defaults to the local bge-small model.
    Inject a different callable (e.g. ``hashing_embedding``) to run without the
    model download.
    """

    def __init__(
        self,
        path: str,
        embed_fn: Callable[[str], list[float]] | None = None,
        collection_name: str = "archival",
    ) -> None:
        self._embed = embed_fn or embedder.embed
        if path != ":memory:":
            Path(path).mkdir(parents=True, exist_ok=True)
        if path == ":memory:":
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self._client = chromadb.PersistentClient(
                path=path, settings=Settings(anonymized_telemetry=False)
            )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # we always pass explicit embeddings
            metadata={"hnsw:space": "cosine"},
        )

    def insert(self, content: str, source: str = "agent", created_at: str | None = None) -> str:
        """Embed and upsert a passage with a timestamp. Returns the passage id."""
        passage_id = uuid.uuid4().hex
        metadata = {"created_at": created_at or _utc_now(), "source": source}
        self._col.add(
            ids=[passage_id],
            embeddings=[self._embed(content)],
            documents=[content],
            metadatas=[metadata],
        )
        return passage_id

    def search(self, query: str, top_k: int = 5, page: int = 0) -> tuple[list[dict], int]:
        """Semantic search. Returns ``(page_items, total_passages)``.

        Chroma has no query offset, so we fetch enough results to cover the
        requested page and slice locally (fine for MVP-scale stores).
        """
        total = self.count()
        if total == 0:
            return [], 0
        # ponytail: fetch the whole collection and page locally. Fetching only
        # top_k*(page+1) is cheaper, but then page 0 sorts a 2-row candidate set
        # while page 1 sorts a 4-row one — with tied distances those disagree and
        # consecutive pages repeat a passage. Fine at MVP scale; the pgvector
        # backend pages in SQL and has no such limit.
        res = self._col.query(
            query_embeddings=[self._embed(query)],
            n_results=total,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        ids = res.get("ids", [[]])[0] or [""] * len(docs)
        items = [
            {
                "id": pid,
                "content": doc,
                "created_at": (meta or {}).get("created_at"),
                "source": (meta or {}).get("source"),
                "distance": dist,
            }
            for pid, doc, meta, dist in zip(ids, docs, metas, dists)
        ]
        # Tie-break on id. Equally-distant passages otherwise come back in
        # whatever order the index felt like this call, so consecutive pages
        # could repeat one passage and drop another.
        items.sort(key=lambda item: (item["distance"], item["id"]))
        start = page * top_k
        return items[start : start + top_k], total

    def count(self) -> int:
        return int(self._col.count())
