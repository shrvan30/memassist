"""Local embedding model for archival memory — bge-small-en-v1.5, 384-d.

Local-only: embeddings never leave the machine and cost nothing. The model is
loaded lazily and cached for the process — the first call pays the load (and,
once ever, the ~130 MB download), every later call is cheap.
"""

from __future__ import annotations

from typing import Any

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

_model: Any = None


def get_model() -> Any:
    """Return the cached SentenceTransformer, loading it on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> list[float]:
    """Embed one string into a 384-d L2-normalized vector.

    Normalizing at encode time means Chroma's cosine space and a plain dot
    product agree.
    """
    vector = get_model().encode(text, normalize_embeddings=True)
    return [float(x) for x in vector]


def embed_many(texts: list[str]) -> list[list[float]]:
    """Batch variant — one forward pass for the whole list (used by migrations)."""
    if not texts:
        return []
    vectors = get_model().encode(texts, normalize_embeddings=True)
    return [[float(x) for x in vector] for vector in vectors]
