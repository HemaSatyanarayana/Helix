"""Retrieval layer — embed the query and search Qdrant for relevant chunks.

Uses the same embedding model as ingestion (query and documents must share a
vector space). Returns chunk text plus provenance (source, headings, pages) so
the UI can cite where each answer came from. Traced by Logfire.
"""

from __future__ import annotations

import os
from functools import lru_cache

from app.telemetry import configure_logfire

logfire = configure_logfire("helix-app")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL)


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """Return the top-k most similar chunks for a question."""
    top_k = top_k or TOP_K
    with logfire.span("retrieve", top_k=top_k) as span:
        client = _qdrant()
        if not client.collection_exists(QDRANT_COLLECTION):
            span.set_attribute("n_hits", 0)
            return []

        vector = _embedder().encode([question], normalize_embeddings=True)[0]
        hits = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector.tolist(),
            limit=top_k,
            with_payload=True,
        ).points

        span.set_attribute("n_hits", len(hits))
        results = []
        for h in hits:
            p = h.payload or {}
            results.append(
                {
                    "text": p.get("text", ""),
                    "source": p.get("source"),
                    "headings": p.get("headings", []),
                    "pages": p.get("pages", []),
                    "score": h.score,
                }
            )
        return results
