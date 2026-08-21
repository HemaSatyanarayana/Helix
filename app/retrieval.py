"""Retrieval layer — search Qdrant for chunks relevant to a question.

Hybrid by default: a dense (semantic) and a sparse (BM25 lexical) ranking are
fused with Reciprocal Rank Fusion inside Qdrant, in a single round trip. See
:mod:`app.embedding` for why product docs need both.

Two behaviours here are deliberate and easy to get wrong in the opposite
direction:

**No relevance floor on the candidate pool.** This stage optimises *recall* —
its only job is to make sure the right chunk is somewhere in the pool the
reranker will see. A cosine cutoff here (the previous default of 0.4) pruned
exactly the passages the cross-encoder exists to rescue, and did it before the
precise scorer ever ran. Relevance is decided later, on the rerank score.
``score_threshold`` remains available for callers that want it.

**Diversity capping.** Chunks are ranked independently, so the top of a hybrid
ranking is routinely four chunks of the same page — measured on this corpus,
every one of the top 4 for "how do I add a new app" came from a single file.
``max_per_document`` interleaves so the context window spans sources.

Falls back to dense-only search against collections that predate the hybrid
schema, so an un-migrated index keeps working (degraded, and logged).
"""

from __future__ import annotations

import os
from functools import lru_cache

from app.embedding import (
    DENSE_VECTOR,
    HYBRID_ENABLED,
    SPARSE_VECTOR,
    embed_dense_query,
    embed_sparse_query,
)
from app.telemetry import configure_logfire

logfire = configure_logfire("helix-app")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
# May name a collection or an alias — ingestion builds into a versioned
# collection and points this alias at it (see ingestion-workers/worker.py).
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
TOP_K = int(os.getenv("TOP_K", "4"))
# Optional. Left unset on purpose: see the module docstring.
_RAW_THRESHOLD = os.getenv("RETRIEVAL_SCORE_THRESHOLD", "").strip()
RETRIEVAL_SCORE_THRESHOLD = float(_RAW_THRESHOLD) if _RAW_THRESHOLD else None
# How many chunks of any one document may appear in the results. 0 = unlimited.
MAX_PER_DOCUMENT = int(os.getenv("MAX_PER_DOCUMENT", "2"))


@lru_cache(maxsize=1)
def _qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


@lru_cache(maxsize=8)
def _collection_is_hybrid(collection: str) -> bool:
    """Whether the collection carries the named dense + sparse vector schema.

    Cached: the schema cannot change under a running process without a restart
    or an alias swap, and this would otherwise add a round trip per query.
    """
    try:
        info = _qdrant().get_collection(collection)
        params = info.config.params
        vectors = params.vectors if isinstance(params.vectors, dict) else {}
        sparse = params.sparse_vectors or {}
        return DENSE_VECTOR in vectors and SPARSE_VECTOR in sparse
    except Exception as exc:  # noqa: BLE001
        logfire.warning("collection_schema_unknown", collection=collection, error=str(exc))
        return False


def _source_filter(sources: list[str] | None):
    """Restrict a search to specific source documents."""
    if not sources:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    return Filter(must=[FieldCondition(key="source", match=MatchAny(any=sources))])


def _cap_per_document(hits: list[dict], max_per_document: int) -> list[dict]:
    """Trim to at most N chunks per source, preserving rank order."""
    if max_per_document <= 0:
        return hits
    seen: dict[str, int] = {}
    kept = []
    for hit in hits:
        source = hit.get("source") or ""
        if seen.get(source, 0) >= max_per_document:
            continue
        seen[source] = seen.get(source, 0) + 1
        kept.append(hit)
    return kept


def _payload_to_hit(point) -> dict:
    p = point.payload or {}
    return {
        "text": p.get("text", ""),
        "source": p.get("source"),
        "doc_id": p.get("doc_id"),
        "headings": p.get("headings", []),
        "pages": p.get("pages", []),
        "score": point.score,
    }


def retrieve(
    question: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
    *,
    sources: list[str] | None = None,
    max_per_document: int | None = None,
    hybrid: bool | None = None,
) -> list[dict]:
    """Return up to ``top_k`` chunks for a question, best first.

    ``score_threshold`` defaults to unset — the candidate pool is not filtered
    by relevance here. Note that with hybrid search the score is an RRF rank
    score, not a cosine similarity, so any threshold must be calibrated against
    the mode actually in use.
    """
    top_k = top_k or TOP_K
    if score_threshold is None:
        score_threshold = RETRIEVAL_SCORE_THRESHOLD
    if max_per_document is None:
        max_per_document = MAX_PER_DOCUMENT
    use_hybrid = HYBRID_ENABLED if hybrid is None else hybrid

    with logfire.span(
        "retrieve", top_k=top_k, score_threshold=score_threshold, hybrid=use_hybrid
    ) as span:
        client = _qdrant()
        if not client.collection_exists(QDRANT_COLLECTION):
            logfire.warning("collection_missing", collection=QDRANT_COLLECTION)
            span.set_attribute("n_hits", 0)
            return []

        query_filter = _source_filter(sources)
        # Over-fetch so the diversity cap has spares to promote; without this,
        # capping would simply return fewer results than asked for.
        fetch = top_k * 3 if max_per_document > 0 else top_k

        if use_hybrid and _collection_is_hybrid(QDRANT_COLLECTION):
            points = _hybrid_search(question, fetch, score_threshold, query_filter)
            span.set_attribute("mode", "hybrid")
        else:
            if use_hybrid:
                logfire.warning(
                    "hybrid_unavailable",
                    collection=QDRANT_COLLECTION,
                    detail="collection lacks named dense+sparse vectors; run scripts/migrate_hybrid.py",
                )
            points = _dense_search(question, fetch, score_threshold, query_filter)
            span.set_attribute("mode", "dense")

        hits = [_payload_to_hit(p) for p in points]
        capped = _cap_per_document(hits, max_per_document)[:top_k]

        span.set_attribute("n_hits", len(capped))
        span.set_attribute("n_documents", len({h["source"] for h in capped}))
        span.set_attribute("n_dropped_by_diversity", len(hits) - len(capped))
        return capped


def _hybrid_search(question: str, limit: int, score_threshold: float | None, query_filter):
    """Dense + BM25 prefetch fused with RRF, server-side in one round trip."""
    from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

    sparse = embed_sparse_query(question)
    # Each branch fetches deeper than the final limit: fusion can only reorder
    # what the branches surfaced, so a shallow prefetch caps hybrid recall.
    branch_limit = max(limit * 2, 50)
    return _qdrant().query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            Prefetch(
                query=embed_dense_query(question),
                using=DENSE_VECTOR,
                limit=branch_limit,
                filter=query_filter,
            ),
            Prefetch(
                query=SparseVector(indices=sparse["indices"], values=sparse["values"]),
                using=SPARSE_VECTOR,
                limit=branch_limit,
                filter=query_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    ).points


def _dense_search(question: str, limit: int, score_threshold: float | None, query_filter):
    """Dense-only search — legacy collections, or HYBRID_ENABLED=false."""
    kwargs = {}
    if _collection_is_hybrid(QDRANT_COLLECTION):
        kwargs["using"] = DENSE_VECTOR
    return _qdrant().query_points(
        collection_name=QDRANT_COLLECTION,
        query=embed_dense_query(question),
        limit=limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
        with_payload=True,
        **kwargs,
    ).points
