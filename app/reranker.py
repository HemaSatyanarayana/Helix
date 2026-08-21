"""Reranking via Cohere Rerank (English).

A cross-encoder reranker re-scores each (query, chunk) pair jointly, which is
far more precise than the bi-encoder vector score. We retrieve a wide candidate
pool from Qdrant and rerank it down to the top-k that actually go to the LLM.

The reranker's score is also what decides whether to answer at all. That
ordering matters and was previously inverted: a cosine floor pruned the
candidate pool *before* the cross-encoder ran — discarding exactly the passages
reranking exists to rescue — while nothing ever checked the precise score the
reranker produced. Now the pool arrives unfiltered and
:data:`RERANK_SCORE_THRESHOLD` decides relevance on the better signal.

Falls back to the vector order (trimmed to top-k) when no COHERE_API_KEY is set
or on transient errors, so the pipeline never hard-fails. That fallback is
reported in :class:`RerankResult.reranked` rather than left silent, because
without it a degraded pipeline is indistinguishable from a healthy one — and
the abstain gate must not apply a cross-encoder threshold to cosine scores.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from app.telemetry import configure_logfire

logfire = configure_logfire("helix-app")

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "rerank-english-v3.0")
# How many candidates to pull from Qdrant before reranking down to RERANK_TOP_K.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "4"))
# Minimum cross-encoder relevance for a chunk to reach the LLM. Cohere v3
# relevance runs 0..1 and is well separated: on-topic chunks score high, weakly
# related ones cluster low. Calibrate with `python evals/sweep_threshold.py`.
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.2"))
# Cap chunks per source document so the context spans more than one page.
RERANK_MAX_PER_DOCUMENT = int(os.getenv("RERANK_MAX_PER_DOCUMENT", "2"))


@dataclass
class RerankResult:
    """Reranked chunks plus the provenance of *how* they were ranked."""

    chunks: list[dict] = field(default_factory=list)
    reranked: bool = False  # False => vector order, scores are not comparable
    n_candidates: int = 0
    n_below_threshold: int = 0

    def __bool__(self) -> bool:
        return bool(self.chunks)


@lru_cache(maxsize=1)
def _client():
    import cohere

    return cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


def _cap_per_document(chunks: list[dict], max_per_document: int) -> list[dict]:
    """Trim to at most N chunks per source, preserving rank order."""
    if max_per_document <= 0:
        return chunks
    seen: dict[str, int] = {}
    kept = []
    for chunk in chunks:
        source = chunk.get("source") or ""
        if seen.get(source, 0) >= max_per_document:
            continue
        seen[source] = seen.get(source, 0) + 1
        kept.append(chunk)
    return kept


def rerank(
    question: str,
    chunks: list[dict],
    top_k: int | None = None,
    *,
    score_threshold: float | None = None,
    max_per_document: int | None = None,
) -> RerankResult:
    """Reorder chunks by cross-encoder relevance and keep the relevant top-k."""
    top_k = top_k or RERANK_TOP_K
    if score_threshold is None:
        score_threshold = RERANK_SCORE_THRESHOLD
    if max_per_document is None:
        max_per_document = RERANK_MAX_PER_DOCUMENT

    if not chunks:
        return RerankResult(chunks=[], reranked=False, n_candidates=0)

    # Fallback: keep vector order when reranking is disabled or unconfigured.
    # No threshold is applied — these are cosine scores, and the cross-encoder
    # threshold would mean something entirely different against them.
    if not RERANK_ENABLED or not os.getenv("COHERE_API_KEY"):
        logfire.warning("rerank_skipped", reason="disabled or no COHERE_API_KEY")
        return RerankResult(
            chunks=_cap_per_document(chunks, max_per_document)[:top_k],
            reranked=False,
            n_candidates=len(chunks),
        )

    with logfire.span(
        "rerank", model=RERANKER_MODEL, n_candidates=len(chunks), top_k=top_k
    ) as span:
        try:
            response = _client().rerank(
                model=RERANKER_MODEL,
                query=question,
                documents=[c["text"] for c in chunks],
                # Rank the whole pool: the threshold and the diversity cap both
                # need the full ordering, not a pre-trimmed head.
                top_n=len(chunks),
            )
        except Exception as exc:  # noqa: BLE001 — fail open to vector order.
            logfire.warning("rerank_unavailable", error=str(exc))
            return RerankResult(
                chunks=_cap_per_document(chunks, max_per_document)[:top_k],
                reranked=False,
                n_candidates=len(chunks),
            )

        ranked = []
        for r in response.results:
            chunk = dict(chunks[r.index])
            chunk["rerank_score"] = r.relevance_score
            ranked.append(chunk)

        relevant = [c for c in ranked if c["rerank_score"] >= score_threshold]
        kept = _cap_per_document(relevant, max_per_document)[:top_k]

        span.set_attribute("n_out", len(kept))
        span.set_attribute("n_below_threshold", len(ranked) - len(relevant))
        span.set_attribute("top_score", ranked[0]["rerank_score"] if ranked else None)
        return RerankResult(
            chunks=kept,
            reranked=True,
            n_candidates=len(chunks),
            n_below_threshold=len(ranked) - len(relevant),
        )
