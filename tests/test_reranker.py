"""Tests for the reranker's relevance gate and diversity cap.

The core regression this guards: the abstain decision must use the
cross-encoder score, not the bi-encoder's, and must never apply a
cross-encoder threshold to vector-order results when Cohere is unavailable.
"""

from __future__ import annotations

import pytest

from app import reranker
from app.reranker import RerankResult, rerank


def _chunks(*, sources, scores=None):
    scores = scores or [1.0] * len(sources)
    return [
        {"text": f"chunk about {s}", "source": s, "score": sc}
        for s, sc in zip(sources, scores)
    ]


@pytest.fixture(autouse=True)
def force_cohere_path(monkeypatch):
    """Point RERANK_ENABLED + a fake key so tests hit the Cohere branch."""
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    monkeypatch.setenv("COHERE_API_KEY", "test-key")


def _stub_client(monkeypatch, scored: list[tuple[int, float]]):
    """scored: [(index, relevance_score), ...] as Cohere would return them."""

    class Result:
        def __init__(self, index, relevance_score):
            self.index = index
            self.relevance_score = relevance_score

    class Response:
        def __init__(self):
            self.results = [Result(i, s) for i, s in scored]

    class Client:
        def rerank(self, **kwargs):
            self.last_call = kwargs
            return Response()

    client = Client()
    monkeypatch.setattr(reranker, "_client", lambda: client)
    return client


# --- No candidates / disabled / unconfigured -------------------------------


def test_no_candidates_short_circuits():
    result = rerank("q", [])
    assert result == RerankResult(chunks=[], reranked=False, n_candidates=0)


def test_disabled_falls_back_to_vector_order(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ENABLED", False)
    chunks = _chunks(sources=["a", "b", "c"])
    result = rerank("q", chunks, top_k=2)
    assert result.reranked is False
    assert [c["source"] for c in result.chunks] == ["a", "b"]


def test_no_key_falls_back_to_vector_order(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    result = rerank("q", _chunks(sources=["a", "b"]))
    assert result.reranked is False


def test_vector_fallback_applies_no_relevance_threshold(monkeypatch):
    """Vector-order results are cosine scores; a cross-encoder threshold would
    mean something different against them, so none is applied."""
    monkeypatch.setattr(reranker, "RERANK_ENABLED", False)
    chunks = _chunks(sources=["a", "b"], scores=[0.01, 0.01])  # far below any bar
    result = rerank("q", chunks, top_k=2)
    assert len(result.chunks) == 2


# --- Cohere path: relevance gate -------------------------------------------


def test_relevance_threshold_filters_the_pool(monkeypatch):
    _stub_client(monkeypatch, [(0, 0.95), (1, 0.05), (2, 0.5)])
    chunks = _chunks(sources=["a", "b", "c"])
    result = rerank("q", chunks, top_k=10, score_threshold=0.2)

    assert result.reranked is True
    assert [c["source"] for c in result.chunks] == ["a", "c"]
    assert result.n_below_threshold == 1


def test_rerank_score_is_attached_to_survivors(monkeypatch):
    _stub_client(monkeypatch, [(0, 0.87)])
    result = rerank("q", _chunks(sources=["a"]), score_threshold=0.2)
    assert result.chunks[0]["rerank_score"] == pytest.approx(0.87)


def test_nothing_clearing_the_bar_returns_empty(monkeypatch):
    """This is the abstain signal: an empty result, not a low-confidence one."""
    _stub_client(monkeypatch, [(0, 0.1), (1, 0.15)])
    result = rerank("q", _chunks(sources=["a", "b"]), score_threshold=0.2)
    assert result.chunks == []
    assert result.reranked is True  # the pipeline ran; nothing was relevant
    assert result.n_below_threshold == 2


def test_cohere_scores_the_whole_pool_not_just_top_k(monkeypatch):
    """top_n must not be capped to top_k before the threshold is applied, or
    a relevant chunk ranked 5th among 20 candidates never gets scored."""
    client = _stub_client(monkeypatch, [(i, 1.0) for i in range(10)])
    rerank("q", _chunks(sources=[str(i) for i in range(10)]), top_k=4)
    assert client.last_call["top_n"] == 10


# --- Cohere path: outage -----------------------------------------------


def test_cohere_outage_fails_open_to_vector_order(monkeypatch):
    class Client:
        def rerank(self, **kwargs):
            raise RuntimeError("503 from Cohere")

    monkeypatch.setattr(reranker, "_client", lambda: Client())
    result = rerank("q", _chunks(sources=["a", "b"]), top_k=2)
    assert result.reranked is False
    assert len(result.chunks) == 2


# --- Diversity cap -----------------------------------------------------


def test_diversity_cap_limits_chunks_per_document(monkeypatch):
    sources = ["a", "a", "a", "b"]
    _stub_client(monkeypatch, [(i, 1.0 - i * 0.01) for i in range(4)])
    result = rerank(
        "q", _chunks(sources=sources), top_k=4, score_threshold=0.0, max_per_document=1
    )
    assert [c["source"] for c in result.chunks] == ["a", "b"]


def test_diversity_cap_preserves_relevance_order(monkeypatch):
    """Capping must not reorder — the best two of "a" beat the third even
    though it would otherwise have filled the second-to-last slot."""
    sources = ["a", "a", "a", "b"]
    _stub_client(monkeypatch, [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)])
    result = rerank(
        "q", _chunks(sources=sources), top_k=4, score_threshold=0.0, max_per_document=2
    )
    assert [c["source"] for c in result.chunks] == ["a", "a", "b"]


def test_diversity_cap_disabled_with_zero(monkeypatch):
    sources = ["a", "a", "a"]
    _stub_client(monkeypatch, [(i, 1.0) for i in range(3)])
    result = rerank(
        "q", _chunks(sources=sources), top_k=3, score_threshold=0.0, max_per_document=0
    )
    assert len(result.chunks) == 3


def test_falsy_bool_when_empty():
    assert not RerankResult(chunks=[])
    assert RerankResult(chunks=[{"text": "x"}])
