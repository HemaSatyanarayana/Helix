"""Evaluation metrics for the Helix retrieval pipeline.

Pure functions over opaque document IDs — no LLM, no network, no model. That
makes every metric here deterministic, free, and fast enough to gate CI on.
(Generation-quality metrics — faithfulness, answer relevance — need a judge and
live elsewhere.)

Three families:

* **Ranking** — ``recall_at_k`` … ``ndcg_at_k``. Did retrieval surface the right
  document, and did reranking order it into the slice the LLM actually sees?
* **Classification** — ``confusion``, ``accuracy``, ``abstention_report``. Did
  the router and the abstain gate make the right call?
* **Aggregation** — ``aggregate`` (mean + bootstrap CI) and ``paired_delta``
  (per-query wins/losses), for comparing two pipeline configurations.

Two conventions worth knowing before reading numbers off these:

**Document IDs, not chunk IDs.** Everything works on opaque strings, and the
runner feeds them source paths. Chunk IDs are content-addressed
(``worker.py:106``) so they survive re-ingestion — but *not* a chunker config
change, which is exactly the experiment you want to measure. Document-level
labels survive re-chunking, re-embedding, and reindexing.

**Undefined ≠ zero.** A query with no relevant documents (an off-topic one, kept
in the set to test abstention) has no meaningful recall. Those metrics return
``None`` rather than a number, and ``aggregate`` excludes them from the mean.
Scoring them 1.0 would silently inflate every headline figure with exactly the
queries retrieval is supposed to fail.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Either an iterable of relevant IDs (binary relevance) or a mapping of
# ID -> graded relevance (0 = irrelevant, 1 = related, 2+ = directly answers).
Relevance = Mapping[str, float] | Iterable[str]


def _grades(relevant: Relevance) -> dict[str, float]:
    """Normalize either relevance form into {doc_id: grade>0}."""
    if isinstance(relevant, Mapping):
        return {d: float(g) for d, g in relevant.items() if g > 0}
    return {d: 1.0 for d in relevant}


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


# --- Ranking metrics ------------------------------------------------------


def recall_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float | None:
    """Fraction of relevant documents present in the top-k.

    The ceiling on everything downstream: a document missing from the candidate
    pool cannot be reranked into the context, and no prompt recovers it. Measure
    at the pool size (k=RERANK_CANDIDATES) to read that ceiling directly.

    Returns None when the query has no relevant documents.
    """
    _check_k(k)
    grades = _grades(relevant)
    if not grades:
        return None
    return len(set(retrieved[:k]) & set(grades)) / len(grades)


def precision_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float | None:
    """Fraction of the top-k that is relevant.

    Denominated by how many documents were actually returned, not by ``k`` — the
    question is how clean the context block sent to the LLM was, and a short
    block should not be scored as an imprecise one. Returns None when nothing
    was retrieved or the query has no relevant documents.
    """
    _check_k(k)
    grades = _grades(relevant)
    top = retrieved[:k]
    if not grades or not top:
        return None
    return sum(1 for d in top if d in grades) / len(top)


def hit_rate_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float | None:
    """1.0 if any relevant document is in the top-k, else 0.0.

    Averaged over a set, this is Success@k — the most interpretable headline
    number ("88% of questions have a correct source in the top 4").
    """
    _check_k(k)
    grades = _grades(relevant)
    if not grades:
        return None
    return 1.0 if set(retrieved[:k]) & set(grades) else 0.0


def reciprocal_rank(
    retrieved: Sequence[str], relevant: Relevance, k: int | None = None
) -> float | None:
    """1 / rank of the first relevant document (0.0 if none in the top-k).

    Averaged over a set this is MRR. Falls off fast by design — rank 1 scores
    1.0, rank 5 only 0.2 — so use it when one source is clearly the right one.
    """
    grades = _grades(relevant)
    if not grades:
        return None
    if k is not None:
        _check_k(k)
        retrieved = retrieved[:k]
    for i, doc in enumerate(retrieved):
        if doc in grades:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    retrieved: Sequence[str], relevant: Relevance, k: int | None = None
) -> float | None:
    """Mean of precision@i over every rank i holding a relevant document.

    Averaged over a set this is MAP. Unlike MRR it rewards finding *all* the
    relevant documents, so it's the better fit for multi-hop questions whose
    answer spans several pages.
    """
    grades = _grades(relevant)
    if not grades:
        return None
    if k is not None:
        _check_k(k)
        retrieved = retrieved[:k]

    hits = 0
    precision_sum = 0.0
    for i, doc in enumerate(retrieved):
        if doc in grades:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(grades)


def dcg_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float:
    """Discounted Cumulative Gain: sum of grade / log2(rank + 1) over the top-k."""
    _check_k(k)
    grades = _grades(relevant)
    return sum(
        grades.get(doc, 0.0) / math.log2(i + 2) for i, doc in enumerate(retrieved[:k])
    )


def ndcg_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float | None:
    """DCG normalized by the best achievable ordering (0..1).

    The best single number for *rerank* quality: reranking is purely an ordering
    problem, and this is the metric that grades ordering. Also handles graded
    relevance, which matters when several pages are partially on-topic.
    """
    _check_k(k)
    grades = _grades(relevant)
    if not grades:
        return None
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return None
    return dcg_at_k(retrieved, grades, k) / idcg


def ranking_report(
    retrieved: Sequence[str],
    relevant: Relevance,
    ks: Sequence[int] = (1, 4, 20),
) -> dict[str, float | None]:
    """Every ranking metric at every k, keyed ``"recall@4"`` and so on.

    Report recall at the pool size and nDCG at the final size together: high
    recall@20 with low nDCG@4 means the reranker is the problem, while low
    recall@20 means retrieval never had a chance.
    """
    report: dict[str, float | None] = {}
    for k in ks:
        report[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        report[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        report[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        report[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    report["mrr"] = reciprocal_rank(retrieved, relevant)
    report["map"] = average_precision(retrieved, relevant)
    return report


# --- Classification metrics -----------------------------------------------


@dataclass(frozen=True)
class Confusion:
    """Binary confusion matrix for one positive class."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def support(self) -> int:
        """Number of genuinely positive cases."""
        return self.tp + self.fn

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float | None:
        predicted = self.tp + self.fp
        return self.tp / predicted if predicted else None

    @property
    def recall(self) -> float | None:
        return self.tp / self.support if self.support else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float | None:
        return (self.tp + self.tn) / self.total if self.total else None


def confusion(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Confusion:
    """Build a confusion matrix from paired boolean labels and predictions."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} labels, {len(y_pred)} predictions")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float | None:
    """Plain multi-class accuracy — used for router labels."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} labels, {len(y_pred)} predictions")
    if not y_true:
        return None
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def confusion_by_label(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> dict[str, Confusion]:
    """One-vs-rest confusion matrix per class label.

    For the router this separates the two failure modes: a product question sent
    down the conversational branch answers without sources, while a greeting sent
    down the technical branch just burns a retrieval.
    """
    labels = sorted(set(y_true) | set(y_pred))
    return {
        label: confusion([t == label for t in y_true], [p == label for p in y_pred])
        for label in labels
    }


@dataclass(frozen=True)
class AbstentionReport:
    """Quality of the abstain-vs-answer decision.

    ``false_answer_rate`` is the dangerous error: an ungrounded answer to a
    question the corpus doesn't cover. ``false_abstain_rate`` is the annoying
    one: deflecting a question the docs do cover. They trade off against each
    other, and the score threshold is the knob — sweep it, plot both, and pick
    the operating point deliberately rather than inheriting a default.
    """

    matrix: Confusion  # positive class = "abstained"

    @property
    def n(self) -> int:
        return self.matrix.total

    @property
    def false_answer_rate(self) -> float | None:
        """P(answered | should have abstained)."""
        return None if not self.matrix.support else self.matrix.fn / self.matrix.support

    @property
    def false_abstain_rate(self) -> float | None:
        """P(abstained | should have answered)."""
        answerable = self.matrix.fp + self.matrix.tn
        return self.matrix.fp / answerable if answerable else None

    @property
    def accuracy(self) -> float | None:
        return self.matrix.accuracy


def abstention_report(
    should_abstain: Sequence[bool], did_abstain: Sequence[bool]
) -> AbstentionReport:
    """Score the abstain gate against per-query labels."""
    return AbstentionReport(matrix=confusion(should_abstain, did_abstain))


# --- Aggregation ----------------------------------------------------------


@dataclass(frozen=True)
class Aggregate:
    """A metric averaged over a query set, with its uncertainty."""

    name: str
    mean: float | None
    n: int  # queries the metric was defined for
    n_undefined: int  # queries skipped (no relevant documents)
    ci_low: float | None = None
    ci_high: float | None = None

    def __str__(self) -> str:
        if self.mean is None:
            return f"{self.name}: n/a (0 of {self.n_undefined} queries defined)"
        ci = ""
        if self.ci_low is not None and self.ci_high is not None:
            ci = f"  [{self.ci_low:.3f}, {self.ci_high:.3f}]"
        return f"{self.name}: {self.mean:.3f}{ci}  (n={self.n})"


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Seeded, so a given value set always yields the same interval. Worth reading
    before celebrating a change: at n=50 the interval on recall is typically
    ±0.1, which is wider than most improvements people report.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(n)) / n for _ in range(iterations)
    )
    lo_idx = int((1 - confidence) / 2 * iterations)
    hi_idx = min(int((1 + confidence) / 2 * iterations), iterations - 1)
    return means[lo_idx], means[hi_idx]


def aggregate(
    name: str,
    values: Sequence[float | None],
    *,
    confidence: float | None = 0.95,
    iterations: int = 2000,
    seed: int = 0,
) -> Aggregate:
    """Mean of the defined values, with a bootstrap CI and a skip count."""
    defined = [v for v in values if v is not None]
    n_undefined = len(values) - len(defined)
    if not defined:
        return Aggregate(name=name, mean=None, n=0, n_undefined=n_undefined)

    ci_low = ci_high = None
    if confidence is not None and len(defined) > 1:
        ci_low, ci_high = bootstrap_ci(defined, confidence, iterations, seed)
    return Aggregate(
        name=name,
        mean=sum(defined) / len(defined),
        n=len(defined),
        n_undefined=n_undefined,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def aggregate_reports(
    reports: Sequence[Mapping[str, float | None]], **kwargs
) -> dict[str, Aggregate]:
    """Aggregate a list of per-query ``ranking_report`` dicts, metric by metric."""
    names: list[str] = []
    for report in reports:
        names.extend(name for name in report if name not in names)
    return {
        name: aggregate(name, [r.get(name) for r in reports], **kwargs) for name in names
    }


@dataclass(frozen=True)
class PairedDelta:
    """Per-query comparison of two configurations on the same query set.

    The aggregate delta hides which queries moved. "recall 0.82 -> 0.85" is
    noise-shaped; "6 fixed, 2 regressed" tells you whether to look at the two
    regressions before shipping.
    """

    name: str
    n: int
    wins: int
    losses: int
    ties: int
    mean_delta: float | None

    def __str__(self) -> str:
        delta = "n/a" if self.mean_delta is None else f"{self.mean_delta:+.3f}"
        return (
            f"{self.name}: {delta}  "
            f"({self.wins} better, {self.losses} worse, {self.ties} same, n={self.n})"
        )


def paired_delta(
    name: str,
    baseline: Sequence[float | None],
    candidate: Sequence[float | None],
) -> PairedDelta:
    """Compare two runs query-by-query; queries undefined in either are skipped."""
    if len(baseline) != len(candidate):
        raise ValueError(
            f"length mismatch: {len(baseline)} baseline, {len(candidate)} candidate"
        )
    pairs = [(b, c) for b, c in zip(baseline, candidate) if b is not None and c is not None]
    if not pairs:
        return PairedDelta(name=name, n=0, wins=0, losses=0, ties=0, mean_delta=None)

    wins = sum(1 for b, c in pairs if c > b)
    losses = sum(1 for b, c in pairs if c < b)
    return PairedDelta(
        name=name,
        n=len(pairs),
        wins=wins,
        losses=losses,
        ties=len(pairs) - wins - losses,
        mean_delta=sum(c - b for b, c in pairs) / len(pairs),
    )


# --- Document identity helpers --------------------------------------------


def document_ranking(chunks: Sequence[Mapping], key: str = "source") -> list[str]:
    """Collapse a chunk ranking into a document ranking, best rank first.

    Retrieval returns chunks and several may share a source, so a top-4 chunk
    list can be two documents. Deduplicating keeps document-level metrics honest
    — and the gap between ``len(chunks)`` and ``len(document_ranking(chunks))``
    is itself the diversity signal worth watching.
    """
    seen: list[str] = []
    for chunk in chunks:
        doc = chunk.get(key)
        if doc and doc not in seen:
            seen.append(doc)
    return seen


def normalize_source(path: str, root: str | Path | None = None) -> str:
    """Normalize a source path so labels and payloads compare equal.

    Ingestion stores whatever path it was handed (``scripts/ingest.py`` passes
    what the caller typed), so the same document can be recorded absolute in one
    run and relative in the next. Labels are written repo-relative and POSIX.
    """
    p = Path(path)
    if root is not None:
        try:
            p = p.resolve().relative_to(Path(root).resolve())
        except ValueError:
            p = Path(path)
    return p.as_posix()
