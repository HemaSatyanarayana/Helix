"""Unit tests for the evaluation metrics.

The metrics are what every later claim about retrieval quality rests on, so the
arithmetic is pinned against hand-computed values rather than against whatever
the implementation happened to return.
"""

from __future__ import annotations

import math

import pytest

from evals.metrics import (
    Confusion,
    abstention_report,
    accuracy,
    aggregate,
    aggregate_reports,
    average_precision,
    bootstrap_ci,
    confusion,
    confusion_by_label,
    dcg_at_k,
    document_ranking,
    hit_rate_at_k,
    ndcg_at_k,
    normalize_source,
    paired_delta,
    precision_at_k,
    ranking_report,
    recall_at_k,
    reciprocal_rank,
)

# Two relevant documents ("a", "b") at ranks 1 and 3 of four results.
RETRIEVED = ["a", "x", "b", "y"]
RELEVANT = {"a", "b", "c"}  # "c" is relevant but never retrieved

LOG2_3 = math.log2(3)  # 1.5849625007211562


# --- Ranking: recall / precision / hit rate --------------------------------


def test_recall_at_k_counts_only_the_top_k():
    assert recall_at_k(RETRIEVED, RELEVANT, 1) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(2 / 3)


def test_recall_is_capped_by_unretrievable_documents():
    """"c" is never retrieved, so recall cannot reach 1.0 at any k."""
    assert recall_at_k(RETRIEVED, RELEVANT, 100) == pytest.approx(2 / 3)


def test_precision_denominated_by_what_was_actually_returned():
    assert precision_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(0.5)
    assert precision_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(0.5)
    # Only one result exists, and it is relevant: 1/1, not 1/4.
    assert precision_at_k(["a"], RELEVANT, 4) == pytest.approx(1.0)


def test_hit_rate_is_binary():
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 1) == 1.0
    assert hit_rate_at_k(["x", "a"], RELEVANT, 1) == 0.0
    assert hit_rate_at_k(["x", "a"], RELEVANT, 2) == 1.0


def test_empty_retrieval_scores_zero_but_precision_is_undefined():
    assert recall_at_k([], RELEVANT, 4) == 0.0
    assert hit_rate_at_k([], RELEVANT, 4) == 0.0
    assert precision_at_k([], RELEVANT, 4) is None


@pytest.mark.parametrize(
    "fn", [recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k]
)
def test_no_relevant_documents_is_undefined_not_zero(fn):
    """Off-topic queries stay in the set for abstention, but must not be
    averaged into retrieval scores in either direction."""
    assert fn(RETRIEVED, set(), 4) is None


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be >= 1"):
        recall_at_k(RETRIEVED, RELEVANT, 0)


# --- Ranking: rank-sensitive metrics ---------------------------------------


def test_reciprocal_rank_uses_the_first_relevant_hit():
    assert reciprocal_rank(["a", "b"], RELEVANT) == pytest.approx(1.0)
    assert reciprocal_rank(["x", "a"], RELEVANT) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y", "z", "a"], RELEVANT) == pytest.approx(0.25)
    assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0


def test_reciprocal_rank_respects_the_k_cutoff():
    assert reciprocal_rank(["x", "y", "a"], RELEVANT, k=2) == 0.0
    assert reciprocal_rank(["x", "y", "a"], RELEVANT, k=3) == pytest.approx(1 / 3)


def test_average_precision_rewards_finding_every_relevant_document():
    # Hits at ranks 1 and 3 -> (1/1 + 2/3) / 3 relevant documents.
    assert average_precision(RETRIEVED, RELEVANT) == pytest.approx((1.0 + 2 / 3) / 3)
    # Same hits one rank later -> strictly worse.
    assert average_precision(["x", "a", "y", "b"], RELEVANT) == pytest.approx(
        (0.5 + 0.5) / 3
    )


def test_average_precision_beats_mrr_on_multi_hop():
    """MRR is blind to the second source; AP is not."""
    one_source = ["a", "x", "y"]
    both_sources = ["a", "b", "y"]
    relevant = {"a", "b"}
    assert reciprocal_rank(one_source, relevant) == reciprocal_rank(both_sources, relevant)
    assert average_precision(both_sources, relevant) > average_precision(one_source, relevant)


# --- Ranking: nDCG ---------------------------------------------------------


def test_dcg_applies_the_positional_discount():
    assert dcg_at_k(["a"], {"a"}, 1) == pytest.approx(1.0)  # 1 / log2(2)
    assert dcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(1 / LOG2_3)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)


def test_ndcg_penalizes_rank_position():
    # Single relevant document demoted to rank 2: nDCG = (1/log2(3)) / 1.
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(0.6309297535714574)


def test_ndcg_normalizes_against_the_achievable_ideal():
    # DCG = 1/log2(3) + 1/log2(4); IDCG = 1/log2(2) + 1/log2(3).
    expected = (1 / LOG2_3 + 0.5) / (1.0 + 1 / LOG2_3)
    assert ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3) == pytest.approx(expected)
    assert 0.0 < expected < 1.0


def test_ndcg_uses_graded_relevance():
    graded = {"a": 1.0, "b": 3.0}
    # Best document second: DCG = 1 + 3/log2(3), IDCG = 3 + 1/log2(3).
    expected = (1.0 + 3 / LOG2_3) / (3.0 + 1 / LOG2_3)
    assert ndcg_at_k(["a", "b"], graded, 2) == pytest.approx(expected)
    # Ordering by grade scores strictly better than the reverse.
    assert ndcg_at_k(["b", "a"], graded, 2) == pytest.approx(1.0)


def test_ndcg_ideal_is_truncated_at_k():
    """With three relevant documents but k=1, retrieving one is a perfect top-1."""
    assert ndcg_at_k(["a"], {"a", "b", "c"}, 1) == pytest.approx(1.0)


# --- ranking_report --------------------------------------------------------


def test_ranking_report_covers_every_k():
    report = ranking_report(RETRIEVED, RELEVANT, ks=(1, 4))
    assert set(report) == {
        "recall@1", "precision@1", "hit_rate@1", "ndcg@1",
        "recall@4", "precision@4", "hit_rate@4", "ndcg@4",
        "mrr", "map",
    }
    assert report["recall@4"] == pytest.approx(2 / 3)
    assert report["mrr"] == pytest.approx(1.0)


# --- Classification --------------------------------------------------------


def test_confusion_counts_each_quadrant():
    m = confusion([True, True, False, False], [True, False, True, False])
    assert (m.tp, m.fn, m.fp, m.tn) == (1, 1, 1, 1)
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(0.5)
    assert m.accuracy == pytest.approx(0.5)
    assert m.support == 2


def test_confusion_undefined_metrics_are_none():
    empty = Confusion(tp=0, fp=0, fn=0, tn=0)
    assert empty.precision is None and empty.recall is None
    assert empty.f1 is None and empty.accuracy is None
    # Nothing predicted positive: precision undefined, recall genuinely 0.
    nothing_predicted = Confusion(tp=0, fp=0, fn=3, tn=1)
    assert nothing_predicted.precision is None
    assert nothing_predicted.recall == 0.0


def test_confusion_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        confusion([True], [True, False])


def test_router_accuracy_and_per_label_breakdown():
    y_true = ["technical", "conversational", "technical", "technical"]
    y_pred = ["technical", "technical", "technical", "conversational"]
    assert accuracy(y_true, y_pred) == pytest.approx(0.5)

    by_label = confusion_by_label(y_true, y_pred)
    assert set(by_label) == {"technical", "conversational"}
    # Two of three technical questions routed correctly; one greeting was not.
    assert by_label["technical"].recall == pytest.approx(2 / 3)
    assert by_label["technical"].precision == pytest.approx(2 / 3)
    assert by_label["conversational"].recall == 0.0


def test_accuracy_of_empty_set_is_undefined():
    assert accuracy([], []) is None


# --- Abstention ------------------------------------------------------------


def test_abstention_separates_the_two_failure_modes():
    #                idx:     0      1      2      3      4
    should_abstain = [True,  True,  False, False, False]
    did_abstain =    [True,  False, False, True,  False]
    report = abstention_report(should_abstain, did_abstain)

    assert report.n == 5
    # idx 1: off-topic question that got answered anyway — the dangerous error.
    assert report.false_answer_rate == pytest.approx(0.5)
    # idx 3: answerable question that got deflected — the annoying error.
    assert report.false_abstain_rate == pytest.approx(1 / 3)
    assert report.accuracy == pytest.approx(3 / 5)


def test_abstention_rates_are_undefined_without_examples_of_that_class():
    all_answerable = abstention_report([False, False], [False, True])
    assert all_answerable.false_answer_rate is None
    assert all_answerable.false_abstain_rate == pytest.approx(0.5)


def test_always_abstaining_is_not_scored_as_safe():
    """A gate that deflects everything has a perfect false-answer rate; the
    false-abstain rate is what stops that from looking good."""
    report = abstention_report([True, False, False], [True, True, True])
    assert report.false_answer_rate == 0.0
    assert report.false_abstain_rate == 1.0


# --- Aggregation -----------------------------------------------------------


def test_aggregate_skips_undefined_queries():
    agg = aggregate("recall@4", [1.0, 0.0, None, 0.5], confidence=None)
    assert agg.mean == pytest.approx(0.5)
    assert agg.n == 3
    assert agg.n_undefined == 1


def test_aggregate_of_all_undefined_has_no_mean():
    agg = aggregate("recall@4", [None, None])
    assert agg.mean is None
    assert agg.n == 0 and agg.n_undefined == 2
    assert "n/a" in str(agg)


def test_aggregate_reports_transposes_per_query_dicts():
    reports = [
        {"recall@4": 1.0, "mrr": 1.0},
        {"recall@4": 0.0, "mrr": 0.5},
        {"recall@4": None, "mrr": None},
    ]
    aggs = aggregate_reports(reports, confidence=None)
    assert aggs["recall@4"].mean == pytest.approx(0.5)
    assert aggs["mrr"].mean == pytest.approx(0.75)
    assert aggs["recall@4"].n_undefined == 1


def test_bootstrap_ci_is_seeded_and_brackets_the_mean():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    low, high = bootstrap_ci(values, seed=0)
    assert low <= sum(values) / len(values) <= high
    assert (low, high) == bootstrap_ci(values, seed=0)  # deterministic


def test_bootstrap_ci_of_a_constant_sample_is_degenerate():
    """Every resample is the same sample, so the interval collapses onto it."""
    assert bootstrap_ci([0.8] * 10, seed=0) == pytest.approx((0.8, 0.8))


def test_bootstrap_ci_narrows_as_the_sample_grows():
    """The reason a 3-point gain at n=50 is not a result."""
    small = bootstrap_ci([1.0, 0.0] * 10, seed=0)
    large = bootstrap_ci([1.0, 0.0] * 200, seed=0)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_ci([])


# --- Paired comparison -----------------------------------------------------


def test_paired_delta_counts_wins_and_losses():
    delta = paired_delta("recall@4", [0.5, 1.0, 0.0], [1.0, 1.0, 0.5])
    assert (delta.wins, delta.losses, delta.ties) == (2, 0, 1)
    assert delta.mean_delta == pytest.approx((0.5 + 0.0 + 0.5) / 3)
    assert delta.n == 3


def test_paired_delta_exposes_regressions_a_mean_would_hide():
    """Same aggregate movement, very different stories."""
    uniform = paired_delta("recall@4", [0.5, 0.5, 0.5], [0.7, 0.7, 0.7])
    mixed = paired_delta("recall@4", [0.5, 0.5, 0.5], [1.0, 1.0, 0.1])
    assert uniform.mean_delta == pytest.approx(mixed.mean_delta)  # identical headline
    assert (uniform.wins, uniform.losses) == (3, 0)
    assert (mixed.wins, mixed.losses) == (2, 1)


def test_paired_delta_skips_queries_undefined_in_either_run():
    delta = paired_delta("recall@4", [1.0, None, 0.5], [1.0, 1.0, None])
    assert delta.n == 1
    assert delta.ties == 1


def test_paired_delta_rejects_misaligned_runs():
    with pytest.raises(ValueError, match="length mismatch"):
        paired_delta("recall@4", [1.0], [1.0, 1.0])


# --- Document identity -----------------------------------------------------


def test_document_ranking_dedupes_and_keeps_best_rank():
    chunks = [
        {"source": "docs/a.md", "text": "..."},
        {"source": "docs/a.md", "text": "..."},
        {"source": "docs/b.md", "text": "..."},
        {"source": "docs/a.md", "text": "..."},
    ]
    assert document_ranking(chunks) == ["docs/a.md", "docs/b.md"]


def test_document_ranking_ignores_chunks_without_a_source():
    chunks = [{"text": "orphan"}, {"source": None}, {"source": "docs/a.md"}]
    assert document_ranking(chunks) == ["docs/a.md"]


def test_normalize_source_makes_labels_and_payloads_comparable(tmp_path):
    absolute = tmp_path / "data" / "guides" / "intro.md"
    assert normalize_source(str(absolute), root=tmp_path) == "data/guides/intro.md"
    # Already relative, or outside the root: left as-is rather than mangled.
    assert normalize_source("data/guides/intro.md", root=tmp_path) == "data/guides/intro.md"
