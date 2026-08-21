#!/usr/bin/env python
"""Sweep RERANK_SCORE_THRESHOLD and report the false-answer / false-abstain
tradeoff at each value.

    uv run python evals/sweep_threshold.py

Retrieves and reranks each golden-set question **once** (the expensive part —
one Cohere call per question) and then replays the cached rerank scores against
every threshold in the sweep, so the API cost is independent of how many
threshold values are tested.

The two rates trade off directly: raise the threshold and false_answer_rate
falls while false_abstain_rate rises. There is no single correct value — it's
a product decision about which error costs more — but this turns "the
threshold is calibrated" from a comment (see ``app/reranker.py``) into a number
you can point at.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.metrics import abstention_report  # noqa: E402
from evals.run_eval import load_dataset  # noqa: E402

THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def collect_scored_runs() -> list[dict]:
    """One retrieve+rerank pass per question, keeping every candidate's score."""
    from app.generation import classify_query, rewrite_query
    from app.guardrails import check_question
    from app.reranker import RERANK_TOP_K, rerank
    from app.retrieval import retrieve

    dataset = load_dataset()
    runs = []
    for item in dataset:
        if "should_abstain" not in item:
            continue  # only labeled-abstention questions inform the sweep

        question, history = item["question"], item.get("history")
        guard = check_question(question, history)
        if not guard.ok:
            # A blocked question abstains at every threshold — score_threshold
            # is irrelevant to it, so it's excluded rather than padding every
            # column with the same constant 1.0.
            continue

        route = classify_query(question, history)
        if route != "technical":
            runs.append({"id": item["id"], "should_abstain": item["should_abstain"], "scores": []})
            continue

        query = rewrite_query(question, history)
        pool = retrieve(query, top_k=20, max_per_document=0)
        # score_threshold=0 keeps every candidate so replay can apply any bar.
        result = rerank(query, pool, top_k=len(pool) or 1, score_threshold=0.0)
        scores = sorted((c["rerank_score"] for c in result.chunks), reverse=True)
        runs.append({"id": item["id"], "should_abstain": item["should_abstain"], "scores": scores[:RERANK_TOP_K]})
        print(f"  scored {item['id']}", file=sys.stderr)
    return runs


def sweep(runs: list[dict], thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        did_abstain = [not any(s >= threshold for s in r["scores"]) for r in runs]
        should_abstain = [r["should_abstain"] for r in runs]
        report = abstention_report(should_abstain, did_abstain)
        rows.append(
            {
                "threshold": threshold,
                "false_answer_rate": report.false_answer_rate,
                "false_abstain_rate": report.false_abstain_rate,
                "accuracy": report.accuracy,
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"\n{'threshold':>10} {'false_answer':>13} {'false_abstain':>14} {'accuracy':>9}")
    for row in rows:
        def fmt(x):
            return "n/a" if x is None else f"{x:.3f}"

        print(
            f"{row['threshold']:>10.2f} {fmt(row['false_answer_rate']):>13} "
            f"{fmt(row['false_abstain_rate']):>14} {fmt(row['accuracy']):>9}"
        )

    from app.reranker import RERANK_SCORE_THRESHOLD

    print(f"\ncurrent RERANK_SCORE_THRESHOLD = {RERANK_SCORE_THRESHOLD}")
    print(
        "false_answer_rate is the dangerous error (an ungrounded answer presented\n"
        "as fact); false_abstain_rate is the annoying one (a real answer withheld).\n"
        "Pick the threshold whose false_answer_rate you can live with, then read off\n"
        "the false_abstain_rate you're trading for it — there is no free lunch here."
    )


def main() -> int:
    print("scoring each question once against Cohere...", file=sys.stderr)
    runs = collect_scored_runs()
    if not runs:
        print("no labeled-abstention questions in the dataset (need `should_abstain`)")
        return 1
    rows = sweep(runs, THRESHOLDS)
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
