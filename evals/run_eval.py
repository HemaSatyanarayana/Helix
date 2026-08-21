#!/usr/bin/env python
"""Run the golden set against the live retrieval pipeline and report metrics.

    uv run python evals/run_eval.py                 # print a report
    uv run python evals/run_eval.py --json out.json  # also write raw results
    uv run python evals/run_eval.py --save-baseline  # write the CI baseline
    uv run python evals/run_eval.py --compare-to baseline.json

Exercises retrieval and reranking directly (not the full chat graph), so it
needs no LLM key — Qdrant and, if configured, Cohere are the only
dependencies. That keeps it fast and free enough to run on every PR; the
groundedness/faithfulness side needs a judge and lives separately (see the
`evals/dataset/README.md` for why the two are split).

Reports two numbers together deliberately: pool-level recall (before rerank)
and post-rerank precision/nDCG. High recall with low nDCG means the reranker
is the problem; low recall means retrieval never had a chance — see
`evals/metrics.py` for the reasoning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from evals.metrics import (  # noqa: E402
    aggregate_reports,
    abstention_report,
    accuracy,
    document_ranking,
    normalize_source,
    paired_delta,
    ranking_report,
)

DATASET = ROOT / "evals" / "dataset" / "golden.yaml"
BASELINE = ROOT / "evals" / "dataset" / "baseline.json"
POOL_K = 20  # matches RERANK_CANDIDATES; the recall ceiling is measured here
FINAL_K = 4  # matches RERANK_TOP_K; nDCG/precision are measured here
# Seconds between questions. Reranking calls Cohere once per question; a trial
# key's rate limit is tripped by firing 64 of them back-to-back (measured: 13
# to 23 degraded to vector order with no pacing). Cheap insurance against a
# multi-minute retry storm — set RERANK_PACING_SECONDS=0 on a paid key.
PACING_SECONDS = float(os.environ.get("RERANK_PACING_SECONDS", "1.5"))


def load_dataset(path: Path = DATASET) -> list[dict]:
    return yaml.safe_load(path.read_text())


def _sources(chunks: list[dict]) -> list[str]:
    return [normalize_source(s, root=ROOT) for s in document_ranking(chunks)]


def run_one(item: dict) -> dict:
    """Retrieve + rerank one golden-set item; return raw outputs for scoring.

    A eval run fires rerank requests back-to-back in a way normal traffic never
    does, which is exactly the pattern a trial-tier rate limit is built to
    catch: on an unpaced first run, 13-23 of 64 Cohere calls failed mid-run and
    silently degraded those queries to vector order. Retrying each failure with
    backoff made it worse, not better — 21s of sleep per degraded query turned
    a 4-minute run into 8. The runner paces every rerank call with a small
    fixed delay instead (see ``main``); a call that still fails is reported as
    degraded (``score()``'s ``degraded`` list) rather than silently averaged
    into the final-stage metrics.
    """
    from app.generation import classify_query, rewrite_query
    from app.guardrails import check_question
    from app.reranker import rerank
    from app.retrieval import retrieve

    question = item["question"]
    history = item.get("history")

    guard = check_question(question, history)
    if not guard.ok:
        return {"id": item["id"], "blocked": True, "route": "blocked", "pool": [], "final": []}

    route = classify_query(question, history)
    if route != "technical":
        return {"id": item["id"], "blocked": False, "route": route, "pool": [], "final": []}

    query = rewrite_query(question, history)
    pool = retrieve(query, top_k=POOL_K, max_per_document=0)
    result = rerank(query, pool, top_k=FINAL_K)

    return {
        "id": item["id"],
        "blocked": False,
        "route": route,
        "query": query,
        "pool": _sources(pool),
        "final": _sources(result.chunks),
        "reranked": result.reranked,
    }


def score(dataset: list[dict], runs: dict[str, dict]) -> dict:
    """Turn raw runs into per-query and aggregate metrics."""
    pool_reports, final_reports = [], []
    route_true, route_pred = [], []
    should_abstain, did_abstain = [], []
    per_query = []

    for item in dataset:
        run = runs[item["id"]]
        expected = set(normalize_source(s, root=ROOT) for s in item.get("expected_sources", []))

        pool_report = ranking_report(run["pool"], expected, ks=(POOL_K,))
        final_report = ranking_report(run["final"], expected, ks=(FINAL_K,))
        pool_reports.append(pool_report)
        final_reports.append(final_report)

        if "expected_route" in item and item["expected_route"] != "blocked":
            route_true.append(item["expected_route"])
            # A blocked run has no route classification of its own to compare;
            # count it as a miss against a non-blocked expectation.
            route_pred.append("blocked" if run["blocked"] else run["route"])

        if "should_abstain" in item:
            should_abstain.append(item["should_abstain"])
            abstained = run["blocked"] or not run["final"]
            did_abstain.append(abstained)

        per_query.append(
            {
                "id": item["id"],
                "type": item.get("type"),
                "question": item["question"],
                f"recall@{POOL_K}": pool_report[f"recall@{POOL_K}"],
                f"precision@{FINAL_K}": final_report[f"precision@{FINAL_K}"],
                f"ndcg@{FINAL_K}": final_report[f"ndcg@{FINAL_K}"],
                "route_ok": (
                    None
                    if "expected_route" not in item
                    else (run["blocked"] if item["expected_route"] == "blocked" else run["route"] == item["expected_route"])
                ),
                # False on a run that reached rerank but got vector order back
                # (Cohere down, or rate-limited) — those queries' final-stage
                # metrics are not comparable to a properly reranked run and
                # must not be silently averaged in as if they were.
                "reranked": run.get("reranked"),
            }
        )

    degraded = [q["id"] for q in per_query if q["reranked"] is False]
    return {
        "pool": {k: v.mean for k, v in aggregate_reports(pool_reports, confidence=None).items()},
        "final": {k: v.mean for k, v in aggregate_reports(final_reports, confidence=None).items()},
        "router_accuracy": accuracy(route_true, route_pred),
        "abstention": abstention_report(should_abstain, did_abstain) if should_abstain else None,
        "degraded": degraded,
        "per_query": per_query,
    }


def _fmt(x, digits=3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def print_report(scored: dict, n: int) -> None:
    print(f"\nHelix retrieval eval — {n} questions\n" + "=" * 44)

    if scored["degraded"]:
        print(
            f"\n  WARNING: {len(scored['degraded'])}/{n} queries fell back to vector "
            f"order (Cohere unavailable or rate-limited, even paced {PACING_SECONDS}s "
            f"apart).\n"
            f"  Final-stage metrics below include these and are not fully trustworthy.\n"
            f"  Raise RERANK_PACING_SECONDS, or check your Cohere rate-limit tier.\n"
            f"  Affected: {', '.join(scored['degraded'][:8])}"
            + (f" (+{len(scored['degraded']) - 8} more)" if len(scored["degraded"]) > 8 else "")
        )

    print(f"\nPool (top {POOL_K}, pre-rerank) — the recall ceiling:")
    for k in (f"recall@{POOL_K}", f"hit_rate@{POOL_K}", "mrr", "map"):
        print(f"  {k:20} {_fmt(scored['pool'].get(k))}")

    print(f"\nFinal (top {FINAL_K}, post-rerank) — what the LLM actually sees:")
    for k in (f"recall@{FINAL_K}", f"precision@{FINAL_K}", f"ndcg@{FINAL_K}"):
        print(f"  {k:20} {_fmt(scored['final'].get(k))}")

    gap = None
    r_pool, r_final = scored["pool"].get(f"recall@{POOL_K}"), scored["final"].get(f"recall@{FINAL_K}")
    if r_pool is not None and r_final is not None:
        gap = r_pool - r_final
    if gap is not None and gap > 0.15:
        print(
            f"\n  note: recall drops {gap:.2f} between pool and final — "
            f"the reranker or diversity cap is discarding relevant chunks."
        )

    print(f"\nRouter accuracy: {_fmt(scored['router_accuracy'])}")

    ab = scored["abstention"]
    if ab:
        print("\nAbstention:")
        print(f"  false_answer_rate   {_fmt(ab.false_answer_rate)}  (answered when it should abstain — the dangerous one)")
        print(f"  false_abstain_rate  {_fmt(ab.false_abstain_rate)}  (deflected an answerable question)")
        print(f"  accuracy            {_fmt(ab.accuracy)}")

    failures = [q for q in scored["per_query"] if (q.get(f"recall@{POOL_K}") or 0) < 1.0 and q.get(f"recall@{POOL_K}") is not None]
    if failures:
        print(f"\n{len(failures)} question(s) with incomplete pool recall:")
        for q in sorted(failures, key=lambda x: x[f"recall@{POOL_K}"])[:15]:
            print(f"  {_fmt(q[f'recall@{POOL_K}'])}  [{q['type']}]  {q['id']}: {q['question'][:60]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--json", type=Path, help="write raw per-query results here")
    ap.add_argument("--save-baseline", action="store_true", help="write evals/dataset/baseline.json")
    ap.add_argument("--compare-to", type=Path, help="print a paired delta against a prior --json run")
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    t0 = time.time()
    runs = {}
    for i, item in enumerate(dataset):
        runs[item["id"]] = run_one(item)
        if PACING_SECONDS and i < len(dataset) - 1:
            time.sleep(PACING_SECONDS)
    elapsed = time.time() - t0

    scored = score(dataset, runs)
    print_report(scored, len(dataset))
    print(f"\n({elapsed:.1f}s, {elapsed / len(dataset):.2f}s/question)")

    output = {
        "pool": scored["pool"],
        "final": scored["final"],
        "router_accuracy": scored["router_accuracy"],
        "degraded": scored["degraded"],
        "per_query": scored["per_query"],
    }
    if scored["abstention"]:
        ab = scored["abstention"]
        output["abstention"] = {
            "false_answer_rate": ab.false_answer_rate,
            "false_abstain_rate": ab.false_abstain_rate,
            "accuracy": ab.accuracy,
        }

    if args.json:
        args.json.write_text(json.dumps(output, indent=2))
        print(f"wrote {args.json}")
    if args.save_baseline:
        BASELINE.write_text(json.dumps(output, indent=2))
        print(f"wrote {BASELINE}")
    if args.compare_to:
        prior = json.loads(args.compare_to.read_text())
        prior_by_id = {q["id"]: q for q in prior["per_query"]}
        print("\nDelta vs", args.compare_to, ":")
        for metric in (f"recall@{POOL_K}", f"ndcg@{FINAL_K}"):
            baseline_vals = [prior_by_id.get(q["id"], {}).get(metric) for q in scored["per_query"]]
            candidate_vals = [q.get(metric) for q in scored["per_query"]]
            print(f"  {paired_delta(metric, baseline_vals, candidate_vals)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
