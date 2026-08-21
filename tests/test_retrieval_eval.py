"""CI regression gate for retrieval quality.

Runs the golden set against live retrieval + reranking and asserts it hasn't
regressed below `evals/dataset/baseline.json`. Marked `slow` and excluded from
the default `pytest` run (see `pyproject.toml`) — it needs a populated Qdrant,
takes ~1-4 minutes depending on the LLM/rerank provider's latency, and calls
whatever COHERE_API_KEY is configured. `make test-eval` runs it explicitly, and
CI runs it as a separate, non-blocking-by-default job (see
`.github/workflows/ci.yml`) so a rerank-provider outage doesn't fail every PR.

Floors are read from the baseline with a tolerance rather than pinned to exact
numbers: the reranker is a live API and its scores are not perfectly
reproducible run to run. A regression that matters is a *drop*, not noise
around the same value — see `evals/dataset/baseline.json` and regenerate it
with `python evals/run_eval.py --save-baseline` after a deliberate change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "evals" / "dataset" / "baseline.json"

# How far below the baseline is tolerated before the gate fails. Wider than
# the bootstrap CI on 64 questions would suggest, deliberately: this guards
# against a real regression (a bad threshold change, a broken rerank call),
# not against the day-to-day wobble of a live cross-encoder API.
TOLERANCE = 0.08


def _qdrant_available() -> bool:
    try:
        from app.retrieval import QDRANT_COLLECTION, _qdrant

        client = _qdrant()
        return client.collection_exists(QDRANT_COLLECTION)
    except Exception:  # noqa: BLE001 — any connection failure means "skip"
        return False


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(f"no baseline at {BASELINE_PATH} — run `python evals/run_eval.py --save-baseline`")
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="module")
def current() -> dict:
    if not _qdrant_available():
        pytest.skip("Qdrant unreachable or the collection is missing — run `make ingest` first")
    from evals.run_eval import load_dataset, run_one, score

    dataset = load_dataset()
    runs = {item["id"]: run_one(item) for item in dataset}
    return score(dataset, runs)


@pytest.mark.parametrize("metric", ["recall@20", "hit_rate@20", "mrr"])
def test_pool_metrics_have_not_regressed(baseline, current, metric):
    """The recall ceiling — did retrieval find the right document at all."""
    base = baseline["pool"].get(metric)
    now = current["pool"].get(metric)
    if base is None or now is None:
        pytest.skip(f"{metric} undefined in baseline or current run")
    assert now >= base - TOLERANCE, (
        f"{metric} regressed: {now:.3f} vs baseline {base:.3f} "
        f"(tolerance {TOLERANCE}) — check retrieval, hybrid search, or the corpus"
    )


@pytest.mark.parametrize("metric", ["recall@4", "precision@4", "ndcg@4"])
def test_final_metrics_have_not_regressed(baseline, current, metric):
    """What the LLM actually sees, after reranking."""
    base = baseline["final"].get(metric)
    now = current["final"].get(metric)
    if base is None or now is None:
        pytest.skip(f"{metric} undefined in baseline or current run")
    assert now >= base - TOLERANCE, (
        f"{metric} regressed: {now:.3f} vs baseline {base:.3f} "
        f"(tolerance {TOLERANCE}) — check the rerank threshold or diversity cap"
    )


def test_router_accuracy_has_not_regressed(baseline, current):
    base, now = baseline.get("router_accuracy"), current.get("router_accuracy")
    if base is None or now is None:
        pytest.skip("router_accuracy undefined in baseline or current run")
    assert now >= base - TOLERANCE, f"router accuracy regressed: {now:.3f} vs baseline {base:.3f}"


def test_false_answer_rate_has_not_worsened(baseline, current):
    """The dangerous error: answering a question the corpus doesn't cover.
    No symmetric tolerance here — this direction only gets a stricter bar."""
    base = baseline.get("abstention", {}).get("false_answer_rate")
    now = current["abstention"].false_answer_rate if current.get("abstention") else None
    if base is None or now is None:
        pytest.skip("false_answer_rate undefined in baseline or current run")
    assert now <= base + TOLERANCE, (
        f"false_answer_rate got worse: {now:.3f} vs baseline {base:.3f} — "
        f"the pipeline is answering more off-topic questions than before"
    )


def test_run_did_not_silently_degrade(current):
    """A run where most reranking fell back to vector order isn't a
    trustworthy signal for the other assertions in this file."""
    degraded = len(current.get("degraded", []))
    total = len(current["per_query"])
    assert degraded / total < 0.25, (
        f"{degraded}/{total} queries fell back to vector order (Cohere down or "
        f"rate-limited) — re-run rather than trusting this result"
    )
