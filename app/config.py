"""Configuration validation and health reporting.

Helix degrades gracefully by design: the guardrail skips without a key, the
reranker falls back to vector order, the rate limiter fails open, the router
falls back to a heuristic. Each fallback is defensible on its own, but together
they mean a misconfigured deployment looks *exactly* like a healthy one — the
Groq migration shut down four subsystems at once and nothing in the response or
the traces said so.

This module makes that state explicit, in two directions:

* :func:`validate_config` runs at startup. In ``ENVIRONMENT=production`` a
  missing requirement raises rather than serving broken answers; elsewhere it
  logs warnings so local development stays frictionless.
* :func:`check_readiness` answers "can this instance actually serve traffic?"
  for ``/readyz``, probing Qdrant, the collection, Redis and the LLM key. A
  load balancer needs this — ``/health`` returning 200 from a process that
  cannot retrieve anything is worse than no probe at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from app.telemetry import configure_logfire

logfire = configure_logfire("helix-app")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


def is_production() -> bool:
    return ENVIRONMENT.strip().lower() in {"production", "prod"}


class ConfigError(RuntimeError):
    """A required setting is missing or invalid."""


@dataclass
class Check:
    """One subsystem's status."""

    name: str
    ok: bool
    detail: str = ""
    required: bool = True  # False => degraded but servable

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "required": self.required}


@dataclass
class Readiness:
    checks: list[Check] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """Ready when every *required* check passes."""
        return all(c.ok for c in self.checks if c.required)

    @property
    def degraded(self) -> list[str]:
        """Optional subsystems that are down — servable, but worth alerting on."""
        return [c.name for c in self.checks if not c.ok and not c.required]

    def as_dict(self) -> dict:
        return {
            "ready": self.ready,
            "environment": ENVIRONMENT,
            "degraded": self.degraded,
            "checks": [c.as_dict() for c in self.checks],
        }


# --- Startup validation ---------------------------------------------------


def _requirements() -> list[Check]:
    """Settings that must be present for the pipeline to work as designed."""
    from app.llm import api_key, provider

    checks: list[Check] = []

    try:
        prov = provider()
        checks.append(
            Check("llm_provider", True, f"{prov.name} @ {prov.base_url}", required=True)
        )
        checks.append(
            Check(
                "llm_api_key",
                bool(api_key()),
                f"set {prov.key_env} or LLM_API_KEY" if not api_key() else "configured",
                required=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 — a bad provider name lands here.
        checks.append(Check("llm_provider", False, str(exc), required=True))
        checks.append(Check("llm_api_key", False, "provider unresolved", required=True))

    # Authentication is only a hard requirement in production, where an open
    # endpoint means the rate limit is decorative.
    from app.auth import API_KEY_REQUIRED, load_keys

    if is_production():
        checks.append(
            Check(
                "auth",
                API_KEY_REQUIRED and bool(load_keys()),
                "set API_KEY_REQUIRED=true and API_KEYS — an unauthenticated "
                "endpoint cannot enforce a per-user rate limit",
                required=True,
            )
        )

    # Reranking and safety are quality/safety layers rather than hard
    # requirements: the pipeline answers without them, but far less well, so
    # they are reported as degraded rather than silently absent.
    from app.guardrails import GUARD_ENABLED
    from app.reranker import RERANK_ENABLED

    rerank_key = bool(os.getenv("COHERE_API_KEY"))
    checks.append(
        Check(
            "reranker",
            not RERANK_ENABLED or rerank_key,
            "COHERE_API_KEY unset — falling back to vector order" if not rerank_key else "configured",
            required=False,
        )
    )
    checks.append(
        Check(
            "safety_guardrail",
            not GUARD_ENABLED or bool(api_key()),
            "no LLM key — safety classification skipped" if GUARD_ENABLED and not api_key() else "configured",
            required=False,
        )
    )
    return checks


def validate_config(*, strict: bool | None = None) -> list[Check]:
    """Check configuration at startup.

    Raises :class:`ConfigError` when a required setting is missing and ``strict``
    (default: production only) is set. Always logs the full picture, so the
    degraded-but-running case is visible in traces from the first second.
    """
    strict = is_production() if strict is None else strict
    checks = _requirements()

    for check in checks:
        if check.ok:
            continue
        if check.required:
            logfire.error("config_invalid", check=check.name, detail=check.detail)
        else:
            logfire.warning("config_degraded", check=check.name, detail=check.detail)

    missing = [c for c in checks if c.required and not c.ok]
    if missing and strict:
        detail = "; ".join(f"{c.name}: {c.detail}" for c in missing)
        raise ConfigError(
            f"refusing to start in ENVIRONMENT={ENVIRONMENT} with invalid configuration — {detail}"
        )

    logfire.info(
        "config_validated",
        environment=ENVIRONMENT,
        degraded=[c.name for c in checks if not c.ok and not c.required],
    )
    return checks


# --- Readiness probing ----------------------------------------------------


def _check_qdrant() -> list[Check]:
    """Qdrant reachable, and the collection present and populated."""
    from app.retrieval import QDRANT_COLLECTION, _qdrant

    try:
        client = _qdrant()
        if not client.collection_exists(QDRANT_COLLECTION):
            return [
                Check("qdrant", True, "reachable"),
                Check("collection", False, f"{QDRANT_COLLECTION!r} does not exist — run `make ingest`"),
            ]
        count = client.count(collection_name=QDRANT_COLLECTION, exact=False).count
        return [
            Check("qdrant", True, "reachable"),
            Check(
                "collection",
                count > 0,
                f"{QDRANT_COLLECTION!r} holds {count} chunks"
                if count
                else f"{QDRANT_COLLECTION!r} is empty — run `make ingest`",
            ),
        ]
    except Exception as exc:  # noqa: BLE001
        return [
            Check("qdrant", False, f"unreachable: {exc}"),
            Check("collection", False, "not checked"),
        ]


def _check_redis() -> Check:
    """Redis backs rate limiting; it fails open, so this is degraded not fatal."""
    from app.ratelimit import _redis

    try:
        _redis().ping()
        return Check("redis", True, "reachable", required=False)
    except Exception as exc:  # noqa: BLE001
        return Check("redis", False, f"unreachable — rate limiting disabled: {exc}", required=False)


def check_readiness() -> Readiness:
    """Probe every dependency. Used by ``/readyz``; never raises."""
    from app.llm import api_key

    with logfire.span("check_readiness") as span:
        checks = [*_check_qdrant(), _check_redis()]
        checks.append(
            Check("llm_api_key", bool(api_key()), "configured" if api_key() else "missing")
        )
        readiness = Readiness(checks=checks)
        span.set_attribute("ready", readiness.ready)
        span.set_attribute("degraded", readiness.degraded)
        return readiness


def warm_up() -> None:
    """Load both retrieval encoders before serving.

    Both the dense (sentence-transformer) and sparse (BM25) encoders are
    lazily constructed on first use, so without this the first request after
    every deploy pays both model loads — measured at 12.3s for the dense model
    alone, against 0.02s warm. Each is warmed independently so one failing
    (e.g. the sparse model download) doesn't skip the other.
    """
    from app.embedding import HYBRID_ENABLED, embed_dense, embed_sparse

    with logfire.span("warm_up"):
        try:
            embed_dense(["warm up"])
            logfire.info("dense_encoder_warm")
        except Exception as exc:  # noqa: BLE001 — never block startup on this.
            logfire.warning("warm_up_failed", encoder="dense", error=str(exc))

        if HYBRID_ENABLED:
            try:
                embed_sparse(["warm up"])
                logfire.info("sparse_encoder_warm")
            except Exception as exc:  # noqa: BLE001
                logfire.warning("warm_up_failed", encoder="sparse", error=str(exc))
