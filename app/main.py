"""Helix RAG API (FastAPI).

Exposes the agentic chat graph: triage (safety + routing, concurrent) → rewrite
→ retrieve → rerank → generate → verify. Instrumented by Logfire.

Three endpoints matter operationally:

* ``/health`` — liveness. Trivially true if the process is up.
* ``/readyz`` — readiness. Probes Qdrant, the collection, Redis and the LLM key,
  and returns 503 when the instance cannot actually serve. A load balancer needs
  this distinction: ``/health`` answering 200 from a process with no index is
  worse than no probe at all.
* ``/chat`` and ``/chat/stream`` — the pipeline, buffered or token-by-token.

Run:  uv run uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth import API_KEY_REQUIRED, Identity, authenticate
from app.config import check_readiness, validate_config, warm_up
from app.graph import answer_question, graph
from app.ratelimit import check_rate_limit
from app.telemetry import configure_logfire

logfire = configure_logfire("helix-app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail loudly here rather than serving degraded answers: in production a
    # missing requirement raises and the container never reports healthy.
    validate_config()
    # The embedder is lazily constructed, so without this the first request
    # after every deploy pays the model load (12.3s measured, against 0.02s
    # warm) — long enough to trip a readiness timeout and cycle the pod.
    warm_up()
    yield


app = FastAPI(title="Helix RAG API", version="0.2.0", lifespan=lifespan)
logfire.instrument_fastapi(app)


def _identity(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> Identity:
    """Establish who is calling, from ``X-API-Key`` or a bearer token."""
    key = x_api_key
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()

    identity = authenticate(key)
    if identity is None:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not identity.authenticated:
        # Anonymous mode: meter by client IP so one host can't exhaust the
        # shared quota for everyone.
        client = request.client.host if request.client else "unknown"
        return Identity(name=f"ip:{client}", authenticated=False)
    return identity


def rate_limit(response: Response, identity: Identity = Depends(_identity)) -> Identity:
    """Per-identity sliding-window rate limit; raises 429 when exceeded."""
    allowed, remaining, retry_after = check_rate_limit(identity.name)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please retry later.",
            headers={"Retry-After": str(retry_after), "X-RateLimit-Remaining": "0"},
        )
    return identity


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Turn] = []


class Source(BaseModel):
    source: str | None = None
    headings: list[str] = []
    pages: list[int] = []
    score: float | None = None
    rerank_score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    blocked: bool = False
    sources: list[Source] = []
    # False when the cross-encoder was unavailable and results are in vector
    # order. Without it a degraded pipeline is indistinguishable from a healthy
    # one from the outside.
    reranked: bool = False
    grounded: bool | None = None


def _sources(result: dict) -> list[Source]:
    return [
        Source(
            source=s.get("source"),
            headings=s.get("headings", []),
            pages=s.get("pages", []),
            score=s.get("score"),
            rerank_score=s.get("rerank_score"),
        )
        for s in result.get("sources", [])
    ]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness — is the process up? Says nothing about its dependencies."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz(response: Response) -> dict:
    """Readiness — can this instance actually serve a request?"""
    readiness = check_readiness()
    if not readiness.ready:
        response.status_code = 503
    return readiness.as_dict()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, identity: Identity = Depends(rate_limit)) -> ChatResponse:
    history = [t.model_dump() for t in req.history]
    result = answer_question(req.question, history)
    return ChatResponse(
        answer=result.get("answer", ""),
        route=result.get("route", ""),
        blocked=result.get("blocked", False),
        sources=_sources(result),
        reranked=bool(result.get("reranked")),
        grounded=result.get("grounded"),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, identity: Identity = Depends(rate_limit)):
    """Stream the pipeline as newline-delimited JSON events.

    The graph runs several sequential LLM calls, so a buffered response leaves
    the user watching a spinner for the whole pipeline. Streaming node updates
    lets the UI show progress ("searching…", sources found) and then the answer,
    rather than nothing until everything finishes.

    Events: ``{"type": "state", ...}`` per completed node, ``{"type": "answer",
    ...}`` with the final payload, ``{"type": "error", ...}`` on failure.
    """
    history = [t.model_dump() for t in req.history]

    def events():
        final: dict = {}
        try:
            for update in graph.stream(
                {"question": req.question, "history": history}, stream_mode="updates"
            ):
                for node, payload in update.items():
                    final.update(payload)
                    yield json.dumps(
                        {
                            "type": "state",
                            "node": node,
                            "route": final.get("route"),
                            "n_sources": len(final.get("sources", [])),
                        }
                    ) + "\n"
            yield json.dumps(
                {
                    "type": "answer",
                    "answer": final.get("answer", ""),
                    "route": final.get("route", ""),
                    "blocked": final.get("blocked", False),
                    "reranked": bool(final.get("reranked")),
                    "grounded": final.get("grounded"),
                    "sources": [s.model_dump() for s in _sources(final)],
                }
            ) + "\n"
        except Exception as exc:  # noqa: BLE001 — the stream owns its errors.
            logfire.error("stream_failed", error=str(exc))
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/config")
def config_summary(identity: Identity = Depends(_identity)) -> dict:
    """Which subsystems are live — the degraded-vs-healthy question, answered.

    Reports names and states only; never key values.
    """
    from app.embedding import EMBEDDING_MODEL, HYBRID_ENABLED, SPARSE_MODEL
    from app.llm import model_for, provider
    from app.reranker import RERANK_ENABLED, RERANKER_MODEL

    return {
        "llm": {
            "provider": provider().name,
            "chat": model_for("chat"),
            "fast": model_for("fast"),
            "guard": model_for("guard"),
        },
        "embedding": {
            "dense": EMBEDDING_MODEL,
            "sparse": SPARSE_MODEL if HYBRID_ENABLED else None,
            "hybrid": HYBRID_ENABLED,
        },
        "reranker": {"enabled": RERANK_ENABLED, "model": RERANKER_MODEL},
        "auth": {"required": API_KEY_REQUIRED},
    }
