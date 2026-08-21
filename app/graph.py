"""Agentic chat graph (LangGraph).

    START
      │
      ▼
  [triage] ──blocked──────────► END
      │  │
      │  ├──conversational───► [conversational] ──► END
      │  └──off_topic────────► [deflect] ─────────► END
      ▼ technical
  [rewrite] ──► [retrieve] ──► [rerank] ──┬──relevant──► [generate] ──► [verify] ──► END
                                          └──nothing───► [deflect] ──► END

**Triage** runs the safety guardrail and the router *concurrently*. They are
independent — the router's answer is discarded when the guardrail blocks — and
running them in series added a whole round trip to every request.

**Off-topic is a route, not just a retrieval outcome.** A general-knowledge
question ("what is the capital of France?") is neither a product question nor
small talk; sending it down the conversational branch had the assistant answer
it from model knowledge, which is exactly the ungrounded answer this pipeline
exists to avoid.

**Relevance is decided on the rerank score.** Retrieval hands over an unfiltered
candidate pool and the cross-encoder decides what is actually relevant, so the
abstain decision uses the precise signal rather than the bi-encoder's. When the
reranker is unavailable the pipeline falls back to vector order and says so, and
the gate is skipped rather than applied to scores it was not calibrated for.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.generation import classify_query, converse, generate_answer, rewrite_query
from app.guardrails import check_groundedness, check_question
from app.llm import LLMError, provider
from app.reranker import RERANK_CANDIDATES, rerank
from app.retrieval import retrieve
from app.telemetry import configure_logfire

logfire = configure_logfire("helix-app")


class ChatState(TypedDict, total=False):
    question: str
    search_query: str  # question, resolved against history
    history: list[dict]
    route: str
    blocked: bool
    reason: str
    contexts: list[dict]
    sources: list[dict]
    answer: str
    reranked: bool  # False => degraded to vector order
    grounded: bool | None  # None => not checked


DEFLECTION = (
    "I couldn't find anything relevant in the product documentation to answer "
    "that. I can only help with questions about the product."
)

UNGROUNDED_NOTICE = (
    "I found related documentation but couldn't produce an answer I can fully "
    "support from it. Could you rephrase the question, or narrow it down?"
)


# --- Nodes ----------------------------------------------------------------


def triage_node(state: ChatState) -> dict:
    """Safety check and routing, concurrently — they don't depend on each other."""
    question = state["question"]
    history = state.get("history")

    with logfire.span("triage"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            guard_future = pool.submit(check_question, question, history)
            route_future = pool.submit(classify_query, question, history)
            guard = guard_future.result()
            route = route_future.result()

    if not guard.ok:
        return {
            "blocked": True,
            "route": "blocked",
            "reason": guard.reason,
            "answer": f"Sorry, I can't help with that — {guard.reason}.",
            "sources": [],
        }
    return {"blocked": False, "route": route}


def _llm_error(exc: Exception) -> str:
    logfire.warning("llm_call_failed", error=str(exc))
    try:
        key_env = provider().key_env
    except LLMError:  # LLM_PROVIDER itself is misconfigured
        key_env = "LLM_API_KEY"
    return f"⚠️ The model is unavailable right now. Is {key_env} set?"


def conversational_node(state: ChatState) -> dict:
    try:
        answer = converse(state["question"], state.get("history"))
    except Exception as exc:  # noqa: BLE001
        answer = _llm_error(exc)
    return {"answer": answer, "sources": []}


def deflect_node(state: ChatState) -> dict:
    """Nothing relevant in the corpus, or the question was never about it."""
    return {"answer": DEFLECTION, "route": "off_topic", "sources": []}


def rewrite_node(state: ChatState) -> dict:
    """Resolve follow-ups into a standalone query before searching."""
    query = rewrite_query(state["question"], state.get("history"))
    return {"search_query": query}


def retrieve_node(state: ChatState) -> dict:
    # Pull a wide, unfiltered candidate pool; the reranker decides relevance.
    query = state.get("search_query") or state["question"]
    contexts = retrieve(query, top_k=RERANK_CANDIDATES, max_per_document=0)
    return {"contexts": contexts}


def rerank_node(state: ChatState) -> dict:
    query = state.get("search_query") or state["question"]
    result = rerank(query, state.get("contexts", []))
    return {
        "contexts": result.chunks,
        "sources": result.chunks,
        "reranked": result.reranked,
    }


def generate_node(state: ChatState) -> dict:
    contexts = state.get("contexts", [])
    try:
        answer = generate_answer(
            state["question"], contexts, history=state.get("history")
        )
    except Exception as exc:  # noqa: BLE001
        return {"answer": _llm_error(exc), "grounded": None}
    return {"answer": answer}


def verify_node(state: ChatState) -> dict:
    """Check the answer against its sources before returning it."""
    result = check_groundedness(state.get("answer", ""), state.get("contexts", []))
    if result.ok:
        return {"grounded": True}
    logfire.warning("answer_withheld", reason=result.reason)
    return {"grounded": False, "answer": UNGROUNDED_NOTICE, "sources": []}


# --- Edges ----------------------------------------------------------------


def _after_triage(state: ChatState) -> str:
    if state.get("blocked"):
        return "blocked"
    route = state.get("route")
    if route == "technical":
        return "technical"
    if route == "off_topic":
        return "off_topic"
    return "conversational"


def _after_rerank(state: ChatState) -> str:
    """Nothing cleared the relevance bar → abstain rather than guess."""
    return "generate" if state.get("contexts") else "deflect"


def build_graph():
    g = StateGraph(ChatState)
    g.add_node("triage", triage_node)
    g.add_node("conversational", conversational_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("deflect", deflect_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)

    g.add_edge(START, "triage")
    g.add_conditional_edges(
        "triage",
        _after_triage,
        {
            "blocked": END,
            "technical": "rewrite",
            "conversational": "conversational",
            "off_topic": "deflect",
        },
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges(
        "rerank", _after_rerank, {"generate": "generate", "deflect": "deflect"}
    )
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    g.add_edge("deflect", END)
    g.add_edge("conversational", END)
    return g.compile()


graph = build_graph()


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    """Run the full chat graph and return the final state."""
    with logfire.span("chat_graph") as span:
        result = graph.invoke({"question": question, "history": history or []})
        span.set_attribute("route", result.get("route"))
        span.set_attribute("reranked", result.get("reranked"))
        span.set_attribute("n_sources", len(result.get("sources", [])))
        return result
