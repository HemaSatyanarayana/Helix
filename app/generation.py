"""RAG answer generation — the LLM layer, observed by Logfire.

Provider choice, model selection and fallback all live in :mod:`app.llm`; this
module only decides *what to say* and which class of model should say it.

Model roles used here: ``chat`` for answers and conversation, ``fast`` for the
router and the query rewriter — both are short, mechanical transformations that
do not need the large model, which on a low-latency provider is most of the
round trip.
"""

from __future__ import annotations

import re

from app.llm import LLMError, complete, has_api_key
from app.prompts import ANSWER, CONVERSE, REWRITE, ROUTER
from app.telemetry import configure_logfire

logfire = configure_logfire("helix-app")

ROUTES = ("technical", "conversational", "off_topic")


# --- Answer generation ----------------------------------------------------


def format_contexts(contexts: list[dict]) -> str:
    """Render retrieved chunks as a numbered, attributed context block.

    Each block carries its source and heading trail. Without them the model has
    only opaque ``[1]``/``[2]`` markers, so its citations cannot be checked
    against a real document and the UI cannot link them anywhere.
    """
    blocks = []
    for i, chunk in enumerate(contexts, start=1):
        source = chunk.get("source") or "unknown source"
        headings = " › ".join(chunk.get("headings") or [])
        header = f"[{i}] {source}"
        if headings:
            header += f" — {headings}"
        blocks.append(f"{header}\n{chunk.get('text', '')}")
    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    contexts: list[dict] | list[str],
    history: list[dict] | None = None,
) -> str:
    """Answer a question grounded in retrieved contexts.

    ``contexts`` are retrieval hits; plain strings are accepted for callers that
    only have text. ``history`` is prior conversation turns for multi-turn chat.
    Raises :class:`~app.llm.LLMError` if every model fails.
    """
    normalized = [{"text": c} if isinstance(c, str) else c for c in contexts]
    context_block = format_contexts(normalized)

    messages = [{"role": "system", "content": ANSWER.system}]
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"}
    )

    with logfire.span(
        "generate_answer",
        n_contexts=len(normalized),
        prompt=ANSWER.name,
        prompt_version=ANSWER.version,
    ):
        return complete(messages, role="chat", span_name="llm_answer")


def converse(question: str, history: list[dict] | None = None) -> str:
    """Answer a conversational message directly (no retrieval)."""
    messages = [{"role": "system", "content": CONVERSE.system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    with logfire.span("converse", prompt=CONVERSE.name, prompt_version=CONVERSE.version):
        return complete(messages, role="chat", span_name="llm_converse")


# --- Routing --------------------------------------------------------------


_CONVERSATIONAL_HINTS = re.compile(
    r"^\s*(hi|hey|hello|thanks|thank you|thx|bye|good (morning|evening|afternoon)|"
    r"how are you|who are you|what can you do|help)\b",
    re.IGNORECASE,
)


def _heuristic_route(question: str) -> str:
    """Keyless fallback router (used when no LLM is configured).

    Never returns ``off_topic``: distinguishing an off-topic question from a
    product one needs world knowledge a regex doesn't have, and routing to
    ``technical`` is the safe error — retrieval then finds nothing and the graph
    abstains, which is the same outcome by a slower path.
    """
    q = (question or "").strip()
    if _CONVERSATIONAL_HINTS.search(q) or len(q.split()) <= 2:
        return "conversational"
    return "technical"


def _parse_route(reply: str) -> str | None:
    """Read the label off the router's reply, or None if it stated none.

    Reads the last non-empty line: if a model's reasoning leaks into the content
    channel, the conclusion is at the end, and an earlier "this isn't technical"
    must not decide the outcome.
    """
    lines = [ln.strip().lower() for ln in reply.splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    # off_topic first: "off_topic" contains neither of the others, but a model
    # writing "off topic, not technical" would otherwise match 'technical'.
    for label, needles in (
        ("off_topic", ("off_topic", "off-topic", "off topic")),
        ("technical", ("technical",)),
        ("conversational", ("conversational",)),
    ):
        if any(n in last for n in needles):
            return label
    return None


def classify_query(question: str, history: list[dict] | None = None) -> str:
    """Route a message to ``technical``, ``conversational`` or ``off_topic``.

    Falls back to the heuristic when no key is configured *or* when the call
    fails — a routing outage should degrade the answer, not drop the request.
    """
    if not has_api_key():
        return _heuristic_route(question)

    messages = [{"role": "system", "content": ROUTER.system}]
    if history:
        messages.extend(history[-4:])
    messages.append({"role": "user", "content": question})

    with logfire.span(
        "classify_query", prompt=ROUTER.name, prompt_version=ROUTER.version
    ) as span:
        try:
            out = complete(
                messages,
                role="fast",
                span_name="llm_router",
                # Room for a reasoning model to think before it answers. A tight
                # cap (the old max_tokens=4) is spent entirely on reasoning
                # tokens and returns empty content, which silently routes every
                # question conversational and disables retrieval outright.
                max_tokens=512,
                # Groq coerces temperature=0 to 1e-8; either way this is greedy.
                temperature=0,
            )
        except LLMError as exc:
            logfire.warning("router_unavailable", error=str(exc))
            span.set_attribute("degraded", True)
            return _heuristic_route(question)

        route = _parse_route(out) or _heuristic_route(question)
        span.set_attribute("route", route)
        return route


# --- Query rewriting ------------------------------------------------------


def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """Resolve a follow-up into a standalone search query.

    Retrieval embeds the query in isolation, so "how do I do that on iOS?"
    otherwise searches for a fragment with no antecedent and matches nothing
    useful. Only runs when there is history to resolve against; returns the
    original question on any failure, since a degraded query still beats none.
    """
    if not history or not has_api_key():
        return question

    messages = [{"role": "system", "content": REWRITE.system}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": question})

    with logfire.span(
        "rewrite_query", prompt=REWRITE.name, prompt_version=REWRITE.version
    ) as span:
        try:
            rewritten = complete(
                messages,
                role="fast",
                span_name="llm_rewrite",
                max_tokens=512,
                temperature=0,
            )
        except LLMError as exc:
            logfire.warning("rewrite_unavailable", error=str(exc))
            return question

        cleaned = _clean_rewrite(rewritten)
        if not cleaned:
            return question
        span.set_attribute("rewritten", cleaned != question)
        span.set_attribute("query", cleaned)
        return cleaned


def _clean_rewrite(reply: str) -> str | None:
    """Take the last non-empty line and strip quoting/labels a model adds."""
    lines = [ln.strip() for ln in (reply or "").splitlines() if ln.strip()]
    if not lines:
        return None
    query = lines[-1]
    query = re.sub(r"^(query|search query|rewritten)\s*:\s*", "", query, flags=re.IGNORECASE)
    return query.strip().strip('"').strip("'").strip() or None
