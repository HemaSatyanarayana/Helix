"""Input guardrails for the user's question (pre-processing).

A layered chain, cheap → expensive:

1. Deterministic checks (no LLM): non-empty, length cap, prompt-injection /
   jailbreak patterns. Fast fail on the obvious stuff.
2. **Safety classifier** (an LLM via :mod:`app.llm`, ``guard`` role): catches
   unsafe content — hate, violence, self-harm, sexual, criminal — that regex
   can't. The model is named by ``LLM_GUARD_MODEL``.

Each check returns a GuardrailResult; the graph short-circuits to a refusal when
one fails. The classifier is skipped when no API key is configured, and by
default fails **open** (allow + log) on transient errors so availability issues
don't take the API down. Set ``GUARDRAIL_FAIL_CLOSED=true`` to refuse instead.

The classifier is prompted (``prompts.SAFETY``) to end its reply with a
``VERDICT:`` line. That contract is deliberately model-agnostic: the older
Llama Guard family emitted a bare ``safe`` / ``unsafe\\n<codes>`` and is now
retired everywhere on Groq, while current safety models are policy-driven
reasoners whose output shape varies. Parsing a required final line works for
both, and an unparseable reply is treated as an availability failure rather
than quietly interpreted.

Suggested future additions: PII/secret detection (Presidio), output-side
groundedness checks, language allowlist.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from app.llm import LLMError, complete, has_api_key
from app.prompts import GROUNDEDNESS, SAFETY
from app.telemetry import configure_logfire

logfire = configure_logfire("helix-app")

MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
GUARD_ENABLED = os.getenv("GUARDRAIL_ENABLED", "true").lower() == "true"
FAIL_CLOSED = os.getenv("GUARDRAIL_FAIL_CLOSED", "false").lower() == "true"
OUTPUT_GUARD_ENABLED = os.getenv("OUTPUT_GUARDRAIL_ENABLED", "false").lower() == "true"

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all|the|your|previous|above|prior)\s+.*instruction",
        r"disregard\s+.*(instruction|prompt|rule)",
        r"forget\s+(everything|all|your)\s+.*(instruction|prompt)",
        r"(reveal|show|print|repeat)\s+.*(system\s*prompt|your\s+instructions)",
        r"you\s+are\s+now\s+",
        r"\bact\s+as\s+(dan|a\s+jailbroken|an?\s+unfiltered)",
        r"developer\s+mode",
    ]
]

# Matches the required final line, and also a bare "safe"/"unsafe (S1)" so a
# terser model still parses.
_VERDICT = re.compile(r"^(?:verdict\s*:\s*)?(safe|unsafe)\b[\s:(]*([^)]*)", re.IGNORECASE)


@dataclass
class GuardrailResult:
    ok: bool
    reason: str = ""


def _unavailable(what: str, detail: str) -> GuardrailResult:
    """Apply the configured fail-open / fail-closed policy to a guard failure."""
    logfire.warning("guardrail_unavailable", check=what, detail=detail, fail_closed=FAIL_CLOSED)
    if FAIL_CLOSED:
        return GuardrailResult(False, "the safety check is unavailable")
    return GuardrailResult(True)


def _parse_verdict(reply: str) -> GuardrailResult | None:
    """Read the classifier's verdict, or None if it didn't state one.

    Scans from the end: reasoning models put their conclusion last, and any
    earlier mention of "unsafe" is deliberation, not the verdict.
    """
    for line in reversed([ln.strip() for ln in reply.splitlines() if ln.strip()]):
        match = _VERDICT.match(line)
        if not match:
            continue
        if match.group(1).lower() == "safe":
            return GuardrailResult(True)
        categories = match.group(2).strip().rstrip(".,")
        reason = "the request was flagged as unsafe"
        if categories:
            reason += f" ({categories})"
        return GuardrailResult(False, reason)
    return None


def check_input_safety(question: str, history: list[dict] | None = None) -> GuardrailResult:
    """Classify the input with the safety model. Safe → ok; unsafe → block.

    Skipped (allowed) when disabled or unconfigured; applies the fail-open /
    fail-closed policy on errors and unparseable replies.
    """
    if not GUARD_ENABLED or not has_api_key():
        return GuardrailResult(True)

    messages: list[dict] = [{"role": "system", "content": SAFETY.system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    with logfire.span(
        "safety_check", prompt=SAFETY.name, prompt_version=SAFETY.version
    ) as span:
        try:
            reply = complete(
                messages,
                role="guard",
                span_name="llm_guard",
                max_tokens=512,  # policy reasoners need room before the verdict
                temperature=0,
            )
        except LLMError as exc:
            return _unavailable("safety_model", str(exc))

        span.set_attribute("verdict_raw", reply.strip()[-120:])
        result = _parse_verdict(reply)
        if result is None:
            return _unavailable("safety_verdict", f"unparseable reply: {reply.strip()[:200]!r}")

        span.set_attribute("safe", result.ok)
        return result


_GROUNDED_VERDICT = re.compile(
    r"^(?:verdict\s*:\s*)?(grounded|ungrounded)\b[\s:(]*([^)]*)", re.IGNORECASE
)


def check_groundedness(answer: str, contexts: list[dict]) -> GuardrailResult:
    """Verify the generated answer is supported by its retrieved context.

    The input guardrail screens what the user sends; this screens what we send
    back. A confidently wrong answer about SDK setup is the failure mode that
    actually costs users something, and retrieval succeeding is no guarantee the
    model stayed inside it.

    Off by default (``OUTPUT_GUARDRAIL_ENABLED``) because it adds an LLM call to
    every answered question — turn it on when correctness matters more than the
    extra latency. Fails open: an unverifiable answer is still an answer, and
    blocking on a classifier outage would be worse than shipping it.
    """
    if not OUTPUT_GUARD_ENABLED or not has_api_key() or not contexts or not answer.strip():
        return GuardrailResult(True)

    from app.generation import format_contexts

    messages = [
        {"role": "system", "content": GROUNDEDNESS.system},
        {
            "role": "user",
            "content": f"CONTEXT:\n{format_contexts(contexts)}\n\nANSWER:\n{answer}",
        },
    ]

    with logfire.span(
        "groundedness_check", prompt=GROUNDEDNESS.name, prompt_version=GROUNDEDNESS.version
    ) as span:
        try:
            reply = complete(
                messages,
                role="guard",
                span_name="llm_groundedness",
                max_tokens=512,
                temperature=0,
            )
        except LLMError as exc:
            logfire.warning("groundedness_unavailable", error=str(exc))
            return GuardrailResult(True)

        for line in reversed([ln.strip() for ln in reply.splitlines() if ln.strip()]):
            match = _GROUNDED_VERDICT.match(line)
            if not match:
                continue
            grounded = match.group(1).lower() == "grounded"
            span.set_attribute("grounded", grounded)
            if grounded:
                return GuardrailResult(True)
            detail = match.group(2).strip().rstrip(".,")
            logfire.warning("answer_ungrounded", detail=detail[:200])
            return GuardrailResult(False, detail or "the answer was not supported by the sources")

        logfire.warning("groundedness_unparseable", reply=reply.strip()[:200])
        return GuardrailResult(True)


def check_question(question: str, history: list[dict] | None = None) -> GuardrailResult:
    """Run the full input guardrail chain over a question."""
    with logfire.span("guardrail_check") as span:
        q = (question or "").strip()

        if not q:
            span.set_attribute("blocked_reason", "empty")
            return GuardrailResult(False, "the question is empty")

        if len(q) > MAX_QUESTION_CHARS:
            span.set_attribute("blocked_reason", "too_long")
            return GuardrailResult(
                False, f"the question exceeds {MAX_QUESTION_CHARS} characters"
            )

        for pat in _INJECTION_PATTERNS:
            if pat.search(q):
                span.set_attribute("blocked_reason", "prompt_injection")
                return GuardrailResult(False, "the request looks like a prompt-injection attempt")

        # LLM safety classifier — nuanced, runs last.
        safety = check_input_safety(q, history)
        if not safety.ok:
            span.set_attribute("blocked_reason", "safety_model")
            return safety

        span.set_attribute("blocked_reason", "")
        return GuardrailResult(True)
