"""RAG answer generation — the LLM layer, observed by Logfire.

Every call here is an LLM generation with a prompt, model, tokens and latency.
Logfire's OpenAI instrumentation captures those automatically, and we route
through OpenRouter with a fallback model list (rate-limit / outage resilience).
"""

from __future__ import annotations

import os

from openai import OpenAI

from app.telemetry import configure_logfire

logfire = configure_logfire("helix-app")
# Auto-trace every OpenAI-compatible call (prompt, model, tokens, latency).
logfire.instrument_openai()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Ordered fallback chain: OpenRouter tries these top-to-bottom until one succeeds.
MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "MODEL_FALLBACKS",
        "anthropic/claude-sonnet-4.5,openai/gpt-4.1,google/gemini-2.5-flash",
    ).split(",")
    if m.strip()
]


def _client() -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


def generate_answer(
    question: str,
    contexts: list[str],
    history: list[dict] | None = None,
) -> str:
    """Answer a question grounded in retrieved contexts.

    ``history`` is prior conversation turns (``{"role", "content"}``) for
    multi-turn chat. Wrapped in a Logfire span; the OpenRouter ``models`` list
    drives fallback across providers.
    """
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    messages = [
        {
            "role": "system",
            "content": (
                "You are Helix, a product assistant for businesses. Answer only "
                "from the provided context; if it is insufficient, say so."
            ),
        }
    ]
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"}
    )

    with logfire.span("generate_answer", model=MODEL_FALLBACKS[0], n_contexts=len(contexts)):
        response = _client().chat.completions.create(
            model=MODEL_FALLBACKS[0],
            messages=messages,
            # OpenRouter-specific: ordered fallback chain.
            extra_body={"models": MODEL_FALLBACKS},
        )

    return response.choices[0].message.content or ""
