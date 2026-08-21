"""Provider-agnostic LLM access — one place that knows who serves the models.

Every LLM call in Helix goes through :func:`complete`. Nothing else in the
codebase names a provider, a base URL, or an API key, so switching vendors is an
env change rather than a diff.

Configured for **Groq** by default. Any OpenAI-compatible endpoint works: set
``LLM_PROVIDER`` to a name in :data:`PROVIDERS`, or point ``LLM_BASE_URL`` and
``LLM_API_KEY`` anywhere at all (vLLM, LiteLLM, Ollama, a gateway).

Models are addressed by **role**, not by name, so each call site asks for the
kind of model it needs and the operator decides which model that is:

* ``chat``  — the answer/conversation model (``LLM_MODEL``)
* ``fast``  — cheap, low-latency classification, e.g. the router (``LLM_FAST_MODEL``)
* ``guard`` — the safety classifier (``LLM_GUARD_MODEL``)

Fallback is client-side: :func:`complete` walks the model chain and moves to the
next one when a call fails. OpenRouter offered this server-side via a custom
``extra_body`` field; doing it here instead means it works on every provider.

Model IDs churn fast — Groq shut down the Llama 3.x line on 2026-08-16 and has
retired every Llama Guard variant — so treat the defaults below as a starting
point and check https://console.groq.com/docs/models when calls start 404ing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from app.telemetry import configure_logfire

# NB: `.env` is loaded by app/__init__.py, which runs before this module.
logfire = configure_logfire("helix-app")
# Auto-trace every OpenAI-compatible call (prompt, model, tokens, latency).
logfire.instrument_openai()


@dataclass(frozen=True)
class Provider:
    """An OpenAI-compatible inference endpoint."""

    name: str
    base_url: str
    key_env: str


PROVIDERS: dict[str, Provider] = {
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": Provider("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "together": Provider("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks": Provider(
        "fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"
    ),
    "ollama": Provider("ollama", "http://localhost:11434/v1", "OLLAMA_API_KEY"),
}

DEFAULT_PROVIDER = "groq"

# Groq defaults, current as of 2026-08-19. Override per role when you switch
# providers — these IDs mean nothing to OpenAI or Anthropic endpoints.
DEFAULT_MODELS = {
    "chat": "openai/gpt-oss-120b",
    "fast": "openai/gpt-oss-20b",
    "guard": "openai/gpt-oss-safeguard-20b",
}
MODEL_ENV_VARS = {
    "chat": "LLM_MODEL",
    "fast": "LLM_FAST_MODEL",
    "guard": "LLM_GUARD_MODEL",
}
DEFAULT_FALLBACKS = "qwen/qwen3.6-27b,openai/gpt-oss-20b"

# Per-role request parameters, as JSON in the matching env var. These tune a
# model family rather than the task, so they live with the provider config.
#
# The gpt-oss models reason before answering, which for a one-word routing
# decision is pure latency — 0.93s of reasoning at the "medium" default, versus
# ~0.5s at "low". Groq's validator accepts only low/medium/high — "none" 400s
# with "reasoning_effort must be one of low, medium, or high" despite briefly
# appearing to work in earlier testing; don't reintroduce it without checking
# against a live call, not just this comment. Not every provider accepts the
# parameter at all; an unsupported one is dropped and retried rather than
# failing the call (see `complete`), and setting the env var to "{}" disables
# it outright.
DEFAULT_ROLE_PARAMS: dict[str, dict] = {
    "chat": {},
    "fast": {"reasoning_effort": "low"},
    "guard": {"reasoning_effort": "low"},
}
PARAMS_ENV_VARS = {
    "chat": "LLM_PARAMS",
    "fast": "LLM_FAST_PARAMS",
    "guard": "LLM_GUARD_PARAMS",
}


class LLMError(RuntimeError):
    """Every model in the chain failed."""


# --- Configuration (resolved per call, so tests and reloads see env changes) --


def provider() -> Provider:
    """The configured provider, or a synthetic one when LLM_BASE_URL is set."""
    name = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    known = PROVIDERS.get(name)
    base_url = os.getenv("LLM_BASE_URL") or (known.base_url if known else None)
    if base_url is None:
        raise LLMError(
            f"unknown LLM_PROVIDER {name!r}; use one of "
            f"{', '.join(sorted(PROVIDERS))} or set LLM_BASE_URL explicitly"
        )
    return Provider(
        name=name,
        base_url=base_url,
        key_env=known.key_env if known else "LLM_API_KEY",
    )


def api_key() -> str | None:
    """Resolve the API key: explicit override first, then the provider's env var."""
    return os.getenv("LLM_API_KEY") or os.getenv(provider().key_env) or None


def has_api_key() -> bool:
    """Whether an LLM call can be attempted at all.

    Call sites use this to degrade deliberately (heuristic routing, skipped
    safety classification) instead of throwing on a missing key.
    """
    return bool(api_key())


def fallbacks() -> list[str]:
    raw = os.getenv("LLM_FALLBACKS", DEFAULT_FALLBACKS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def model_for(role: str = "chat") -> str:
    """The configured model for a role (``chat`` / ``fast`` / ``guard``)."""
    if role not in MODEL_ENV_VARS:
        raise ValueError(f"unknown model role {role!r}; expected one of {sorted(MODEL_ENV_VARS)}")
    return os.getenv(MODEL_ENV_VARS[role]) or DEFAULT_MODELS[role]


def model_chain(role: str = "chat") -> list[str]:
    """Ordered models to try for a role: the role's model, then the fallbacks.

    The guard role gets no fallback chain. A safety classifier that quietly
    swaps in a different model is worse than one that fails and says so — the
    replacement may not share the verdict format the parser expects.
    """
    primary = model_for(role)
    if role == "guard":
        return [primary]
    return [primary] + [m for m in fallbacks() if m != primary]


def role_params(role: str = "chat") -> dict:
    """Extra request parameters for a role, from env JSON or the defaults."""
    if role not in PARAMS_ENV_VARS:
        raise ValueError(f"unknown model role {role!r}; expected one of {sorted(PARAMS_ENV_VARS)}")
    raw = os.getenv(PARAMS_ENV_VARS[role])
    if raw is None:
        return dict(DEFAULT_ROLE_PARAMS[role])
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"{PARAMS_ENV_VARS[role]} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"{PARAMS_ENV_VARS[role]} must be a JSON object, got {type(parsed).__name__}")
    return parsed


@lru_cache(maxsize=8)
def _client(base_url: str, key: str | None) -> OpenAI:
    # Cached per (base_url, key) so a swapped env var builds a fresh client.
    # Local servers accept any key; send a placeholder rather than None so the
    # SDK doesn't raise before the request is made.
    return OpenAI(base_url=base_url, api_key=key or "no-key")


# --- The one call path ----------------------------------------------------


def complete(
    messages: list[dict],
    *,
    role: str = "chat",
    models: list[str] | None = None,
    span_name: str = "llm_complete",
    **kwargs,
) -> str:
    """Run a chat completion, walking the model chain until one succeeds.

    Extra keyword arguments pass straight through to the completions API.
    Returns the message content (empty string if the model returned none).

    Raises :class:`LLMError` if no key is configured or every model failed —
    callers decide whether to fail open or surface it.
    """
    chain = models or model_chain(role)
    if not chain:
        raise LLMError(f"no models configured for role {role!r}")

    key = api_key()
    if not key:
        prov = provider()
        raise LLMError(f"no API key: set {prov.key_env} (provider {prov.name!r}) or LLM_API_KEY")

    prov = provider()
    client = _client(prov.base_url, key)
    last_error: Exception | None = None

    # Role defaults tune the model family; an explicit kwarg from the caller
    # always wins over them.
    extras = {k: v for k, v in role_params(role).items() if k not in kwargs}
    params = {**extras, **kwargs}

    with logfire.span(span_name, provider=prov.name, role=role, models=chain) as span:
        for attempt, model in enumerate(chain):
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, **params
                )
            except Exception as exc:  # noqa: BLE001
                # A provider that doesn't know a role parameter rejects the
                # request outright. Drop the extras and try the same model once
                # more, so switching providers degrades instead of breaking.
                if extras:
                    logfire.warning(
                        "llm_params_rejected",
                        model=model,
                        provider=prov.name,
                        dropped=sorted(extras),
                        error=str(exc),
                    )
                    try:
                        response = client.chat.completions.create(
                            model=model, messages=messages, **kwargs
                        )
                    except Exception as retry_exc:  # noqa: BLE001 — next model.
                        last_error = retry_exc
                        logfire.warning(
                            "llm_model_failed",
                            model=model,
                            provider=prov.name,
                            error=str(retry_exc),
                        )
                        continue
                    span.set_attribute("dropped_params", sorted(extras))
                else:
                    last_error = exc
                    logfire.warning(
                        "llm_model_failed", model=model, provider=prov.name, error=str(exc)
                    )
                    continue

            # Which model actually answered — without this a degraded chain is
            # invisible, and every latency/cost number gets attributed wrong.
            span.set_attribute("model_used", model)
            span.set_attribute("fell_back", attempt > 0)
            return response.choices[0].message.content or ""

    raise LLMError(
        f"all {len(chain)} model(s) failed on provider {prov.name!r}: {last_error}"
    ) from last_error
