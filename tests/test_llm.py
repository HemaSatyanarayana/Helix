"""Tests for the provider-agnostic LLM layer.

No network: the OpenAI client is replaced with a stub that records calls and
fails on demand, so provider resolution and fallback behaviour are checked
without a key or an endpoint.
"""

from __future__ import annotations

import pytest

from app import llm

LLM_ENV = [
    "LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
    "LLM_FAST_MODEL", "LLM_GUARD_MODEL", "LLM_FALLBACKS",
    "LLM_PARAMS", "LLM_FAST_PARAMS", "LLM_GUARD_PARAMS",
    "GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start each test from a bare environment — .env is loaded on import."""
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    llm._client.cache_clear()


class StubCompletions:
    """Stands in for client.chat.completions, failing for chosen models.

    ``reject_params`` mimics a provider that 400s on a parameter it doesn't
    know — the way a non-Groq endpoint reacts to ``reasoning_effort``.
    """

    def __init__(self, fail_for=(), content="ok", reject_params=()):
        self.fail_for = set(fail_for)
        self.content = content
        self.reject_params = set(reject_params)
        self.calls: list[dict] = []

    def create(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        if model in self.fail_for:
            raise RuntimeError(f"model {model} unavailable")
        unknown = self.reject_params & set(kwargs)
        if unknown:
            raise RuntimeError(f"unrecognized request argument: {sorted(unknown)}")

        message = type("Message", (), {"content": self.content})
        choice = type("Choice", (), {"message": message})
        return type("Response", (), {"choices": [choice]})


@pytest.fixture
def stub(monkeypatch):
    """Install a stub client; returns the completions recorder."""
    completions = StubCompletions()

    def install(fail_for=(), content="ok", reject_params=()):
        completions.fail_for = set(fail_for)
        completions.content = content
        completions.reject_params = set(reject_params)
        client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})})
        monkeypatch.setattr(llm, "_client", lambda base_url, key: client)
        return completions

    install()
    return install


# --- Provider resolution ---------------------------------------------------


def test_defaults_to_groq():
    assert llm.provider().name == "groq"
    assert llm.provider().base_url == "https://api.groq.com/openai/v1"
    assert llm.provider().key_env == "GROQ_API_KEY"


def test_switching_provider_is_one_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert llm.provider().base_url == "https://openrouter.ai/api/v1"
    assert llm.provider().key_env == "OPENROUTER_API_KEY"


def test_provider_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  GROQ ")
    assert llm.provider().name == "groq"


def test_base_url_override_supports_unknown_endpoints(monkeypatch):
    """A local vLLM/LiteLLM server needs no entry in the registry."""
    monkeypatch.setenv("LLM_PROVIDER", "my-vllm")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    prov = llm.provider()
    assert prov.base_url == "http://localhost:8000/v1"
    assert prov.key_env == "LLM_API_KEY"


def test_unknown_provider_without_a_base_url_is_an_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nonesuch")
    with pytest.raises(llm.LLMError, match="unknown LLM_PROVIDER"):
        llm.provider()


def test_base_url_override_wins_over_a_known_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_BASE_URL", "http://proxy.internal/v1")
    assert llm.provider().base_url == "http://proxy.internal/v1"


# --- Key resolution --------------------------------------------------------


def test_api_key_comes_from_the_provider_variable(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    assert llm.api_key() == "gsk-test"
    assert llm.has_api_key()


def test_explicit_key_overrides_the_provider_variable(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("LLM_API_KEY", "override")
    assert llm.api_key() == "override"


def test_another_providers_key_is_not_used(monkeypatch):
    """An OpenRouter key left over in .env must not look like Groq config."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-old")
    assert llm.api_key() is None
    assert not llm.has_api_key()


def test_empty_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert not llm.has_api_key()


# --- Model roles -----------------------------------------------------------


def test_each_role_has_a_groq_default():
    assert llm.model_for("chat") == "openai/gpt-oss-120b"
    assert llm.model_for("fast") == "openai/gpt-oss-20b"
    assert llm.model_for("guard") == "openai/gpt-oss-safeguard-20b"


def test_roles_are_independently_overridable(monkeypatch):
    monkeypatch.setenv("LLM_FAST_MODEL", "tiny-model")
    assert llm.model_for("fast") == "tiny-model"
    assert llm.model_for("chat") == "openai/gpt-oss-120b"


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="unknown model role"):
        llm.model_for("embedding")


def test_chain_appends_fallbacks_without_duplicating_the_primary(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "primary")
    monkeypatch.setenv("LLM_FALLBACKS", "primary, backup , ")
    assert llm.model_chain("chat") == ["primary", "backup"]


def test_guard_role_has_no_fallback_chain(monkeypatch):
    """A substituted safety model may not share the verdict format."""
    monkeypatch.setenv("LLM_FALLBACKS", "some-other-model")
    assert llm.model_chain("guard") == ["openai/gpt-oss-safeguard-20b"]


# --- complete() ------------------------------------------------------------


def test_complete_returns_content_from_the_primary_model(monkeypatch, stub):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    completions = stub(content="hello")

    assert llm.complete([{"role": "user", "content": "hi"}]) == "hello"
    assert len(completions.calls) == 1
    assert completions.calls[0]["model"] == "openai/gpt-oss-120b"


def test_complete_walks_the_chain_past_failures(monkeypatch, stub):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("LLM_MODEL", "broken")
    monkeypatch.setenv("LLM_FALLBACKS", "also-broken,working")
    completions = stub(fail_for=["broken", "also-broken"], content="recovered")

    assert llm.complete([{"role": "user", "content": "hi"}]) == "recovered"
    assert [c["model"] for c in completions.calls] == ["broken", "also-broken", "working"]


def test_complete_raises_when_every_model_fails(monkeypatch, stub):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("LLM_MODEL", "a")
    monkeypatch.setenv("LLM_FALLBACKS", "b")
    stub(fail_for=["a", "b"])

    with pytest.raises(llm.LLMError, match="all 2 model"):
        llm.complete([{"role": "user", "content": "hi"}])


def test_complete_without_a_key_names_the_variable_to_set(monkeypatch, stub):
    stub()
    with pytest.raises(llm.LLMError, match="GROQ_API_KEY"):
        llm.complete([{"role": "user", "content": "hi"}])


def test_complete_forwards_extra_arguments(monkeypatch, stub):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    completions = stub()

    llm.complete([{"role": "user", "content": "hi"}], role="fast", max_tokens=4, temperature=0)
    call = completions.calls[0]
    assert call["model"] == "openai/gpt-oss-20b"
    assert call["max_tokens"] == 4 and call["temperature"] == 0


def test_complete_accepts_an_explicit_model_list(monkeypatch, stub):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    completions = stub()

    llm.complete([{"role": "user", "content": "hi"}], models=["pinned"])
    assert completions.calls[0]["model"] == "pinned"


def test_complete_tolerates_a_null_content_response(monkeypatch, stub):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    stub(content=None)
    assert llm.complete([{"role": "user", "content": "hi"}]) == ""


# --- Role parameters -------------------------------------------------------


def test_reasoning_is_low_by_default_for_the_router():
    """gpt-oss reasons before answering; a one-word classification needs less
    of it. "none" 400s on Groq's validator, so "low" is the default."""
    assert llm.role_params("fast") == {"reasoning_effort": "low"}
    assert llm.role_params("chat") == {}


def test_role_params_can_be_overridden_or_cleared(monkeypatch):
    monkeypatch.setenv("LLM_FAST_PARAMS", '{"reasoning_effort": "high"}')
    assert llm.role_params("fast") == {"reasoning_effort": "high"}
    monkeypatch.setenv("LLM_FAST_PARAMS", "{}")
    assert llm.role_params("fast") == {}


def test_malformed_role_params_are_reported_not_ignored(monkeypatch):
    monkeypatch.setenv("LLM_FAST_PARAMS", "reasoning_effort=none")
    with pytest.raises(llm.LLMError, match="not valid JSON"):
        llm.role_params("fast")

    monkeypatch.setenv("LLM_FAST_PARAMS", '["nope"]')
    with pytest.raises(llm.LLMError, match="must be a JSON object"):
        llm.role_params("fast")


def test_role_params_are_applied_to_the_request(monkeypatch, stub):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    completions = stub()
    llm.complete([{"role": "user", "content": "hi"}], role="fast")
    assert completions.calls[0]["reasoning_effort"] == "low"


def test_caller_kwargs_win_over_role_params(monkeypatch, stub):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    completions = stub()
    llm.complete([{"role": "user", "content": "hi"}], role="fast", reasoning_effort="high")
    assert completions.calls[0]["reasoning_effort"] == "high"


def test_unsupported_role_params_are_dropped_and_retried(monkeypatch, stub):
    """Switching to a provider that rejects reasoning_effort must degrade, not
    break — the same model is retried once without the extras."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    completions = stub(reject_params=["reasoning_effort"], content="recovered")

    assert llm.complete([{"role": "user", "content": "hi"}], role="fast") == "recovered"
    assert len(completions.calls) == 2
    assert "reasoning_effort" in completions.calls[0]
    assert "reasoning_effort" not in completions.calls[1]


def test_retry_without_params_does_not_mask_a_dead_model(monkeypatch, stub):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("LLM_FAST_MODEL", "dead")
    monkeypatch.setenv("LLM_FALLBACKS", "alive")
    completions = stub(fail_for=["dead"], reject_params=["reasoning_effort"])

    assert llm.complete([{"role": "user", "content": "hi"}], role="fast") == "ok"
    # Each model gets the params attempt then the bare retry, so a dead model
    # is retired properly instead of the param retry hiding it.
    assert [c["model"] for c in completions.calls] == ["dead", "dead", "alive", "alive"]
    assert "reasoning_effort" not in completions.calls[-1]
