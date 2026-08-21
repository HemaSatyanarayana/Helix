"""Tests for routing and answer assembly.

The route parser is the regression guard for a bug the Groq migration exposed:
reasoning models spend a tight ``max_tokens`` budget on reasoning and return
empty content, which silently routed every question conversational and turned
retrieval off for the whole application.
"""

from __future__ import annotations

import pytest

from app import generation
from app.generation import (
    _clean_rewrite,
    _heuristic_route,
    _parse_route,
    classify_query,
    format_contexts,
    rewrite_query,
)


# --- Route parsing ---------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("technical", "technical"),
        ("Technical", "technical"),
        ("  conversational  ", "conversational"),
        ("conversational.", "conversational"),
        ("off_topic", "off_topic"),
        ("off-topic", "off_topic"),
        ("this is off topic", "off_topic"),
    ],
)
def test_plain_labels_parse(reply, expected):
    assert _parse_route(reply) == expected


def test_off_topic_is_not_mistaken_for_topical():
    """"off_topic" must not accidentally match a substring check for "topic"
    or get shadowed by "technical" appearing earlier in reasoning text."""
    reply = "This seems technical at first glance but is unrelated to the product.\noff_topic"
    assert _parse_route(reply) == "off_topic"


def test_empty_content_states_no_route():
    """What a reasoning model returns when the token budget ran out."""
    assert _parse_route("") is None
    assert _parse_route("   \n  ") is None


def test_unrecognized_reply_states_no_route():
    assert _parse_route("I'm not sure what you mean.") is None


def test_leaked_reasoning_does_not_decide_the_route():
    """An early "not technical" must not outvote the conclusion."""
    reply = (
        "The user says hello, so this is not technical.\n"
        "conversational"
    )
    assert _parse_route(reply) == "conversational"

    reply = "This could be conversational, but they ask about setup.\ntechnical"
    assert _parse_route(reply) == "technical"


# --- Heuristic fallback ----------------------------------------------------


@pytest.mark.parametrize(
    "question", ["hi", "hey there", "thanks!", "who are you?", "help"]
)
def test_heuristic_recognizes_small_talk(question):
    assert _heuristic_route(question) == "conversational"


def test_heuristic_defaults_to_technical():
    assert _heuristic_route("How do I add a new app in Apxor?") == "technical"


# --- classify_query --------------------------------------------------------


def test_routing_without_a_key_uses_the_heuristic(monkeypatch):
    monkeypatch.setattr(generation, "has_api_key", lambda: False)

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("the router called the LLM without a key")

    monkeypatch.setattr(generation, "complete", explode)
    assert classify_query("How do I add a new app?") == "technical"


def test_router_outage_falls_back_to_the_heuristic(monkeypatch):
    monkeypatch.setattr(generation, "has_api_key", lambda: True)

    def fail(*args, **kwargs):
        raise generation.LLMError("all models failed")

    monkeypatch.setattr(generation, "complete", fail)
    assert classify_query("How do I add a new app?") == "technical"
    assert classify_query("hey there") == "conversational"


def test_empty_model_reply_falls_back_to_the_heuristic(monkeypatch):
    """The exact failure the Groq migration hit: rather than defaulting every
    question to conversational, an unusable reply defers to the heuristic."""
    monkeypatch.setattr(generation, "has_api_key", lambda: True)
    monkeypatch.setattr(generation, "complete", lambda *a, **k: "")
    assert classify_query("How do I add a new app in Apxor?") == "technical"


def test_router_uses_the_fast_role_with_room_to_answer(monkeypatch):
    seen = {}

    monkeypatch.setattr(generation, "has_api_key", lambda: True)

    def record(messages, **kwargs):
        seen.update(kwargs)
        return "technical"

    monkeypatch.setattr(generation, "complete", record)
    classify_query("How do I add a new app?")
    assert seen["role"] == "fast"
    assert seen["max_tokens"] >= 512  # room for reasoning tokens


# --- Query rewriting --------------------------------------------------------


def test_rewrite_is_skipped_without_history():
    """No history to resolve pronouns against — the question is already
    standalone, and skipping avoids a pointless LLM call on every turn."""
    assert rewrite_query("How do I add a new app?", history=None) == "How do I add a new app?"
    assert rewrite_query("How do I add a new app?", history=[]) == "How do I add a new app?"


def test_rewrite_is_skipped_without_a_key(monkeypatch):
    monkeypatch.setattr(generation, "has_api_key", lambda: False)
    history = [{"role": "user", "content": "how do I integrate the SDK?"}]
    assert rewrite_query("how do I do that on iOS?", history=history) == "how do I do that on iOS?"


def test_rewrite_resolves_a_follow_up(monkeypatch):
    monkeypatch.setattr(generation, "has_api_key", lambda: True)
    monkeypatch.setattr(
        generation, "complete", lambda *a, **k: "How do I integrate the SDK on iOS?"
    )
    history = [{"role": "user", "content": "how do I integrate the SDK?"}]
    out = rewrite_query("how do I do that on iOS?", history=history)
    assert out == "How do I integrate the SDK on iOS?"


def test_rewrite_outage_returns_the_original_question(monkeypatch):
    """A degraded query still beats no query — never drop the search."""
    monkeypatch.setattr(generation, "has_api_key", lambda: True)

    def fail(*a, **k):
        raise generation.LLMError("down")

    monkeypatch.setattr(generation, "complete", fail)
    history = [{"role": "user", "content": "how do I integrate the SDK?"}]
    assert rewrite_query("how do I do that on iOS?", history=history) == (
        "how do I do that on iOS?"
    )


def test_rewrite_empty_reply_returns_the_original_question(monkeypatch):
    monkeypatch.setattr(generation, "has_api_key", lambda: True)
    monkeypatch.setattr(generation, "complete", lambda *a, **k: "")
    history = [{"role": "user", "content": "how do I integrate the SDK?"}]
    assert rewrite_query("how do I do that on iOS?", history=history) == (
        "how do I do that on iOS?"
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("How do I set up the SDK on iOS?", "How do I set up the SDK on iOS?"),
        ('"How do I set up the SDK on iOS?"', "How do I set up the SDK on iOS?"),
        ("Query: How do I set up the SDK on iOS?", "How do I set up the SDK on iOS?"),
        ("Reasoning...\nHow do I set up the SDK on iOS?", "How do I set up the SDK on iOS?"),
        ("", None),
        ("   ", None),
    ],
)
def test_clean_rewrite_strips_labels_and_quoting(reply, expected):
    assert _clean_rewrite(reply) == expected


# --- Context formatting ------------------------------------------------------


def test_format_contexts_attributes_each_chunk_to_its_source():
    chunks = [
        {"source": "docs/a.md", "headings": ["Setup", "iOS"], "text": "Do X."},
        {"source": "docs/b.md", "headings": [], "text": "Do Y."},
    ]
    block = format_contexts(chunks)
    assert "[1] docs/a.md — Setup › iOS" in block
    assert "Do X." in block
    assert "[2] docs/b.md" in block
    assert "Do Y." in block


def test_format_contexts_handles_missing_source():
    block = format_contexts([{"text": "orphan chunk"}])
    assert "[1] unknown source" in block
