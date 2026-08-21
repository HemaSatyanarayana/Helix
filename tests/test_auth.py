"""Tests for API key authentication.

The regression this guards: the rate limit used to key off a client-supplied
``X-User-Id`` header that nobody verified, so it was bypassed by changing a
string. Identity now has to be established by a key before it can be metered.
"""

from __future__ import annotations

import pytest

from app import auth
from app.auth import Identity, authenticate, load_keys


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setattr(auth, "API_KEY_REQUIRED", False)


# --- Key parsing ------------------------------------------------------------


def test_labelled_and_bare_keys_both_parse(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alice:sk-alice, sk-bare-one ,bob:sk-bob")
    keys = load_keys()
    assert keys["sk-alice"] == "alice"
    assert keys["sk-bob"] == "bob"
    assert keys["sk-bare-one"].startswith("key-")  # unlabelled -> hashed id


def test_bare_key_identity_is_stable_and_non_reversible(monkeypatch):
    monkeypatch.setenv("API_KEYS", "sk-secret-value")
    label = load_keys()["sk-secret-value"]
    assert label.startswith("key-")
    assert "sk-secret-value" not in label


def test_empty_api_keys_parses_to_nothing(monkeypatch):
    monkeypatch.setenv("API_KEYS", "")
    assert load_keys() == {}


# --- Optional mode (API_KEY_REQUIRED=false) --------------------------------


def test_optional_mode_without_a_key_is_anonymous():
    identity = authenticate(None)
    assert identity == Identity(name="anonymous", authenticated=False)


def test_optional_mode_with_a_valid_key_is_still_authenticated(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alice:sk-alice")
    identity = authenticate("sk-alice")
    assert identity == Identity(name="alice", authenticated=True)


def test_optional_mode_with_a_wrong_key_falls_back_to_anonymous(monkeypatch):
    """Optional auth degrades gracefully rather than rejecting the request."""
    monkeypatch.setenv("API_KEYS", "alice:sk-alice")
    identity = authenticate("not-a-real-key")
    assert identity is not None
    assert identity.authenticated is False


# --- Required mode (API_KEY_REQUIRED=true) ---------------------------------


def test_required_mode_rejects_a_missing_key(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY_REQUIRED", True)
    monkeypatch.setenv("API_KEYS", "alice:sk-alice")
    assert authenticate(None) is None


def test_required_mode_rejects_a_wrong_key(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY_REQUIRED", True)
    monkeypatch.setenv("API_KEYS", "alice:sk-alice")
    assert authenticate("wrong") is None


def test_required_mode_with_no_keys_configured_rejects_everything(monkeypatch):
    """Required-but-unconfigured must fail closed, not open the gate."""
    monkeypatch.setattr(auth, "API_KEY_REQUIRED", True)
    monkeypatch.setenv("API_KEYS", "")
    assert authenticate("anything") is None


def test_required_mode_accepts_the_right_key(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY_REQUIRED", True)
    monkeypatch.setenv("API_KEYS", "alice:sk-alice,bob:sk-bob")
    assert authenticate("sk-alice") == Identity(name="alice", authenticated=True)
    assert authenticate("sk-bob") == Identity(name="bob", authenticated=True)


def test_a_client_supplied_label_cannot_impersonate_another_caller(monkeypatch):
    """The old X-User-Id bypass: nothing the caller sends except the secret
    itself should be able to select an identity."""
    monkeypatch.setattr(auth, "API_KEY_REQUIRED", True)
    monkeypatch.setenv("API_KEYS", "alice:sk-alice")
    assert authenticate("bob") is None
    assert authenticate("alice") is None
