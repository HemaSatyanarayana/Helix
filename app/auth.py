"""API key authentication.

The rate limiter previously keyed off an ``X-User-Id`` header the client sent
and nobody verified, so the per-user quota was bypassed by changing a string.
Identity has to be *established* before it can be metered.

Keys are configured as ``API_KEYS=alice:sk-live-…,bob:sk-live-…`` — the label
before the colon becomes the rate-limit identity and appears in traces, so a
quota can be attributed and a single caller revoked without rotating everyone's
key. A bare ``API_KEYS=sk-…,sk-…`` also works; those callers are identified by a
hash prefix of their key.

Disabled by default (``API_KEY_REQUIRED=false``) so local development and the
Streamlit UI keep working, but :func:`app.config.validate_config` refuses to
start a production environment with authentication off.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

from app.telemetry import configure_logfire

logfire = configure_logfire("helix-app")

API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
ANONYMOUS = "anonymous"


@dataclass(frozen=True)
class Identity:
    """Who is calling, and whether that was proven."""

    name: str
    authenticated: bool

    def __str__(self) -> str:
        return self.name


def _key_id(key: str) -> str:
    """Stable, non-reversible label for an unlabelled key."""
    return "key-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def load_keys() -> dict[str, str]:
    """Parse ``API_KEYS`` into {secret: identity}."""
    raw = os.getenv("API_KEYS", "")
    keys: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        label, _, secret = entry.partition(":")
        if secret:
            keys[secret.strip()] = label.strip()
        else:
            keys[label] = _key_id(label)
    return keys


def authenticate(api_key: str | None) -> Identity | None:
    """Resolve a key to an identity, or None when it is absent or wrong.

    Returns an anonymous identity when authentication is not required, so the
    same call path serves both modes.
    """
    keys = load_keys()

    if not API_KEY_REQUIRED:
        if api_key and api_key in keys:
            return Identity(name=keys[api_key], authenticated=True)
        return Identity(name=ANONYMOUS, authenticated=False)

    if not api_key or not keys:
        return None

    # Compare against every key rather than a dict lookup, so response time
    # doesn't reveal whether a prefix was right.
    matched: str | None = None
    for secret, label in keys.items():
        if hmac.compare_digest(api_key, secret):
            matched = label
    if matched is None:
        logfire.warning("auth_rejected", key_prefix=(api_key or "")[:6])
        return None
    return Identity(name=matched, authenticated=True)
