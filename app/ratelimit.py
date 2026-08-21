"""Redis-backed per-user rate limiting (sliding window).

Uses a single atomic Lua script (sorted-set sliding window) so concurrent
requests can't race past the limit. Fails **open** — if Redis is unreachable the
request is allowed and a warning is logged, so a Redis blip never takes the API
down. Configure via RATE_LIMIT_MAX / RATE_LIMIT_WINDOW_SECONDS.
"""

from __future__ import annotations

import os
import time
import uuid
from functools import lru_cache

from app.telemetry import configure_logfire

logfire = configure_logfire("helix-app")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# KEYS[1]=bucket key; ARGV: now_ms, window_ms, limit, unique_member
# Returns {allowed(1/0), remaining, retry_after_seconds}
_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count < limit then
  redis.call('ZADD', key, now, ARGV[4])
  redis.call('PEXPIRE', key, window)
  return {1, limit - count - 1, 0}
else
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local retry = 1
  if oldest[2] then
    retry = math.ceil((tonumber(oldest[2]) + window - now) / 1000)
  end
  return {0, 0, retry}
end
"""


@lru_cache(maxsize=1)
def _redis():
    import redis

    return redis.Redis.from_url(REDIS_URL)


@lru_cache(maxsize=1)
def _script():
    return _redis().register_script(_LUA)


def check_rate_limit(
    user_id: str,
    limit: int = RATE_LIMIT_MAX,
    window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
) -> tuple[bool, int, int]:
    """Return (allowed, remaining, retry_after_seconds) for a user."""
    now_ms = int(time.time() * 1000)
    key = f"ratelimit:chat:{user_id}"
    member = f"{now_ms}-{uuid.uuid4().hex}"
    try:
        allowed, remaining, retry_after = _script()(
            keys=[key], args=[now_ms, window_seconds * 1000, limit, member]
        )
        return bool(allowed), int(remaining), int(retry_after)
    except Exception as exc:  # noqa: BLE001 — fail open on Redis issues.
        logfire.warning("rate_limit_unavailable", error=str(exc))
        return True, limit, 0
