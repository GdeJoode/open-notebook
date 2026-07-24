"""Per-key rate limiting for the agent API (Track G.1, decision G-D3).

The agent routes reuse the app's existing slowapi ``Limiter`` but with a per-route
``key_func`` that buckets by the **X-API-Key** header (so each agent key gets its
own quota, not a shared per-IP one) and a dynamic limit string from the config
default ``AGENT_RATE_LIMIT_RPM``.

Per-key ``rate_limit_rpm`` OVERRIDE (a key carrying its own cap) is a documented
follow-up: slowapi evaluates the limit before the auth dependency resolves the key
row, so honouring a per-key cap needs a pre-auth key lookup (cache). G.1 buckets
per key at the uniform default; G-follow-up adds the per-key override.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List

from fastapi import HTTPException
from starlette.requests import Request

from app_main.config import get_agent_rate_limit_rpm


def agent_key_func(request: Request) -> str:
    """Rate-limit bucket identity: the X-API-Key header, else the client IP."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"agentkey:{api_key}"
    from slowapi.util import get_remote_address

    return get_remote_address(request)


def agent_default_limit() -> str:
    """The default per-key limit as a slowapi limit string (``"60/minute"``)."""
    return f"{get_agent_rate_limit_rpm()}/minute"


# --- Per-IP PRE-AUTH throttle (fixes the auth-failure-flood gap) -------------
#
# The per-key slowapi limit is a decorator on the endpoint BODY, so it runs only
# AFTER `require_agent_key` resolves — an invalid-key request 401s before it, and
# each such request costs a DB read + an audit write. This dependency runs at the
# ROUTER level (before the auth dependency) and buckets by client IP, so a flood
# of bad keys from one host is bounded regardless of auth outcome. In-memory
# sliding window (same durability as the existing per-IP slowapi limiter);
# self-pruning so memory stays bounded.

_WINDOW_SEC = 60.0
_ip_hits: Dict[str, List[float]] = defaultdict(list)
_PRUNE_ABOVE = 10_000


def agent_ip_throttle():
    """Router dependency: per-IP requests/min cap that runs BEFORE agent auth."""

    async def _dep(request: Request) -> None:
        limit = get_agent_rate_limit_rpm()
        from slowapi.util import get_remote_address

        ip = get_remote_address(request)
        now = time.monotonic()
        hits = [t for t in _ip_hits.get(ip, []) if now - t < _WINDOW_SEC]
        if len(hits) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"},
            )
        hits.append(now)
        _ip_hits[ip] = hits
        if len(_ip_hits) > _PRUNE_ABOVE:
            for k in [k for k, v in list(_ip_hits.items()) if not v]:
                _ip_hits.pop(k, None)

    return _dep


__all__ = ["agent_default_limit", "agent_ip_throttle", "agent_key_func"]
