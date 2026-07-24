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


__all__ = ["agent_default_limit", "agent_key_func"]
