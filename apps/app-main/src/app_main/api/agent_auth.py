"""Agent API-key authentication dependency (Track G.1, decision G-D1).

The SOLE authenticator for ``/api/v1/agents/*`` routes. Those routes are excluded
from the shared-password middleware (``auth.py``), so this dependency is the only
gate — and it is **fail-closed and password-independent**: a missing / invalid /
revoked key is always a 401, regardless of whether ``OPEN_NOTEBOOK_PASSWORD`` is
set. That is a net security improvement over the shared-password middleware, which
is fail-OPEN when no password is configured.

Scope: a route declares a minimum permission via :func:`require_agent_key`; a key
whose scope is >= it passes (``read`` < ``write`` < ``admin``). The resolved key
context is attached to ``request.state.agent_key`` so the sub-app's audit
middleware can record the call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app_main.services.agents.key_service import AgentKeyService

_AGENT_PREFIX = "/api/v1/agents"

_PERMISSION_LEVEL = {"read": 0, "write": 1, "admin": 2}


@dataclass(frozen=True)
class AgentKeyContext:
    """The resolved, authenticated key for a request."""

    key_id: str
    agent_id: str
    permission: str
    rate_limit_rpm: Optional[int] = None


def _level(permission: str) -> int:
    return _PERMISSION_LEVEL.get(permission, -1)


def require_agent_key(min_permission: str = "read"):
    """FastAPI dependency factory gating an agent route by ``min_permission``.

    Reads ``X-API-Key``, authenticates it (SHA-256 lookup of a non-revoked key),
    enforces the scope, attaches the :class:`AgentKeyContext` to
    ``request.state.agent_key``, and returns it. Raises 401 (missing / invalid /
    revoked) or 403 (insufficient scope). Never falls open.
    """
    required = _level(min_permission)

    async def _dependency(
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> AgentKeyContext:
        # Reuse a key already resolved by a broader gate this request (the
        # router-level read gate), so a route that additionally requires a higher
        # scope (e.g. write) re-checks the SCOPE without a second DB lookup +
        # last_used_at write. Fixes the round-1 N4 double-authenticate.
        cached = getattr(request.state, "agent_key", None)
        if cached is not None:
            if _level(cached.permission) < required:
                raise HTTPException(
                    status_code=403,
                    detail=f"API key lacks '{min_permission}' scope",
                )
            return cached
        if not x_api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing X-API-Key",
                headers={"WWW-Authenticate": "X-API-Key"},
            )
        row = await AgentKeyService().authenticate(x_api_key)
        if row is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or revoked API key",
                headers={"WWW-Authenticate": "X-API-Key"},
            )
        permission = str(row.get("permission", "read"))
        if _level(permission) < required:
            raise HTTPException(
                status_code=403,
                detail=f"API key lacks '{min_permission}' scope",
            )
        ctx = AgentKeyContext(
            key_id=str(row.get("id")),
            agent_id=str(row.get("agent_id")),
            permission=permission,
            rate_limit_rpm=row.get("rate_limit_rpm"),
        )
        # Expose to the sub-app audit middleware + the per-key rate limiter.
        request.state.agent_key = ctx
        return ctx

    return _dependency


class AgentAuditMiddleware(BaseHTTPMiddleware):
    """Append-only per-call audit for the agent API (G.1).

    Records every ``/api/v1/agents/*`` call — including auth failures (401/403,
    where no key context was attached) — with method / path / status / latency.
    Best-effort: an audit write never affects the response. Reads the resolved
    key context off ``request.state.agent_key`` (set by :func:`require_agent_key`
    on the success path).
    """

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith(_AGENT_PREFIX):
            return await call_next(request)
        start = time.time()
        response = await call_next(request)
        latency_ms = int((time.time() - start) * 1000)
        # Do NOT audit a throttled (429) request. The per-IP pre-auth throttle
        # already bounds those to the per-minute cap; writing one agent_audit_log
        # row per rejected flood request would itself be a write-amplification DoS
        # (the throttle bounds the auth DB READ, this bounds the audit DB WRITE).
        # The ~N/min requests that pass the throttle and reach auth ARE recorded
        # (incl. 401s), so brute-force visibility is preserved — and bounded.
        if response.status_code == 429:
            return response
        ctx = getattr(request.state, "agent_key", None)
        try:
            from app_main.services.agents.audit_service import AgentAuditService

            await AgentAuditService().record(
                agent_id=ctx.agent_id if ctx else "unknown",
                agent_key_id=ctx.key_id if ctx else None,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=latency_ms,
            )
        except Exception:  # noqa: BLE001 — audit must never affect the response
            pass
        return response


__all__ = ["AgentAuditMiddleware", "AgentKeyContext", "require_agent_key"]
