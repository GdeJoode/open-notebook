"""Security tests for the agent-key auth dependency (Track G.1, G-D1).

`require_agent_key` is the SOLE authenticator for /api/v1/agents/* — fail-closed
and password-independent. These lock its contract: no key → 401, invalid/revoked
→ 401, insufficient scope → 403, valid + sufficient scope → the key context. The
DB lookup is mocked (AgentKeyService.authenticate), so this is pure dependency logic.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import app_main.api.agent_auth as agent_auth
import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.asyncio


def _request() -> SimpleNamespace:
    # require_agent_key only touches request.state.
    return SimpleNamespace(state=SimpleNamespace())


def _patch_auth(monkeypatch, return_value):
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(return_value=return_value),
    )


async def test_missing_key_is_401(monkeypatch):
    dep = agent_auth.require_agent_key("read")
    with pytest.raises(HTTPException) as exc:
        await dep(request=_request(), x_api_key=None)
    assert exc.value.status_code == 401


async def test_invalid_or_revoked_key_is_401(monkeypatch):
    _patch_auth(monkeypatch, None)  # authenticate returns None for unknown/revoked
    dep = agent_auth.require_agent_key("read")
    with pytest.raises(HTTPException) as exc:
        await dep(request=_request(), x_api_key="ak_bogus")
    assert exc.value.status_code == 401


async def test_insufficient_scope_is_403(monkeypatch):
    _patch_auth(
        monkeypatch,
        {"id": "agent_keys:1", "agent_id": "a1", "permission": "read"},
    )
    dep = agent_auth.require_agent_key("write")  # route needs write; key is read
    with pytest.raises(HTTPException) as exc:
        await dep(request=_request(), x_api_key="ak_good")
    assert exc.value.status_code == 403


async def test_valid_key_returns_context_and_attaches_state(monkeypatch):
    _patch_auth(
        monkeypatch,
        {
            "id": "agent_keys:1",
            "agent_id": "a1",
            "permission": "admin",
            "rate_limit_rpm": 30,
        },
    )
    req = _request()
    dep = agent_auth.require_agent_key("write")  # admin >= write → ok
    ctx = await dep(request=req, x_api_key="ak_good")
    assert ctx.agent_id == "a1"
    assert ctx.permission == "admin"
    assert ctx.rate_limit_rpm == 30
    # The context is attached for the audit middleware.
    assert req.state.agent_key is ctx


async def test_read_scope_route_accepts_read_key(monkeypatch):
    _patch_auth(
        monkeypatch,
        {"id": "agent_keys:1", "agent_id": "a1", "permission": "read"},
    )
    dep = agent_auth.require_agent_key("read")
    ctx = await dep(request=_request(), x_api_key="ak_good")
    assert ctx.permission == "read"


# -- per-IP pre-auth throttle (round-1 review fix #1) ------------------------


async def test_agent_ip_throttle_429s_after_limit(monkeypatch):
    import app_main.api.agent_rate_limit as arl

    monkeypatch.setenv("AGENT_RATE_LIMIT_RPM", "2")
    arl._ip_hits.clear()  # isolate from other tests (module-level state)

    dep = arl.agent_ip_throttle()
    req = SimpleNamespace(client=SimpleNamespace(host="9.9.9.9"), headers={})

    await dep(req)  # 1
    await dep(req)  # 2
    with pytest.raises(HTTPException) as exc:
        await dep(req)  # 3 → over the cap
    assert exc.value.status_code == 429
    assert exc.value.headers.get("Retry-After") == "60"
