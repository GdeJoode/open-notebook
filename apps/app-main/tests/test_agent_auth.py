"""Tests for the key check per route (F2-01, ADR 0002 A/B).

``require_right`` is the right check per route; ``authenticate_request`` reads
``Authorization: Bearer``. Contract: no key → 401, invalid/revoked/old right →
401, insufficient right → 403, sufficient right → the key context. The DB lookup
is mocked (AgentKeyService.authenticate), so this is pure dependency logic.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import app_main.api.agent_auth as agent_auth
import pytest
from fastapi import HTTPException
from shared.models.agents import ClassificationScope, Right

pytestmark = [pytest.mark.asyncio, pytest.mark.echte_sleutels]


def _request(authorization: str | None = None) -> SimpleNamespace:
    # require_right only touches request.state and the Authorization header.
    headers = {"Authorization": authorization} if authorization else {}
    return SimpleNamespace(state=SimpleNamespace(), headers=headers)


def _patch_auth(monkeypatch, return_value):
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(return_value=return_value),
    )


def _row(permission: str, **extra) -> dict:
    return {"id": "agent_keys:1", "agent_id": "a1", "permission": permission, **extra}


async def test_missing_key_is_401(monkeypatch):
    dep = agent_auth.require_right(Right.LEZEN_EXTERN)
    with pytest.raises(HTTPException) as exc:
        await dep(request=_request())
    assert exc.value.status_code == 401


@pytest.mark.parametrize("header", ["ak_good", "Basic ak_good", "Bearer "])
async def test_non_bearer_header_is_401(monkeypatch, header):
    _patch_auth(monkeypatch, _row("beheerder", person="P"))
    dep = agent_auth.require_right(Right.LEZEN_EXTERN)
    with pytest.raises(HTTPException) as exc:
        await dep(request=_request(header))
    assert exc.value.status_code == 401


async def test_invalid_or_revoked_key_is_401(monkeypatch):
    _patch_auth(monkeypatch, None)  # authenticate returns None for unknown/revoked
    dep = agent_auth.require_right(Right.LEZEN_EXTERN)
    with pytest.raises(HTTPException) as exc:
        await dep(request=_request("Bearer ak_bogus"))
    assert exc.value.status_code == 401


@pytest.mark.parametrize("old", ["read", "write", "admin"])
async def test_old_right_is_401(monkeypatch, old):
    _patch_auth(monkeypatch, _row(old))
    dep = agent_auth.require_right(Right.LEZEN_EXTERN)
    with pytest.raises(HTTPException) as exc:
        await dep(request=_request("Bearer ak_old"))
    assert exc.value.status_code == 401


async def test_beheerder_without_person_is_401(monkeypatch):
    _patch_auth(monkeypatch, _row("beheerder", person="  "))
    dep = agent_auth.require_right(Right.BEHEERDER)
    with pytest.raises(HTTPException) as exc:
        await dep(request=_request("Bearer ak_b"))
    assert exc.value.status_code == 401


#: (minimal right of the route, right of the key, allowed?)
_MATRIX = [
    (route, key, key in allowed)
    for route, allowed in {
        Right.LEZEN_EXTERN: {"lezen_extern", "lezen_intern", "pijplijn", "beheerder"},
        Right.LEZEN_INTERN: {"lezen_intern", "pijplijn", "beheerder"},
        Right.PIJPLIJN: {"pijplijn", "beheerder"},
        Right.OPHALEN: {"ophalen", "beheerder"},
        Right.BEHEERDER: {"beheerder"},
    }.items()
    for key in ["lezen_extern", "lezen_intern", "pijplijn", "ophalen", "beheerder"]
]


@pytest.mark.parametrize("route_right, key_right, allowed", _MATRIX)
async def test_right_matrix(monkeypatch, route_right, key_right, allowed):
    _patch_auth(monkeypatch, _row(key_right, person="Testpersoon"))
    dep = agent_auth.require_right(route_right)
    if allowed:
        ctx = await dep(request=_request("Bearer ak_k"))
        assert ctx.permission == Right(key_right)
    else:
        with pytest.raises(HTTPException) as exc:
            await dep(request=_request("Bearer ak_k"))
        assert exc.value.status_code == 403


async def test_valid_key_returns_context_and_attaches_state(monkeypatch):
    _patch_auth(
        monkeypatch, _row("beheerder", person="Testpersoon", rate_limit_rpm=30)
    )
    req = _request("Bearer ak_good")
    ctx = await agent_auth.require_right(Right.PIJPLIJN)(request=req)
    assert ctx.agent_id == "a1"
    assert ctx.permission == Right.BEHEERDER
    assert ctx.person == "Testpersoon"
    assert ctx.rate_limit_rpm == 30
    # The context is attached for the audit middleware.
    assert req.state.agent_key is ctx


async def test_key_from_middleware_is_reused_without_lookup(monkeypatch):
    lookup = AsyncMock()
    monkeypatch.setattr(agent_auth.AgentKeyService, "authenticate", lookup)
    req = _request()
    req.state.agent_key = agent_auth.AgentKeyContext(
        key_id="agent_keys:1", agent_id="a1", permission=Right.LEZEN_EXTERN
    )
    ctx = await agent_auth.require_right(Right.LEZEN_EXTERN)(request=req)
    assert ctx is req.state.agent_key
    lookup.assert_not_awaited()


@pytest.mark.parametrize(
    "right, scope",
    [
        (Right.LEZEN_EXTERN, ClassificationScope.EXTERN),
        (Right.LEZEN_INTERN, ClassificationScope.ALLES),
        (Right.PIJPLIJN, ClassificationScope.ALLES),
        (Right.OPHALEN, ClassificationScope.LEEG),
        (Right.BEHEERDER, ClassificationScope.ALLES),
    ],
)
async def test_scope_follows_right(right, scope):
    ctx = agent_auth.AgentKeyContext(key_id="k", agent_id="a", permission=right)
    assert ctx.scope == scope


async def test_require_right_is_one_function_per_right():
    assert agent_auth.require_right(Right.BEHEERDER) is agent_auth.require_right(
        "beheerder"
    )
    assert agent_auth.require_right(Right.BEHEERDER).required_right == Right.BEHEERDER


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


async def test_revoke_malformed_id_returns_false_no_raise():
    # N5 fix: a colon-less id must honour the idempotent bool contract, not raise.
    from app_main.services.agents.key_service import AgentKeyService

    assert await AgentKeyService().revoke("nocolon") is False
