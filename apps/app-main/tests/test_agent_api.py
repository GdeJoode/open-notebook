"""Endpoint tests for the agent capability API (Track G.1).

A minimal app wires the agents router + the slowapi limiter + the audit
middleware; the key lookup and the extraction are mocked so this stays an HTTP
contract test (no DB, no LLM). Covers AC4: no key → 401, a valid read key → 200
with an entities array.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import app_main.api.agent_auth as agent_auth
import app_main.services.entity_extraction_service as ees
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app_main.api.agent_auth import AgentAuditMiddleware
from app_main.api.rate_limit import limiter
from app_main.api.routers import agents


def _make_client() -> TestClient:
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(AgentAuditMiddleware)
    app.include_router(agents.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _mock_extraction(monkeypatch):
    monkeypatch.setattr(
        ees,
        "extract_from_text",
        AsyncMock(
            return_value=[
                SimpleNamespace(name="Alice", entity_type="Person", confidence=0.9)
            ]
        ),
    )


def test_extract_entities_requires_a_key():
    client = _make_client()
    resp = client.post(
        "/api/v1/agents/extract-entities", json={"text": "Alice works at Acme."}
    )
    assert resp.status_code == 401


def test_extract_entities_rejects_an_unknown_key(monkeypatch):
    monkeypatch.setattr(
        agent_auth.AgentKeyService, "authenticate", AsyncMock(return_value=None)
    )
    client = _make_client()
    resp = client.post(
        "/api/v1/agents/extract-entities",
        json={"text": "Alice works at Acme."},
        headers={"X-API-Key": "ak_bogus"},
    )
    assert resp.status_code == 401


def test_extract_entities_with_valid_read_key_returns_entities(monkeypatch):
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(
            return_value={
                "id": "agent_keys:1",
                "agent_id": "agent-x",
                "permission": "read",
            }
        ),
    )
    client = _make_client()
    resp = client.post(
        "/api/v1/agents/extract-entities",
        json={"text": "Alice works at Acme."},
        headers={"X-API-Key": "ak_good"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["entities"][0]["name"] == "Alice"
    assert body["entities"][0]["entity_type"] == "Person"


def test_throttled_429_is_not_audited(monkeypatch):
    # N1 fix: a request rejected by the pre-auth throttle must NOT write an audit
    # row (else a flood is a write-amplification DoS). The throttle bounds writes
    # to the ~N/min that reach auth.
    import app_main.api.agent_rate_limit as arl
    import app_main.services.agents.audit_service as aud

    monkeypatch.setenv("AGENT_RATE_LIMIT_RPM", "1")
    arl._ip_hits.clear()
    record = AsyncMock()
    monkeypatch.setattr(aud.AgentAuditService, "record", record)
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(return_value={"id": "agent_keys:1", "agent_id": "a", "permission": "read"}),
    )
    client = _make_client()

    r1 = client.post(
        "/api/v1/agents/extract-entities", json={"text": "x"}, headers={"X-API-Key": "ak"}
    )
    r2 = client.post(
        "/api/v1/agents/extract-entities", json={"text": "x"}, headers={"X-API-Key": "ak"}
    )
    assert r1.status_code == 200
    assert r2.status_code == 429
    # The 200 is audited; the 429 is NOT.
    audited = [c.kwargs.get("status") for c in record.await_args_list]
    assert 200 in audited
    assert 429 not in audited
