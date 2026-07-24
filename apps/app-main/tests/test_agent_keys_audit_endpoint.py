"""HTTP contract tests for the G.4 session-authed key audit-log endpoint.

`GET /api/agent-keys/{key_id}/audit-log` resolves a key to its agent_id
server-side and returns that agent's trail. The key service + audit service are
mocked (no DB): these assert the endpoint wiring — resolve, 404-on-missing,
limit clamp — not the SurrealQL (that lives in the service tests).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import app_main.services.agents.audit_service as aud
from app_main.api.routers import agent_keys
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(key_row) -> TestClient:
    app = FastAPI()
    app.include_router(agent_keys.router, prefix="/api")
    stub = AsyncMock()
    stub.get = AsyncMock(return_value=key_row)
    app.dependency_overrides[agent_keys.get_agent_key_service] = lambda: stub
    return TestClient(app, raise_server_exceptions=False)


def test_audit_log_resolves_agent_and_returns_entries(monkeypatch):
    lister = AsyncMock(
        return_value=[{"method": "POST", "path": "/process-url", "status": 200}]
    )
    monkeypatch.setattr(aud.AgentAuditService, "list_for_agent", lister)

    client = _client({"id": "agent_keys:1", "agent_id": "bot", "permission": "read"})
    resp = client.get("/api/agent-keys/agent_keys:1/audit-log")

    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_id"] == "bot"
    assert body["key_id"] == "agent_keys:1"
    assert body["entries"][0]["method"] == "POST"
    # Resolved server-side to the KEY's agent_id (not a client-supplied one).
    assert lister.await_args.args[0] == "bot"
    assert lister.await_args.kwargs["limit"] == 100


def test_audit_log_clamps_limit(monkeypatch):
    lister = AsyncMock(return_value=[])
    monkeypatch.setattr(aud.AgentAuditService, "list_for_agent", lister)

    client = _client({"id": "agent_keys:1", "agent_id": "bot", "permission": "read"})
    resp = client.get("/api/agent-keys/agent_keys:1/audit-log?limit=99999")
    assert resp.status_code == 200
    assert lister.await_args.kwargs["limit"] == 500  # clamped to the max


def test_audit_log_404_for_missing_key():
    client = _client(None)  # service.get() → None
    resp = client.get("/api/agent-keys/agent_keys:missing/audit-log")
    assert resp.status_code == 404
