"""G.7 — the published agent OpenAPI spec is valid + externally consumable.

Rather than shell out to openapi-generator, assert the spec `GET
/api/v1/agents/openapi.json` returns is a well-formed OpenAPI 3.x document that
covers the agent surface (the capability + poll + audit routes) with a resolvable
components section — the properties an external client generator needs to succeed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import app_main.api.agent_auth as agent_auth
from app_main.api.routers import agents
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(monkeypatch) -> TestClient:
    app = FastAPI(title="Open Notebook")
    app.include_router(agents.router)
    # Router-level gate: a valid (admin) key passes; the per-IP throttle allows a
    # single request. No app.state.limiter needed for one call.
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(
            return_value={"id": "agent_keys:1", "agent_id": "t", "permission": "admin"}
        ),
    )
    return TestClient(app, raise_server_exceptions=False)


def test_agent_openapi_is_valid_and_covers_the_surface(monkeypatch):
    resp = _client(monkeypatch).get(
        "/api/v1/agents/openapi.json", headers={"X-API-Key": "ak"}
    )
    assert resp.status_code == 200
    spec = resp.json()

    # Well-formed OpenAPI 3.x envelope.
    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"] == "Open Notebook Agent API"
    assert isinstance(spec.get("paths"), dict) and spec["paths"]

    # Only agent paths are published (the filter holds).
    assert all(p.startswith("/api/v1/agents") for p in spec["paths"])

    # The capability + lifecycle surface is present.
    expected = {
        "/api/v1/agents/extract-entities",
        "/api/v1/agents/generate-summary",
        "/api/v1/agents/process-url",
        "/api/v1/agents/process-document",
        "/api/v1/agents/process-audio",
        "/api/v1/agents/jobs/{job_id}",
        "/api/v1/agents/audit-log",
    }
    assert expected.issubset(set(spec["paths"]))

    # A generator needs resolvable request/response schemas.
    assert "schemas" in spec.get("components", {})
    post = spec["paths"]["/api/v1/agents/process-url"]["post"]
    assert "requestBody" in post
    assert "200" in post["responses"]


def test_agent_openapi_response_models_referenced(monkeypatch):
    spec = _client(monkeypatch).get(
        "/api/v1/agents/openapi.json", headers={"X-API-Key": "ak"}
    ).json()
    schemas = spec["components"]["schemas"]
    # The agent request/response models are emitted into the components.
    assert "ProcessUrlRequest" in schemas
    assert "ProcessSourceResponse" in schemas
    assert "ExtractEntitiesResponse" in schemas
