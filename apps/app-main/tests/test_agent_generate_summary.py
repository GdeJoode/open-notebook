"""Tests for the G.2 generate-summary agent capability.

The service method (`summarize_text`) is unit-tested with the workflow mocked
(no LLM, no DB); the endpoint is contract-tested with auth + service mocked
(write→200, read→403, no key→401).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import app_main.api.agent_auth as agent_auth
import pytest
import summarization.workflow as sumwf
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app_main.api.agent_auth import AgentAuditMiddleware
from app_main.api.rate_limit import limiter
from app_main.api.routers import agents
from app_main.dependencies import get_summarization_service
from app_main.services.summarization_service import SummarizationService

pytestmark = pytest.mark.asyncio


# -- service method ---------------------------------------------------------


async def test_summarize_text_no_db_write(monkeypatch):
    repo = AsyncMock()
    svc = SummarizationService(summary_repo=repo)
    monkeypatch.setattr(svc, "_build_routed_model", AsyncMock(return_value=None))

    class _FakeWF:
        def __init__(self, config):
            pass

        async def process(self, chunks):
            return SimpleNamespace(
                document_summary="A concise summary.",
                strategy=SimpleNamespace(value="naive"),
            )

    monkeypatch.setattr(sumwf, "SummarizationWorkflow", _FakeWF)

    out = await svc.summarize_text("some long text to summarize", strategy="naive")

    assert out["summary"] == "A concise summary."
    assert out["strategy"] == "naive"
    repo.create_summary.assert_not_called()  # no DB write on the raw-text path


async def test_summarize_text_unknown_strategy_raises():
    svc = SummarizationService(summary_repo=AsyncMock())
    with pytest.raises(ValueError):
        await svc.summarize_text("x", strategy="does-not-exist")


async def test_summarize_text_empty_text_is_empty_summary():
    svc = SummarizationService(summary_repo=AsyncMock())
    out = await svc.summarize_text("   ", strategy="naive")
    assert out["summary"] == ""


# -- endpoint ---------------------------------------------------------------


def _make_client(summary_svc) -> TestClient:
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(AgentAuditMiddleware)
    app.include_router(agents.router)
    app.dependency_overrides[get_summarization_service] = lambda: summary_svc
    return TestClient(app, raise_server_exceptions=False)


def _auth_returns(monkeypatch, permission):
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(
            return_value={"id": "agent_keys:1", "agent_id": "a", "permission": permission}
        ),
    )


def test_generate_summary_write_key_returns_200(monkeypatch):
    _auth_returns(monkeypatch, "write")
    svc = AsyncMock()
    svc.summarize_text = AsyncMock(
        return_value={"summary": "Sum.", "strategy": "naive"}
    )
    client = _make_client(svc)
    resp = client.post(
        "/api/v1/agents/generate-summary",
        json={"text": "text", "strategy": "naive"},
        headers={"X-API-Key": "ak_w"},
    )
    assert resp.status_code == 200
    assert resp.json()["summary"] == "Sum."


def test_generate_summary_read_key_is_403(monkeypatch):
    _auth_returns(monkeypatch, "read")  # read < write → forbidden
    client = _make_client(AsyncMock())
    resp = client.post(
        "/api/v1/agents/generate-summary",
        json={"text": "text"},
        headers={"X-API-Key": "ak_r"},
    )
    assert resp.status_code == 403


def test_generate_summary_no_key_is_401():
    client = _make_client(AsyncMock())
    resp = client.post(
        "/api/v1/agents/generate-summary", json={"text": "text"}
    )
    assert resp.status_code == 401
