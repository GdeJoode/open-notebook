"""Tests for the G.3b agent upload façade — process-document / process-audio.

Multipart-upload siblings of process-url: the reuse (_create_source_impl) is
mocked, so these are HTTP contract tests (no DB, no disk) — the façade maps the
multipart request onto the shipped `type="upload"` chain, parses transformations,
and returns a pollable job id. The upload guards live inside the mocked impl and
are covered by the I.H1 upload tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import app_main.api.agent_auth as agent_auth
import app_main.api.routers.sources_upload as sources_upload
from app_main.api.agent_auth import AgentAuditMiddleware
from app_main.api.rate_limit import limiter
from app_main.api.routers import agents
from app_main.dependencies import (
    get_notebook_service,
    get_source_service,
    get_transformation_service,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware


def _make_client(source_svc=None) -> TestClient:
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(AgentAuditMiddleware)
    app.include_router(agents.router)
    app.dependency_overrides[get_source_service] = lambda: source_svc or AsyncMock()
    app.dependency_overrides[get_notebook_service] = lambda: AsyncMock()
    app.dependency_overrides[get_transformation_service] = lambda: AsyncMock()
    return TestClient(app, raise_server_exceptions=False)


def _auth(monkeypatch, permission, agent_id="a1"):
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(
            return_value={
                "id": "agent_keys:1",
                "agent_id": agent_id,
                "permission": permission,
            }
        ),
    )


def _mock_impl(monkeypatch, source_svc):
    create_impl = AsyncMock(return_value=SimpleNamespace(id="source:x"))
    monkeypatch.setattr(sources_upload, "_create_source_impl", create_impl)
    source_svc.get = AsyncMock(return_value=SimpleNamespace(command="cmd:job1"))
    return create_impl


# -- process-document -------------------------------------------------------


def test_process_document_reuses_upload_chain_and_returns_job_id(monkeypatch):
    _auth(monkeypatch, "write")
    source_svc = AsyncMock()
    create_impl = _mock_impl(monkeypatch, source_svc)

    client = _make_client(source_svc)
    resp = client.post(
        "/api/v1/agents/process-document",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"notebook_id": "notebook:n", "transformations": '["t1", "t2"]'},
        headers={"X-API-Key": "ak_w"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "cmd:job1"
    assert body["source_id"] == "source:x"
    assert body["status"] == "queued"
    # The façade fed the shipped chain an upload SourceCreate + the UploadFile.
    sc = create_impl.await_args.args[0]
    assert sc.type == "upload"
    assert sc.notebook_id == "notebook:n"
    assert sc.transformations == ["t1", "t2"]
    assert sc.async_processing is True
    assert create_impl.await_args.args[1] is not None  # the upload file


def test_process_document_read_key_is_403(monkeypatch):
    _auth(monkeypatch, "read")  # write-gated
    client = _make_client()
    resp = client.post(
        "/api/v1/agents/process-document",
        files={"file": ("d.txt", b"hi", "text/plain")},
        data={"notebook_id": "notebook:n"},
        headers={"X-API-Key": "ak_r"},
    )
    assert resp.status_code == 403


def test_process_document_parses_comma_separated_transformations(monkeypatch):
    _auth(monkeypatch, "write")
    source_svc = AsyncMock()
    create_impl = _mock_impl(monkeypatch, source_svc)

    client = _make_client(source_svc)
    resp = client.post(
        "/api/v1/agents/process-document",
        files={"file": ("d.txt", b"hi", "text/plain")},
        data={"notebook_id": "notebook:n", "transformations": "t1, t2 , t3"},
        headers={"X-API-Key": "ak_w"},
    )
    assert resp.status_code == 200
    assert create_impl.await_args.args[0].transformations == ["t1", "t2", "t3"]


def test_process_document_no_transformations_is_none(monkeypatch):
    _auth(monkeypatch, "write")
    source_svc = AsyncMock()
    create_impl = _mock_impl(monkeypatch, source_svc)

    client = _make_client(source_svc)
    resp = client.post(
        "/api/v1/agents/process-document",
        files={"file": ("d.txt", b"hi", "text/plain")},
        data={"notebook_id": "notebook:n"},
        headers={"X-API-Key": "ak_w"},
    )
    assert resp.status_code == 200
    assert create_impl.await_args.args[0].transformations is None


# -- process-audio ----------------------------------------------------------


def test_process_audio_reuses_same_upload_chain(monkeypatch):
    _auth(monkeypatch, "write")
    source_svc = AsyncMock()
    create_impl = _mock_impl(monkeypatch, source_svc)

    client = _make_client(source_svc)
    resp = client.post(
        "/api/v1/agents/process-audio",
        files={"file": ("talk.mp3", b"ID3 fake", "audio/mpeg")},
        data={"notebook_id": "notebook:n"},
        headers={"X-API-Key": "ak_w"},
    )
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "cmd:job1"
    assert create_impl.await_args.args[0].type == "upload"


def test_process_audio_read_key_is_403(monkeypatch):
    _auth(monkeypatch, "read")
    client = _make_client()
    resp = client.post(
        "/api/v1/agents/process-audio",
        files={"file": ("talk.mp3", b"ID3 fake", "audio/mpeg")},
        data={"notebook_id": "notebook:n"},
        headers={"X-API-Key": "ak_r"},
    )
    assert resp.status_code == 403
