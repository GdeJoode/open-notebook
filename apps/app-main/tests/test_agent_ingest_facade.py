"""Tests for the G.3 agent ingest façade + job status + audit-log read.

The ingest reuse (_create_source_impl) and the job/audit services are mocked, so
these are HTTP contract tests (no DB, no real ingest): process-url maps the
request onto the shipped chain and returns a job id; jobs passthrough;
audit-log scoping (no cross-agent leak for a read key).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import app_main.api.agent_auth as agent_auth
import app_main.api.routers.sources_upload as sources_upload
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app_main.api.agent_auth import AgentAuditMiddleware
from app_main.api.rate_limit import limiter
from app_main.api.routers import agents
from app_main.dependencies import (
    get_notebook_service,
    get_source_service,
    get_transformation_service,
)


# F2-01: these tests exercise the real key check (no test key from conftest).
pytestmark = pytest.mark.echte_sleutels


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
                "person": "Testpersoon",
            }
        ),
    )


# -- process-url ------------------------------------------------------------


def test_process_url_reuses_ingest_and_returns_job_id(monkeypatch):
    _auth(monkeypatch, "beheerder")
    create_impl = AsyncMock(return_value=SimpleNamespace(id="source:x"))
    monkeypatch.setattr(sources_upload, "_create_source_impl", create_impl)
    source_svc = AsyncMock()
    source_svc.get = AsyncMock(return_value=SimpleNamespace(command="cmd:job1"))

    client = _make_client(source_svc)
    resp = client.post(
        "/api/v1/agents/process-url",
        json={"url": "https://example.com", "notebook_id": "notebook:n"},
        headers={"Authorization": "Bearer ak_w"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "cmd:job1"
    assert body["source_id"] == "source:x"
    assert body["status"] == "queued"
    # The façade fed the shipped chain a link SourceCreate with the url + notebook.
    sc = create_impl.await_args.args[0]
    assert sc.type == "link" and sc.url == "https://example.com"
    assert sc.notebook_id == "notebook:n"


def test_process_url_pijplijn_key_is_403(monkeypatch):
    _auth(monkeypatch, "pijplijn")  # beheerder-gated
    client = _make_client()
    resp = client.post(
        "/api/v1/agents/process-url",
        json={"url": "https://x", "notebook_id": "notebook:n"},
        headers={"Authorization": "Bearer ak_r"},
    )
    assert resp.status_code == 403


def test_process_url_rejects_ssrf_target(monkeypatch):
    """A beheerder key cannot make the server fetch a loopback/metadata target."""
    _auth(monkeypatch, "beheerder")
    create_impl = AsyncMock(return_value=SimpleNamespace(id="source:x"))
    monkeypatch.setattr(sources_upload, "_create_source_impl", create_impl)

    client = _make_client()
    for bad in (
        "http://169.254.169.254/latest/meta-data/",  # canonical metadata IP
        "http://127.0.0.1:9000/admin",  # canonical loopback
        "http://localhost/internal",  # loopback hostname
        "file:///etc/passwd",  # non-http(s) scheme
        # Non-canonical IP encodings that the round-1 guard let through —
        # glibc resolves each to the SAME blocked address (127.0.0.1 / metadata).
        "http://2130706433/",  # decimal loopback
        "http://0177.0.0.1/",  # octal loopback
        "http://0x7f.0.0.1/",  # hex loopback
        "http://127.0.0.1./admin",  # trailing-dot loopback
        "http://2852039166/latest/meta-data/",  # decimal cloud-metadata IP
        "http://127.1/",  # short-form loopback
        # Unicode homoglyph-dot / fullwidth encodings: the fetcher IDNA/NFKC-
        # collapses these to the same blocked address before connecting.
        "http://169。254。169。254/latest/meta-data/",  # ideographic dots
        "http://127｡0｡0｡1/",  # halfwidth ideographic dots
        "http://127．0．0．1/",  # fullwidth full stops
        "http://１２７．0．0．1/",  # fullwidth digits 127.0.0.1
    ):
        resp = client.post(
            "/api/v1/agents/process-url",
            json={"url": bad, "notebook_id": "notebook:n"},
            headers={"Authorization": "Bearer ak_w"},
        )
        assert resp.status_code == 422, bad
    # Unicode-16 "outlined digits" (U+1CCF0–U+1CCF9) → 127.0.0.1 under UTS-46.
    outlined_127 = "http://" + "".join(chr(0x1CCF0 + d) for d in (1, 2, 7)) + ".0.0.1/"
    resp = client.post(
        "/api/v1/agents/process-url",
        json={"url": outlined_127, "notebook_id": "notebook:n"},
        headers={"Authorization": "Bearer ak_w"},
    )
    assert resp.status_code == 422, outlined_127
    # The ingest chain was never entered for any blocked URL.
    create_impl.assert_not_awaited()


# -- jobs/{id} --------------------------------------------------------------


def test_job_status_passthrough(monkeypatch):
    _auth(monkeypatch, "beheerder")
    import app_main.services.command_service as cs

    monkeypatch.setattr(
        cs.CommandService,
        "get_command_status",
        AsyncMock(return_value={"job_id": "j1", "status": "running", "result": None}),
    )
    client = _make_client()
    resp = client.get("/api/v1/agents/jobs/j1", headers={"Authorization": "Bearer ak_r"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


def test_job_status_unknown_is_200(monkeypatch):
    _auth(monkeypatch, "beheerder")
    import app_main.services.command_service as cs

    monkeypatch.setattr(
        cs.CommandService,
        "get_command_status",
        AsyncMock(return_value={"job_id": "nope", "status": "unknown"}),
    )
    client = _make_client()
    resp = client.get("/api/v1/agents/jobs/nope", headers={"Authorization": "Bearer ak_r"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "unknown"


def test_job_status_non_beheerder_is_403_and_does_not_leak(monkeypatch):
    """F2-01: only `beheerder` reaches the agent routes; status never read."""
    _auth(monkeypatch, "pijplijn")
    import app_main.services.command_service as cs

    status_fn = AsyncMock(
        return_value={"job_id": "victim", "status": "running", "result": {"x": 1}}
    )
    monkeypatch.setattr(cs.CommandService, "get_command_status", status_fn)

    client = _make_client()
    resp = client.get("/api/v1/agents/jobs/victim", headers={"Authorization": "Bearer ak_p"})
    assert resp.status_code == 403
    status_fn.assert_not_awaited()


# -- audit-log --------------------------------------------------------------


def test_audit_log_non_beheerder_is_403(monkeypatch):
    _auth(monkeypatch, "pijplijn", agent_id="a1")
    import app_main.services.agents.audit_service as aud

    lister = AsyncMock(return_value=[])
    monkeypatch.setattr(aud.AgentAuditService, "list_for_agent", lister)

    client = _make_client()
    resp = client.get(
        "/api/v1/agents/audit-log?agent_id=other", headers={"Authorization": "Bearer ak_r"}
    )
    assert resp.status_code == 403
    lister.assert_not_awaited()


def test_audit_log_beheerder_can_read_other_agent(monkeypatch):
    _auth(monkeypatch, "beheerder", agent_id="admin1")
    import app_main.services.agents.audit_service as aud

    lister = AsyncMock(return_value=[])
    monkeypatch.setattr(aud.AgentAuditService, "list_for_agent", lister)

    client = _make_client()
    resp = client.get(
        "/api/v1/agents/audit-log?agent_id=other", headers={"Authorization": "Bearer ak_admin"}
    )
    assert resp.status_code == 200
    assert resp.json()["agent_id"] == "other"
