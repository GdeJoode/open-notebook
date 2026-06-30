"""Track PL.5 — the sync ingest path is decoupled from the shared worker.

Before PL.5 the sync upload path enqueued ``process_source`` onto the shared
single-worker queue and then polled its status for up to 300s. That both
blocked the request for the whole parse AND starved every other queued job
behind the one busy worker slot (a heavy sync ingest serialized all other
work).

PL.5 runs the SAME ``DOCUMENT_PARSE`` handler in-request via the registry, so
the heavy parse executes in the request's own coroutine and never occupies the
shared worker. These tests pin that contract:

* the sync path calls ``registry.execute(JobType.DOCUMENT_PARSE, ...)`` once;
* it does NOT submit ``process_source`` onto the shared queue;
* it does NOT poll ``get_command_status`` (no 300s shared-queue poll);
* the caller still gets a fully-populated ``SourceResponse`` (sync contract
  preserved);
* a handler failure surfaces as a 500 and cleans up the half-created source.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app_main.api.rate_limit import limiter
from app_main.api.routers.sources_upload import router
from app_main.dependencies import (
    get_notebook_service,
    get_source_service,
    get_transformation_service,
)
from app_main.services.notebook_service import NotebookService
from app_main.services.source_service import SourceService
from app_main.services.transformation_service import TransformationService
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.types.enums import JobType


@pytest.fixture(autouse=True)
def _generous_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_RPM", "100000")


def _make_client(source_svc: AsyncMock) -> TestClient:
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router, prefix="/sources")
    app.dependency_overrides[get_source_service] = lambda: source_svc
    app.dependency_overrides[get_notebook_service] = lambda: AsyncMock(
        spec=NotebookService
    )
    app.dependency_overrides[get_transformation_service] = lambda: AsyncMock(
        spec=TransformationService
    )
    return TestClient(app)


def _processed_stub() -> MagicMock:
    src = MagicMock()
    src.id = "source:abc"
    src.title = "hello world"
    src.topics = []
    src.asset = None
    src.full_text = "hello world"
    src.created = "2026-06-21T00:00:00Z"
    src.updated = "2026-06-21T00:00:00Z"
    src.private = False
    return src


def _source_svc() -> AsyncMock:
    svc = AsyncMock(spec=SourceService)
    created = MagicMock()
    created.id = "source:abc"
    created.title = "Processing..."
    created.topics = []
    created.created = "2026-06-21T00:00:00Z"
    created.updated = "2026-06-21T00:00:00Z"
    svc.create.return_value = created
    svc.get.return_value = _processed_stub()
    svc.get_embedding_count.return_value = 3
    return svc


def test_sync_path_runs_handler_in_request_not_on_shared_queue():
    """The sync path executes DOCUMENT_PARSE in-request and never submits to /
    polls the shared queue."""
    source_svc = _source_svc()
    client = _make_client(source_svc)

    registry = MagicMock()
    registry.execute = AsyncMock(return_value={"success": True, "chunk_count": 3})

    submit = AsyncMock(return_value="command:xyz")
    status = AsyncMock(return_value={"status": "completed"})

    with patch(
        "app_main.services.command_service.get_registry", return_value=registry
    ), patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=submit,
    ), patch(
        "app_main.services.command_service.CommandService.get_command_status",
        new=status,
    ), patch(
        "surrealdb_service.connection.execute_query",
        new=AsyncMock(return_value=None),
    ):
        resp = client.post(
            "/sources/",
            data={
                "type": "text",
                "content": "hello world",
                "async_processing": "false",  # SYNC path
            },
        )

    assert resp.status_code == 200, resp.text

    # The parse ran in-request via the registry, exactly once, as DOCUMENT_PARSE.
    registry.execute.assert_awaited_once()
    assert registry.execute.await_args.args[0] == JobType.DOCUMENT_PARSE
    handler_payload = registry.execute.await_args.args[1]
    assert handler_payload["source_id"] == "source:abc"
    assert handler_payload["command_name"] == "process_source"

    # It did NOT submit process_source onto the shared queue, and did NOT poll.
    for call in submit.await_args_list:
        assert call.args[1] != "process_source"
    status.assert_not_awaited()

    # The sync contract is preserved: a fully-populated SourceResponse.
    body = resp.json()
    assert body["id"] == "source:abc"
    assert body["embedded_chunks"] == 3
    assert body["embedded"] is True


def test_sync_path_handler_failure_returns_500_and_cleans_up():
    """A parse failure in-request → 500 + the half-created source is deleted."""
    source_svc = _source_svc()
    client = _make_client(source_svc)

    registry = MagicMock()
    registry.execute = AsyncMock(side_effect=RuntimeError("parse boom"))

    with patch(
        "app_main.services.command_service.get_registry", return_value=registry
    ), patch(
        "surrealdb_service.connection.execute_query",
        new=AsyncMock(return_value=None),
    ):
        resp = client.post(
            "/sources/",
            data={
                "type": "text",
                "content": "hello world",
                "async_processing": "false",
            },
        )

    assert resp.status_code == 500, resp.text
    assert "parse boom" in resp.text
    # The half-created source was cleaned up.
    source_svc.delete.assert_awaited_with("source:abc")
