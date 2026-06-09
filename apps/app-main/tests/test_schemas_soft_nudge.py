"""Tests for the Phase B.3c soft-nudge + pause-toggle endpoints.

Three endpoints land in the schemas router:

* ``POST /api/notebooks/{id}/schema/review_required`` — toggles the
  per-notebook review_required flag.
* ``POST /api/notebooks/{id}/schema/dismiss_nudge`` — sets
  ``soft_nudge_dismissed=true``.
* ``POST /api/notebooks/{id}/extraction/resume`` — re-queues
  ``PAUSED_FOR_REVIEW`` jobs for sources in the notebook and (when the
  review-gate predicate would still fire) appends a sentinel entry to
  ``accepted_extensions``.

Plus the new notebook_events router:

* ``GET  /api/notebooks/{id}/events`` — list with optional type filter
  and ``unread=true``.
* ``POST /api/notebooks/{id}/events/{event_id}/mark_read`` — flip
  ``read_at``.

The tests use ``FastAPI.dependency_overrides`` + ``AsyncMock`` so no
real DB / job-queue is required.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.schemas import router as schemas_router
from app_main.api.routers.notebook_events import router as events_router
from app_main.dependencies import (
    get_notebook_schema_repo,
    get_notebook_service,
    get_source_repo,
)
from app_main.services.notebook_service import NotebookService
from shared.models import Notebook
from shared.models.notebook_schema import NotebookSchema
from surrealdb_service.repositories.notebook_schema import (
    NotebookSchemaRepository,
)
from surrealdb_service.repositories.source import SourceRepository


_NOW = datetime(2026, 6, 1, 12, 0, 0)


def _make_notebook(notebook_id: str = "notebook:b3c") -> Notebook:
    return Notebook(
        id=notebook_id,
        name="B.3c test notebook",
        description="",
        archived=False,
        created=_NOW,
        updated=_NOW,
    )


def _make_app(
    notebook_svc: AsyncMock,
    schema_repo: AsyncMock,
    source_repo: AsyncMock | None = None,
    include_events: bool = False,
) -> TestClient:
    app = FastAPI()
    app.include_router(schemas_router, prefix="/api")
    if include_events:
        app.include_router(events_router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_notebook_schema_repo] = lambda: schema_repo
    if source_repo is not None:
        app.dependency_overrides[get_source_repo] = lambda: source_repo
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /schema/review_required
# ---------------------------------------------------------------------------


class TestReviewRequiredEndpoint:
    def test_sets_review_required_true_on_existing_row(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        existing_schema = NotebookSchema(
            notebook="notebook:b3c",
            base_ontology="scholarly",
            review_required=False,
        )
        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = existing_schema
        schema_repo.upsert.return_value = "notebook_schema:abc"

        client = _make_app(notebook_svc, schema_repo)
        resp = client.post(
            "/api/notebooks/notebook:b3c/schema/review_required",
            json={"enabled": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["review_required"] is True
        assert body["notebook_id"] == "notebook:b3c"

        # The repo upsert should be called with review_required=True.
        schema_repo.upsert.assert_awaited_once()
        sent = schema_repo.upsert.await_args.args[0]
        assert sent.review_required is True

    def test_creates_schema_row_when_absent(self):
        """No existing schema row → endpoint materialises a default one."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = None
        schema_repo.upsert.return_value = "notebook_schema:new"

        client = _make_app(notebook_svc, schema_repo)
        resp = client.post(
            "/api/notebooks/notebook:b3c/schema/review_required",
            json={"enabled": True},
        )
        assert resp.status_code == 200
        assert resp.json()["review_required"] is True
        schema_repo.upsert.assert_awaited_once()

    def test_disable_persists_false(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:b3c",
            base_ontology="scholarly",
            review_required=True,
        )
        schema_repo.upsert.return_value = "notebook_schema:abc"

        client = _make_app(notebook_svc, schema_repo)
        resp = client.post(
            "/api/notebooks/notebook:b3c/schema/review_required",
            json={"enabled": False},
        )
        assert resp.status_code == 200
        assert resp.json()["review_required"] is False

    def test_404_for_unknown_notebook(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None
        schema_repo = AsyncMock(spec=NotebookSchemaRepository)

        client = _make_app(notebook_svc, schema_repo)
        resp = client.post(
            "/api/notebooks/notebook:missing/schema/review_required",
            json={"enabled": True},
        )
        assert resp.status_code == 404
        schema_repo.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# POST /schema/dismiss_nudge
# ---------------------------------------------------------------------------


class TestDismissNudgeEndpoint:
    def test_sets_soft_nudge_dismissed_true(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:b3c",
            base_ontology="scholarly",
            soft_nudge_dismissed=False,
        )
        schema_repo.upsert.return_value = "notebook_schema:abc"

        client = _make_app(notebook_svc, schema_repo)
        resp = client.post("/api/notebooks/notebook:b3c/schema/dismiss_nudge")
        assert resp.status_code == 200
        assert resp.json()["soft_nudge_dismissed"] is True
        sent = schema_repo.upsert.await_args.args[0]
        assert sent.soft_nudge_dismissed is True

    def test_idempotent_when_already_dismissed(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:b3c",
            base_ontology="scholarly",
            soft_nudge_dismissed=True,
        )
        schema_repo.upsert.return_value = "notebook_schema:abc"

        client = _make_app(notebook_svc, schema_repo)
        resp = client.post("/api/notebooks/notebook:b3c/schema/dismiss_nudge")
        assert resp.status_code == 200
        # Still upserts to bump last_modified_at.
        schema_repo.upsert.assert_awaited_once()

    def test_404_for_unknown_notebook(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None
        schema_repo = AsyncMock(spec=NotebookSchemaRepository)

        client = _make_app(notebook_svc, schema_repo)
        resp = client.post(
            "/api/notebooks/notebook:missing/schema/dismiss_nudge"
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /extraction/resume
# ---------------------------------------------------------------------------


class TestResumeExtractionEndpoint:
    """The resume flow has two sub-effects: sentinel append + job re-queue.

    We exercise the sentinel path against a mocked schema repo and patch
    the job-queue repository import inside the router so we don't need
    a real SurrealDB.
    """

    def test_appends_sentinel_when_gate_would_fire(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:b3c",
            base_ontology="scholarly",
            review_required=True,
            accepted_extensions=[],
        )
        schema_repo.upsert.return_value = "notebook_schema:abc"

        source_repo = AsyncMock(spec=SourceRepository)
        source_repo.list_with_metadata.return_value = []

        client = _make_app(notebook_svc, schema_repo, source_repo)
        resp = client.post("/api/notebooks/notebook:b3c/extraction/resume")
        assert resp.status_code == 200
        body = resp.json()
        assert body["sentinel_added"] is True
        assert body["resumed_count"] == 0

        # The upsert payload carries the new sentinel.
        sent = schema_repo.upsert.await_args.args[0]
        assert len(sent.accepted_extensions) == 1
        sentinel = sent.accepted_extensions[0]
        assert sentinel["type_name"] == "_resumed_without_extensions"
        assert sentinel["is_resume_sentinel"] is True
        assert "created_at" in sentinel

    def test_no_sentinel_when_gate_would_not_fire(self):
        """When ``review_required=False`` the gate isn't blocking."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:b3c",
            base_ontology="scholarly",
            review_required=False,
            accepted_extensions=[],
        )

        source_repo = AsyncMock(spec=SourceRepository)
        source_repo.list_with_metadata.return_value = []

        client = _make_app(notebook_svc, schema_repo, source_repo)
        resp = client.post("/api/notebooks/notebook:b3c/extraction/resume")
        assert resp.status_code == 200
        assert resp.json()["sentinel_added"] is False
        schema_repo.upsert.assert_not_called()

    def test_no_sentinel_when_accepted_extensions_non_empty(self):
        """Gate already lifted by a real accepted extension."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:b3c",
            base_ontology="scholarly",
            review_required=True,
            accepted_extensions=[
                {"extension_id": "e1", "type_name": "ClinicalTrial"}
            ],
        )

        source_repo = AsyncMock(spec=SourceRepository)
        source_repo.list_with_metadata.return_value = []

        client = _make_app(notebook_svc, schema_repo, source_repo)
        resp = client.post("/api/notebooks/notebook:b3c/extraction/resume")
        assert resp.status_code == 200
        assert resp.json()["sentinel_added"] is False
        schema_repo.upsert.assert_not_called()

    def test_requeues_paused_jobs(self):
        """Paused entity-extract jobs are flipped back to QUEUED."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:b3c",
            base_ontology="scholarly",
            review_required=False,
            accepted_extensions=[],
        )

        source_repo = AsyncMock(spec=SourceRepository)
        source_repo.list_with_metadata.return_value = [
            {"id": "source:s1"},
            {"id": "source:s2"},
        ]

        # Fake paused jobs for source:s1 and source:s2; one foreign job
        # for source:s3 (not in this notebook) should NOT be touched.
        paused_jobs = [
            SimpleNamespace(id="job:p1", source_id="source:s1"),
            SimpleNamespace(id="job:p2", source_id="source:s2"),
            SimpleNamespace(id="job:p3", source_id="source:s3"),
        ]
        job_repo = MagicMock()
        job_repo.list_jobs = AsyncMock(return_value=paused_jobs)
        job_repo.update_status = AsyncMock(return_value=None)

        with patch(
            "job_queue.repository.JobRepository",
            return_value=job_repo,
        ):
            client = _make_app(notebook_svc, schema_repo, source_repo)
            resp = client.post("/api/notebooks/notebook:b3c/extraction/resume")

        assert resp.status_code == 200
        body = resp.json()
        assert body["resumed_count"] == 2

        # Both paused jobs for sources in this notebook are re-queued.
        called_job_ids = {
            call.args[0] for call in job_repo.update_status.await_args_list
        }
        assert called_job_ids == {"job:p1", "job:p2"}

    def test_404_for_unknown_notebook(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None
        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        source_repo = AsyncMock(spec=SourceRepository)

        client = _make_app(notebook_svc, schema_repo, source_repo)
        resp = client.post("/api/notebooks/notebook:missing/extraction/resume")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Sentinel filtering — TTL + JSON endpoints
# ---------------------------------------------------------------------------


class TestSentinelFiltering:
    """Resume sentinels must not leak into the schema browse / TTL outputs."""

    def test_json_schema_response_hides_sentinel(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:b3c",
            base_ontology="scholarly",
            accepted_extensions=[
                {
                    "type_name": "_resumed_without_extensions",
                    "is_resume_sentinel": True,
                    "created_at": "2026-06-01T00:00:00Z",
                },
                {
                    "extension_id": "ext-1",
                    "type_name": "ClinicalTrial",
                },
            ],
        )

        client = _make_app(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:b3c/schema")
        assert resp.status_code == 200
        body = resp.json()
        accepted_names = [e["type_name"] for e in body["accepted_extensions"]]
        assert "ClinicalTrial" in accepted_names
        assert "_resumed_without_extensions" not in accepted_names

    def test_ttl_export_skips_sentinel(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:b3c",
            base_ontology="scholarly",
            accepted_extensions=[
                {
                    "type_name": "_resumed_without_extensions",
                    "is_resume_sentinel": True,
                },
            ],
        )

        client = _make_app(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:b3c/schema.ttl")
        assert resp.status_code == 200
        # The sentinel must NOT appear as a class label/URI in the TTL.
        assert "_resumed_without_extensions" not in resp.text


# ---------------------------------------------------------------------------
# notebook_events router
# ---------------------------------------------------------------------------


class TestNotebookEventsRouter:
    """The B.3c GET/POST surface; the underlying repo is on B.3b.

    These tests patch the import path so they exercise the router
    contract without depending on B.3b's repo existing on this branch.
    """

    def test_lists_events_when_repo_available(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        client = _make_app(
            notebook_svc, AsyncMock(spec=NotebookSchemaRepository),
            include_events=True,
        )

        # The router does ``from surrealdb_service.repositories.notebook_event
        # import NotebookEventRepository`` inside the handler — patch the
        # repo class at the module path it tries to import.
        fake_repo = MagicMock()
        fake_repo.list_for_notebook = AsyncMock(
            return_value=[
                SimpleNamespace(
                    id="notebook_event:e1",
                    event_type="extension_suggested",
                    message="Coverage 0.85 — 2 extensions proposed",
                    source_id="source:s1",
                    created_at=datetime(2026, 6, 1, 12, 0, 0),
                    read_at=None,
                ),
            ]
        )

        fake_module = MagicMock()
        fake_module.NotebookEventRepository = MagicMock(return_value=fake_repo)

        import sys
        sys.modules[
            "surrealdb_service.repositories.notebook_event"
        ] = fake_module
        try:
            resp = client.get(
                "/api/notebooks/notebook:b3c/events"
                "?type=extension_suggested,schema_mismatch&unread=true"
            )
        finally:
            del sys.modules["surrealdb_service.repositories.notebook_event"]

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["event_type"] == "extension_suggested"
        assert body[0]["source_id"] == "source:s1"

        # Repo got the parsed type list + unread flag.
        call = fake_repo.list_for_notebook.await_args
        assert call.args[0] == "notebook:b3c"
        assert call.kwargs["event_types"] == [
            "extension_suggested",
            "schema_mismatch",
        ]
        assert call.kwargs["unread_only"] is True

    def test_returns_empty_when_repo_unavailable(self):
        """B.3b not merged yet → graceful empty response."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        client = _make_app(
            notebook_svc, AsyncMock(spec=NotebookSchemaRepository),
            include_events=True,
        )

        # The router catches ImportError; nothing to mock since the real
        # module doesn't exist on this branch yet.
        resp = client.get("/api/notebooks/notebook:b3c/events?unread=true")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_mark_read_calls_repo(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        client = _make_app(
            notebook_svc, AsyncMock(spec=NotebookSchemaRepository),
            include_events=True,
        )

        fake_repo = MagicMock()
        fake_repo.mark_read = AsyncMock(return_value=True)
        fake_module = MagicMock()
        fake_module.NotebookEventRepository = MagicMock(return_value=fake_repo)

        import sys
        sys.modules[
            "surrealdb_service.repositories.notebook_event"
        ] = fake_module
        try:
            resp = client.post(
                "/api/notebooks/notebook:b3c/events/notebook_event:e1/mark_read"
            )
        finally:
            del sys.modules["surrealdb_service.repositories.notebook_event"]

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["event_id"] == "notebook_event:e1"
        fake_repo.mark_read.assert_awaited_once_with("notebook_event:e1")

    def test_mark_read_no_op_when_repo_unavailable(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        client = _make_app(
            notebook_svc, AsyncMock(spec=NotebookSchemaRepository),
            include_events=True,
        )

        resp = client.post(
            "/api/notebooks/notebook:b3c/events/notebook_event:e1/mark_read"
        )
        assert resp.status_code == 200
        # Best-effort: returns success so the client doesn't loop.
        assert resp.json()["success"] is True

    def test_events_404_for_unknown_notebook(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None

        client = _make_app(
            notebook_svc, AsyncMock(spec=NotebookSchemaRepository),
            include_events=True,
        )
        resp = client.get("/api/notebooks/notebook:missing/events")
        assert resp.status_code == 404
