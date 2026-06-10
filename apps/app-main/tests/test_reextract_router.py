"""Router tests for the Phase B.3d re-extract endpoints.

Two new endpoints on ``apps/app-main/src/app_main/api/routers/schemas.py``:

* GET ``/api/notebooks/{id}/schema/reextract_candidates`` — affected
  source ids (V1: all notebook sources).
* POST ``/api/notebooks/{id}/schema/reextract`` — enqueue ENTITY_EXTRACT
  jobs for the requested source ids; idempotent dedup.

Also covers the minimal ``GET /api/notebooks/{id}/events`` surface the
ReextractPromptBanner polls — see the schemas.py docstring for why the
events endpoint lives in this phase.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.schemas import router
from app_main.dependencies import (
    get_notebook_event_repo,
    get_notebook_service,
    get_reextract_service,
    get_source_repo,
)
from app_main.services.notebook_service import NotebookService
from app_main.services.reextract_service import (
    ReextractResult,
    ReextractService,
)
from shared.models import Notebook, NotebookEvent


_NOW = datetime(2026, 6, 5, 12, 0, 0, tzinfo=timezone.utc)
NOTEBOOK_ID = "notebook:reextract-router"


def _make_notebook() -> Notebook:
    return Notebook(
        id=NOTEBOOK_ID,
        name="Reextract Router Notebook",
        description="",
        archived=False,
        created=_NOW,
        updated=_NOW,
    )


def _make_app(
    notebook_svc: AsyncMock,
    *,
    source_repo: AsyncMock | None = None,
    reextract_svc: AsyncMock | None = None,
    event_repo: AsyncMock | None = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    if source_repo is not None:
        app.dependency_overrides[get_source_repo] = lambda: source_repo
    if reextract_svc is not None:
        app.dependency_overrides[get_reextract_service] = lambda: reextract_svc
    if event_repo is not None:
        app.dependency_overrides[get_notebook_event_repo] = lambda: event_repo
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /schema/reextract_candidates
# ---------------------------------------------------------------------------


class TestReextractCandidates:
    def test_returns_all_notebook_sources(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        source_repo = AsyncMock()
        source_repo.list_with_metadata.return_value = [
            {"id": "source:a"},
            {"id": "source:b"},
            {"id": "source:c"},
        ]

        client = _make_app(notebook_svc, source_repo=source_repo)
        resp = client.get(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reextract_candidates"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["notebook_id"] == NOTEBOOK_ID
        assert body["source_ids"] == ["source:a", "source:b", "source:c"]
        assert body["count"] == 3
        # list_with_metadata is called with notebook_id filter.
        source_repo.list_with_metadata.assert_awaited_once()
        _, kwargs = source_repo.list_with_metadata.call_args
        assert kwargs.get("notebook_id") == NOTEBOOK_ID

    def test_returns_empty_list_when_no_sources(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        source_repo = AsyncMock()
        source_repo.list_with_metadata.return_value = []

        client = _make_app(notebook_svc, source_repo=source_repo)
        resp = client.get(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reextract_candidates"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["source_ids"] == []
        assert body["count"] == 0

    def test_returns_404_when_notebook_missing(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None

        client = _make_app(notebook_svc, source_repo=AsyncMock())
        resp = client.get(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reextract_candidates"
        )

        assert resp.status_code == 404

    def test_resilient_to_source_repo_error(self):
        """A transient list_with_metadata failure should degrade to an
        empty candidate list, not crash the banner."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        source_repo = AsyncMock()
        source_repo.list_with_metadata.side_effect = RuntimeError("db down")

        client = _make_app(notebook_svc, source_repo=source_repo)
        resp = client.get(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reextract_candidates"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["source_ids"] == []
        assert body["count"] == 0


# ---------------------------------------------------------------------------
# POST /schema/reextract
# ---------------------------------------------------------------------------


class TestReextractEnqueue:
    def test_enqueues_jobs_for_all_sources_happy_path(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        reextract_svc = AsyncMock(spec=ReextractService)
        reextract_svc.enqueue_reextract_jobs.return_value = ReextractResult(
            jobs_enqueued=2,
            source_ids=["source:a", "source:b"],
            enqueued_source_ids=["source:a", "source:b"],
            skipped_source_ids=[],
        )

        client = _make_app(notebook_svc, reextract_svc=reextract_svc)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reextract",
            json={"source_ids": ["source:a", "source:b"]},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["notebook_id"] == NOTEBOOK_ID
        assert body["jobs_enqueued"] == 2
        assert body["source_ids"] == ["source:a", "source:b"]
        assert body["enqueued_source_ids"] == ["source:a", "source:b"]
        assert body["skipped_source_ids"] == []
        reextract_svc.enqueue_reextract_jobs.assert_awaited_once()
        _, kwargs = reextract_svc.enqueue_reextract_jobs.call_args
        assert kwargs.get("notebook_id") == NOTEBOOK_ID
        assert kwargs.get("source_ids") == ["source:a", "source:b"]

    def test_idempotent_repeat_returns_zero_with_skipped(self):
        """Simulates the second click while the first batch is still running."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        reextract_svc = AsyncMock(spec=ReextractService)
        reextract_svc.enqueue_reextract_jobs.return_value = ReextractResult(
            jobs_enqueued=0,
            source_ids=["source:a"],
            enqueued_source_ids=[],
            skipped_source_ids=["source:a"],
        )

        client = _make_app(notebook_svc, reextract_svc=reextract_svc)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reextract",
            json={"source_ids": ["source:a"]},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["jobs_enqueued"] == 0
        assert body["enqueued_source_ids"] == []
        assert body["skipped_source_ids"] == ["source:a"]

    def test_empty_source_list_is_noop_200(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        reextract_svc = AsyncMock(spec=ReextractService)
        reextract_svc.enqueue_reextract_jobs.return_value = ReextractResult(
            jobs_enqueued=0,
            source_ids=[],
            enqueued_source_ids=[],
            skipped_source_ids=[],
        )

        client = _make_app(notebook_svc, reextract_svc=reextract_svc)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reextract",
            json={"source_ids": []},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["jobs_enqueued"] == 0

    def test_returns_404_when_notebook_missing(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None

        reextract_svc = AsyncMock(spec=ReextractService)
        client = _make_app(notebook_svc, reextract_svc=reextract_svc)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reextract",
            json={"source_ids": ["source:a"]},
        )

        assert resp.status_code == 404
        reextract_svc.enqueue_reextract_jobs.assert_not_awaited()


# ---------------------------------------------------------------------------
# GET /events (used by the banner to poll schema_changed events)
# ---------------------------------------------------------------------------


def _evt(event_id: str, event_type: str, payload: dict | None = None) -> NotebookEvent:
    return NotebookEvent(
        id=event_id,
        notebook=NOTEBOOK_ID,
        event_type=event_type,
        payload=payload or {},
        created_at=_NOW,
        read_at=None,
    )


class TestNotebookEventsEndpoint:
    def test_returns_unread_events_filtered_by_type(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        event_repo = AsyncMock()
        event_repo.list_unread.return_value = [
            _evt("notebook_event:e1", "schema_changed", {"op": "rename"}),
            _evt("notebook_event:e2", "schema_changed", {"op": "merge"}),
        ]

        client = _make_app(notebook_svc, event_repo=event_repo)
        resp = client.get(
            f"/api/notebooks/{NOTEBOOK_ID}/events?type=schema_changed"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert body[0]["event_type"] == "schema_changed"
        assert body[0]["id"] == "notebook_event:e1"
        assert body[0]["payload"]["op"] == "rename"
        event_repo.list_unread.assert_awaited_once_with(
            NOTEBOOK_ID, event_type="schema_changed"
        )

    def test_no_type_filter_returns_all_unread(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        event_repo = AsyncMock()
        event_repo.list_unread.return_value = []

        client = _make_app(notebook_svc, event_repo=event_repo)
        resp = client.get(f"/api/notebooks/{NOTEBOOK_ID}/events")

        assert resp.status_code == 200
        assert resp.json() == []
        event_repo.list_unread.assert_awaited_once_with(
            NOTEBOOK_ID, event_type=None
        )

    def test_unread_only_false_uses_list_by_notebook(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        event_repo = AsyncMock()
        event_repo.list_by_notebook.return_value = []

        client = _make_app(notebook_svc, event_repo=event_repo)
        resp = client.get(
            f"/api/notebooks/{NOTEBOOK_ID}/events?unread_only=false"
        )

        assert resp.status_code == 200
        event_repo.list_by_notebook.assert_awaited_once()
        event_repo.list_unread.assert_not_awaited()

    def test_returns_404_when_notebook_missing(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None

        client = _make_app(notebook_svc, event_repo=AsyncMock())
        resp = client.get(f"/api/notebooks/{NOTEBOOK_ID}/events")
        assert resp.status_code == 404

    def test_mark_read_calls_repo_with_reconstructed_id(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        event_repo = AsyncMock()
        event_repo.mark_read.return_value = True

        client = _make_app(notebook_svc, event_repo=event_repo)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/events/abc-123/mark_read"
        )

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        event_repo.mark_read.assert_awaited_once_with("notebook_event:abc-123")

    def test_mark_read_returns_404_when_event_missing(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        event_repo = AsyncMock()
        event_repo.mark_read.return_value = False

        client = _make_app(notebook_svc, event_repo=event_repo)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/events/missing/mark_read"
        )
        assert resp.status_code == 404
