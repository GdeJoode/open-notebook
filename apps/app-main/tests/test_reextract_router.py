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
    get_notebook_service,
    get_reextract_service,
    get_source_repo,
)
from app_main.services.notebook_service import NotebookService
from app_main.services.reextract_service import (
    ReextractResult,
    ReextractService,
)
from shared.models import Notebook


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
) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    if source_repo is not None:
        app.dependency_overrides[get_source_repo] = lambda: source_repo
    if reextract_svc is not None:
        app.dependency_overrides[get_reextract_service] = lambda: reextract_svc
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /schema/reextract_candidates
# ---------------------------------------------------------------------------


class TestReextractCandidates:
    def test_returns_all_notebook_sources(self):
        """V1 returns ALL sources, ignoring the schema_changed event payload's
        op/type_name. Future work narrows via entity → source index. When that
        lands, this test must evolve — assert that the candidate list is a
        SUBSET of all sources scoped to the entities of the modified type.
        Until then, the conservative "every source is potentially affected"
        invariant is what the banner relies on.
        """
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

    def test_filters_out_cross_notebook_source_ids(self):
        """Minor 1 (attempt 2): an out-of-notebook source id in the
        payload is silently dropped before reaching the service. The
        candidate list (via `list_with_metadata`) is the source of
        truth; anything not on it is treated as a bogus client payload.
        """
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        # Notebook owns only source:a — source:foreign is from a
        # different notebook (or simply doesn't exist).
        source_repo = AsyncMock()
        source_repo.list_with_metadata.return_value = [{"id": "source:a"}]

        reextract_svc = AsyncMock(spec=ReextractService)
        reextract_svc.enqueue_reextract_jobs.return_value = ReextractResult(
            jobs_enqueued=1,
            source_ids=["source:a"],
            enqueued_source_ids=["source:a"],
            skipped_source_ids=[],
        )

        client = _make_app(
            notebook_svc,
            source_repo=source_repo,
            reextract_svc=reextract_svc,
        )
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reextract",
            json={"source_ids": ["source:a", "source:foreign"]},
        )

        assert resp.status_code == 200
        # The service was called with only the in-notebook id; the
        # foreign id was dropped at the router layer before reaching
        # the service.
        _, kwargs = reextract_svc.enqueue_reextract_jobs.call_args
        assert kwargs.get("source_ids") == ["source:a"]


# ---------------------------------------------------------------------------
# /events and /events/{id}/mark_read coverage lives in
# `test_schemas_soft_nudge.py` (B.3c). Attempt 1 of B.3d carried a
# duplicate handler pair in this router; both were dropped during the
# rebase. The canonical surface is `notebook_events.py` on main.
# ---------------------------------------------------------------------------
