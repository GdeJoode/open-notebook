"""Tests for the notebooks router."""

from unittest.mock import AsyncMock

import pytest
from app_main.api.routers.notebooks import router
from app_main.dependencies import (
    get_notebook_merge_service,
    get_notebook_service,
    get_source_service,
)
from app_main.services.notebook_merge_service import (
    NotebookMergeReport,
    NotebookMergeService,
)
from app_main.services.notebook_service import NotebookService
from app_main.services.source_service import SourceService
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.conftest import make_notebook, make_source


def _make_app(notebook_svc, source_svc=None, merge_svc=None):
    """Wire a test app with FastAPI dependency_overrides."""
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    if source_svc:
        app.dependency_overrides[get_source_service] = lambda: source_svc
    if merge_svc:
        app.dependency_overrides[get_notebook_merge_service] = lambda: merge_svc
    return TestClient(app)


class TestListNotebooks:

    def test_list_notebooks(self):
        svc = AsyncMock(spec=NotebookService)
        svc.get_all_with_counts.return_value = [
            {
                "id": "notebook:1",
                "name": "NB1",
                "description": "Desc",
                "archived": False,
                "created": "2025-01-01",
                "updated": "2025-01-01",
                "source_count": 3,
                "note_count": 2,
            }
        ]
        client = _make_app(svc)

        resp = client.get("/api/notebooks")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "NB1"
        assert data[0]["source_count"] == 3


class TestCreateNotebook:

    def test_create_notebook(self):
        svc = AsyncMock(spec=NotebookService)
        nb = make_notebook()
        svc.create.return_value = nb
        svc.get_with_counts.return_value = {
            "id": "notebook:test1",
            "name": "Test Notebook",
            "description": "A test notebook",
            "archived": False,
            "created": str(nb.created),
            "updated": str(nb.updated),
            "source_count": 0,
            "note_count": 0,
        }
        client = _make_app(svc)

        resp = client.post("/api/notebooks", json={"name": "Test Notebook"})

        assert resp.status_code == 201
        assert resp.json()["name"] == "Test Notebook"


class TestGetNotebook:

    def test_get_notebook(self):
        svc = AsyncMock(spec=NotebookService)
        svc.get_with_counts.return_value = {
            "id": "notebook:1",
            "name": "NB",
            "description": "",
            "archived": False,
            "created": "2025-01-01",
            "updated": "2025-01-01",
            "source_count": 0,
            "note_count": 0,
        }
        client = _make_app(svc)

        resp = client.get("/api/notebooks/notebook:1")

        assert resp.status_code == 200

    def test_get_notebook_not_found(self):
        svc = AsyncMock(spec=NotebookService)
        svc.get_with_counts.return_value = None
        client = _make_app(svc)

        resp = client.get("/api/notebooks/notebook:missing")

        assert resp.status_code == 404


class TestNotebookPrivacyMode:
    """Track J.3: per-notebook privacy_mode persists through update + response."""

    def test_update_persists_privacy_mode(self):
        svc = AsyncMock(spec=NotebookService)
        svc.get.return_value = make_notebook()
        svc.get_with_counts.return_value = {
            "id": "notebook:1",
            "name": "NB",
            "description": "",
            "archived": False,
            "created": "2025-01-01",
            "updated": "2025-01-01",
            "source_count": 0,
            "note_count": 0,
            "privacy_mode": "private",
        }
        client = _make_app(svc)

        resp = client.put(
            "/api/notebooks/notebook:1", json={"privacy_mode": "private"}
        )

        assert resp.status_code == 200
        # The update payload forwarded privacy_mode to the service.
        svc.update.assert_awaited_once()
        update_arg = svc.update.await_args.args[1]
        assert update_arg["privacy_mode"] == "private"
        # And it surfaces on the response.
        assert resp.json()["privacy_mode"] == "private"

    def test_response_privacy_mode_defaults_none(self):
        svc = AsyncMock(spec=NotebookService)
        svc.get_with_counts.return_value = {
            "id": "notebook:1",
            "name": "NB",
            "description": "",
            "archived": False,
            "created": "2025-01-01",
            "updated": "2025-01-01",
            "source_count": 0,
            "note_count": 0,
        }
        client = _make_app(svc)

        resp = client.get("/api/notebooks/notebook:1")

        assert resp.status_code == 200
        assert resp.json()["privacy_mode"] is None


class TestDeleteNotebook:

    def test_delete_notebook(self):
        svc = AsyncMock(spec=NotebookService)
        svc.get.return_value = make_notebook()
        svc.delete.return_value = True
        client = _make_app(svc)

        resp = client.delete("/api/notebooks/notebook:1")

        assert resp.status_code == 204

    def test_delete_notebook_not_found(self):
        svc = AsyncMock(spec=NotebookService)
        svc.get.return_value = None
        client = _make_app(svc)

        resp = client.delete("/api/notebooks/notebook:missing")

        assert resp.status_code == 404


class TestAddSourceToNotebook:

    def test_add_source(self):
        nb_svc = AsyncMock(spec=NotebookService)
        nb_svc.get.return_value = make_notebook()
        nb_svc.add_source.return_value = True
        src_svc = AsyncMock(spec=SourceService)
        src_svc.get.return_value = make_source()
        client = _make_app(nb_svc, src_svc)

        resp = client.post("/api/notebooks/notebook:1/sources/source:1")

        assert resp.status_code == 204


class TestMergeNotebooks:
    """Tests for ``POST /api/notebooks/merge`` (B.6).

    Covers the API-boundary guards added in attempt 2:

    - B2 blocker: ``target_id`` overlap with ``source_ids`` → 422
    - Minor 3: archived notebooks rejected at the boundary → 422
    - Pre-existing 404 / 422 paths preserved
    """

    @staticmethod
    def _make_merge_svc(report: NotebookMergeReport | None = None):
        svc = AsyncMock(spec=NotebookMergeService)
        if report is None:
            report = NotebookMergeReport(
                entities_merged=3,
                relations_created=2,
                conflicts=[],
                source_notebook_ids=["notebook:src"],
                target_notebook_id="notebook:tgt",
                dry_run=False,
            )
        svc.merge_notebooks.return_value = report
        return svc

    def test_merge_happy_path(self):
        nb_svc = AsyncMock(spec=NotebookService)
        nb_svc.get.side_effect = [
            make_notebook(id="notebook:tgt"),  # target lookup
            make_notebook(id="notebook:src"),  # source lookup
        ]
        merge_svc = self._make_merge_svc()
        client = _make_app(nb_svc, merge_svc=merge_svc)

        resp = client.post(
            "/api/notebooks/merge",
            json={
                "source_ids": ["notebook:src"],
                "target_id": "notebook:tgt",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["entities_merged"] == 3
        assert body["relations_created"] == 2
        merge_svc.merge_notebooks.assert_awaited_once()

    def test_merge_rejects_overlapping_target_and_source(self):
        """B2: target_id in source_ids → 422 (silent self-merge guard)."""
        nb_svc = AsyncMock(spec=NotebookService)
        merge_svc = self._make_merge_svc()
        client = _make_app(nb_svc, merge_svc=merge_svc)

        resp = client.post(
            "/api/notebooks/merge",
            json={
                "source_ids": ["notebook:n1"],
                "target_id": "notebook:n1",
            },
        )

        assert resp.status_code == 422
        assert "target_id" in resp.json()["detail"].lower()
        # Service must not be touched when the guard fires.
        merge_svc.merge_notebooks.assert_not_called()

    def test_merge_rejects_target_in_mixed_source_list(self):
        """B2: target overlapping with one of multiple sources → still 422."""
        nb_svc = AsyncMock(spec=NotebookService)
        merge_svc = self._make_merge_svc()
        client = _make_app(nb_svc, merge_svc=merge_svc)

        resp = client.post(
            "/api/notebooks/merge",
            json={
                "source_ids": ["notebook:a", "notebook:tgt", "notebook:b"],
                "target_id": "notebook:tgt",
            },
        )

        assert resp.status_code == 422
        merge_svc.merge_notebooks.assert_not_called()

    def test_merge_rejects_empty_source_ids(self):
        """Empty source_ids → 422 (pydantic min_length=1 + explicit guard)."""
        nb_svc = AsyncMock(spec=NotebookService)
        merge_svc = self._make_merge_svc()
        client = _make_app(nb_svc, merge_svc=merge_svc)

        resp = client.post(
            "/api/notebooks/merge",
            json={
                "source_ids": [],
                "target_id": "notebook:tgt",
            },
        )

        assert resp.status_code == 422
        merge_svc.merge_notebooks.assert_not_called()

    def test_merge_404_when_target_missing(self):
        nb_svc = AsyncMock(spec=NotebookService)
        nb_svc.get.return_value = None
        merge_svc = self._make_merge_svc()
        client = _make_app(nb_svc, merge_svc=merge_svc)

        resp = client.post(
            "/api/notebooks/merge",
            json={
                "source_ids": ["notebook:src"],
                "target_id": "notebook:missing",
            },
        )

        assert resp.status_code == 404
        merge_svc.merge_notebooks.assert_not_called()

    def test_merge_404_when_source_missing(self):
        nb_svc = AsyncMock(spec=NotebookService)
        # target found, source not
        nb_svc.get.side_effect = [
            make_notebook(id="notebook:tgt"),
            None,
        ]
        merge_svc = self._make_merge_svc()
        client = _make_app(nb_svc, merge_svc=merge_svc)

        resp = client.post(
            "/api/notebooks/merge",
            json={
                "source_ids": ["notebook:missing"],
                "target_id": "notebook:tgt",
            },
        )

        assert resp.status_code == 404
        assert "source" in resp.json()["detail"].lower()
        merge_svc.merge_notebooks.assert_not_called()

    def test_merge_rejects_archived_target(self):
        """Minor 3: archived target → 422 (avoid resurrecting archived data)."""
        nb_svc = AsyncMock(spec=NotebookService)
        nb_svc.get.return_value = make_notebook(
            id="notebook:tgt", archived=True
        )
        merge_svc = self._make_merge_svc()
        client = _make_app(nb_svc, merge_svc=merge_svc)

        resp = client.post(
            "/api/notebooks/merge",
            json={
                "source_ids": ["notebook:src"],
                "target_id": "notebook:tgt",
            },
        )

        assert resp.status_code == 422
        assert "archived" in resp.json()["detail"].lower()
        merge_svc.merge_notebooks.assert_not_called()

    def test_merge_rejects_archived_source(self):
        nb_svc = AsyncMock(spec=NotebookService)
        nb_svc.get.side_effect = [
            make_notebook(id="notebook:tgt"),  # target ok
            make_notebook(id="notebook:src", archived=True),  # source archived
        ]
        merge_svc = self._make_merge_svc()
        client = _make_app(nb_svc, merge_svc=merge_svc)

        resp = client.post(
            "/api/notebooks/merge",
            json={
                "source_ids": ["notebook:src"],
                "target_id": "notebook:tgt",
            },
        )

        assert resp.status_code == 422
        assert "archived" in resp.json()["detail"].lower()
        merge_svc.merge_notebooks.assert_not_called()
