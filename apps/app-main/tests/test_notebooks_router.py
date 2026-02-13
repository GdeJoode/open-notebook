"""Tests for the notebooks router."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.notebooks import router
from app_main.dependencies import get_notebook_service, get_source_service
from app_main.services.notebook_service import NotebookService
from app_main.services.source_service import SourceService
from tests.conftest import make_notebook, make_source


def _make_app(notebook_svc, source_svc=None):
    """Wire a test app with FastAPI dependency_overrides."""
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    if source_svc:
        app.dependency_overrides[get_source_service] = lambda: source_svc
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
