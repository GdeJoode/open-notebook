"""Router tests for the Phase B.3b edit-ops endpoints.

Six new endpoints on ``apps/app-main/src/app_main/api/routers/schemas.py``:

* POST ``/api/notebooks/{id}/schema/extensions/{type_name}/accept``
* POST ``/api/notebooks/{id}/schema/extensions/{type_name}/reject``
* POST ``/api/notebooks/{id}/schema/rename``
* POST ``/api/notebooks/{id}/schema/merge``
* POST ``/api/notebooks/{id}/schema/split``
* DELETE ``/api/notebooks/{id}/schema/types/{type_name}``

These tests mock the underlying SchemaEditService so we exercise:

* The notebook-existence guard (404 on unknown notebook).
* Translation of service exceptions to HTTP status codes
  (NotebookSchemaNotFoundError → 404, UnknownExtensionError → 404,
  ValueError → 422).
* The response shape — every endpoint returns an updated
  ``NotebookSchemaResponse`` including ``excluded_types``.

The end-to-end behaviour (idempotency, event emission, state diffing)
lives in ``test_schema_edit_service.py`` — keeping the layered
responsibilities separate.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.schemas import router
from app_main.dependencies import (
    get_notebook_schema_repo,
    get_notebook_service,
    get_schema_edit_service,
)
from app_main.services.notebook_service import NotebookService
from app_main.services.schema_edit_service import (
    NotebookSchemaNotFoundError,
    SchemaEditService,
    UnknownExtensionError,
)
from shared.models import Notebook, NotebookSchema
from surrealdb_service.repositories.notebook_schema import (
    NotebookSchemaRepository,
)


_NOW = datetime(2026, 6, 1, 12, 0, 0)
NOTEBOOK_ID = "notebook:edit-router"


def _make_notebook() -> Notebook:
    return Notebook(
        id=NOTEBOOK_ID,
        name="Edit Router Notebook",
        description="",
        archived=False,
        created=_NOW,
        updated=_NOW,
    )


def _make_schema(
    *,
    accepted=None,
    pending=None,
    excluded=None,
) -> NotebookSchema:
    return NotebookSchema(
        notebook=NOTEBOOK_ID,
        base_ontology="scholarly",
        accepted_extensions=list(accepted or []),
        pending_extensions=list(pending or []),
        excluded_types=list(excluded or []),
    )


def _make_app(
    notebook_svc: AsyncMock,
    edit_service: AsyncMock,
    schema_repo: AsyncMock | None = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_schema_edit_service] = lambda: edit_service
    if schema_repo is not None:
        # The Accept/Reject handlers don't actually use schema_repo
        # directly (they go through edit_service), but the response
        # serialiser does — we override defensively in case future
        # endpoints add direct reads.
        app.dependency_overrides[get_notebook_schema_repo] = lambda: schema_repo
    return TestClient(app)


# ---------------------------------------------------------------------------
# Accept / reject endpoints
# ---------------------------------------------------------------------------


class TestAcceptEndpoint:
    def test_returns_200_with_updated_schema(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.accept_extension.return_value = _make_schema(
            accepted=[{"extension_id": "ext-1", "type_name": "Cohort"}]
        )

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/extensions/Cohort/accept"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted_extensions"][0]["type_name"] == "Cohort"
        edit_service.accept_extension.assert_awaited_once_with(
            NOTEBOOK_ID, "Cohort"
        )

    def test_returns_404_when_notebook_missing(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None

        edit_service = AsyncMock(spec=SchemaEditService)

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/extensions/X/accept"
        )

        assert resp.status_code == 404
        edit_service.accept_extension.assert_not_awaited()

    def test_returns_404_when_extension_unknown(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.accept_extension.side_effect = UnknownExtensionError("nope")

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/extensions/Nope/accept"
        )

        assert resp.status_code == 404


class TestRejectEndpoint:
    def test_returns_200_with_updated_schema(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.reject_extension.return_value = _make_schema()

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/extensions/Cohort/reject"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["pending_extensions"] == []
        edit_service.reject_extension.assert_awaited_once_with(
            NOTEBOOK_ID, "Cohort"
        )


# ---------------------------------------------------------------------------
# Rename endpoint
# ---------------------------------------------------------------------------


class TestRenameEndpoint:
    def test_returns_200_with_rename_entry(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.rename_type.return_value = _make_schema(
            accepted=[
                {
                    "op": "rename",
                    "old_name": "Researcher",
                    "new_name": "ResearchFellow",
                    "type_name": "ResearchFellow",
                }
            ]
        )

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/rename",
            json={"old_name": "Researcher", "new_name": "ResearchFellow"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted_extensions"][0]["type_name"] == "ResearchFellow"
        edit_service.rename_type.assert_awaited_once_with(
            NOTEBOOK_ID, "Researcher", "ResearchFellow"
        )

    def test_returns_422_when_old_name_empty(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/rename",
            json={"old_name": "", "new_name": "Y"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Merge endpoint
# ---------------------------------------------------------------------------


class TestMergeEndpoint:
    def test_returns_200_with_merge_entry(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.merge_types.return_value = _make_schema(
            accepted=[
                {
                    "op": "merge",
                    "source_types": ["Author", "Editor"],
                    "merged_name": "Contributor",
                    "type_name": "Contributor",
                }
            ]
        )

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/merge",
            json={
                "type_names": ["Author", "Editor"],
                "merged_name": "Contributor",
            },
        )

        assert resp.status_code == 200
        edit_service.merge_types.assert_awaited_once_with(
            NOTEBOOK_ID, ["Author", "Editor"], "Contributor"
        )

    def test_returns_422_when_service_raises_valueerror(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.merge_types.side_effect = ValueError("need 2+")

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/merge",
            json={"type_names": ["A", "A"], "merged_name": "Z"},
        )
        # Pydantic itself wouldn't catch "duplicates" — the service raises
        # ValueError after dedupe + length check.
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Split endpoint
# ---------------------------------------------------------------------------


class TestSplitEndpoint:
    def test_returns_200_with_split_entry(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.split_type.return_value = _make_schema(
            accepted=[
                {
                    "op": "split",
                    "source_type": "Cohort",
                    "into": ["StudyCohort", "ControlCohort"],
                    "criterion": "by trial role",
                    "type_name": "Cohort",
                }
            ]
        )

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/split",
            json={
                "type_name": "Cohort",
                "into": ["StudyCohort", "ControlCohort"],
                "criterion": "by trial role",
            },
        )

        assert resp.status_code == 200
        edit_service.split_type.assert_awaited_once_with(
            NOTEBOOK_ID,
            "Cohort",
            ["StudyCohort", "ControlCohort"],
            "by trial role",
        )


# ---------------------------------------------------------------------------
# Delete endpoint
# ---------------------------------------------------------------------------


class TestDeleteEndpoint:
    def test_returns_200_with_excluded_types_populated(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.delete_type.return_value = _make_schema(
            excluded=["Methodology"]
        )

        client = _make_app(notebook_svc, edit_service)
        resp = client.delete(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/types/Methodology"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["excluded_types"] == ["Methodology"]
        edit_service.delete_type.assert_awaited_once_with(
            NOTEBOOK_ID, "Methodology"
        )

    def test_returns_404_when_schema_row_missing(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.delete_type.side_effect = NotebookSchemaNotFoundError(
            "no row"
        )

        client = _make_app(notebook_svc, edit_service)
        resp = client.delete(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/types/X"
        )

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Reparent endpoint (Track N.4d.3)
# ---------------------------------------------------------------------------


class TestReparentEndpoint:
    def test_returns_200_and_forwards_the_payload(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.reparent_type.return_value = _make_schema(
            accepted=[
                {
                    "reparent_id": "reparent::Article->Publication",
                    "op": "reparent",
                    "type_name": "Article",
                    "new_parent": "Publication",
                    "parent_type": "Publication",
                }
            ]
        )

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reparent",
            json={"type_names": ["Article", "Report"], "new_parent": "Publication"},
        )

        assert resp.status_code == 200
        edit_service.reparent_type.assert_awaited_once_with(
            NOTEBOOK_ID, ["Article", "Report"], "Publication"
        )

    def test_returns_422_when_the_service_rejects_the_move(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.reparent_type.side_effect = ValueError("under itself")

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reparent",
            json={"type_names": ["Article"], "new_parent": "Article"},
        )
        assert resp.status_code == 422

    def test_returns_422_when_no_type_is_named(self):
        """Caught by the request model before the service is reached."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()
        edit_service = AsyncMock(spec=SchemaEditService)

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reparent",
            json={"type_names": [], "new_parent": "Publication"},
        )
        assert resp.status_code == 422
        edit_service.reparent_type.assert_not_awaited()

    def test_returns_404_when_the_schema_row_is_missing(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.reparent_type.side_effect = NotebookSchemaNotFoundError("none")

        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reparent",
            json={"type_names": ["Article"], "new_parent": "Publication"},
        )
        assert resp.status_code == 404
