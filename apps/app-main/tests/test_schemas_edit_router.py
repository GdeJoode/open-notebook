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


class TestEditResponsesReportTheEffectiveSchema:
    """The POST half of the same contract the GET carries.

    The first attempt asserted this on the GET only, and the review measured
    that reverting the POST filter left all fifty router tests green — so the
    duplicate row and the `ext:${type_name}` key collision were reintroducible on
    the path a client refreshes from after an edit.
    """

    @staticmethod
    def _response_for(accepted):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()
        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.reparent_type.return_value = _make_schema(accepted=accepted)
        client = _make_app(notebook_svc, edit_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reparent",
            json={"type_names": ["ScholarlyArticle"], "new_parent": "Thesis"},
        )
        assert resp.status_code == 200
        return resp.json()

    @staticmethod
    def _move(type_name, new_parent):
        return {
            "reparent_id": f"reparent::{type_name}->{new_parent}",
            "op": "reparent",
            "type_name": type_name,
            "new_parent": new_parent,
            "parent_type": new_parent,
        }

    def test_the_declared_parent_is_the_baseline(self):
        body = self._response_for([])
        node = next(
            t for t in body["base_ontology_types"] if t["name"] == "ScholarlyArticle"
        )
        assert node["parent_type"] == "Article"

    def test_a_reparent_is_not_returned_as_an_extension(self):
        body = self._response_for([self._move("ScholarlyArticle", "Thesis")])
        assert body["accepted_extensions"] == []

    def test_the_row_reports_the_new_parent(self):
        body = self._response_for([self._move("ScholarlyArticle", "Thesis")])
        node = next(
            t for t in body["base_ontology_types"] if t["name"] == "ScholarlyArticle"
        )
        assert node["parent_type"] == "Thesis"

    def test_two_moves_of_one_type_cannot_collide(self):
        body = self._response_for(
            [
                self._move("ScholarlyArticle", "Thesis"),
                self._move("ScholarlyArticle", "Periodical"),
            ]
        )
        assert body["accepted_extensions"] == []
        node = next(
            t for t in body["base_ontology_types"] if t["name"] == "ScholarlyArticle"
        )
        assert node["parent_type"] == "Periodical"


class TestAcceptSurfacesThePlacement:
    """N.4d.3 — accepting an extension shows the curator where it landed.

    A report, never an applied change: the endpoint re-parents nothing, and the
    curator applies a move by posting it to `/schema/reparent`.
    """

    @staticmethod
    def _app(edit_service, placement_service):
        from app_main.dependencies import get_type_placement_service

        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()
        client = _make_app(notebook_svc, edit_service)
        client.app.dependency_overrides[get_type_placement_service] = (
            lambda: placement_service
        )
        return client

    def test_the_response_carries_the_placement(self):
        from app_main.services.type_placement_service import DECIDED, PlacementReport

        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.accept_extension.return_value = _make_schema(
            accepted=[
                {"extension_id": "e1", "type_name": "Tranche", "parent_type": "Deal"}
            ]
        )
        placement_service = AsyncMock()
        placement_service.placement_for.return_value = PlacementReport(
            type_name="Tranche",
            verdict="PLACED",
            reason_code="declared_parent_resolves",
            evidence="the declared parent resolves",
            parent="Deal",
            candidates=("RegioDeal", "Woondeal"),
            selected=("RegioDeal",),
            judge_status=DECIDED,
            vocabulary=("deals",),
        )

        client = self._app(edit_service, placement_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/extensions/Tranche/accept"
        )

        assert resp.status_code == 200
        placement = resp.json()["placement"]
        assert placement["verdict"] == "PLACED"
        assert placement["candidates"] == ["RegioDeal", "Woondeal"]
        assert placement["selected"] == ["RegioDeal"]
        assert placement["judged"] is True
        assert placement["judge_status"] == "decided"
        # The declared parent from the accepted entry is what gets validated —
        # the service validates a claim, it does not invent one.
        assert placement_service.placement_for.await_args.args[1:3] == (
            "Tranche",
            "Deal",
        )

    def test_a_placement_failure_does_not_fail_the_accept(self):
        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.accept_extension.return_value = _make_schema(
            accepted=[
                {"extension_id": "e1", "type_name": "Tranche", "parent_type": "Deal"}
            ]
        )
        placement_service = AsyncMock()
        placement_service.placement_for.side_effect = RuntimeError("no ontology dir")

        client = self._app(edit_service, placement_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/extensions/Tranche/accept"
        )

        assert resp.status_code == 200
        assert resp.json()["placement"] is None

    def test_the_other_edit_endpoints_carry_no_placement(self):
        """The field is populated by accept and nowhere else, so a client cannot
        read a stale verdict off an unrelated edit.
        """
        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.reparent_type.return_value = _make_schema()
        placement_service = AsyncMock()

        client = self._app(edit_service, placement_service)
        resp = client.post(
            f"/api/notebooks/{NOTEBOOK_ID}/schema/reparent",
            json={"type_names": ["Article"], "new_parent": "Publication"},
        )
        assert resp.status_code == 200
        assert resp.json()["placement"] is None
        placement_service.placement_for.assert_not_awaited()


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
