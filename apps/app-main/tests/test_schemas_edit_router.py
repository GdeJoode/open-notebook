"""Router tests for the notebook's proposal queue (B.3b, OE.5).

Endpoints on ``apps/app-main/src/app_main/api/routers/schemas.py``:

* GET  ``/api/notebooks/{id}/schema/extensions/{type_name}/accept`` (target)
* POST ``/api/notebooks/{id}/schema/extensions/{type_name}/accept/preview``
* POST ``/api/notebooks/{id}/schema/extensions/{type_name}/accept``
* POST ``/api/notebooks/{id}/schema/extensions/{type_name}/reject``

OE.5 retired the per-notebook edit routes (rename, merge, split, re-parent,
delete): those are edits of the shared ontology, made in the ontology editor.

These tests mock the SchemaEditService, so they exercise the notebook guard, the
translation of service errors (the ontology editor's 409 / 422 pass through with
their detail) and the response shape. The behaviour against a real database is
in ``test_schema_edit_service.py``.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.schemas import router
from app_main.dependencies import (
    get_notebook_service,
    get_schema_edit_service,
)
from app_main.services.notebook_service import NotebookService
from app_main.services.ontology_edit_service import OntologyEditError
from app_main.services.schema_edit_service import (
    SchemaEditService,
    UnknownExtensionError,
)
from shared.models import Notebook, NotebookSchema


_NOW = datetime(2026, 6, 1, 12, 0, 0)
NOTEBOOK_ID = "notebook:edit-router"
BASE = f"/api/notebooks/{NOTEBOOK_ID}/schema"


def _make_notebook() -> Notebook:
    return Notebook(
        id=NOTEBOOK_ID,
        name="Edit Router Notebook",
        description="",
        archived=False,
        created=_NOW,
        updated=_NOW,
    )


def _make_schema(*, accepted=None, pending=None) -> NotebookSchema:
    return NotebookSchema(
        notebook=NOTEBOOK_ID,
        base_ontology="scholarly",
        accepted_extensions=list(accepted or []),
        pending_extensions=list(pending or []),
    )


def _client(edit_service: AsyncMock, notebook=True) -> TestClient:
    notebook_svc = AsyncMock(spec=NotebookService)
    notebook_svc.get.return_value = _make_notebook() if notebook else None
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_schema_edit_service] = lambda: edit_service
    return TestClient(app)


class TestAcceptEndpoint:
    def test_publishes_and_returns_the_queue_and_the_version(self):
        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.accept_extension.return_value = (
            _make_schema(accepted=[{"type_name": "Cohort", "published_to": "scholarly"}]),
            {"name": "scholarly", "version": "1.0+x", "warnings": [], "retype": {}},
        )
        resp = _client(edit_service).post(
            f"{BASE}/extensions/Cohort/accept",
            json={"ontology": "scholarly", "parent_type": "Group"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["published"]["version"] == "1.0+x"
        assert body["notebook_schema"]["pending_extensions"] == []
        assert "accepted_extensions" not in body["notebook_schema"]
        edit_service.accept_extension.assert_awaited_once_with(
            NOTEBOOK_ID, "Cohort", "scholarly", "Group"
        )

    def test_the_ontology_is_required(self):
        edit_service = AsyncMock(spec=SchemaEditService)
        resp = _client(edit_service).post(f"{BASE}/extensions/Cohort/accept", json={})
        assert resp.status_code == 422
        edit_service.accept_extension.assert_not_awaited()

    def test_a_blocking_finding_is_a_422_with_the_findings(self):
        edit_service = AsyncMock(spec=SchemaEditService)
        findings = [{"rule": "orphan", "severity": "block", "subject": "Cohort"}]
        edit_service.accept_extension.side_effect = OntologyEditError(
            422, {"message": "the new type has blocking findings", "findings": findings}
        )
        resp = _client(edit_service).post(
            f"{BASE}/extensions/Cohort/accept", json={"ontology": "scholarly"}
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["findings"] == findings

    def test_an_open_draft_is_a_409(self):
        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.accept_extension.side_effect = OntologyEditError(409, "open draft")
        resp = _client(edit_service).post(
            f"{BASE}/extensions/Cohort/accept", json={"ontology": "scholarly"}
        )
        assert resp.status_code == 409

    def test_returns_404_when_notebook_missing(self):
        edit_service = AsyncMock(spec=SchemaEditService)
        resp = _client(edit_service, notebook=False).post(
            f"{BASE}/extensions/X/accept", json={"ontology": "scholarly"}
        )
        assert resp.status_code == 404
        edit_service.accept_extension.assert_not_awaited()

    def test_returns_404_when_extension_unknown(self):
        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.accept_extension.side_effect = UnknownExtensionError("nope")
        resp = _client(edit_service).post(
            f"{BASE}/extensions/Nope/accept", json={"ontology": "scholarly"}
        )
        assert resp.status_code == 404


class TestAcceptTargetAndPreview:
    def test_the_target_names_the_default_ontology(self):
        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.accept_target.return_value = {
            "type_name": "Cohort", "parent_type": "Group",
            "default_ontology": "scholarly", "ontologies": ["base", "scholarly"],
        }
        resp = _client(edit_service).get(f"{BASE}/extensions/Cohort/accept")
        assert resp.status_code == 200
        assert resp.json()["default_ontology"] == "scholarly"

    def test_the_preview_is_passed_through(self):
        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.preview_accept.return_value = {
            "findings": [], "blocking": False, "draft_open": False,
        }
        resp = _client(edit_service).post(
            f"{BASE}/extensions/Cohort/accept/preview",
            json={"ontology": "scholarly", "parent_type": ""},
        )
        assert resp.status_code == 200
        assert resp.json()["blocking"] is False
        edit_service.preview_accept.assert_awaited_once_with(
            NOTEBOOK_ID, "Cohort", "scholarly", ""
        )


class TestRejectEndpoint:
    def test_returns_200_with_updated_schema(self):
        edit_service = AsyncMock(spec=SchemaEditService)
        edit_service.reject_extension.return_value = _make_schema()
        resp = _client(edit_service).post(f"{BASE}/extensions/Cohort/reject")
        assert resp.status_code == 200
        assert resp.json()["pending_extensions"] == []
        edit_service.reject_extension.assert_awaited_once_with(NOTEBOOK_ID, "Cohort")


@pytest.mark.parametrize("method, path, payload", [
    ("post", "/rename", {"old_name": "A", "new_name": "B"}),
    ("post", "/merge", {"type_names": ["A", "B"], "merged_name": "C"}),
    ("post", "/split", {"type_name": "A", "into": ["B", "C"], "criterion": "x"}),
    ("post", "/reparent", {"type_names": ["A"], "new_parent": "B"}),
    ("delete", "/types/A", None),
])
def test_the_per_notebook_edit_routes_are_gone(method, path, payload):
    """OE.5: nothing writes a per-notebook edit any more."""
    edit_service = AsyncMock(spec=SchemaEditService)
    client = _client(edit_service)
    call = getattr(client, method)
    resp = call(f"{BASE}{path}", json=payload) if payload else call(f"{BASE}{path}")
    assert resp.status_code in (404, 405)
