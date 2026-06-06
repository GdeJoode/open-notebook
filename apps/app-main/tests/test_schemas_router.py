"""Tests for the schemas router (Phase B.2b).

Covers ``GET /api/notebooks/{id}/schema.ttl``:

* happy path with two accepted extensions → 2 extra ``owl:Class`` declarations
* empty-extensions path (no notebook_schema row) → base ontology unchanged
* 404 for unknown notebook id
* response headers: ``Content-Type: text/turtle`` and an attachment filename
* roundtrip: returned body parses back via ``rdflib.Graph().parse(format="turtle")``

The tests use FastAPI's dependency_overrides plus AsyncMock for the
notebook service and the notebook_schema repository — no DB needed.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from rdflib import Graph
from rdflib.namespace import OWL, RDF

from app_main.api.routers.schemas import (
    get_notebook_schema_repo,
    router,
)
from app_main.dependencies import get_notebook_service
from app_main.services.notebook_service import NotebookService
from shared.models import Notebook
from shared.models.notebook_schema import NotebookSchema
from surrealdb_service.repositories.notebook_schema import (
    NotebookSchemaRepository,
)


_NOW = datetime(2026, 1, 1, 12, 0, 0)


def _make_notebook(notebook_id: str = "notebook:test1") -> Notebook:
    return Notebook(
        id=notebook_id,
        name="Test Notebook",
        description="",
        archived=False,
        created=_NOW,
        updated=_NOW,
    )


def _make_app(notebook_svc: AsyncMock, schema_repo: AsyncMock) -> TestClient:
    """Wire a FastAPI test client with the two dependencies overridden."""
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_notebook_schema_repo] = lambda: schema_repo
    return TestClient(app)


# ---------------------------------------------------------------------------
# Happy path — notebook with accepted extensions
# ---------------------------------------------------------------------------


class TestExportWithAcceptedExtensions:
    def test_returns_ttl_with_extra_owl_classes(self):
        """Two accepted extensions → 2 additional ``owl:Class`` triples."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:test1",
            base_ontology="scholarly",
            accepted_extensions=[
                {
                    "extension_id": "ext-1",
                    "type_name": "PreprintServer",
                    "parent_type": "Organization",
                    "properties": [
                        {"name": "preprintUrl", "data_type": "url"},
                    ],
                },
                {
                    "extension_id": "ext-2",
                    "type_name": "Cohort",
                    "description": "A study cohort.",
                },
            ],
            pending_extensions=[],
        )

        client = _make_app(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:test1/schema.ttl")

        assert resp.status_code == 200
        body = resp.text
        assert body.lstrip().startswith("@prefix")

        # Roundtrip — parses cleanly
        with_ext = Graph()
        with_ext.parse(data=body, format="turtle")

        # Baseline: load scholarly directly with NO extensions and compare
        # the count of owl:Class subjects.
        from ontology_manager.rdf_owl_shacl import load_yaml_ontology
        from app_main.api.routers.schemas import _ontologies_dir

        baseline = load_yaml_ontology(_ontologies_dir() / "scholarly.yaml")
        base_classes = set(baseline.subjects(RDF.type, OWL.Class))
        merged_classes = set(with_ext.subjects(RDF.type, OWL.Class))

        # The two new classes show up as additional subjects.
        new_classes = merged_classes - base_classes
        assert len(new_classes) == 2

        labels = {str(s).rsplit("/", 1)[-1] for s in new_classes}
        assert labels == {"PreprintServer", "Cohort"}


# ---------------------------------------------------------------------------
# 404 path — unknown notebook
# ---------------------------------------------------------------------------


class TestUnknownNotebook:
    def test_returns_404_for_unknown_notebook(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        # Should not be called — guarded by the notebook lookup.

        client = _make_app(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:missing/schema.ttl")

        assert resp.status_code == 404
        schema_repo.get_by_notebook.assert_not_called()


# ---------------------------------------------------------------------------
# Empty path — notebook exists but no notebook_schema row
# ---------------------------------------------------------------------------


class TestNotebookWithoutSchemaRow:
    def test_returns_base_ontology_when_schema_row_absent(self):
        """B.1c hasn't run yet → return the default base ontology."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = None

        client = _make_app(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:test1/schema.ttl")

        assert resp.status_code == 200
        body = resp.text
        assert body.lstrip().startswith("@prefix")

        # The output equals the default base ontology with no extensions
        # added. Compare the class set to the scholarly baseline.
        from ontology_manager.rdf_owl_shacl import load_yaml_ontology
        from app_main.api.routers.schemas import _ontologies_dir

        returned = Graph()
        returned.parse(data=body, format="turtle")
        baseline = load_yaml_ontology(_ontologies_dir() / "scholarly.yaml")

        assert set(returned.subjects(RDF.type, OWL.Class)) == set(
            baseline.subjects(RDF.type, OWL.Class)
        )


# ---------------------------------------------------------------------------
# Response-header contract
# ---------------------------------------------------------------------------


class TestResponseHeaders:
    def test_content_type_is_turtle(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()
        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = None

        client = _make_app(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:test1/schema.ttl")

        # Starlette appends ``; charset=utf-8`` to text/* media types —
        # assert the prefix rather than equality.
        assert resp.headers["content-type"].startswith("text/turtle")

    def test_content_disposition_is_attachment_with_filename(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()
        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = None

        client = _make_app(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:test1/schema.ttl")

        cd = resp.headers["content-disposition"]
        assert cd.startswith("attachment;")
        # Colon in record id is sanitised to underscore so older clients
        # don't choke on the save dialog.
        assert 'filename="notebook_test1.ttl"' in cd


# ---------------------------------------------------------------------------
# Roundtrip sanity — the body is well-formed Turtle
# ---------------------------------------------------------------------------


class TestRoundtripParse:
    def test_body_parses_via_rdflib(self):
        """rdflib parsing succeeds — proxy for Protégé import compatibility."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:test1",
            base_ontology="scholarly",
            accepted_extensions=[
                {"extension_id": "ext-x", "type_name": "DatasetVersion"}
            ],
        )

        client = _make_app(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:test1/schema.ttl")

        assert resp.status_code == 200
        # If this raises, the body is malformed.
        g = Graph()
        g.parse(data=resp.text, format="turtle")
        # Triples > 0 — sanity that we got real content not an empty doc.
        assert len(g) > 0
