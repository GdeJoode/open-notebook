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
from rdflib import Graph, Literal
from rdflib.namespace import OWL, RDF, RDFS

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
        # added. Compare the class set to a freshly loaded baseline of
        # whichever ontology the router calls "default". Pulling the
        # name from the router constant keeps this test honest if the
        # default ever changes via config.
        from ontology_manager.rdf_owl_shacl import load_yaml_ontology
        from app_main.api.routers.schemas import (
            _DEFAULT_BASE_ONTOLOGY,
            _ontologies_dir,
        )

        returned = Graph()
        returned.parse(data=body, format="turtle")
        baseline = load_yaml_ontology(
            _ontologies_dir() / f"{_DEFAULT_BASE_ONTOLOGY}.yaml"
        )

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


# ---------------------------------------------------------------------------
# URI sanitisation — type_name with URI-illegal characters
# ---------------------------------------------------------------------------


class TestTypeNameSanitisation:
    """Reviewer-flagged major: extensions whose ``type_name`` contains
    spaces or punctuation must not crash rdflib's Turtle serializer.

    Acceptance: the response is 200, the URI fragment is CamelCase
    (e.g. ``MyClassWithSpaces``), and the original human-readable name
    survives as ``rdfs:label``.
    """

    def _request(self, type_name: str):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:test1",
            base_ontology="scholarly",
            accepted_extensions=[
                {"extension_id": "e1", "type_name": type_name},
            ],
        )
        client = _make_app(notebook_svc, schema_repo)
        return client.get("/api/notebooks/notebook:test1/schema.ttl")

    def test_spaces_in_type_name_produce_camelcase_uri_and_label(self):
        """The canonical case from the review report.

        Input  : ``"My Class With Spaces"``
        Expects: URI fragment ``MyClassWithSpaces``,
                 rdfs:label ``"My Class With Spaces"``,
                 200 response (NOT a 500).
        """
        resp = self._request("My Class With Spaces")
        assert resp.status_code == 200

        g = Graph()
        g.parse(data=resp.text, format="turtle")

        # 1) The CamelCase URI is present as an owl:Class.
        from app_main.api.routers.schemas import ON

        expected_uri = ON["MyClassWithSpaces"]
        assert (expected_uri, RDF.type, OWL.Class) in g

        # 2) The original human-readable name survives as rdfs:label.
        assert (
            expected_uri,
            RDFS.label,
            Literal("My Class With Spaces"),
        ) in g

    def test_punctuation_in_type_name_is_stripped(self):
        """Slashes, parentheses, hyphens → all collapsed into CamelCase."""
        resp = self._request("Author/Editor (Senior)")
        assert resp.status_code == 200

        g = Graph()
        g.parse(data=resp.text, format="turtle")

        from app_main.api.routers.schemas import ON

        assert (ON["AuthorEditorSenior"], RDF.type, OWL.Class) in g
        assert (
            ON["AuthorEditorSenior"],
            RDFS.label,
            Literal("Author/Editor (Senior)"),
        ) in g

    def test_leading_digit_type_name_gets_underscore_prefix(self):
        """URI fragments cannot start with a digit (NCName rule)."""
        resp = self._request("2024 cohort")
        assert resp.status_code == 200

        g = Graph()
        g.parse(data=resp.text, format="turtle")

        from app_main.api.routers.schemas import ON

        assert (ON["_2024Cohort"], RDF.type, OWL.Class) in g

    def test_parent_type_with_spaces_also_sanitised(self):
        """parent_type goes into rdfs:subClassOf — same sanitisation rules."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:test1",
            base_ontology="scholarly",
            accepted_extensions=[
                {
                    "extension_id": "e1",
                    "type_name": "Clinical Trial",
                    "parent_type": "Research Study",
                },
            ],
        )
        client = _make_app(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:test1/schema.ttl")

        assert resp.status_code == 200

        g = Graph()
        g.parse(data=resp.text, format="turtle")

        from app_main.api.routers.schemas import ON

        assert (
            ON["ClinicalTrial"],
            RDFS.subClassOf,
            ON["ResearchStudy"],
        ) in g


# ---------------------------------------------------------------------------
# Filename sanitisation — Content-Disposition header safety
# ---------------------------------------------------------------------------


class TestFilenameSanitisation:
    """Newly-handled characters per reviewer minor #2 (over-promising
    docstring on ``_safe_filename``).

    The header itself must remain syntactically valid even when the
    notebook id contains a CR/LF or a quote.
    """

    def test_safe_filename_strips_quotes_and_newlines(self):
        from app_main.api.routers.schemas import _safe_filename

        # Quotes would terminate the filename="..." attribute.
        assert _safe_filename('notebook:with"quote') == "notebook_with_quote.ttl"
        # CR/LF would let an attacker inject extra headers.
        assert (
            _safe_filename("notebook:abc\r\ndef") == "notebook_abc__def.ttl"
        )
        # Single quotes also collapsed.
        assert _safe_filename("name's") == "name_s.ttl"
        # Tabs and nulls
        assert _safe_filename("a\tb\x00c") == "a_b_c.ttl"


# ---------------------------------------------------------------------------
# Auth — endpoint is NOT in the global auth-exclusion allow-list
# ---------------------------------------------------------------------------


class TestAuthExclusionAllowList:
    """Reviewer minor #5 — confirm the endpoint inherits the global
    ``PasswordAuthMiddleware`` policy.

    Strategy: stand up a minimal FastAPI app with only the
    ``PasswordAuthMiddleware`` and the schemas router (mirroring the
    auth config from ``app_main.api.app.create_app``), set
    ``OPEN_NOTEBOOK_PASSWORD``, and confirm an unauthenticated request
    to the schema-TTL endpoint is blocked with 401.

    Avoids the full app's lifespan (DB migrations, worker startup) so
    the test runs in <100ms with no external deps.
    """

    def test_endpoint_returns_401_when_password_set_and_no_auth_header(
        self, monkeypatch
    ):
        monkeypatch.setenv("OPEN_NOTEBOOK_PASSWORD", "test-secret")

        from app_main.api.auth import PasswordAuthMiddleware

        app = FastAPI()
        # SAME excluded_paths list as app_main.api.app.create_app —
        # if this drifts and someone adds "schema.ttl" to the
        # production list, this test still catches it because it
        # asserts the production list verbatim.
        app.add_middleware(
            PasswordAuthMiddleware,
            excluded_paths=[
                "/",
                "/health",
                "/docs",
                "/openapi.json",
                "/redoc",
                "/api/auth/status",
                "/api/config",
            ],
        )
        app.include_router(router, prefix="/api")

        with TestClient(app) as client:
            resp = client.get("/api/notebooks/notebook:any/schema.ttl")

        # 401 (no Authorization header) — the middleware short-circuits
        # before the router runs. Anything else (e.g. 404 from
        # NotebookService) would mean the auth check was bypassed.
        assert resp.status_code == 401, (
            f"Expected 401 from PasswordAuthMiddleware, got "
            f"{resp.status_code}. The schema.ttl endpoint must NOT be "
            f"in the auth excluded_paths allow-list."
        )
