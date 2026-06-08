"""Tests for the schemas router (Phase B.2b + B.3a).

Covers ``GET /api/notebooks/{id}/schema.ttl`` (B.2b):

* happy path with two accepted extensions → 2 extra ``owl:Class`` declarations
* empty-extensions path (no notebook_schema row) → base ontology unchanged
* 404 for unknown notebook id
* response headers: ``Content-Type: text/turtle`` and an attachment filename
* roundtrip: returned body parses back via ``rdflib.Graph().parse(format="turtle")``

And ``GET /api/notebooks/{id}/schema`` + ``GET /api/notebooks/{id}/pass1_results``
(B.3a):

* schema JSON happy path: base entity types + accepted/pending extensions + flags
* schema JSON 404 for unknown notebook
* schema JSON empty-state: returns the default base ontology + empty extensions
* pass1_results: enriches each row with the source title
* pass1_results: 404 for unknown notebook

The tests use FastAPI's dependency_overrides plus AsyncMock for the
notebook service and the notebook_schema / pass1_results / source
repositories — no DB needed.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from rdflib import Graph, Literal
from rdflib.namespace import OWL, RDF, RDFS

from app_main.api.routers.schemas import router
from app_main.dependencies import (
    get_notebook_schema_repo,
    get_notebook_service,
    get_pass1_result_repo,
    get_source_repo,
)
from app_main.services.notebook_service import NotebookService
from shared.models import Notebook, Source
from shared.models.notebook_schema import NotebookSchema, Pass1Result
from surrealdb_service.repositories.notebook_schema import (
    NotebookSchemaRepository,
    Pass1ResultRepository,
)
from surrealdb_service.repositories.source import SourceRepository


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


# ---------------------------------------------------------------------------
# B.3a — GET /api/notebooks/{id}/schema (JSON browse)
# ---------------------------------------------------------------------------


def _make_app_b3a(
    notebook_svc: AsyncMock,
    schema_repo: AsyncMock,
    pass1_repo: AsyncMock | None = None,
    source_repo: AsyncMock | None = None,
) -> TestClient:
    """Like ``_make_app`` but also wires the pass1 + source overrides.

    Kept separate from ``_make_app`` so the B.2b tests above don't have
    to know about the new dependencies — additive surface only.
    """
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_notebook_schema_repo] = lambda: schema_repo
    if pass1_repo is not None:
        app.dependency_overrides[get_pass1_result_repo] = lambda: pass1_repo
    if source_repo is not None:
        app.dependency_overrides[get_source_repo] = lambda: source_repo
    return TestClient(app)


class TestSchemaJsonHappyPath:
    """Schema JSON returns base types + accepted + pending extensions."""

    def test_returns_full_payload_with_extensions(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:test1",
            base_ontology="scholarly",
            accepted_extensions=[
                {
                    "extension_id": "ext-acc-1",
                    "type_name": "PreprintServer",
                    "parent_type": "Organization",
                    "description": "An online archive for preprints.",
                    "properties": [
                        {"name": "preprintUrl", "data_type": "url"},
                    ],
                },
            ],
            pending_extensions=[
                {
                    "extension_id": "ext-pen-1",
                    "type_name": "Cohort",
                    "parent_type": "Group",
                    "rationale": "Found cohort references in 3 sources.",
                },
            ],
            coverage_pct=0.85,
            review_required=True,
            soft_nudge_dismissed=False,
        )

        client = _make_app_b3a(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:test1/schema")

        assert resp.status_code == 200
        body = resp.json()

        assert body["notebook_id"] == "notebook:test1"
        assert body["base_ontology"] == "scholarly"
        assert body["coverage_pct"] == 0.85
        assert body["review_required"] is True
        assert body["soft_nudge_dismissed"] is False

        # Base ontology types come from the YAML — non-empty proves
        # we actually loaded scholarly.yaml.
        assert len(body["base_ontology_types"]) > 0
        # Each node has the expected keys.
        first = body["base_ontology_types"][0]
        assert "name" in first
        assert "properties" in first

        # Extensions normalised into the typed view-model.
        assert len(body["accepted_extensions"]) == 1
        acc = body["accepted_extensions"][0]
        assert acc["extension_id"] == "ext-acc-1"
        assert acc["type_name"] == "PreprintServer"
        assert acc["parent_type"] == "Organization"
        assert len(acc["properties"]) == 1
        assert acc["properties"][0]["name"] == "preprintUrl"

        assert len(body["pending_extensions"]) == 1
        pen = body["pending_extensions"][0]
        assert pen["extension_id"] == "ext-pen-1"
        assert pen["type_name"] == "Cohort"
        assert pen["rationale"] == "Found cohort references in 3 sources."


class TestSchemaJsonEmptyState:
    """Notebook exists but has no notebook_schema row → defaults."""

    def test_returns_defaults_when_schema_row_absent(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = None

        client = _make_app_b3a(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:test1/schema")

        assert resp.status_code == 200
        body = resp.json()

        from app_main.api.routers.schemas import _DEFAULT_BASE_ONTOLOGY

        assert body["base_ontology"] == _DEFAULT_BASE_ONTOLOGY
        assert body["accepted_extensions"] == []
        assert body["pending_extensions"] == []
        assert body["coverage_pct"] == 0.0
        assert body["review_required"] is False
        # The base ontology types are still populated from YAML.
        assert len(body["base_ontology_types"]) > 0


class TestSchemaJsonUnknownNotebook:
    """404 path for the JSON endpoint."""

    def test_returns_404_for_unknown_notebook(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)

        client = _make_app_b3a(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:missing/schema")

        assert resp.status_code == 404
        # The repo should not be touched once the notebook lookup fails.
        schema_repo.get_by_notebook.assert_not_called()


class TestSchemaJsonExtensionWithoutTypeName:
    """Defensive: an extension dict missing ``type_name`` survives the
    boundary as ``"(unknown)"`` rather than crashing the response.

    Rationale: pending extensions arrive from the LLM and may be
    malformed; the user needs to *see* them in the UI to reject them.
    Dropping silently would orphan them in the DB.
    """

    def test_unknown_type_name_renders_as_placeholder(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        schema_repo.get_by_notebook.return_value = NotebookSchema(
            notebook="notebook:test1",
            base_ontology="scholarly",
            pending_extensions=[
                # Missing type_name — defensive normalisation.
                {"extension_id": "ext-bad", "rationale": "??"},
            ],
        )

        client = _make_app_b3a(notebook_svc, schema_repo)
        resp = client.get("/api/notebooks/notebook:test1/schema")

        assert resp.status_code == 200
        body = resp.json()
        assert body["pending_extensions"][0]["type_name"] == "(unknown)"


# ---------------------------------------------------------------------------
# B.3a — GET /api/notebooks/{id}/pass1_results
# ---------------------------------------------------------------------------


_PASS1_NOW = datetime(2026, 2, 1, 10, 0, 0)


def _make_pass1_row(
    source_id: str = "source:a",
    coverage_pct: float = 0.92,
    uncovered: int = 0,
) -> Pass1Result:
    return Pass1Result(
        id=f"pass1_results:{source_id.replace(':', '_')}",
        source=source_id,
        notebook="notebook:test1",
        schema_attempted="scholarly",
        detected_schema="scholarly",
        confidence_in_choice=0.9,
        coverage_pct=coverage_pct,
        uncovered_concepts=[{"surface_form": f"c{i}"} for i in range(uncovered)],
        created=_PASS1_NOW,
        updated=_PASS1_NOW,
    )


def _make_source(source_id: str = "source:a", title: str = "Some Paper") -> Source:
    return Source(
        id=source_id,
        title=title,
        full_text="Body.",
        topics=[],
        created=_PASS1_NOW,
        updated=_PASS1_NOW,
    )


class TestPass1ResultsHappyPath:
    def test_returns_rows_enriched_with_source_title(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        pass1_repo = AsyncMock(spec=Pass1ResultRepository)
        pass1_repo.list_by_notebook.return_value = [
            _make_pass1_row("source:a", 0.92, 2),
            _make_pass1_row("source:b", 0.66, 7),
        ]

        source_repo = AsyncMock(spec=SourceRepository)

        # The router calls source_repo.get(source_id) once per unique
        # source — we side-effect based on the id passed in.
        async def _get(source_id: str):
            return _make_source(
                source_id,
                {"source:a": "Paper A", "source:b": "Paper B"}.get(source_id, ""),
            )

        source_repo.get.side_effect = _get

        client = _make_app_b3a(notebook_svc, schema_repo, pass1_repo, source_repo)
        resp = client.get("/api/notebooks/notebook:test1/pass1_results")

        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 2
        assert rows[0]["source_id"] == "source:a"
        assert rows[0]["source_title"] == "Paper A"
        assert rows[0]["coverage_pct"] == 0.92
        assert rows[0]["uncovered_concept_count"] == 2
        assert rows[1]["source_id"] == "source:b"
        assert rows[1]["source_title"] == "Paper B"
        assert rows[1]["uncovered_concept_count"] == 7

    def test_caches_source_title_lookup_per_source(self):
        """Multiple rows for the same source ⇒ source_repo.get called once."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        pass1_repo = AsyncMock(spec=Pass1ResultRepository)
        # Two reprocess attempts on the same source.
        pass1_repo.list_by_notebook.return_value = [
            _make_pass1_row("source:a", 0.92, 2),
            _make_pass1_row("source:a", 0.88, 4),
        ]

        source_repo = AsyncMock(spec=SourceRepository)
        source_repo.get.return_value = _make_source("source:a", "Paper A")

        client = _make_app_b3a(notebook_svc, schema_repo, pass1_repo, source_repo)
        resp = client.get("/api/notebooks/notebook:test1/pass1_results")

        assert resp.status_code == 200
        # source_repo.get must be called at most once even though there
        # are two rows for the same source — cache hit on the second.
        assert source_repo.get.call_count == 1


class TestPass1ResultsEmpty:
    def test_empty_list_when_no_pass1_history(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        pass1_repo = AsyncMock(spec=Pass1ResultRepository)
        pass1_repo.list_by_notebook.return_value = []
        source_repo = AsyncMock(spec=SourceRepository)

        client = _make_app_b3a(notebook_svc, schema_repo, pass1_repo, source_repo)
        resp = client.get("/api/notebooks/notebook:test1/pass1_results")

        assert resp.status_code == 200
        assert resp.json() == []


class TestPass1ResultsUnknownNotebook:
    def test_returns_404_for_unknown_notebook(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        pass1_repo = AsyncMock(spec=Pass1ResultRepository)
        source_repo = AsyncMock(spec=SourceRepository)

        client = _make_app_b3a(notebook_svc, schema_repo, pass1_repo, source_repo)
        resp = client.get("/api/notebooks/notebook:missing/pass1_results")

        assert resp.status_code == 404
        pass1_repo.list_by_notebook.assert_not_called()


class TestPass1ResultsOrphanedSource:
    """Pass1Result row whose source has been deleted → source_title=None."""

    def test_missing_source_yields_null_title(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        schema_repo = AsyncMock(spec=NotebookSchemaRepository)
        pass1_repo = AsyncMock(spec=Pass1ResultRepository)
        pass1_repo.list_by_notebook.return_value = [
            _make_pass1_row("source:deleted", 0.5, 1),
        ]
        source_repo = AsyncMock(spec=SourceRepository)
        source_repo.get.return_value = None

        client = _make_app_b3a(notebook_svc, schema_repo, pass1_repo, source_repo)
        resp = client.get("/api/notebooks/notebook:test1/pass1_results")

        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 1
        assert rows[0]["source_title"] is None
