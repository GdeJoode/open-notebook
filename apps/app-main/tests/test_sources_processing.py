"""Tests for the sources_processing router — specifically the
B.1f ``/{source_id}/run-entities`` endpoint contract.

Pins three properties the B.3c UI keys off:

1. ``409 Conflict`` with body ``{"code": "schema_review_pending",
   "notebook_id": ..., "pending_count": ...}`` when the owning
   notebook has ``review_required=True`` and no accepted extensions.
2. ``multi_schema_enabled: false`` passed through to the queued job
   payload (kill-switch must reach the handler).
3. langextract-specific options forwarded into the job payload.

All filesystem / DB / queue interactions are mocked at the
``CommandService.submit_command_job`` /
``NotebookSchemaRepository.get_by_notebook`` /
``SourceRepository.get_notebook_id`` seams — no SurrealDB connection
is opened.
"""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.models.notebook_schema import NotebookSchema

from app_main.api.routers.sources_processing import router
from app_main.dependencies import get_source_service
from app_main.services.source_service import SourceService
from tests.conftest import make_source


def _make_app(source_svc) -> TestClient:
    """Wire a test app with the sources_processing router and a
    FastAPI dependency override on :func:`get_source_service`."""
    app = FastAPI()
    app.include_router(router, prefix="/api/sources")
    app.dependency_overrides[get_source_service] = lambda: source_svc
    return TestClient(app)


def _source_svc_returning(source) -> AsyncMock:
    """Helper to spin up a SourceService stub that returns ``source``."""
    svc = AsyncMock(spec=SourceService)
    svc.get = AsyncMock(return_value=source)
    return svc


class TestRunEntitiesSchemaReviewGate:
    """The 409 pre-check kicks in when the notebook has the review
    toggle set and no accepted extensions yet.

    AC #3 (router-side enforcement): without this the UI would have
    to wait for the worker to pause the job before learning about
    the gate.
    """

    def test_returns_409_with_contract_body_when_review_required(self):
        """The B.3c UI keys off this exact body shape — pin every
        field name and value the UI consumes."""
        source = make_source(id="source:abc")
        svc = _source_svc_returning(source)

        nb_schema = NotebookSchema(
            notebook="notebook:xyz",
            base_ontology="scholarly",
            review_required=True,
            accepted_extensions=[],
            pending_extensions=[
                {"type_name": "EarlyCareerResearcher", "schema_name": "scholarly"},
                {"type_name": "PostDoc", "schema_name": "scholarly"},
            ],
        )

        # Patch the two collaborators the pre-check consults: the
        # source repo (resolves source → notebook id) and the
        # notebook-schema repo (fetches the review state).
        mock_source_repo = MagicMock()
        mock_source_repo.get_notebook_id = AsyncMock(return_value="notebook:xyz")

        nb_schema_repo_instance = MagicMock()
        nb_schema_repo_instance.get_by_notebook = AsyncMock(
            return_value=nb_schema
        )

        with patch(
            "app_main.dependencies.get_source_repo",
            return_value=mock_source_repo,
        ), patch(
            "surrealdb_service.repositories.NotebookSchemaRepository",
            return_value=nb_schema_repo_instance,
        ):
            client = _make_app(svc)
            resp = client.post(
                "/api/sources/source:abc/run-entities",
                json={
                    "ontology_name": "scholarly",
                    "extractor_type": "llm",
                    "multi_schema_enabled": True,
                },
            )

        assert resp.status_code == 409, resp.text
        body = resp.json()
        # FastAPI wraps the detail dict under "detail".
        detail = body["detail"]
        assert detail["code"] == "schema_review_pending"
        assert detail["notebook_id"] == "notebook:xyz"
        assert detail["pending_count"] == 2
        # Human-readable message is present (UI may render it).
        assert "message" in detail and isinstance(detail["message"], str)

    def test_no_409_when_accepted_extensions_present(self):
        """Once the user has accepted at least one extension, the gate
        opens and the job is queued normally."""
        source = make_source(id="source:abc")
        svc = _source_svc_returning(source)

        nb_schema = NotebookSchema(
            notebook="notebook:xyz",
            base_ontology="scholarly",
            review_required=True,
            accepted_extensions=[
                {"type_name": "EarlyCareerResearcher", "schema_name": "scholarly"}
            ],
        )

        mock_source_repo = MagicMock()
        mock_source_repo.get_notebook_id = AsyncMock(return_value="notebook:xyz")

        nb_schema_repo_instance = MagicMock()
        nb_schema_repo_instance.get_by_notebook = AsyncMock(
            return_value=nb_schema
        )

        with patch(
            "app_main.dependencies.get_source_repo",
            return_value=mock_source_repo,
        ), patch(
            "surrealdb_service.repositories.NotebookSchemaRepository",
            return_value=nb_schema_repo_instance,
        ), patch(
            "app_main.services.command_service.CommandService.submit_command_job",
            new=AsyncMock(return_value="job:queued1"),
        ):
            client = _make_app(svc)
            resp = client.post(
                "/api/sources/source:abc/run-entities",
                json={
                    "ontology_name": "scholarly",
                    "extractor_type": "llm",
                    "multi_schema_enabled": True,
                },
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["command_id"] == "job:queued1"
        assert body["status"] == "queued"

    def test_a_reparent_alone_still_returns_409(self):
        """N.4d.3 — the router-side half of the same gate.

        A re-parent moves a type that already exists; it is not a review of the
        pending proposals. Counting it would resume extraction for a notebook
        whose extensions nobody has looked at.
        """
        source = make_source(id="source:abc")
        svc = _source_svc_returning(source)

        nb_schema = NotebookSchema(
            notebook="notebook:xyz",
            base_ontology="scholarly",
            review_required=True,
            accepted_extensions=[
                {
                    "reparent_id": "reparent::Researcher->Institution",
                    "op": "reparent",
                    "type_name": "Researcher",
                    "new_parent": "Institution",
                }
            ],
            pending_extensions=[
                {"type_name": "EarlyCareerResearcher", "schema_name": "scholarly"}
            ],
        )

        mock_source_repo = MagicMock()
        mock_source_repo.get_notebook_id = AsyncMock(return_value="notebook:xyz")

        nb_schema_repo_instance = MagicMock()
        nb_schema_repo_instance.get_by_notebook = AsyncMock(return_value=nb_schema)

        with patch(
            "app_main.dependencies.get_source_repo",
            return_value=mock_source_repo,
        ), patch(
            "surrealdb_service.repositories.NotebookSchemaRepository",
            return_value=nb_schema_repo_instance,
        ):
            client = _make_app(svc)
            resp = client.post(
                "/api/sources/source:abc/run-entities",
                json={
                    "ontology_name": "scholarly",
                    "extractor_type": "llm",
                    "multi_schema_enabled": True,
                },
            )

        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"]["code"] == "schema_review_pending"

    def test_404_when_source_not_found(self):
        """Pre-flight 404 happens before the schema-review check."""
        svc = _source_svc_returning(None)
        client = _make_app(svc)
        resp = client.post(
            "/api/sources/source:missing/run-entities",
            json={},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestRunEntitiesPayloadForwarding:
    """The router builds the command payload — fields that affect
    handler behaviour MUST be forwarded verbatim. Pin via spy on
    :func:`CommandService.submit_command_job`."""

    def test_multi_schema_enabled_false_forwarded_to_job_payload(self):
        """Ops kill-switch path: when the request body has
        ``multi_schema_enabled: false``, the job payload carries it
        verbatim. Without this the handler would default to multi-
        schema mode and the kill-switch would be silently dropped.
        """
        source = make_source(id="source:abc")
        svc = _source_svc_returning(source)

        mock_source_repo = MagicMock()
        # When kill-switch is OFF the pre-check is skipped entirely —
        # ``get_notebook_id`` is not called. Set it anyway in case the
        # ordering changes.
        mock_source_repo.get_notebook_id = AsyncMock(return_value=None)

        submit_spy = AsyncMock(return_value="job:queued1")

        with patch(
            "app_main.dependencies.get_source_repo",
            return_value=mock_source_repo,
        ), patch(
            "app_main.services.command_service.CommandService.submit_command_job",
            new=submit_spy,
        ):
            client = _make_app(svc)
            resp = client.post(
                "/api/sources/source:abc/run-entities",
                json={
                    "ontology_name": "scholarly",
                    "extractor_type": "llm",
                    "multi_schema_enabled": False,
                },
            )

        assert resp.status_code == 200, resp.text
        submit_spy.assert_awaited_once()
        # submit_command_job(module, command, payload)
        args, kwargs = submit_spy.await_args.args, submit_spy.await_args.kwargs
        payload = args[2] if len(args) >= 3 else kwargs.get("command_args")
        assert payload is not None
        assert payload["multi_schema_enabled"] is False
        assert payload["ontology_name"] == "scholarly"
        assert payload["extractor_type"] == "llm"
        assert payload["source_id"] == "source:abc"

    def test_langextract_options_forwarded(self):
        """langextract-specific options must reach the handler so the
        worker can build a LangExtractExtractor with the requested
        config — without forwarding, the UI's per-run model override
        is silently lost.
        """
        source = make_source(id="source:abc")
        svc = _source_svc_returning(source)

        mock_source_repo = MagicMock()
        mock_source_repo.get_notebook_id = AsyncMock(return_value=None)

        submit_spy = AsyncMock(return_value="job:queued1")

        with patch(
            "app_main.dependencies.get_source_repo",
            return_value=mock_source_repo,
        ), patch(
            "app_main.services.command_service.CommandService.submit_command_job",
            new=submit_spy,
        ):
            client = _make_app(svc)
            resp = client.post(
                "/api/sources/source:abc/run-entities",
                json={
                    "ontology_name": "general",
                    "extractor_type": "langextract",
                    "langextract_model_id": "gemma-7b",
                    "langextract_model_url": "http://gpu:8000",
                    "langextract_temperature": 0.1,
                    "langextract_use_schema_constraints": True,
                    "langextract_fence_output": False,
                },
            )

        assert resp.status_code == 200, resp.text
        args, kwargs = submit_spy.await_args.args, submit_spy.await_args.kwargs
        payload = args[2] if len(args) >= 3 else kwargs.get("command_args")
        assert payload is not None
        assert payload["extractor_type"] == "langextract"
        assert payload["langextract_model_id"] == "gemma-7b"
        assert payload["langextract_model_url"] == "http://gpu:8000"
        assert payload["langextract_temperature"] == 0.1
        assert payload["langextract_use_schema_constraints"] is True
        assert payload["langextract_fence_output"] is False

    def test_langextract_path_skips_review_gate(self):
        """The review-gate check only applies to ``extractor_type=llm``
        — langextract runs an entirely separate extraction path that
        does not consult notebook-schema state. Verify the pre-check
        is not even called for non-llm extractors.
        """
        source = make_source(id="source:abc")
        svc = _source_svc_returning(source)

        mock_source_repo = MagicMock()
        # Spy will detect any call.
        mock_source_repo.get_notebook_id = AsyncMock(return_value="notebook:xyz")

        submit_spy = AsyncMock(return_value="job:queued1")

        with patch(
            "app_main.dependencies.get_source_repo",
            return_value=mock_source_repo,
        ), patch(
            "app_main.services.command_service.CommandService.submit_command_job",
            new=submit_spy,
        ):
            client = _make_app(svc)
            resp = client.post(
                "/api/sources/source:abc/run-entities",
                json={
                    "ontology_name": "general",
                    "extractor_type": "langextract",
                },
            )

        assert resp.status_code == 200
        # Pre-check should be skipped — the langextract branch never
        # consults the notebook-schema state.
        mock_source_repo.get_notebook_id.assert_not_called()
