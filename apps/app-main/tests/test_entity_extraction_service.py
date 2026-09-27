"""Tests for EntityExtractionService (Phase B.1f).

Covers the wiring contract:

1. ``notebook_id=None`` → legacy single-schema path (regression guard).
2. ``notebook_id`` + ``multi_schema_enabled=True`` → multi-schema
   orchestrator invoked.
3. ``review_required=True`` with no accepted extensions → raises
   :class:`SchemaReviewPendingError`.
4. ``multi_schema_enabled=False`` forces single-schema even when
   ``notebook_id`` set (ops kill-switch).

All paths are mocked at the workflow / orchestrator seam — no LLM,
no SurrealDB, no esperanto model construction.
"""

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app_main.services.entity_extraction_service import (
    NO_DECLARED_BASE,
    EntityExtractionService,
    SchemaReviewPendingError,
)
from shared.models import Chunk
from shared.models.extraction import ExtractedEntity, ExtractionResult, FilteredResult
from shared.models.notebook_schema import NotebookSchema


def _make_chunk(text: str = "hello world", chunk_id: str = "chunk:1") -> Chunk:
    """Build a minimal Chunk that survives Pydantic validation."""
    return Chunk(
        id=chunk_id,
        source="source:test",
        text=text,
        order=0,
        physical_page=0,
        element_type="paragraph",
        positions=[],
        section_path=[],
        section_level=0,
    )


@pytest.fixture
def base_source_repo():
    """Stub SourceRepository with two chunks and a basic source row."""
    repo = AsyncMock()
    repo.get_chunks = AsyncMock(return_value=[_make_chunk()])
    repo.get = AsyncMock(return_value=MagicMock(id="source:test", metadata={}))
    return repo


@pytest.fixture
def notebook_schema_repo_fixture():
    repo = AsyncMock()
    repo.get_by_notebook = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def pass1_repo_fixture():
    repo = AsyncMock()
    repo.record = AsyncMock(return_value="pass1_results:1")
    return repo


class TestRunExtractionLegacyPath:
    """notebook_id omitted → legacy single-schema workflow path.

    The multi-schema orchestrator must NOT be invoked. Regression
    guard for CLI / test callers that predate B.1f.
    """

    @pytest.mark.asyncio
    async def test_no_notebook_id_uses_single_schema(
        self, base_source_repo
    ):
        """Without notebook_id, multi-schema is bypassed entirely."""
        svc = EntityExtractionService(source_repo=base_source_repo)

        # The seam: ExtractionWorkflow.extract called with no
        # ``mode="multi"`` kwarg → single-schema path.
        mock_extract = AsyncMock(return_value=ExtractionResult(metadata={}))
        with patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch.object(
            svc, "_save_result", AsyncMock()
        ):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                ontology_name="general",
                run_filtering=False,
            )

        # The single-schema call site invokes extract(chunks) with no
        # ``mode`` kwarg (default "single").
        assert mock_extract.await_count == 1
        call_kwargs = mock_extract.await_args.kwargs
        assert "mode" not in call_kwargs or call_kwargs["mode"] != "multi"

    @pytest.mark.asyncio
    async def test_no_chunks_short_circuits(self, base_source_repo):
        """Empty chunks → no extraction call at all."""
        base_source_repo.get_chunks = AsyncMock(return_value=[])
        svc = EntityExtractionService(source_repo=base_source_repo)

        with patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls:
            result = await svc.run_extraction(source_id="source:test")
            mock_workflow_cls.assert_not_called()

        assert result["entity_count"] == 0
        assert result["relation_count"] == 0


class TestRunExtractionMultiSchemaPath:
    """notebook_id provided + multi_schema_enabled=True → orchestrator."""

    @pytest.mark.asyncio
    async def test_notebook_id_routes_to_multi_schema(
        self,
        base_source_repo,
        notebook_schema_repo_fixture,
        pass1_repo_fixture,
    ):
        """The multi-schema branch is taken — assert via _run_multi_schema spy."""
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        with patch.object(
            svc, "_run_multi_schema", AsyncMock(return_value=ExtractionResult())
        ) as spy, patch.object(svc, "_save_result", AsyncMock()):
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        spy.assert_awaited_once()
        # Notebook id flows through unchanged.
        assert spy.await_args.kwargs["notebook_id"] == "notebook:abc"
        assert spy.await_args.kwargs["source_id"] == "source:test"

    @pytest.mark.asyncio
    async def test_multi_schema_enabled_false_forces_single_schema(
        self,
        base_source_repo,
        notebook_schema_repo_fixture,
        pass1_repo_fixture,
    ):
        """Even with notebook_id, the kill-switch routes to single-schema."""
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        with patch.object(
            svc, "_run_multi_schema", AsyncMock(return_value=ExtractionResult())
        ) as spy, patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = AsyncMock(return_value=ExtractionResult())
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                multi_schema_enabled=False,
                run_filtering=False,
            )

        # The orchestrator must NOT have been called.
        spy.assert_not_awaited()
        # The single-schema workflow.extract was called instead.
        mock_workflow.extract.assert_awaited_once()


class TestSchemaReviewPendingError:
    """``review_required=True`` + no accepted extensions → exception."""

    @pytest.mark.asyncio
    async def test_review_required_raises(
        self,
        base_source_repo,
        notebook_schema_repo_fixture,
        pass1_repo_fixture,
    ):
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc",
                base_ontology="scholarly",
                review_required=True,
                accepted_extensions=[],
                pending_extensions=[
                    {"type_name": "EarlyCareerResearcher", "schema_name": "scholarly"}
                ],
            )
        )

        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        with pytest.raises(SchemaReviewPendingError) as excinfo:
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        # Exception carries the context the handler / API will surface.
        err = excinfo.value
        assert err.notebook_id == "notebook:abc"
        assert err.source_id == "source:test"
        assert err.pending_count == 1

    @pytest.mark.asyncio
    async def test_review_required_with_accepted_extensions_does_not_raise(
        self,
        base_source_repo,
        notebook_schema_repo_fixture,
        pass1_repo_fixture,
    ):
        """Once the user has accepted at least one extension, the gate
        no longer blocks (the user has acted on the review prompt).

        Pins the review-gate's two-part condition: ``review_required``
        AND ``not accepted_extensions`` — failing either side opens
        the gate.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc",
                base_ontology="scholarly",
                review_required=True,
                accepted_extensions=[
                    {"type_name": "EarlyCareerResearcher", "schema_name": "scholarly"}
                ],
            )
        )

        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        # Patch the orchestrator off so we just verify no exception.
        with patch.object(
            svc, "_run_multi_schema", AsyncMock(return_value=ExtractionResult())
        ), patch.object(svc, "_save_result", AsyncMock()):
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

    @pytest.mark.asyncio
    async def test_a_reparent_alone_does_not_lift_the_review_gate(
        self,
        base_source_repo,
        notebook_schema_repo_fixture,
        pass1_repo_fixture,
    ):
        """N.4d.3 — the new entry kind must not silently open a paused gate.

        The predicate is "the curator has reviewed the schema this notebook
        proposes". A re-parent moves a type that ALREADY exists and says nothing
        about the pending proposals, so a curator who adjusts one parent while
        leaving forty extensions unreviewed must still be paused.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc",
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
        )

        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        with pytest.raises(SchemaReviewPendingError):
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

    @pytest.mark.asyncio
    async def test_a_reparent_beside_a_real_acceptance_still_opens_the_gate(
        self,
        base_source_repo,
        notebook_schema_repo_fixture,
        pass1_repo_fixture,
    ):
        """The exclusion is reparent-specific, not a wholesale re-close: a real
        acceptance in the same list still satisfies the gate.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc",
                base_ontology="scholarly",
                review_required=True,
                accepted_extensions=[
                    {
                        "reparent_id": "reparent::Researcher->Institution",
                        "op": "reparent",
                        "type_name": "Researcher",
                        "new_parent": "Institution",
                    },
                    {"type_name": "EarlyCareerResearcher", "schema_name": "scholarly"},
                ],
            )
        )

        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        with patch.object(
            svc, "_run_multi_schema", AsyncMock(return_value=ExtractionResult())
        ), patch.object(svc, "_save_result", AsyncMock()):
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

    @pytest.mark.asyncio
    async def test_paused_error_subclasses_job_paused(
        self,
        notebook_schema_repo_fixture,
    ):
        """The error subclasses JobPausedForReviewError so the generic
        job-queue worker handles it without knowing about extraction.

        Static-type pin — important because the worker decides
        ``JobStatus.PAUSED_FOR_REVIEW`` vs ``FAILED`` based on this
        type discrimination.
        """
        from job_queue import JobPausedForReviewError

        err = SchemaReviewPendingError(
            notebook_id="notebook:n",
            source_id="source:s",
            pending_count=2,
        )
        assert isinstance(err, JobPausedForReviewError)


class TestMakeDefaultLLMCaller:
    """B.1f: the LLM caller factory routes through ModelManager.

    Pin via spy that ``ModelManager.get_model_from_config`` is invoked
    and the returned async callable dispatches via ``achat_complete``.
    """

    @pytest.mark.asyncio
    async def test_caller_routes_through_model_manager_and_achat_complete(self):
        from app_main.services.entity_extraction_service import make_default_llm_caller
        from shared.models import DefaultModels, Model

        # Mock the LanguageModel returned by the factory.
        mock_lm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
        mock_lm.achat_complete = AsyncMock(return_value=mock_response)

        defaults = DefaultModels(default_chat_model="model:default")
        model_record = Model(
            id="model:default",
            name="gpt-test",
            provider="openai",
            type="language",
        )

        mock_mm = MagicMock()
        mock_mm.get_defaults = MagicMock(return_value=defaults)
        mock_mm.get_model_from_config = MagicMock(return_value=mock_lm)

        mock_model_repo = MagicMock()
        mock_model_repo.get = AsyncMock(return_value=model_record)

        # Patch esperanto.LanguageModel so isinstance() check inside
        # make_default_llm_caller passes against our MagicMock.
        with patch(
            "app_main.services.entity_extraction_service.get_model_manager",
            create=True,
        ) as mock_get_mm, patch(
            "app_main.dependencies.get_model_repo"
        ) as mock_get_model_repo, patch(
            "app_main.services.entity_extraction_service.get_model_manager",
            return_value=mock_mm,
            create=True,
        ):
            # Override the lazy import inside the factory.
            import llm_manager
            with patch.object(llm_manager, "get_model_manager", return_value=mock_mm):
                mock_get_model_repo.return_value = mock_model_repo
                with patch("esperanto.LanguageModel", new=type(mock_lm)):
                    caller = await make_default_llm_caller()
                    result = await caller("sys", "usr", "default")

        # The factory's caller routed our prompts through achat_complete.
        mock_lm.achat_complete.assert_awaited_once()
        sent_messages = mock_lm.achat_complete.await_args.kwargs.get("messages")
        assert sent_messages is not None
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[0]["content"] == "sys"
        assert sent_messages[1]["content"] == "usr"
        # The caller unwraps the esperanto ChatCompletion shape.
        assert result == "OK"

    async def _resolve_model_id(self, defaults, default_field, monkeypatch):
        """Invoke a J.4-routed caller with mocked deps and return the head model
        id it resolved (the id the local route candidate loaded from model_repo).

        With no cloud keys set the CLOUD route is single-entry [ollama]; its head
        model id is the per-task default. B.8's extraction-model independence is
        preserved: the head id still comes from ``default_extraction_model`` (or
        ``default_chat_model`` when the former is unset)."""
        from app_main.services.entity_extraction_service import make_default_llm_caller
        from shared.models import Model

        # No cloud keys -> CLOUD resolves to the local-only single-entry route.
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

        mock_lm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
        mock_lm.achat_complete = AsyncMock(return_value=mock_response)

        captured = {}

        async def _get(model_id):
            captured["id"] = model_id
            return Model(id=model_id, name="n", provider="ollama", type="language")

        mock_mm = MagicMock()
        mock_mm.get_defaults = MagicMock(return_value=defaults)
        mock_mm.get_model_from_config = MagicMock(return_value=mock_lm)
        mock_model_repo = MagicMock()
        mock_model_repo.get = AsyncMock(side_effect=_get)

        with patch(
            "app_main.dependencies.get_model_repo", return_value=mock_model_repo
        ):
            import llm_manager
            with patch.object(llm_manager, "get_model_manager", return_value=mock_mm):
                with patch("esperanto.LanguageModel", new=type(mock_lm)):
                    caller = await make_default_llm_caller(default_field=default_field)
                    await caller("sys", "usr", "default")
        return captured.get("id")

    @pytest.mark.asyncio
    async def test_default_field_resolves_extraction_model(self, monkeypatch):
        """B.8a: default_field='default_extraction_model' selects the extraction
        model independently of the chat model (now the route head)."""
        from shared.models import DefaultModels

        defaults = DefaultModels(
            default_chat_model="model:chat",
            default_extraction_model="model:extract",
        )
        resolved = await self._resolve_model_id(
            defaults, "default_extraction_model", monkeypatch
        )
        assert resolved == "model:extract"

    @pytest.mark.asyncio
    async def test_extraction_model_falls_back_to_chat_when_unset(self, monkeypatch):
        """B.8a: when default_extraction_model is unset, extraction falls back to
        the chat model (back-compat)."""
        from shared.models import DefaultModels

        defaults = DefaultModels(
            default_chat_model="model:chat",
            default_extraction_model=None,
        )
        resolved = await self._resolve_model_id(
            defaults, "default_extraction_model", monkeypatch
        )
        assert resolved == "model:chat"

    @pytest.mark.asyncio
    async def test_per_call_model_override_becomes_route_head(self, monkeypatch):
        """J.4: a per-call model id on the B.8 caller shape now flows into the
        resolved route as the HEAD model id (B.8 override precedence), rather
        than being logged-and-ignored as in B.1f.

        Replaces B.1f's warning-branch test: the override is no longer inert —
        it overrides the per-task default for that call."""
        from app_main.services.entity_extraction_service import make_default_llm_caller
        from shared.models import DefaultModels, Model

        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

        mock_lm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
        mock_lm.achat_complete = AsyncMock(return_value=mock_response)

        defaults = DefaultModels(default_chat_model="model:bound")

        captured = {}

        async def _get(model_id):
            captured["id"] = model_id
            return Model(
                id=model_id, name="n", provider="ollama", type="language"
            )

        mock_mm = MagicMock()
        mock_mm.get_defaults = MagicMock(return_value=defaults)
        mock_mm.get_model_from_config = MagicMock(return_value=mock_lm)

        mock_model_repo = MagicMock()
        mock_model_repo.get = AsyncMock(side_effect=_get)

        import llm_manager
        with patch.object(
            llm_manager, "get_model_manager", return_value=mock_mm
        ), patch(
            "app_main.dependencies.get_model_repo",
            return_value=mock_model_repo,
        ), patch("esperanto.LanguageModel", new=type(mock_lm)):
            caller = await make_default_llm_caller()
            # A per-call override id wins over the configured default.
            result = await caller("sys", "usr", "model:other-id")

        assert result == "OK"
        assert captured.get("id") == "model:other-id"
        assert caller.served_model_id == "model:other-id"


# ---------------------------------------------------------------------------
# B.1f Attempt-2: cover the _run_multi_schema body end-to-end
# ---------------------------------------------------------------------------
#
# The attempt-1 tests patched ``_run_multi_schema`` itself, so the
# schema-discovery / applicability-detection / accepted-extensions
# broadcast / LLM-caller-wiring branches inside the body were never
# exercised. Reviewer's Major #3.
#
# These tests build a real ``EntityExtractionService`` and patch only
# at the *seams* the service crosses:
#
#   * ``get_ontology_manager`` — provides the list of candidate
#     ontologies (we hand-build minimal Ontology instances).
#   * ``detect_applicable_schemas`` — the scorer; mocked so we can
#     drive its return value per test.
#   * ``ExtractionWorkflow.extract`` — the seam between service and
#     orchestrator; spied so we assert what was forwarded.
#   * ``make_default_llm_caller`` — production LLM-caller factory;
#     spied to verify it was invoked.
# ---------------------------------------------------------------------------


def _make_ontology(name: str):
    """Tiny Ontology instance suitable for orchestrator wiring tests."""
    from ontology_manager.schema import Ontology, OntologyMetadata

    return Ontology(
        metadata=OntologyMetadata(name=name, version="0.0.1"),
        entity_types={},
        relationship_types={},
    )


class TestRunMultiSchemaBody:
    """End-to-end coverage of ``_run_multi_schema`` (Major #3).

    Each test patches at a different seam to pin a single branch:

    - ``test_calls_detect_applicable_schemas_with_correct_args``
    - ``test_falls_back_to_single_schema_when_no_applicable``
    - ``test_invokes_default_llm_caller_factory_for_multi_path``
    - ``test_accepted_extensions_with_schema_name_routed_to_one_schema``
    - ``test_accepted_extensions_without_schema_name_broadcast_to_all``
    """

    @pytest.fixture
    def svc(self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture):
        return EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

    @pytest.mark.asyncio
    async def test_calls_detect_applicable_schemas_with_correct_args(
        self, svc, base_source_repo
    ):
        """Schema discovery: the service calls ``detect_applicable_schemas``
        with the document_type from source metadata, a sample of the
        first chunk's text, and the ontologies discovered via
        ``get_ontology_manager().list_ontologies()``.
        """
        # Source has a document_type → should be forwarded.
        base_source_repo.get = AsyncMock(
            return_value=MagicMock(
                id="source:test",
                metadata={"document_type": "scholarly_paper"},
            )
        )
        # Chunks the source repo returns — we want the text to be
        # forwarded as ``sample_text`` to the scorer.
        base_source_repo.get_chunks = AsyncMock(
            return_value=[_make_chunk(text="introduction body" * 20)]
        )

        # Mock the manager so we control the candidate ontologies.
        scholarly = _make_ontology("scholarly")
        general = _make_ontology("general")

        mock_manager = MagicMock()
        mock_manager.list_ontologies = AsyncMock(
            return_value=["scholarly", "general"]
        )
        mock_manager.get_ontology = AsyncMock(
            side_effect=lambda name: {"scholarly": scholarly, "general": general}.get(
                name
            )
        )

        # Spy on detect_applicable_schemas. Return an applicable
        # schema so the body continues into the workflow.extract call.
        detect_spy = AsyncMock(return_value=[(scholarly, 0.92)])

        # ExtractionWorkflow.extract: the seam we don't actually call.
        mock_extract = AsyncMock(return_value=ExtractionResult())

        with patch(
            "ontology_manager.get_ontology_manager",
            return_value=mock_manager,
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect_spy,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        detect_spy.assert_awaited_once()
        kwargs = detect_spy.await_args.kwargs
        # Document-type forwarded from source metadata.
        assert kwargs["document_type"] == "scholarly_paper"
        # Sample text comes from the first chunk (truncated to 2000
        # chars by the service).
        assert kwargs["document_text"] is not None
        assert "introduction body" in kwargs["document_text"]
        assert len(kwargs["document_text"]) <= 2000
        # Both candidate ontologies were forwarded.
        names = [o.metadata.name for o in kwargs["ontologies"]]
        assert "scholarly" in names
        assert "general" in names
        # The orchestrator was reached.
        mock_extract.assert_awaited_once()
        extract_kwargs = mock_extract.await_args.kwargs
        assert extract_kwargs["mode"] == "multi"
        assert extract_kwargs["notebook_id"] == "notebook:abc"
        assert extract_kwargs["source_id"] == "source:test"

    @pytest.mark.asyncio
    async def test_falls_back_to_single_schema_when_no_applicable(
        self, svc
    ):
        """When ``detect_applicable_schemas`` returns ``[]``, the service
        falls back to the legacy single-schema ``workflow.extract(chunks)``
        path. Otherwise the orchestrator would return an empty result
        with no entities — strictly worse than the legacy default.
        """
        mock_manager = MagicMock()
        mock_manager.list_ontologies = AsyncMock(return_value=["general"])
        mock_manager.get_ontology = AsyncMock(
            return_value=_make_ontology("general")
        )

        # Empty applicable → fallback path.
        detect_spy = AsyncMock(return_value=[])
        mock_extract = AsyncMock(return_value=ExtractionResult())

        with patch(
            "ontology_manager.get_ontology_manager",
            return_value=mock_manager,
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect_spy,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        # Fallback: workflow.extract called positionally with chunks
        # (no ``mode="multi"`` kwarg).
        mock_extract.assert_awaited_once()
        call_kwargs = mock_extract.await_args.kwargs
        # Either mode is absent (positional chunks call) or it's not
        # ``"multi"``.
        assert "mode" not in call_kwargs or call_kwargs.get("mode") != "multi"

    @pytest.mark.asyncio
    async def test_invokes_default_llm_caller_factory_for_multi_path(
        self, svc
    ):
        """The multi-schema body builds a single production LLM caller
        via ``make_default_llm_caller()`` and forwards it to the
        workflow so Pass-1 and Pass-2 share one bound LanguageModel.

        Without this wiring the orchestrator would fall back to its
        lazy-default empty-result path (canary log). This is the
        wiring fix the whole B.1f sequence depends on.
        """
        mock_manager = MagicMock()
        scholarly = _make_ontology("scholarly")
        mock_manager.list_ontologies = AsyncMock(return_value=["scholarly"])
        mock_manager.get_ontology = AsyncMock(return_value=scholarly)

        detect_spy = AsyncMock(return_value=[(scholarly, 0.92)])
        mock_extract = AsyncMock(return_value=ExtractionResult())

        # Spy on the LLM-caller factory.
        sentinel_caller = AsyncMock()
        caller_factory = AsyncMock(return_value=sentinel_caller)

        with patch(
            "ontology_manager.get_ontology_manager",
            return_value=mock_manager,
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect_spy,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=caller_factory,
        ), patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        # The factory was invoked once for this run.
        caller_factory.assert_awaited_once()
        # The resulting caller was threaded through to workflow.extract.
        forwarded = mock_extract.await_args.kwargs["llm_caller"]
        assert forwarded is sentinel_caller

    @pytest.mark.asyncio
    async def test_llm_caller_factory_failure_is_non_fatal(self, svc):
        """When the LLM caller factory raises (e.g. esperanto provider
        misconfigured), the service logs a warning and continues with
        ``llm_caller=None`` — Pass-1/Pass-2 will use their lazy-default
        empty-result paths rather than blowing up the whole job.

        Pins the try/except block at
        ``entity_extraction_service.py:337-344``.
        """
        mock_manager = MagicMock()
        scholarly = _make_ontology("scholarly")
        mock_manager.list_ontologies = AsyncMock(return_value=["scholarly"])
        mock_manager.get_ontology = AsyncMock(return_value=scholarly)

        detect_spy = AsyncMock(return_value=[(scholarly, 0.92)])
        mock_extract = AsyncMock(return_value=ExtractionResult())

        with patch(
            "ontology_manager.get_ontology_manager",
            return_value=mock_manager,
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect_spy,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(side_effect=RuntimeError("no provider")),
        ), patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            # Should NOT raise.
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        # Workflow still called; llm_caller is None (lazy default
        # takes over).
        assert mock_extract.await_args.kwargs["llm_caller"] is None

    @pytest.mark.asyncio
    async def test_accepted_extensions_with_schema_name_routed_to_one_schema(
        self, svc, notebook_schema_repo_fixture
    ):
        """An accepted extension carrying ``schema_name`` is delivered
        only to that schema's accepted-extensions bucket.

        Schemas without a name-tagged extension get NO entries (the
        caller is being explicit about scope).
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc",
                base_ontology="scholarly",
                accepted_extensions=[
                    {
                        "type_name": "Postdoc",
                        "schema_name": "scholarly",
                    }
                ],
            )
        )

        mock_manager = MagicMock()
        scholarly = _make_ontology("scholarly")
        general = _make_ontology("general")
        mock_manager.list_ontologies = AsyncMock(
            return_value=["scholarly", "general"]
        )
        mock_manager.get_ontology = AsyncMock(
            side_effect=lambda name: {"scholarly": scholarly, "general": general}.get(
                name
            )
        )

        detect_spy = AsyncMock(
            return_value=[(scholarly, 0.92), (general, 0.65)]
        )
        mock_extract = AsyncMock(return_value=ExtractionResult())

        with patch(
            "ontology_manager.get_ontology_manager",
            return_value=mock_manager,
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect_spy,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        forwarded = mock_extract.await_args.kwargs[
            "accepted_extensions_by_schema"
        ]
        assert forwarded is not None
        assert "scholarly" in forwarded
        assert len(forwarded["scholarly"]) == 1
        assert forwarded["scholarly"][0]["type_name"] == "Postdoc"
        # General got nothing — the extension was schema-scoped.
        assert "general" not in forwarded or forwarded["general"] == []

    @pytest.mark.asyncio
    async def test_accepted_extensions_without_schema_name_broadcast_to_all(
        self, svc, notebook_schema_repo_fixture
    ):
        """An accepted extension WITHOUT ``schema_name`` is broadcast
        to every applicable schema. Conservative default — caller
        didn't restrict scope, so every schema gets a copy.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc",
                base_ontology="scholarly",
                accepted_extensions=[
                    {"type_name": "GlobalExt"}  # no schema_name
                ],
            )
        )

        mock_manager = MagicMock()
        scholarly = _make_ontology("scholarly")
        general = _make_ontology("general")
        mock_manager.list_ontologies = AsyncMock(
            return_value=["scholarly", "general"]
        )
        mock_manager.get_ontology = AsyncMock(
            side_effect=lambda name: {"scholarly": scholarly, "general": general}.get(
                name
            )
        )

        detect_spy = AsyncMock(
            return_value=[(scholarly, 0.92), (general, 0.65)]
        )
        mock_extract = AsyncMock(return_value=ExtractionResult())

        with patch(
            "ontology_manager.get_ontology_manager",
            return_value=mock_manager,
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect_spy,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        forwarded = mock_extract.await_args.kwargs[
            "accepted_extensions_by_schema"
        ]
        assert forwarded is not None
        # Both schemas received the broadcast extension.
        assert "scholarly" in forwarded and "general" in forwarded
        assert forwarded["scholarly"][0]["type_name"] == "GlobalExt"
        assert forwarded["general"][0]["type_name"] == "GlobalExt"

    @pytest.mark.asyncio
    async def test_run_multi_schema_filters_resume_sentinel(
        self, svc, notebook_schema_repo_fixture
    ):
        """B.3c attempt-2 — Blocker B1.

        The resume sentinel appended by ``POST /extraction/resume``
        (shape: ``{type_name: "_resumed_without_extensions",
        is_resume_sentinel: True, ...}``) must NEVER reach the
        per-schema ``accepted_extensions_by_schema`` map. If it does,
        ``_format_accepted_extensions`` renders it into the LLM prompt
        as a first-class entity type — the LLM is then instructed to
        extract instances of ``_resumed_without_extensions``, which
        pollutes Pass-2 output for the rest of the notebook's life.

        Real extension types in the same list MUST still come through —
        the filter is sentinel-specific.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc",
                base_ontology="scholarly",
                accepted_extensions=[
                    # Real extension — must survive the filter.
                    {
                        "type_name": "X",
                        "schema_name": "scholarly",
                        "parent_type": "Researcher",
                    },
                    # Resume sentinel — must be dropped.
                    {
                        "type_name": "_resumed_without_extensions",
                        "is_resume_sentinel": True,
                        "created_at": "2026-06-09T12:00:00Z",
                    },
                ],
            )
        )

        mock_manager = MagicMock()
        scholarly = _make_ontology("scholarly")
        mock_manager.list_ontologies = AsyncMock(return_value=["scholarly"])
        mock_manager.get_ontology = AsyncMock(return_value=scholarly)

        detect_spy = AsyncMock(return_value=[(scholarly, 0.92)])
        mock_extract = AsyncMock(return_value=ExtractionResult())

        with patch(
            "ontology_manager.get_ontology_manager",
            return_value=mock_manager,
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect_spy,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        forwarded = mock_extract.await_args.kwargs[
            "accepted_extensions_by_schema"
        ]
        assert forwarded is not None
        # The real extension X is present under its schema.
        assert "scholarly" in forwarded
        type_names = [ext["type_name"] for ext in forwarded["scholarly"]]
        assert "X" in type_names
        # The sentinel was filtered out at the service seam — no entry
        # in any schema bucket should carry ``is_resume_sentinel=True``
        # or the marker ``type_name``.
        for schema_name, bucket in forwarded.items():
            for ext in bucket:
                assert ext.get("is_resume_sentinel") is not True, (
                    f"Sentinel leaked into schema={schema_name!r}: {ext!r}"
                )
                assert ext.get("type_name") != "_resumed_without_extensions", (
                    f"Sentinel type_name leaked into schema={schema_name!r}: "
                    f"{ext!r}"
                )


    @pytest.mark.asyncio
    async def test_run_multi_schema_projects_the_notebooks_accepted_edits(
        self, svc, notebook_schema_repo_fixture
    ):
        """N.4d.3 — the call site itself, not just the helper.

        `_project_notebook_edits` is pure and tested against the real vocabulary
        elsewhere, but the review measured that DELETING its one call left all
        1632 app-main tests green: every guarantee in the phase flows through a
        line nothing asserted. The workflow must receive the PROJECTED ontology,
        so the same objects Pass 2 renders and the persist path bridges through
        carry the curator's edit.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc",
                base_ontology="scholarly",
                accepted_extensions=[
                    {
                        "extension_id": "ext-1",
                        "type_name": "PreprintServer",
                        "schema_name": "scholarly",
                        "parent_type": "Organization",
                    },
                ],
            )
        )

        scholarly = _make_ontology("scholarly")
        mock_manager = MagicMock()
        mock_manager.list_ontologies = AsyncMock(return_value=["scholarly"])
        mock_manager.get_ontology = AsyncMock(return_value=scholarly)
        detect_spy = AsyncMock(return_value=[(scholarly, 0.92)])
        mock_extract = AsyncMock(return_value=ExtractionResult())

        with patch(
            "ontology_manager.get_ontology_manager",
            return_value=mock_manager,
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect_spy,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        applied = mock_extract.await_args.kwargs["applicable_schemas"]
        forwarded_types = applied[0][0].entity_types
        assert "PreprintServer" in forwarded_types, (
            "the accepted extension never reached the workflow's ontology — "
            "the projection is not wired into _run_multi_schema"
        )
        # And the registry's own object is untouched, so the next notebook in
        # this process does not inherit this one's vocabulary.
        assert "PreprintServer" not in scholarly.entity_types
        # The persist path reads the same stashed list.
        assert "PreprintServer" in svc._applicable_schemas[0].entity_types

    @pytest.mark.asyncio
    async def test_an_empty_base_ontology_forces_no_schema(
        self, svc, notebook_schema_repo_fixture
    ):
        """N.4d.3 / D-N4-13 point 2 — the gate this decision rests on.

        `_apply_notebook_schema_default` runs only for a TRUTHY `base_ontology`,
        which the Regio-Deal notebooks leave empty. That is why the forced set is
        not always a subset of what the runtime applies, and it is what
        `TypePlacementService` reports around.

        The discriminator is an accepted extension naming a schema
        auto-detection did NOT pick: with the gate, nothing is forced and the
        applied set is exactly what `detect_applicable_schemas` returned; without
        it, `scholarly` is forced in on the extension's `schema_name`. A review
        measured that the placement-side test asserted this half in a form that
        could not fail — `assert service._apply_notebook_schema_default is not
        None` is true of any object with that attribute — so the gate is
        exercised here instead.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc",
                base_ontology="",
                accepted_extensions=[
                    {"extension_id": "e1", "type_name": "X", "schema_name": "scholarly"}
                ],
            )
        )

        deals = _make_ontology("deals")
        scholarly = _make_ontology("scholarly")
        by_name = {"deals": deals, "scholarly": scholarly}

        mock_manager = MagicMock()
        mock_manager.list_ontologies = AsyncMock(return_value=list(by_name))
        mock_manager.get_ontology = AsyncMock(side_effect=lambda n: by_name.get(n))
        detect_spy = AsyncMock(return_value=[(deals, 0.92)])
        mock_extract = AsyncMock(return_value=ExtractionResult())

        with patch(
            "ontology_manager.get_ontology_manager",
            return_value=mock_manager,
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect_spy,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        applied = [
            ontology.metadata.name
            for ontology, _conf in mock_extract.await_args.kwargs["applicable_schemas"]
        ]
        assert applied == ["deals"], (
            f"an empty base_ontology forced a schema in: {applied}"
        )
        # And with no declared vocabulary there is no name to file gaps under —
        # the aligner falls back to the applied schema's own. The pair of
        # assertions is what makes "the name comes from the notebook" falsifiable
        # in both directions.
        assert svc._gap_ontology_name is None

    @pytest.mark.asyncio
    async def test_a_configured_base_ontology_does_force_its_schemas(
        self, svc, notebook_schema_repo_fixture
    ):
        """The vacuity guard for the test above: with a base set, the same
        extension's schema IS forced, so "nothing was forced" is a statement
        about the gate rather than about a mechanism that never fires.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc",
                base_ontology="deals",
                accepted_extensions=[
                    {"extension_id": "e1", "type_name": "X", "schema_name": "scholarly"}
                ],
            )
        )

        deals = _make_ontology("deals")
        scholarly = _make_ontology("scholarly")
        by_name = {"deals": deals, "scholarly": scholarly}

        mock_manager = MagicMock()
        mock_manager.list_ontologies = AsyncMock(return_value=list(by_name))
        mock_manager.get_ontology = AsyncMock(side_effect=lambda n: by_name.get(n))
        detect_spy = AsyncMock(return_value=[(deals, 0.92)])
        mock_extract = AsyncMock(return_value=ExtractionResult())

        with patch(
            "ontology_manager.get_ontology_manager",
            return_value=mock_manager,
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect_spy,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        applied = [
            ontology.metadata.name
            for ontology, _conf in mock_extract.await_args.kwargs["applicable_schemas"]
        ]
        assert "scholarly" in applied
        # N.4d.4 / D-N4-14: the name every gap row is filed under comes from the
        # notebook's DECLARED vocabulary, read HERE, in the real body. A review
        # measured that the seam test one layer up asserts a literal its own stub
        # assigned, so it round-trips the stub rather than this mechanism.
        assert svc._gap_ontology_name == "deals"

    @pytest.mark.asyncio
    async def test_run_multi_schema_does_not_forward_a_reparent_as_a_type(
        self, svc, notebook_schema_repo_fixture
    ):
        """N.4d.3 — the seam filter, guarded the way the sentinel's is.

        A re-parent records that an EXISTING type moved, not that a new type
        exists. Forwarded, `_format_accepted_extensions` renders it under
        "Accepted Extension Types" and instructs the LLM to treat a base type as
        a curator addition. Real extensions in the same list must still come
        through — the filter is reparent-specific.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc",
                base_ontology="scholarly",
                accepted_extensions=[
                    {
                        "extension_id": "ext-1",
                        "type_name": "PreprintServer",
                        "schema_name": "scholarly",
                        "parent_type": "Organization",
                    },
                    {
                        "reparent_id": "reparent::Researcher->Organization",
                        "op": "reparent",
                        "type_name": "Researcher",
                        "new_parent": "Organization",
                        "parent_type": "Organization",
                        "schema_name": "scholarly",
                    },
                ],
            )
        )

        scholarly = _make_ontology("scholarly")
        mock_manager = MagicMock()
        mock_manager.list_ontologies = AsyncMock(return_value=["scholarly"])
        mock_manager.get_ontology = AsyncMock(return_value=scholarly)
        detect_spy = AsyncMock(return_value=[(scholarly, 0.92)])
        mock_extract = AsyncMock(return_value=ExtractionResult())

        with patch(
            "ontology_manager.get_ontology_manager",
            return_value=mock_manager,
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect_spy,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(svc, "_save_result", AsyncMock()):
            mock_workflow = MagicMock()
            mock_workflow.extract = mock_extract
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        forwarded = mock_extract.await_args.kwargs["accepted_extensions_by_schema"]
        names = [ext["type_name"] for ext in forwarded["scholarly"]]
        assert "PreprintServer" in names
        assert "Researcher" not in names, (
            f"a re-parent reached the Pass-2 extension map: {forwarded!r}"
        )


class TestConceptAlignmentIsReachable:
    """Track N.4d.4 / D-N4-8 — the flag and the wiring.

    The previous judge tier shipped behind a config default nothing set and never
    ran in production; the decision exists so that cannot repeat. These assert the
    stage is reachable from an env flag and that its collaborators actually arrive,
    not that a helper returns the right dict in isolation.
    """

    def _service(self):
        return EntityExtractionService.__new__(EntityExtractionService)

    @pytest.mark.asyncio
    async def test_the_flag_is_off_by_default(self, monkeypatch):
        from app_main.config import get_concept_alignment_enabled

        monkeypatch.delenv("ENABLE_CONCEPT_ALIGNMENT", raising=False)
        assert get_concept_alignment_enabled() is False

    @pytest.mark.asyncio
    async def test_the_flag_turns_it_on(self, monkeypatch):
        from app_main.config import get_concept_alignment_enabled

        monkeypatch.setenv("ENABLE_CONCEPT_ALIGNMENT", "true")
        assert get_concept_alignment_enabled() is True

    @pytest.mark.asyncio
    async def test_a_disabled_stage_builds_no_collaborators(self):
        """Not merely "returns None": nothing is constructed, so a run with the
        flag off pays no repo, model or ontology cost.
        """
        from entity_filtering.config import ConceptAlignmentConfig, FilteringConfig

        service = self._service()
        service._applicable_schemas = ["an-ontology"]
        service._gap_ontology_name = "deals"
        deps = await service._concept_alignment_deps(
            FilteringConfig(concept_alignment=ConceptAlignmentConfig(enabled=False))
        )
        assert set(deps) == {
            "entity_repo",
            "schemas",
            "gap_recorder",
            "llm_caller",
            "gap_ontology_name",
        }
        assert all(value is None for value in deps.values())

    @pytest.mark.asyncio
    async def test_an_enabled_stage_wires_all_four(self):
        from entity_filtering.config import ConceptAlignmentConfig, FilteringConfig

        service = self._service()
        service._applicable_schemas = ["first", "second", "third"]
        service._gap_ontology_name = "deals"
        with patch.object(
            EntityExtractionService,
            "_make_routed_caller",
            AsyncMock(return_value="a-caller"),
        ):
            deps = await service._concept_alignment_deps(
                FilteringConfig(concept_alignment=ConceptAlignmentConfig(enabled=True))
            )
        assert deps["entity_repo"] is not None
        # All three, not `[0]`: `detect_applicable_schemas` uses top_k=3.
        assert deps["schemas"] == ["first", "second", "third"]
        assert deps["gap_ontology_name"] == "deals"
        assert deps["gap_recorder"] is not None
        assert deps["llm_caller"] == "a-caller"

    @pytest.mark.asyncio
    async def test_a_collaborator_that_cannot_be_built_degrades_that_tier_only(self):
        """A judge that cannot be reached must not cost the graph query. Nothing
        here raises — a wiring failure degrades a tier, it does not fail the
        extraction.
        """
        from entity_filtering.config import ConceptAlignmentConfig, FilteringConfig

        service = self._service()
        service._applicable_schemas = ["an-ontology"]
        service._gap_ontology_name = None
        with patch.object(
            EntityExtractionService,
            "_make_routed_caller",
            AsyncMock(side_effect=RuntimeError("no route")),
        ):
            deps = await service._concept_alignment_deps(
                FilteringConfig(concept_alignment=ConceptAlignmentConfig(enabled=True))
            )
        assert deps["llm_caller"] is None
        assert deps["entity_repo"] is not None and deps["gap_recorder"] is not None

    @pytest.mark.asyncio
    async def test_the_judge_caller_is_not_built_when_the_judge_is_off(self):
        from entity_filtering.config import ConceptAlignmentConfig, FilteringConfig

        service = self._service()
        service._applicable_schemas = None
        service._gap_ontology_name = None
        caller = AsyncMock(return_value="a-caller")
        with patch.object(EntityExtractionService, "_make_routed_caller", caller):
            deps = await service._concept_alignment_deps(
                FilteringConfig(
                    concept_alignment=ConceptAlignmentConfig(
                        enabled=True, judge_enabled=False
                    )
                )
            )
        assert deps["llm_caller"] is None
        caller.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_collaborators_reach_the_workflow(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture,
        monkeypatch,
    ):
        """The seam, not the helper: deleting the wiring at the call site leaves
        the helper's own tests green, so this asserts what FilteringWorkflow was
        actually constructed with and what `process` was called with.
        """
        monkeypatch.setenv("ENABLE_CONCEPT_ALIGNMENT", "true")
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        extracted = ExtractionResult(
            entities=[ExtractedEntity(text="X", label="L")], relations=[]
        )
        captured = {}

        class _Workflow:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            async def process(self, result, **kwargs):
                captured["process"] = kwargs
                return FilteredResult(entities=result.entities, relations=[])

        async def _extract_and_stash(**_kwargs):
            # The real `_run_multi_schema` stashes the applied set and the
            # notebook's declared vocabulary; a stub that skips that leaves
            # `_applicable_schemas` at None, and then an assertion about the
            # ontology reaching the workflow cannot fail. A review caught exactly
            # that here.
            svc._applicable_schemas = ["first-ontology", "second-ontology"]
            svc._gap_ontology_name = "deals"
            return extracted

        with patch.object(
            svc, "_run_multi_schema", AsyncMock(side_effect=_extract_and_stash)
        ), patch.object(svc, "_save_result", AsyncMock()), patch.object(
            svc, "_embed_entities", AsyncMock()
        ), patch.object(
            svc, "_persistence", AsyncMock()
        ), patch(
            "app_main.services.entity_extraction_service.FilteringWorkflow",
            _Workflow,
        ), patch.object(
            EntityExtractionService,
            "_make_routed_caller",
            AsyncMock(return_value="a-caller"),
        ):
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=True,
            )

        assert captured["init"]["config"].concept_alignment.enabled is True
        assert captured["init"]["entity_repo"] is not None
        assert captured["init"]["gap_recorder"] is not None
        # ALL applied schemas, not just the first: a type declared in the second
        # or third would otherwise fail to resolve and license no gap.
        assert captured["init"]["alignment_schemas"] == [
            "first-ontology",
            "second-ontology",
        ]
        # And the gap name is the notebook's DECLARED vocabulary, never a member
        # of the applied set — that set is ranked per document, and gaps are
        # keyed on (entity_text, ontology_name).
        assert captured["init"]["gap_ontology_name"] == "deals"
        # source_id is the gap rows' provenance — without it a concept recurring
        # across documents is indistinguishable from one seen once.
        assert captured["process"]["source_id"] == "source:test"
        assert captured["process"]["alignment_llm_caller"] == "a-caller"

    @pytest.mark.asyncio
    async def test_the_gap_counters_reach_the_persisted_stats(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture,
        monkeypatch,
    ):
        """The counters are this phase's only operator-visible output. On the
        alignment report alone they reach nobody: `filtering_stats` is what lands
        in `extraction_result.metadata["filtering"]`.
        """
        monkeypatch.setenv("ENABLE_CONCEPT_ALIGNMENT", "true")
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )
        extracted = ExtractionResult(
            entities=[ExtractedEntity(text="X", label="L")], relations=[]
        )

        class _Workflow:
            def __init__(self, **kwargs):
                pass

            async def process(self, result, **kwargs):
                return FilteredResult(
                    entities=result.entities,
                    relations=[],
                    concept_alignment_report={
                        "aligned_count": 3,
                        "judged_count": 1,
                        "gap_eligible": 2,
                        "gaps_recorded": 1,
                        "gaps_unrecorded": 1,
                        "gap_duplicates_suppressed": 2,
                        "gap_recorder_wired": True,
                        "gap_statistics": {"total": 9},
                        "gap_statistics_status": "ok",
                    },
                )

        with patch.object(
            svc, "_run_multi_schema", AsyncMock(return_value=extracted)
        ), patch.object(svc, "_save_result", AsyncMock()), patch.object(
            svc, "_embed_entities", AsyncMock()
        ), patch.object(svc, "_persistence", AsyncMock()), patch(
            "app_main.services.entity_extraction_service.FilteringWorkflow",
            _Workflow,
        ), patch.object(
            EntityExtractionService,
            "_make_routed_caller",
            AsyncMock(return_value="a-caller"),
        ):
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=True,
            )

        stats = extracted.metadata["filtering"]["concept_alignment"]
        # Both numbers, because a null id from `record_gap` is not success: with
        # only one of them a run where nothing was written reads exactly like one
        # where everything was.
        assert stats["gap_eligible"] == 2
        assert stats["gaps_recorded"] == 1
        assert stats["gaps_unrecorded"] == 1
        assert stats["gap_duplicates_suppressed"] == 2
        assert stats["gap_recorder_wired"] is True
        # The standing totals are the carried N.4c scope item; on the in-memory
        # report alone they stop at a FilteredResult the caller discards.
        assert stats["gap_statistics"] == {"total": 9}
        assert stats["gap_statistics_status"] == "ok"

    @pytest.mark.asyncio
    async def test_no_alignment_report_leaves_the_stats_untouched(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture,
        monkeypatch,
    ):
        """Vacuity guard: the key appears because a report was produced, not
        unconditionally.
        """
        monkeypatch.delenv("ENABLE_CONCEPT_ALIGNMENT", raising=False)
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )
        extracted = ExtractionResult(
            entities=[ExtractedEntity(text="X", label="L")], relations=[]
        )

        class _Workflow:
            def __init__(self, **kwargs):
                pass

            async def process(self, result, **kwargs):
                return FilteredResult(entities=result.entities, relations=[])

        with patch.object(
            svc, "_run_multi_schema", AsyncMock(return_value=extracted)
        ), patch.object(svc, "_save_result", AsyncMock()), patch.object(
            svc, "_embed_entities", AsyncMock()
        ), patch.object(svc, "_persistence", AsyncMock()), patch(
            "app_main.services.entity_extraction_service.FilteringWorkflow",
            _Workflow,
        ):
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=True,
            )

        assert "concept_alignment" not in extracted.metadata["filtering"]

    @pytest.mark.asyncio
    async def test_a_refilter_does_not_inherit_a_previous_runs_schemas(self):
        """A reused service instance must not resolve source B's types against
        source A's detected schemas, or file B's gaps under A's vocabulary.

        `run_filtering_only` re-detects nothing, so whatever sits on the instance
        belongs to the previous run. The L.1 comment twenty lines below already
        passes `applicable_schemas=None` to persist for exactly this reason; the
        alignment side has to make the same choice rather than rely on the DI
        provider happening to build a fresh instance per call.
        """
        svc = EntityExtractionService.__new__(EntityExtractionService)
        svc._applicable_schemas = ["source-A-ontology"]
        svc._gap_ontology_name = "source-A-vocabulary"
        captured = {}

        class _Workflow:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            async def process(self, result, **kwargs):
                captured["process"] = kwargs
                return FilteredResult(entities=[], relations=[])

        row = {
            "id": "extraction_result:1",
            "entities": [{"text": "X", "label": "L"}],
            "relations": [],
            "metadata": {},
        }
        with patch(
            "app_main.services.entity_extraction_service.execute_query",
            AsyncMock(return_value=[row]),
        ), patch(
            "app_main.services.entity_extraction_service.FilteringWorkflow",
            _Workflow,
        ), patch.object(
            EntityExtractionService, "_embed_entities", AsyncMock()
        ), patch.object(
            EntityExtractionService, "_persistence", AsyncMock(), create=True
        ):
            await svc.run_filtering_only(source_id="source:B")

        assert captured["init"]["alignment_schemas"] is None
        assert captured["init"]["gap_ontology_name"] is None
        assert svc._applicable_schemas is None
        assert svc._gap_ontology_name is None
        # And the re-filter wires the same collaborators rather than a bare
        # workflow, so enabling the stage there is not quietly a no-op.
        assert set(captured["init"]) == {
            "config",
            "entity_repo",
            "alignment_schemas",
            "gap_recorder",
            "gap_ontology_name",
        }
        assert captured["process"]["source_id"] == "source:B"

    @pytest.mark.asyncio
    async def test_the_flag_off_leaves_the_stage_disabled_at_the_seam(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture,
        monkeypatch,
    ):
        """The vacuity guard for the test above: same path, flag off, stage off."""
        monkeypatch.delenv("ENABLE_CONCEPT_ALIGNMENT", raising=False)
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )
        extracted = ExtractionResult(
            entities=[ExtractedEntity(text="X", label="L")], relations=[]
        )
        captured = {}

        class _Workflow:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            async def process(self, result, **kwargs):
                return FilteredResult(entities=result.entities, relations=[])

        with patch.object(
            svc, "_run_multi_schema", AsyncMock(return_value=extracted)
        ), patch.object(svc, "_save_result", AsyncMock()), patch.object(
            svc, "_embed_entities", AsyncMock()
        ), patch.object(svc, "_persistence", AsyncMock()), patch(
            "app_main.services.entity_extraction_service.FilteringWorkflow",
            _Workflow,
        ):
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=True,
            )

        assert captured["init"]["config"].concept_alignment.enabled is False
        assert captured["init"]["gap_recorder"] is None


class TestTheCountersCrossOutOfTheService:
    """Track N.5d — the seam a review found missing between N.5a and the gate.

    N.5a made the multi-schema merge carry N.3's counters into
    `ExtractionResult.metadata`. `run_extraction` returned only entity and
    relation counts plus the filtering stats, so nothing that reads a RUN ever
    saw them — including the regression gate, whose two cost dimensions read a
    key no producer in the repository wrote. Half a gate, structurally unable to
    fail, while three documents said re-measuring would populate it.
    """

    def test_the_counters_are_lifted_out_of_the_metadata(self):
        from app_main.services.entity_extraction_service import (
            _observability_counters,
        )

        counters = _observability_counters(
            {
                "chunk_count": 20,
                "entities_extracted": 14,
                "entities_kept": 5,
                "not_a_concept_removed": 9,
                "not_a_concept_judged": 7,
                "abstained_chunks": 9,
                "parse_failures": 0,
                "merged_from_schemas": ["deals", "policy"],  # not a counter
            }
        )
        assert counters == {
            "chunk_count": 20,
            "entities_extracted": 14,
            "entities_kept": 5,
            "not_a_concept_removed": 9,
            "not_a_concept_judged": 7,
            "abstained_chunks": 9,
            "parse_failures": 0,
        }

    def test_absent_counters_yield_no_key_at_all(self):
        """"Not measured" and "measured zero" must stay distinguishable — the
        gate's SKIPPED-versus-PASSED rule is built on exactly that difference.
        """
        from app_main.services.entity_extraction_service import (
            _observability_counters,
        )

        assert _observability_counters({}) == {}
        assert _observability_counters(None) == {}
        assert _observability_counters("not a dict") == {}
        # A measured zero survives.
        assert _observability_counters({"abstained_chunks": 0}) == {
            "abstained_chunks": 0
        }

    def test_a_non_numeric_counter_is_dropped_not_raised(self):
        from app_main.services.entity_extraction_service import (
            _observability_counters,
        )

        got = _observability_counters(
            {"chunk_count": "many", "entities_extracted": 4}
        )
        assert got == {"entities_extracted": 4}

    @pytest.mark.asyncio
    async def test_a_run_summary_carries_them_to_the_gate(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture
    ):
        """The seam itself, driven end to end: what `run_extraction` RETURNS is
        what the harness writes and the gate reads, so the assertion is on the
        summary rather than on the metadata it came from.
        """
        from shared.regression import summarise_run

        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(notebook="notebook:abc", base_ontology="deals")
        )
        notebook_schema_repo_fixture.merge_pending_extensions = AsyncMock(return_value=0)
        notebook_schema_repo_fixture.set_coverage_pct = AsyncMock(return_value=True)
        notebook_schema_repo_fixture.ensure_row = AsyncMock(return_value=False)
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        deals = _make_ontology("deals")
        mock_manager = MagicMock()
        mock_manager.list_ontologies = AsyncMock(return_value=["deals"])
        mock_manager.get_ontology = AsyncMock(return_value=deals)
        extracted = ExtractionResult(
            entities=[ExtractedEntity(text="X", label="L")],
            relations=[],
            metadata={
                "chunk_count": 20,
                "entities_extracted": 14,
                "entities_kept": 5,
                "abstained_chunks": 9,
            },
        )

        with patch(
            "ontology_manager.get_ontology_manager", return_value=mock_manager
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=AsyncMock(return_value=[(deals, 0.9)]),
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(svc, "_save_result", AsyncMock()), patch.object(
            svc, "_embed_entities", AsyncMock()
        ), patch.object(svc, "_persistence", AsyncMock()), patch.object(
            EntityExtractionService,
            "_resolve_privacy_mode",
            AsyncMock(return_value=None),
        ), patch.object(
            EntityExtractionService,
            "_pack_chunks_for_route_head",
            AsyncMock(side_effect=lambda chunks: chunks),
        ):
            workflow = MagicMock()
            workflow.extract = AsyncMock(return_value=extracted)
            workflow_cls.return_value = workflow
            summary = await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        assert summary["counters"]["entities_extracted"] == 14
        # And the gate, reading exactly this shape, now measures something.
        measured = summarise_run([{"result": summary}])
        assert measured["over_generation_rate"] == pytest.approx(9 / 14)
        assert measured["abstain_rate"] == pytest.approx(9 / 20)


class TestARefilterDoesNotEraseWhatItDidNotMeasure:
    """PC.1b / W3b — the boundary a review proved had no guard at all.

    `run_filtering_only` builds a six-key `stats` dict and assigned it over
    `metadata["filtering"]`. The extraction path's version of that key also
    carries a `concept_alignment` block with the alignment counters, so any
    re-filter destroyed them — silently, and with nothing to notice.

    The review reverted the fix and ran both suites: 1839 passed, 69 passed, zero
    failures. The phase's own AC says every wired boundary has a test that fails
    when the state stops crossing, and this one did not. This is that test.
    """

    @staticmethod
    async def _refilter(previous_metadata):
        """Drive the real `run_filtering_only` and return what it wrote back."""
        svc = EntityExtractionService(source_repo=AsyncMock())
        written = {}

        class _Workflow:
            def __init__(self, **kwargs):
                pass

            async def process(self, result, **kwargs):
                return FilteredResult(entities=[], relations=[])

        row = {
            "id": "extraction_result:1",
            "entities": [{"text": "X", "label": "L"}],
            "relations": [],
            "metadata": previous_metadata,
        }

        async def fake_query(query, params=None, *a, **kw):
            if query.strip().startswith("UPDATE"):
                written.update(params or {})
                return []
            return [row]

        with patch(
            "app_main.services.entity_extraction_service.execute_query",
            AsyncMock(side_effect=fake_query),
        ), patch(
            "app_main.services.entity_extraction_service.FilteringWorkflow",
            _Workflow,
        ), patch.object(
            EntityExtractionService, "_embed_entities", AsyncMock()
        ), patch.object(
            EntityExtractionService, "_persistence", AsyncMock(), create=True
        ):
            await svc.run_filtering_only(source_id="source:B")
        return written.get("metadata", {})

    @pytest.mark.asyncio
    async def test_the_alignment_counters_survive_a_refilter(self):
        metadata = await self._refilter(
            {
                "filtering": {
                    "entities_before": 9,
                    "concept_alignment": {"aligned": 4, "gaps_recorded": 2},
                }
            }
        )
        assert metadata["filtering"]["concept_alignment"] == {
            "aligned": 4,
            "gaps_recorded": 2,
        }

    @pytest.mark.asyncio
    async def test_the_keys_this_path_recomputes_still_win(self):
        """Vacuity guard. A merge that preferred the OLD values would keep the
        alignment block and also keep a stale `entities_before`, which is worse
        than the overwrite it replaced.
        """
        metadata = await self._refilter(
            {"filtering": {"entities_before": 999, "concept_alignment": {"aligned": 4}}}
        )
        assert metadata["filtering"]["entities_before"] == 1
        assert metadata["filtering"]["concept_alignment"] == {"aligned": 4}

    @pytest.mark.asyncio
    async def test_a_row_with_no_previous_filtering_block_is_unaffected(self):
        metadata = await self._refilter({})
        assert metadata["filtering"]["entities_before"] == 1
        assert "concept_alignment" not in metadata["filtering"]


class TestWhatWasWrittenReachesTheSummary:
    """PC.1b / W3 — the difference between extracted and persisted, expressible.

    `persist_filtered_result` has always returned five counts; only
    `persisted_entity_ids` was read. So `summary["entity_count"]` is the count the
    LLM produced, and the pipeline review's own measurement — 124 entities
    extracted against 117 rows in the graph — could not be reported by the system
    that produced it. PC.3's AC ("materially fewer than 117 rows, with a named
    figure") has no instrument without this.
    """

    @staticmethod
    def _persistence(**counts):
        persistence = AsyncMock()
        persistence.persist_filtered_result = AsyncMock(
            return_value={
                "persisted_entity_ids": ["entity:1"],
                "entities_upserted": counts.get("entities_upserted", 3),
                "entities_failed": counts.get("entities_failed", 1),
                "relations_created": counts.get("relations_created", 2),
                "relations_merged": counts.get("relations_merged", 0),
                "candidates_stored": counts.get("candidates_stored", 0),
            }
        )
        return persistence

    async def _run(self, svc, persistence, save_mock=None):
        """`save_mock` for the same reason `TestPass1OutcomeReachesTheCurator._run`
        takes one: a caller patching `_save_result` from the OUTSIDE has its patch
        shadowed by the one below, so its assertions run against a double nothing
        ever called. That trap was found by review in PC.1 and it caught this
        class too — a helper that patches a collaborator must let the caller
        supply it.
        """
        deals = _make_ontology("deals")
        mock_manager = MagicMock()
        mock_manager.list_ontologies = AsyncMock(return_value=["deals"])
        mock_manager.get_ontology = AsyncMock(return_value=deals)
        extracted = ExtractionResult(
            entities=[ExtractedEntity(text="X", label="L")], relations=[], metadata={}
        )
        with patch(
            "ontology_manager.get_ontology_manager", return_value=mock_manager
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=AsyncMock(return_value=[(deals, 0.9)]),
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(
            svc, "_save_result", save_mock or AsyncMock()
        ), patch.object(
            svc, "_embed_entities", AsyncMock()
        ), patch.object(svc, "_persistence", persistence), patch.object(
            EntityExtractionService,
            "_resolve_privacy_mode",
            AsyncMock(return_value=None),
        ), patch.object(
            EntityExtractionService,
            "_pack_chunks_for_route_head",
            AsyncMock(side_effect=lambda chunks: chunks),
        ):
            workflow = MagicMock()
            workflow.extract = AsyncMock(return_value=extracted)
            workflow_cls.return_value = workflow
            return await svc.run_extraction(source_id="source:test", notebook_id=None)

    @pytest.mark.asyncio
    async def test_the_summary_reports_what_was_written(self, base_source_repo):
        svc = EntityExtractionService(source_repo=base_source_repo)
        summary = await self._run(svc, self._persistence())

        assert summary["persisted"]["entities_upserted"] == 3
        assert summary["persisted"]["entities_failed"] == 1
        assert summary["persisted"]["relations_created"] == 2

    @pytest.mark.asyncio
    async def test_extracted_and_persisted_are_separately_visible(
        self, base_source_repo
    ):
        """The property the review needed and could not express: a run whose LLM
        produced more than the graph accepted must be able to SAY so.
        """
        svc = EntityExtractionService(source_repo=base_source_repo)
        summary = await self._run(
            svc, self._persistence(entities_upserted=0, entities_failed=1)
        )

        assert summary["entity_count"] == 1  # what the LLM produced
        assert summary["persisted"]["entities_upserted"] == 0  # what was written
        assert summary["entity_count"] != summary["persisted"]["entities_upserted"]

    def test_the_counts_helper_distinguishes_absent_from_zero(self):
        from app_main.services.entity_extraction_service import _persisted_counts

        assert _persisted_counts(None) == {}
        assert _persisted_counts("not a dict") == {}
        assert _persisted_counts({"persisted_entity_ids": []}) == {}
        # A measured zero survives; it is not the same as "persistence did not run".
        assert _persisted_counts({"entities_upserted": 0}) == {"entities_upserted": 0}

    @pytest.mark.asyncio
    async def test_the_counts_are_persisted_where_something_reads_them(
        self, base_source_repo
    ):
        """`summary` lands in `job.result`, which nothing in the repo reads.
        Writing only there would have created a fresh producer with no consumer,
        in the phase whose point is to stop doing that. The metadata copy is
        durable on `extraction_result` and IS read, by the PC.1b probe.
        """
        svc = EntityExtractionService(source_repo=base_source_repo)
        saved = AsyncMock()
        await self._run(svc, self._persistence(), save_mock=saved)

        assert saved.await_count == 1
        _source_id, persisted_result = saved.await_args.args
        assert persisted_result.metadata["persisted"]["entities_upserted"] == 3

    @pytest.mark.asyncio
    async def test_a_run_that_persisted_nothing_says_nothing(self, base_source_repo):
        """Vacuity guard. With filtering off, nothing is written, and the key is
        absent rather than a fabricated set of zeros — "not measured" and
        "measured zero" must stay distinguishable, the same rule N.5d rests on.
        """
        svc = EntityExtractionService(source_repo=base_source_repo)
        deals = _make_ontology("deals")
        mock_manager = MagicMock()
        mock_manager.list_ontologies = AsyncMock(return_value=["deals"])
        mock_manager.get_ontology = AsyncMock(return_value=deals)
        with patch(
            "ontology_manager.get_ontology_manager", return_value=mock_manager
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(svc, "_save_result", AsyncMock()), patch.object(
            svc, "_embed_entities", AsyncMock()
        ), patch.object(
            EntityExtractionService,
            "_resolve_privacy_mode",
            AsyncMock(return_value=None),
        ), patch.object(
            EntityExtractionService,
            "_pack_chunks_for_route_head",
            AsyncMock(side_effect=lambda chunks: chunks),
        ):
            workflow = MagicMock()
            workflow.extract = AsyncMock(
                return_value=ExtractionResult(
                    entities=[ExtractedEntity(text="X", label="L")], relations=[]
                )
            )
            workflow_cls.return_value = workflow
            summary = await svc.run_extraction(
                source_id="source:test", notebook_id=None, run_filtering=False
            )

        assert "persisted" not in summary


class TestTheSoftNudgeReachesItsBanner:
    """PC.1b / W1 — the cleanest producer/consumer pair in the repo, connected.

    `_decide_soft_nudge` has run on every document since B.1e; B.3c built the
    `notebook_event` table, its repository, a router filtering on
    `_KNOWN_NUDGE_EVENT_TYPES` and `SchemaSoftNudge.tsx` to poll it. The two
    halves never met: `workflow.py` discards the decision into `_decision` and the
    table held ZERO rows. Measured before this landed, all five sources carrying
    Pass-1 rows would have fired a banner.

    These assert on what the REPOSITORY was asked to write, not on the decision
    being computed — the decision was always computed. The gap was the write.
    """

    @staticmethod
    def _svc(base_source_repo, event_repo, **kw):
        return EntityExtractionService(
            source_repo=base_source_repo, notebook_event_repo=event_repo, **kw
        )

    @staticmethod
    def _event_repo(unread=None):
        repo = AsyncMock()
        repo.list_unread = AsyncMock(return_value=unread or [])
        repo.record = AsyncMock(return_value="notebook_event:1")
        return repo

    @pytest.mark.asyncio
    async def test_a_mismatch_verdict_is_written_as_an_event(self, base_source_repo):
        event_repo = self._event_repo()
        svc = self._svc(base_source_repo, event_repo)

        await svc._emit_soft_nudge(
            {
                "soft_nudge": "schema_mismatch",
                "best_coverage": 0.45,
                "schemas_attempted": ["deals"],
            },
            "notebook:abc",
            "source:x",
        )

        event_repo.record.assert_awaited_once()
        nb, event_type, payload = event_repo.record.await_args.args
        assert nb == "notebook:abc"
        assert event_type == "schema_mismatch"
        assert payload["source_id"] == "source:x"
        assert payload["best_coverage"] == 0.45

    @pytest.mark.asyncio
    async def test_the_event_type_is_exactly_what_the_router_filters_on(
        self, base_source_repo
    ):
        """The enum's docstring promises its values "round-trip through
        `notebook_event.type` without manual conversion". If either side is
        renamed, the banner silently stops appearing — which is the state this
        phase found. So assert against the ROUTER's own set, not a literal.
        """
        from app_main.api.routers.notebook_events import _KNOWN_NUDGE_EVENT_TYPES
        from ontology_extraction.multi_schema_orchestrator import SoftNudgeDecision

        for decision in (
            SoftNudgeDecision.SCHEMA_MISMATCH,
            SoftNudgeDecision.EXTENSION_SUGGESTED,
        ):
            event_repo = self._event_repo()
            svc = self._svc(base_source_repo, event_repo)
            await svc._emit_soft_nudge(
                {"soft_nudge": decision.value}, "notebook:abc", "source:x"
            )
            _nb, event_type, _payload = event_repo.record.await_args.args
            assert event_type in _KNOWN_NUDGE_EVENT_TYPES

    @pytest.mark.asyncio
    async def test_a_fitting_schema_raises_nothing(self, base_source_repo):
        """NONE writes no row. An event the banner would ignore is still a row a
        curator has to dismiss.
        """
        event_repo = self._event_repo()
        svc = self._svc(base_source_repo, event_repo)

        for metadata in ({"soft_nudge": "none"}, {}, {"soft_nudge": ""}):
            await svc._emit_soft_nudge(metadata, "notebook:abc", "source:x")
        event_repo.record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_unread_banner_is_not_duplicated(self, base_source_repo):
        """Re-extracting a notebook's documents would otherwise queue one banner
        per document saying the same thing.
        """
        event_repo = self._event_repo(unread=[MagicMock()])
        svc = self._svc(base_source_repo, event_repo)

        await svc._emit_soft_nudge(
            {"soft_nudge": "schema_mismatch"}, "notebook:abc", "source:x"
        )
        event_repo.record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failing_write_costs_no_entities(self, base_source_repo):
        """Extraction has already succeeded by the time this runs."""
        event_repo = self._event_repo()
        event_repo.record = AsyncMock(side_effect=RuntimeError("db down"))
        svc = self._svc(base_source_repo, event_repo)

        await svc._emit_soft_nudge(
            {"soft_nudge": "schema_mismatch"}, "notebook:abc", "source:x"
        )  # must not raise

    @pytest.mark.asyncio
    async def test_the_decision_travels_from_the_run_to_the_repository(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture
    ):
        """The seam, not the unit. Everything above could pass while
        `_record_pass1_outcome` never called `_emit_soft_nudge` — which is the
        exact shape that made PC.1's first attempt fully green and inert. This
        drives the recorder with a real result payload.
        """
        event_repo = self._event_repo()
        notebook_schema_repo_fixture.merge_pending_extensions = AsyncMock(return_value=0)
        notebook_schema_repo_fixture.set_coverage_pct = AsyncMock(return_value=True)
        pass1_repo_fixture.list_by_notebook = AsyncMock(return_value=[])
        svc = self._svc(
            base_source_repo,
            event_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        result = ExtractionResult(
            entities=[],
            relations=[],
            metadata={"soft_nudge": "extension_suggested", "best_coverage": 0.85},
        )
        await svc._record_pass1_outcome(
            result,
            "notebook:abc",
            notebook_schema_repo_fixture,
            pass1_repo_fixture,
            "source:test",
        )

        event_repo.record.assert_awaited_once()
        _nb, event_type, payload = event_repo.record.await_args.args
        assert event_type == "extension_suggested"
        assert payload["source_id"] == "source:test"


class TestApplicabilitySample:
    """PC.1 — what the schema detector is allowed to see.

    The sample used to be the first chunk capped at 2000 characters. Measured on
    the project's corpus, the first chunk of a parsed PDF is a title fragment
    with a MEDIAN LENGTH OF 66 CHARACTERS, so detection fired for 2 of 14
    sources; with a sample spread over the body it fires for 13 of 14. That was
    not a scoring nicety: a document with no detected schema takes the legacy
    path where Pass 1 never runs, so the curator queue stayed empty for reasons
    that had nothing to do with the queue.
    """

    @pytest.mark.asyncio
    async def test_the_extraction_path_actually_uses_the_spread_sample(self):
        """The helper being right is worth nothing if production does not call it.

        A mutation proved that: reverting `_run_multi_schema` to the old
        first-chunk sample left every test in this class green, because they all
        exercised `_applicability_sample` directly and nothing pinned the WIRING.
        That is the same shape as the guards this track keeps having to fix — a
        correct unit with no seam behind it.

        So this drives `run_extraction` and inspects the text the detector was
        actually handed.
        """
        source_repo = AsyncMock()
        source_repo.get = AsyncMock(
            return_value=MagicMock(id="source:test", metadata={})
        )
        # A title page, then a body. The old rule would hand over "Convenant".
        source_repo.get_chunks = AsyncMock(
            return_value=[_make_chunk("Convenant", "chunk:0")]
            + [
                _make_chunk(f"brede welvaart paragraaf {i}", f"chunk:{i + 1}")
                for i in range(300)
            ]
        )

        deals = _make_ontology("deals")
        mock_manager = MagicMock()
        mock_manager.list_ontologies = AsyncMock(return_value=["deals"])
        mock_manager.get_ontology = AsyncMock(return_value=deals)
        detect = AsyncMock(return_value=[(deals, 0.9)])

        schema_repo = AsyncMock()
        schema_repo.get_by_notebook = AsyncMock(return_value=None)
        schema_repo.ensure_row = AsyncMock(return_value=True)
        schema_repo.merge_pending_extensions = AsyncMock(return_value=0)
        schema_repo.set_coverage_pct = AsyncMock(return_value=True)
        pass1_repo = AsyncMock()
        pass1_repo.list_by_notebook = AsyncMock(return_value=[])
        svc = EntityExtractionService(
            source_repo=source_repo,
            notebook_schema_repo=schema_repo,
            pass1_repo=pass1_repo,
        )

        with patch(
            "ontology_manager.get_ontology_manager", return_value=mock_manager
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=detect,
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(svc, "_save_result", AsyncMock()), patch.object(
            svc, "_embed_entities", AsyncMock()
        ), patch.object(svc, "_persistence", AsyncMock()), patch.object(
            EntityExtractionService,
            "_resolve_privacy_mode",
            AsyncMock(return_value=None),
        ), patch.object(
            EntityExtractionService,
            "_pack_chunks_for_route_head",
            AsyncMock(side_effect=lambda chunks: chunks),
        ):
            workflow = MagicMock()
            workflow.extract = AsyncMock(
                return_value=ExtractionResult(entities=[], relations=[], metadata={})
            )
            workflow_cls.return_value = workflow
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        detect.assert_awaited()
        scored = detect.await_args.kwargs["document_text"]
        assert "Convenant" in scored, "the head is still scored"
        assert "paragraaf 299" in scored, (
            "the detector must see the end of the document, not only its cover"
        )
        # The old rule handed over the first chunk alone — the word "Convenant"
        # and nothing else. Count how much of the document is actually
        # represented rather than asserting a length, which would only be
        # measuring this fixture's chunk size.
        represented = sum(f"paragraaf {i} " in scored + " " for i in range(300))
        assert represented >= 20, (
            f"only {represented} of 300 body paragraphs reached the detector"
        )

    def test_the_sample_spans_the_document_not_just_its_first_chunk(self):
        from app_main.services.entity_extraction_service import (
            _applicability_sample,
        )

        # A title page, then a body whose vocabulary is what a scorer needs.
        chunks = [{"text": "Convenant"}] + [
            {"text": f"brede welvaart leefbaarheid paragraaf {i}"} for i in range(200)
        ]
        sample = _applicability_sample(chunks)

        assert sample is not None
        assert "Convenant" in sample, "the head is still represented"
        assert "paragraaf 199" in sample, "and so is the tail"
        # The old rule would have returned 9 characters for this document.
        assert len(sample) > 200

    def test_windows_are_spread_rather_than_taken_from_the_front(self):
        from app_main.services.entity_extraction_service import (
            _SAMPLE_MAX_WINDOWS,
            _applicability_sample,
        )

        chunks = [{"text": f"chunk-{i}"} for i in range(500)]
        sample = _applicability_sample(chunks)

        # A prefix-taking implementation contains the head and nothing from the
        # far end. A plain `range(0, n, step)` gets closer but still stops short:
        # for 500 chunks it ends at 480, so the conclusions and annexes — where a
        # policy document names what it is about — are never scored.
        assert "chunk-0 " in sample
        assert "chunk-499" in sample
        assert sample.count("chunk-") <= _SAMPLE_MAX_WINDOWS

    def test_no_text_returns_none_not_an_empty_string(self):
        """`detect_applicable_schemas` treats None as "skip the content path".
        An empty string would score every ontology at 0 and reach the same
        outcome, but through a different branch; None is what it documents.
        """
        from app_main.services.entity_extraction_service import (
            _applicability_sample,
        )

        assert _applicability_sample([]) is None
        assert _applicability_sample([{"text": ""}, {"text": "   "}]) is None
        assert _applicability_sample([{"no_text_key": 1}]) is None

    def test_the_budget_is_respected(self):
        from app_main.services.entity_extraction_service import (
            _SAMPLE_BUDGET_CHARS,
            _applicability_sample,
        )

        chunks = [{"text": "x" * 5000} for _ in range(200)]
        assert len(_applicability_sample(chunks)) <= _SAMPLE_BUDGET_CHARS

    @pytest.mark.parametrize("chunk_length", [60, 600, 1500, 3000, 8000])
    def test_long_chunks_do_not_starve_the_tail_of_the_document(self, chunk_length):
        """The budget must bound the WORK, not truncate the spread.

        An earlier version capped each window at 1200 characters and stopped
        once the running total reached the budget. That break fires in ascending
        index order, so it re-introduced the exact head bias the spread exists to
        remove — a review measured 17 of 40 windows surviving at 1500-character
        chunks, scoring the document on its first 41%, one order of magnitude
        worse than the cover-page problem this all started as.

        The two tests around it could not see this. `test_windows_are_spread`
        uses 9-character chunks, where the budget never binds; the budget test
        used 5000-character chunks but asserted only the LENGTH of the result.
        Between them they touched the failing input and the failing property and
        never at the same time — which is why this one is parametrised over chunk
        size and asserts coverage.
        """
        from app_main.services.entity_extraction_service import (
            _SAMPLE_BUDGET_CHARS,
            _SAMPLE_MAX_WINDOWS,
            _applicability_sample,
        )

        n = 200
        chunks = [{"text": f"MARK{i}-" + ("x" * chunk_length)} for i in range(n)]
        sample = _applicability_sample(chunks)

        kept = [i for i in range(n) if f"MARK{i}-" in sample]
        assert len(kept) == _SAMPLE_MAX_WINDOWS, (
            f"{len(kept)} of {_SAMPLE_MAX_WINDOWS} windows survived at "
            f"chunk_length={chunk_length}"
        )
        assert kept[0] == 0 and kept[-1] == n - 1, (
            "the spread must still reach both ends of the document"
        )
        assert len(sample) <= _SAMPLE_BUDGET_CHARS


class TestNotebookHistoryFallback:
    """PC.1 — what happens when a document detects nothing.

    Per-document detection decides; only when it genuinely fails does the
    notebook's own history get a vote. Measured on the project's corpus: 13 of 14
    documents detect for themselves, and the one that does not sits in a notebook
    whose 17 Pass-1 rows all name `policy_themes`.
    """

    @staticmethod
    def _ont(name):
        return _make_ontology(name)

    @staticmethod
    def _repo(rows):
        repo = AsyncMock()
        repo.list_by_notebook = AsyncMock(return_value=rows)
        return repo

    @pytest.mark.asyncio
    async def test_the_most_attempted_schema_wins(self):
        rows = [
            _pass1_row("source:a", "policy_themes", 0.4),
            _pass1_row("source:b", "policy_themes", 0.5),
            _pass1_row("source:c", "scholarly", 0.9),
        ]
        got = await EntityExtractionService._notebook_fallback_schemas(
            "notebook:abc",
            self._repo(rows),
            [self._ont("policy_themes"), self._ont("scholarly")],
        )
        # `scholarly` scored higher ONCE; `policy_themes` is what this notebook
        # keeps turning out to be. Frequency is the notebook's verdict.
        assert [o.metadata.name for o, _c in got] == ["policy_themes"]

    @pytest.mark.asyncio
    async def test_a_tie_is_broken_by_mean_coverage(self):
        rows = [
            _pass1_row("source:a", "deals", 0.2),
            _pass1_row("source:b", "policy_themes", 0.8),
        ]
        got = await EntityExtractionService._notebook_fallback_schemas(
            "notebook:abc",
            self._repo(rows),
            [self._ont("deals"), self._ont("policy_themes")],
        )
        assert [o.metadata.name for o, _c in got] == ["policy_themes"]

    @pytest.mark.asyncio
    async def test_only_one_schema_is_returned(self):
        """Every extra schema is another Pass-1 and Pass-2 LLM call, and a
        fallback that guessed three would assert more than the evidence carries.
        """
        rows = [
            _pass1_row("source:a", "policy_themes", 0.5),
            _pass1_row("source:b", "deals", 0.5),
            _pass1_row("source:c", "scholarly", 0.5),
        ]
        got = await EntityExtractionService._notebook_fallback_schemas(
            "notebook:abc",
            self._repo(rows),
            [self._ont("policy_themes"), self._ont("deals"), self._ont("scholarly")],
        )
        assert len(got) == 1

    @pytest.mark.asyncio
    async def test_no_history_means_no_fallback(self):
        """A brand-new notebook drops to the legacy path exactly as before."""
        got = await EntityExtractionService._notebook_fallback_schemas(
            "notebook:abc", self._repo([]), [self._ont("deals")]
        )
        assert got == []

    @pytest.mark.asyncio
    async def test_a_schema_no_longer_in_the_registry_is_skipped(self):
        """History can name an ontology that has since been removed. The next
        most attempted one is used rather than crashing or returning it.
        """
        rows = [
            _pass1_row("source:a", "removed_schema", 0.9),
            _pass1_row("source:b", "removed_schema", 0.9),
            _pass1_row("source:c", "deals", 0.1),
        ]
        got = await EntityExtractionService._notebook_fallback_schemas(
            "notebook:abc", self._repo(rows), [self._ont("deals")]
        )
        assert [o.metadata.name for o, _c in got] == ["deals"]

    @pytest.mark.asyncio
    async def test_a_failing_history_read_falls_through_quietly(self):
        """Extraction must not fail because a fallback could not be computed."""
        repo = AsyncMock()
        repo.list_by_notebook = AsyncMock(side_effect=RuntimeError("db down"))
        got = await EntityExtractionService._notebook_fallback_schemas(
            "notebook:abc", repo, [self._ont("deals")]
        )
        assert got == []

    @pytest.mark.asyncio
    async def test_the_fallback_ranks_below_a_curators_declaration(self):
        """A curator's `base_ontology` is forced at 0.85. History is weaker
        evidence than a declaration and must sort below it, while still clearing
        `MIN_APPLICABLE_CONFIDENCE` so it is applied at all.
        """
        from shared.config import MIN_APPLICABLE_CONFIDENCE

        conf = EntityExtractionService._NOTEBOOK_FALLBACK_CONFIDENCE
        assert MIN_APPLICABLE_CONFIDENCE <= conf < 0.85


def _pass1_row(source, schema, coverage, run=None):
    """A `pass1_results` row as `list_by_notebook` returns it.

    `run` is the `pass1_metadata["run_id"]` the orchestrator stamps (PC.1). Rows
    written before PC.1 have none, which is what `run=None` reproduces — so a
    test that omits it is testing the legacy path, deliberately.
    """
    return SimpleNamespace(
        source=source,
        schema_attempted=schema,
        coverage_pct=coverage,
        pass1_metadata={"run_id": run} if run else {},
    )


class TestPass1OutcomeReachesTheCurator:
    """Track PC.1 — the queue the whole review surface reads from.

    A pipeline review measured that `notebook_schema.pending_extensions` had no
    production writer at all: `add_pending_extension`'s only callers were in its
    own roundtrip test, so after eight documents and fourteen `pass1_results`
    rows the queue was empty and `coverage_pct` was 0.0. Everything downstream
    reads those two fields — accept, reject, the panel, and through the accept
    step the whole of N.4d.1-N.4d.3.

    These drive `run_extraction`, not the repository. The repository method
    already worked; nobody called it, and only a test at the seam can see that.
    """

    @staticmethod
    def _ontologies():
        deals = _make_ontology("deals")
        return {"deals": deals}, deals

    async def _run(
        self, svc, schema_repo, pass1_rows, proposals, coverage=0.8, save_mock=None
    ):
        """Drive a full `run_extraction`.

        `save_mock` exists because a review measured that a caller patching
        `_save_result` from the OUTSIDE had its patch shadowed by the one below,
        so the `await_count == 0` it asserted could not fail. A caller that wants
        to inspect what was persisted passes its double in here instead.
        """
        by_name, deals = self._ontologies()
        extracted = ExtractionResult(
            entities=[ExtractedEntity(text="X", label="L")],
            relations=[],
            metadata={"proposed_extensions": proposals, "best_coverage": coverage},
        )

        mock_manager = MagicMock()
        mock_manager.list_ontologies = AsyncMock(return_value=list(by_name))
        mock_manager.get_ontology = AsyncMock(side_effect=lambda n: by_name.get(n))
        pass1_repo = AsyncMock()
        pass1_repo.list_by_notebook = AsyncMock(return_value=pass1_rows)
        svc._pass1_repo = pass1_repo

        with patch(
            "ontology_manager.get_ontology_manager", return_value=mock_manager
        ), patch(
            "app_main.services.entity_extraction_service.detect_applicable_schemas",
            new=AsyncMock(return_value=[(deals, 0.9)]),
        ), patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as workflow_cls, patch(
            "app_main.services.entity_extraction_service.make_default_llm_caller",
            new=AsyncMock(return_value=AsyncMock()),
        ), patch.object(
            svc, "_save_result", save_mock or AsyncMock()
        ), patch.object(
            # Both of these reach OUT of the process - `_embed_entities` to the
            # embedding model and `_persistence` to the database. Left unpatched
            # they made a single test take minutes instead of milliseconds, and
            # the assertions here are about the Pass-1 queue, not about either.
            svc, "_embed_entities", AsyncMock()
        ), patch.object(svc, "_persistence", AsyncMock()), patch.object(
            # `_resolve_privacy_mode` reads the database and swallows failure.
            # With a reachable DB that is milliseconds; with an unreachable one
            # it waits out the connection timeout — measured at 43.6s, which was
            # the ENTIRE cost of each test here. A unit test about the Pass-1
            # queue must not depend on a database being up to run quickly.
            EntityExtractionService, "_resolve_privacy_mode", AsyncMock(return_value=None)
        ), patch.object(
            # Same shape, second call site: the context packer resolves a model
            # route, and on failure logs a WARNING and falls back to un-packed
            # chunks. Measured at 45.2s against an unreachable database. Neither
            # of these two is what this class asserts.
            EntityExtractionService,
            "_pack_chunks_for_route_head",
            AsyncMock(side_effect=lambda chunks: chunks),
        ):
            workflow = MagicMock()
            workflow.extract = AsyncMock(return_value=extracted)
            workflow_cls.return_value = workflow
            await svc.run_extraction(
                source_id="source:test", notebook_id="notebook:abc",
                run_filtering=False,
            )
        return schema_repo

    @pytest.mark.asyncio
    async def test_proposals_reach_the_pending_queue(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture
    ):
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(
                notebook="notebook:abc", base_ontology="deals"
            )
        )
        notebook_schema_repo_fixture.merge_pending_extensions = AsyncMock(return_value=2)
        notebook_schema_repo_fixture.set_coverage_pct = AsyncMock(return_value=True)
        notebook_schema_repo_fixture.ensure_row = AsyncMock(return_value=False)
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        proposals = [
            {"type_name": "Method", "parent_type": "Concept"},
            {"type_name": "GrantFundingSource", "parent_type": "Organization"},
        ]
        await self._run(svc, notebook_schema_repo_fixture, [], proposals)

        notebook_schema_repo_fixture.merge_pending_extensions.assert_awaited_once()
        args = notebook_schema_repo_fixture.merge_pending_extensions.await_args.args
        assert args[0] == "notebook:abc"
        assert [p["type_name"] for p in args[1]] == ["Method", "GrantFundingSource"]

    @pytest.mark.asyncio
    async def test_the_schema_row_is_created_when_it_does_not_exist(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture
    ):
        """Nothing in production ever created this row.

        Measured on the live corpus: 17 `pass1_results` rows carrying 111
        proposals across 79 distinct type names, and ZERO `notebook_schema` rows.
        Every writer — including PC.1's own `merge_pending_extensions` —
        correctly reported "no row" and did nothing, so the fix was inert on
        exactly the data that motivated it.

        The row is created with NO declared base — see the sibling test.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(return_value=None)
        notebook_schema_repo_fixture.ensure_row = AsyncMock(return_value=True)
        notebook_schema_repo_fixture.merge_pending_extensions = AsyncMock(return_value=1)
        notebook_schema_repo_fixture.set_coverage_pct = AsyncMock(return_value=True)
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        await self._run(
            svc, notebook_schema_repo_fixture, [],
            [{"type_name": "Method", "parent_type": "Concept"}],
        )

        notebook_schema_repo_fixture.ensure_row.assert_awaited_once()
        nb, base = notebook_schema_repo_fixture.ensure_row.await_args.args
        assert nb == "notebook:abc"
        assert base == NO_DECLARED_BASE
        # And the queue write still happens, on the row that now exists.
        notebook_schema_repo_fixture.merge_pending_extensions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_created_row_declares_no_base_ontology(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture
    ):
        """Whatever lands in `base_ontology` is FORCED onto every later run.

        Two attempts guessed a value here and both were wrong for the same
        reason. `config.ontology_name` is a per-request parameter defaulting to
        "general" that nothing sets; `DEFAULT_BASE_ONTOLOGY` is "scholarly", a
        constant chosen for the TTL-export read path. Either one, once written,
        is merged into the applied set of every subsequent extraction at
        confidence 0.85, ahead of what the document itself detected — and
        `scholarly` is detected for ZERO of the project's fourteen sources while
        all 17 `pass1_results` rows ran against `policy_themes`.

        So the row declares nothing. The schema is established per document by
        detection, with the notebook's own history as the fallback. This asserts
        the negative — that neither candidate value was written — because that
        is the property that would have caught both attempts.
        """
        from app_main.api.routers.schemas import _DEFAULT_BASE_ONTOLOGY
        from app_main.services.entity_extraction_service import ExtractionConfig

        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(return_value=None)
        notebook_schema_repo_fixture.ensure_row = AsyncMock(return_value=True)
        notebook_schema_repo_fixture.merge_pending_extensions = AsyncMock(return_value=1)
        notebook_schema_repo_fixture.set_coverage_pct = AsyncMock(return_value=True)
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        await self._run(
            svc, notebook_schema_repo_fixture, [],
            [{"type_name": "Method", "parent_type": "Concept"}],
        )

        _nb, base = notebook_schema_repo_fixture.ensure_row.await_args.args
        assert base == NO_DECLARED_BASE
        assert base != ExtractionConfig().ontology_name  # the request parameter
        assert base != _DEFAULT_BASE_ONTOLOGY  # the read-path constant
        # And an empty base forces nothing: the rule that would have applied it
        # skips a falsy name, so detection stays in charge.
        svc_forced = EntityExtractionService(source_repo=base_source_repo)
        assert svc_forced._apply_notebook_schema_default(
            [], [], NotebookSchema(notebook="notebook:abc", base_ontology=base)
        ) == []

    @pytest.mark.asyncio
    async def test_no_proposals_writes_nothing_to_the_queue(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture
    ):
        """Vacuity guard: the call happens because there were proposals, not on
        every run.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(notebook="notebook:abc", base_ontology="deals")
        )
        notebook_schema_repo_fixture.merge_pending_extensions = AsyncMock(return_value=0)
        notebook_schema_repo_fixture.set_coverage_pct = AsyncMock(return_value=True)
        notebook_schema_repo_fixture.ensure_row = AsyncMock(return_value=False)
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )
        await self._run(svc, notebook_schema_repo_fixture, [], [])
        notebook_schema_repo_fixture.merge_pending_extensions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_coverage_is_the_mean_of_each_sources_best(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture
    ):
        """Per source the MAX (Pass 1 runs once per applied schema, and the
        notebook's coverage is how well its best-fitting schema did), then the
        mean across sources.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(notebook="notebook:abc", base_ontology="deals")
        )
        notebook_schema_repo_fixture.merge_pending_extensions = AsyncMock(return_value=0)
        notebook_schema_repo_fixture.set_coverage_pct = AsyncMock(return_value=True)
        notebook_schema_repo_fixture.ensure_row = AsyncMock(return_value=False)
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        # `list_by_notebook` returns newest-first. `pass1_results` is append-only,
        # so a re-extraction ADDS rows: the 0.9 below belongs to source:a's
        # PREVIOUS run and must not count, or a coverage regression could never
        # lower the number.
        rows = [
            _pass1_row("source:a", "deals", 0.8, run="r2"),
            _pass1_row("source:a", "policy_themes", 0.5, run="r2"),
            _pass1_row("source:b", "deals", 0.6, run="r2"),
            _pass1_row("source:a", "deals", 0.9, run="r1"),
        ]
        await self._run(svc, notebook_schema_repo_fixture, rows, [])

        notebook_schema_repo_fixture.set_coverage_pct.assert_awaited_once()
        _nb, value = notebook_schema_repo_fixture.set_coverage_pct.await_args.args
        # source:a -> max(0.8 deals, 0.5 policy_themes) within run r2 = 0.8;
        # source:b -> 0.6. The superseded 0.9 is ignored, which is the point.
        assert value == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_coverage_falls_when_a_schema_edit_makes_it_worse(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture
    ):
        """The flow the soft-nudge exists to drive, and the one two earlier
        versions of this rule could not express.

        Low coverage -> the curator edits the schema set -> re-extract -> the
        number must be free to FALL. Grouping on the newest row per
        `(source, schema_attempted)` cannot do that: a schema edit is exactly
        when `schema_attempted` changes, so the abandoned schema's row is never
        superseded, only aged, and a review measured it still reporting 0.9 while
        the current run scored 0.30. Grouping on the newest RUN can.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(notebook="notebook:abc", base_ontology="deals")
        )
        notebook_schema_repo_fixture.merge_pending_extensions = AsyncMock(return_value=0)
        notebook_schema_repo_fixture.set_coverage_pct = AsyncMock(return_value=True)
        notebook_schema_repo_fixture.ensure_row = AsyncMock(return_value=False)
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )
        rows = [
            # Newest run: the curator switched the notebook to `scholarly`.
            _pass1_row("source:a", "scholarly", 0.3, run="r2"),
            # History: the schemas that ran before the edit, both better.
            _pass1_row("source:a", "deals", 0.9, run="r1"),
            _pass1_row("source:a", "policy_themes", 0.4, run="r1"),
        ]
        await self._run(svc, notebook_schema_repo_fixture, rows, [])

        _nb, value = notebook_schema_repo_fixture.set_coverage_pct.await_args.args
        assert value == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_rows_written_before_the_run_id_still_report_their_best(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture
    ):
        """The 17 rows already in the live database carry no `run_id`.

        They are grouped as ONE run per source rather than one run each, so a
        legacy source reports its best schema instead of whichever row happens to
        sort newest — and a single real run supersedes the whole legacy tail,
        which is how a notebook leaves this state.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(notebook="notebook:abc", base_ontology="deals")
        )
        notebook_schema_repo_fixture.merge_pending_extensions = AsyncMock(return_value=0)
        notebook_schema_repo_fixture.set_coverage_pct = AsyncMock(return_value=True)
        notebook_schema_repo_fixture.ensure_row = AsyncMock(return_value=False)
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )
        legacy_only = [
            _pass1_row("source:a", "policy_themes", 0.4),
            _pass1_row("source:a", "deals", 0.9),
        ]
        await self._run(svc, notebook_schema_repo_fixture, legacy_only, [])
        _nb, value = notebook_schema_repo_fixture.set_coverage_pct.await_args.args
        assert value == pytest.approx(0.9)

        notebook_schema_repo_fixture.set_coverage_pct.reset_mock()
        stamped_then_legacy = [
            _pass1_row("source:a", "scholarly", 0.3, run="r1"),
            _pass1_row("source:a", "deals", 0.9),
        ]
        await self._run(svc, notebook_schema_repo_fixture, stamped_then_legacy, [])
        _nb, value = notebook_schema_repo_fixture.set_coverage_pct.await_args.args
        assert value == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_no_pass1_rows_writes_no_coverage(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture
    ):
        """Writing 0.0 would claim a measurement nobody made."""
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(notebook="notebook:abc", base_ontology="deals")
        )
        notebook_schema_repo_fixture.merge_pending_extensions = AsyncMock(return_value=0)
        notebook_schema_repo_fixture.set_coverage_pct = AsyncMock(return_value=True)
        notebook_schema_repo_fixture.ensure_row = AsyncMock(return_value=False)
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )
        await self._run(svc, notebook_schema_repo_fixture, [], [])
        notebook_schema_repo_fixture.set_coverage_pct.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failing_queue_write_costs_no_entities(
        self, base_source_repo, notebook_schema_repo_fixture, pass1_repo_fixture
    ):
        """Extraction has already succeeded by the time this runs; surfacing a
        proposal must never lose the entities.
        """
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=NotebookSchema(notebook="notebook:abc", base_ontology="deals")
        )
        notebook_schema_repo_fixture.merge_pending_extensions = AsyncMock(
            side_effect=RuntimeError("db down")
        )
        notebook_schema_repo_fixture.set_coverage_pct = AsyncMock(
            side_effect=RuntimeError("db down")
        )
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )
        rows = [_pass1_row("source:a", "deals", 0.5, run="r1")]
        saved = AsyncMock()
        await self._run(
            svc, notebook_schema_repo_fixture, rows,
            [{"type_name": "Method", "parent_type": "Concept"}],
            save_mock=saved,
        )
        # Both writes were attempted and both raised...
        notebook_schema_repo_fixture.merge_pending_extensions.assert_awaited()
        notebook_schema_repo_fixture.set_coverage_pct.assert_awaited()
        # ...and the extraction still completed with its entity intact. Two
        # earlier versions of this ending were weaker than they read: the first
        # asserted nothing at all, the second asserted on a double that the
        # helper had shadowed, so it could not fail. This one inspects what was
        # actually handed to `_save_result`.
        assert saved.await_count == 1
        _source_id, persisted = saved.await_args.args
        assert [e.text for e in persisted.entities] == ["X"]


# ---------------------------------------------------------------------------
# B.4: telemetry hook
# ---------------------------------------------------------------------------


class TestExtractionTelemetryHook:
    """run_extraction emits exactly one ``extraction.complete`` metric per call.

    Closes the AC#3 acceptance gate for Phase B.4. The autouse
    ``_disable_metrics_by_default`` conftest fixture means the real
    ``record_metric`` would short-circuit (env flag) — these tests patch
    the symbol at import-site so we can spy on calls without needing
    the env to be unset.
    """

    @pytest.fixture
    def patched_record_metric(self):
        """Patch ``record_metric`` at its callsite in entity_extraction_service.

        Patching the imported reference (not the source module) ensures
        the spy intercepts the call even though it's bound at module
        load time. Returns the AsyncMock so tests can assert on it.
        """
        with patch(
            "app_main.services.entity_extraction_service.record_metric",
            new_callable=AsyncMock,
        ) as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_happy_path_emits_one_extraction_complete(
        self, base_source_repo, patched_record_metric
    ):
        """Standard single-schema run → exactly one metric, with the right payload."""
        svc = EntityExtractionService(source_repo=base_source_repo)

        with patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ) as mock_workflow_cls, patch.object(
            svc, "_save_result", AsyncMock()
        ):
            mock_workflow = MagicMock()
            mock_workflow.extract = AsyncMock(
                return_value=ExtractionResult(metadata={})
            )
            mock_workflow_cls.return_value = mock_workflow

            await svc.run_extraction(
                source_id="source:test",
                ontology_name="general",
                run_filtering=False,
            )

        patched_record_metric.assert_awaited_once()
        call = patched_record_metric.await_args
        assert call.args[0] == "extraction.complete"
        payload = call.args[1]
        assert payload["entity_count"] == 0
        assert payload["relation_count"] == 0
        assert payload["multi_schema"] is False
        assert payload["avg_confidence"] == 0.0
        # Context flows through unchanged.
        assert call.kwargs["source"] == "source:test"
        assert call.kwargs["notebook"] is None

    @pytest.mark.asyncio
    async def test_no_chunks_still_emits_one_metric(
        self, base_source_repo, patched_record_metric
    ):
        """Zero-chunk early-return must still write a metric (AC#3)."""
        base_source_repo.get_chunks = AsyncMock(return_value=[])
        svc = EntityExtractionService(source_repo=base_source_repo)

        with patch(
            "app_main.services.entity_extraction_service.ExtractionWorkflow"
        ):
            await svc.run_extraction(source_id="source:test")

        patched_record_metric.assert_awaited_once()
        call = patched_record_metric.await_args
        assert call.args[0] == "extraction.complete"
        payload = call.args[1]
        assert payload["no_chunks"] is True
        assert payload["entity_count"] == 0
        assert call.kwargs["source"] == "source:test"

    @pytest.mark.asyncio
    async def test_multi_schema_path_marks_payload(
        self,
        base_source_repo,
        notebook_schema_repo_fixture,
        pass1_repo_fixture,
        patched_record_metric,
    ):
        """multi_schema=True in the payload when the orchestrator was used."""
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        with patch.object(
            svc, "_run_multi_schema", AsyncMock(return_value=ExtractionResult())
        ), patch.object(svc, "_save_result", AsyncMock()):
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        patched_record_metric.assert_awaited_once()
        call = patched_record_metric.await_args
        assert call.args[1]["multi_schema"] is True
        assert call.kwargs["notebook"] == "notebook:abc"

    @pytest.mark.asyncio
    async def test_review_paused_does_not_emit_metric(
        self,
        base_source_repo,
        notebook_schema_repo_fixture,
        pass1_repo_fixture,
        patched_record_metric,
    ):
        """SchemaReviewPendingError must not be followed by a metric write.

        The job is parked, not completed. Emitting here would inflate
        the extraction.complete count and pollute dashboards.
        """
        nb_schema = NotebookSchema(
            notebook="notebook:abc",
            base_ontology="scholarly",
            review_required=True,
            pending_extensions=[{"extension_id": "ext-1"}],
            accepted_extensions=[],
        )
        notebook_schema_repo_fixture.get_by_notebook = AsyncMock(
            return_value=nb_schema
        )
        svc = EntityExtractionService(
            source_repo=base_source_repo,
            notebook_schema_repo=notebook_schema_repo_fixture,
            pass1_repo=pass1_repo_fixture,
        )

        with pytest.raises(SchemaReviewPendingError):
            await svc.run_extraction(
                source_id="source:test",
                notebook_id="notebook:abc",
                run_filtering=False,
            )

        patched_record_metric.assert_not_awaited()


class TestSaveResultNonFatal:
    """B.8d: _save_result must not fail the extraction job — by the time it
    runs, entities + relations are already persisted to the KG; the
    extraction_result record is only a secondary re-filter cache."""

    @pytest.mark.asyncio
    async def test_save_result_swallows_failure(self, base_source_repo):
        svc = EntityExtractionService(source_repo=base_source_repo)
        result = MagicMock(
            entities=[], relations=[], metadata={}, entity_count=0,
            relation_count=0,
        )
        with patch(
            "app_main.services.entity_extraction_service.execute_query",
            new_callable=AsyncMock,
            # the real-world failure: surrealdb async-ws KeyError on a large payload
            side_effect=KeyError("ce2a54c9-9c40-456f-8a47-bd06d3d3537a"),
        ):
            # Must NOT raise — a raised exception here would mark a successful
            # extraction "failed" and dead-letter it.
            await svc._save_result("source:test", result)
