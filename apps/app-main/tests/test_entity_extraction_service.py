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

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.models import Chunk
from shared.models.extraction import ExtractionResult
from shared.models.notebook_schema import NotebookSchema

from app_main.services.entity_extraction_service import (
    EntityExtractionService,
    SchemaReviewPendingError,
)


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
