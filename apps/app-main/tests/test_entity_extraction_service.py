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
