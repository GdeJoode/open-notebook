"""Tests for ontology_extraction.workflow module.

All tests mock the ontology manager and extractor to avoid needing a live
LLM or ontology-manager service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from ontology_extraction.config import ExtractionConfig
from ontology_extraction.workflow import ExtractionWorkflow


def _make_mock_manager(ontology=None):
    """Create a mock ontology manager that returns the given ontology."""
    manager = MagicMock()
    if ontology is None:
        ontology = MagicMock()  # non-None by default so tests pass the check
    manager.get_ontology = AsyncMock(return_value=ontology)
    return manager


def _make_mock_extractor(result=None):
    """Create a mock extractor returning the given ExtractionResult."""
    if result is None:
        result = ExtractionResult()
    extractor = AsyncMock()
    extractor.extract = AsyncMock(return_value=result)
    return extractor


class TestExtractionWorkflowExtract:
    """Tests for ExtractionWorkflow.extract method."""

    async def test_extract_empty_chunks(self):
        """Empty chunk list returns empty result with metadata."""
        mock_manager = _make_mock_manager()
        mock_extractor = _make_mock_extractor()

        with patch(
            "ontology_extraction.workflow.get_ontology_manager",
            return_value=mock_manager,
        ):
            workflow = ExtractionWorkflow()
            workflow._extractor = mock_extractor
            result = await workflow.extract([])

        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert result.metadata["chunk_count"] == 0
        mock_extractor.extract.assert_not_called()

    async def test_extract_skips_empty_text(self):
        """Chunks with empty or whitespace-only text are skipped."""
        entity = ExtractedEntity(text="Found", label="THING", confidence=0.9)
        extraction_result = ExtractionResult(entities=[entity])

        mock_manager = _make_mock_manager()
        mock_extractor = _make_mock_extractor(result=extraction_result)

        with patch(
            "ontology_extraction.workflow.get_ontology_manager",
            return_value=mock_manager,
        ):
            workflow = ExtractionWorkflow()
            workflow._extractor = mock_extractor
            result = await workflow.extract(
                [
                    {"text": "", "id": "empty"},
                    {"text": "   ", "id": "whitespace"},
                    {"text": "Some real text", "id": "valid"},
                ]
            )

        # Only the valid chunk was processed
        mock_extractor.extract.assert_called_once()
        assert len(result.entities) == 1
        assert result.entities[0].text == "Found"

    async def test_extract_ontology_not_found(self):
        """Returns error result when the configured ontology does not exist."""
        mock_manager = MagicMock()
        mock_manager.get_ontology = AsyncMock(return_value=None)

        with patch(
            "ontology_extraction.workflow.get_ontology_manager",
            return_value=mock_manager,
        ):
            workflow = ExtractionWorkflow()
            result = await workflow.extract([{"text": "hello"}])

        assert "error" in result.metadata
        assert "not found" in result.metadata["error"].lower()

    async def test_extract_tags_entities_with_chunk_id(self):
        """Entities and relations are tagged with the source chunk id."""
        entity = ExtractedEntity(text="E", label="X")
        relation = ExtractedRelation(
            source_entity="A", target_entity="B", relation_type="R"
        )
        extraction_result = ExtractionResult(entities=[entity], relations=[relation])

        mock_manager = _make_mock_manager()
        mock_extractor = _make_mock_extractor(result=extraction_result)

        with patch(
            "ontology_extraction.workflow.get_ontology_manager",
            return_value=mock_manager,
        ):
            workflow = ExtractionWorkflow()
            workflow._extractor = mock_extractor
            result = await workflow.extract(
                [{"text": "chunk text", "id": "chunk-42"}]
            )

        assert result.entities[0].source_chunk_id == "chunk-42"
        assert result.relations[0].source_chunk_id == "chunk-42"

    async def test_extract_multiple_chunks_aggregated(self):
        """Results from multiple chunks are aggregated into a single result."""
        mock_manager = _make_mock_manager()

        call_count = 0

        async def side_effect(text, ontology):
            nonlocal call_count
            call_count += 1
            return ExtractionResult(
                entities=[
                    ExtractedEntity(text=f"E{call_count}", label="X")
                ],
                relations=[
                    ExtractedRelation(
                        source_entity=f"S{call_count}",
                        target_entity=f"T{call_count}",
                        relation_type="R",
                    )
                ],
            )

        mock_extractor = AsyncMock()
        mock_extractor.extract = AsyncMock(side_effect=side_effect)

        with patch(
            "ontology_extraction.workflow.get_ontology_manager",
            return_value=mock_manager,
        ):
            workflow = ExtractionWorkflow()
            workflow._extractor = mock_extractor
            result = await workflow.extract(
                [
                    {"text": "First chunk", "id": "c1"},
                    {"text": "Second chunk", "id": "c2"},
                ]
            )

        assert len(result.entities) == 2
        assert len(result.relations) == 2
        assert result.metadata["total_entities"] == 2
        assert result.metadata["total_relations"] == 2
        assert result.metadata["chunk_count"] == 2

    async def test_extract_metadata_contains_ontology_name(self):
        """Result metadata includes the ontology name used for extraction."""
        mock_manager = _make_mock_manager()
        mock_extractor = _make_mock_extractor()

        config = ExtractionConfig(ontology_name="scholarly")

        with patch(
            "ontology_extraction.workflow.get_ontology_manager",
            return_value=mock_manager,
        ):
            workflow = ExtractionWorkflow(config=config)
            workflow._extractor = mock_extractor
            result = await workflow.extract([{"text": "content"}])

        assert result.metadata["ontology_name"] == "scholarly"

    async def test_extract_respects_batch_size(self):
        """Chunks are processed in batches of config.batch_size."""
        mock_manager = _make_mock_manager()
        mock_extractor = _make_mock_extractor()

        config = ExtractionConfig(batch_size=2)

        with patch(
            "ontology_extraction.workflow.get_ontology_manager",
            return_value=mock_manager,
        ):
            workflow = ExtractionWorkflow(config=config)
            workflow._extractor = mock_extractor

            chunks = [{"text": f"chunk {i}", "id": str(i)} for i in range(5)]
            await workflow.extract(chunks)

        # All 5 chunks should have been processed
        assert mock_extractor.extract.call_count == 5


class TestExtractionWorkflowExtractSingle:
    """Tests for ExtractionWorkflow.extract_single convenience method."""

    async def test_extract_single_wraps_text_in_chunk(self):
        """extract_single wraps the text into a single-element chunk list."""
        entity = ExtractedEntity(text="Result", label="THING")
        extraction_result = ExtractionResult(entities=[entity])

        mock_manager = _make_mock_manager()
        mock_extractor = _make_mock_extractor(result=extraction_result)

        with patch(
            "ontology_extraction.workflow.get_ontology_manager",
            return_value=mock_manager,
        ):
            workflow = ExtractionWorkflow()
            workflow._extractor = mock_extractor
            result = await workflow.extract_single("Some text", chunk_id="s1")

        assert len(result.entities) == 1
        assert result.entities[0].source_chunk_id == "s1"

    async def test_extract_single_without_chunk_id(self):
        """extract_single works when chunk_id is omitted."""
        mock_manager = _make_mock_manager()
        mock_extractor = _make_mock_extractor()

        with patch(
            "ontology_extraction.workflow.get_ontology_manager",
            return_value=mock_manager,
        ):
            workflow = ExtractionWorkflow()
            workflow._extractor = mock_extractor
            result = await workflow.extract_single("Text only")

        assert len(result.entities) == 0
        mock_extractor.extract.assert_called_once()


class TestExtractionWorkflowInit:
    """Tests for ExtractionWorkflow constructor and lazy extractor creation."""

    def test_default_config(self):
        """Workflow uses default ExtractionConfig when none provided."""
        workflow = ExtractionWorkflow()
        assert workflow._config.ontology_name == "general"
        assert workflow._config.batch_size == 10

    def test_custom_config(self):
        """Workflow stores the provided config."""
        config = ExtractionConfig(ontology_name="medical", batch_size=3)
        workflow = ExtractionWorkflow(config=config)
        assert workflow._config.ontology_name == "medical"
        assert workflow._config.batch_size == 3

    def test_extractor_initially_none(self):
        """Extractor is not created until first use (lazy init)."""
        workflow = ExtractionWorkflow()
        assert workflow._extractor is None
