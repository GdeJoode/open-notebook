"""Tests for FilteringWorkflow end-to-end pipeline orchestration."""

import pytest

from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    FilteredResult,
)

from entity_filtering.config import FilteringConfig
from entity_filtering.workflow import FilteringWorkflow


def _extraction_result(entities=None, relations=None, metadata=None):
    """Helper to build an ExtractionResult."""
    return ExtractionResult(
        entities=[ExtractedEntity(**e) for e in (entities or [])],
        relations=[ExtractedRelation(**r) for r in (relations or [])],
        metadata=metadata or {},
    )


def _entity(text, label="MISC", confidence=0.9):
    return {
        "text": text,
        "label": label,
        "properties": {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


def _relation(source, target, rel_type="RELATED_TO", confidence=0.8):
    return {
        "source_entity": source,
        "target_entity": target,
        "relation_type": rel_type,
        "properties": {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


class TestWorkflowFullPipeline:
    async def test_full_pipeline_with_extraction_result(self):
        entities = [
            _entity("John Doe", "PERSON", 0.9),
            _entity("Microsoft", "ORG", 0.95),
            _entity("123"),
        ]
        relations = [
            _relation("John Doe", "Microsoft", "WORKS_AT"),
            _relation("123", "Microsoft", "RELATED_TO"),
        ]
        extraction = _extraction_result(entities, relations)
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        assert isinstance(result, FilteredResult)
        # "123" should be removed (pure number noise)
        result_texts = {e.text for e in result.entities}
        assert "John Doe" in result_texts
        assert "Microsoft" in result_texts
        assert "123" not in result_texts
        # Relation referencing "123" should be removed
        assert len(result.relations) == 1
        assert result.relations[0].source_entity == "John Doe"

    async def test_empty_input_returns_empty_filtered_result(self):
        extraction = _extraction_result()
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        assert isinstance(result, FilteredResult)
        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert len(result.removed_entities) == 0
        assert len(result.merged_entity_groups) == 0
        assert len(result.predicted_edges) == 0


class TestWorkflowNoiseRemoval:
    async def test_noise_entities_are_removed(self):
        entities = [
            _entity("Valid Entity", "MISC"),
            _entity("---"),
            _entity("et al."),
            _entity("https://example.com"),
        ]
        extraction = _extraction_result(entities)
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        result_texts = {e.text for e in result.entities}
        assert "Valid Entity" in result_texts
        assert "---" not in result_texts
        assert "et al." not in result_texts
        assert "https://example.com" not in result_texts

    async def test_removed_entities_tracked(self):
        entities = [
            _entity("Good", "MISC"),
            _entity("123"),
        ]
        extraction = _extraction_result(entities)
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        removed_texts = {e.text for e in result.removed_entities}
        assert "123" in removed_texts


class TestWorkflowNormalization:
    async def test_normalization_merges_article_variants(self):
        entities = [
            _entity("The United Nations", "ORG", 0.8),
            _entity("United Nations", "ORG", 0.9),
            _entity("United Nations", "ORG", 0.85),
        ]
        extraction = _extraction_result(entities)
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        # All three should normalize to the same key and merge
        un_entities = [e for e in result.entities if "United Nations" in e.text]
        assert len(un_entities) == 1


class TestWorkflowDeduplication:
    async def test_deduplication_merges_case_variants(self):
        entities = [
            _entity("john doe", "PERSON", 0.8),
            _entity("John Doe", "PERSON", 0.9),
        ]
        extraction = _extraction_result(entities)
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        person_entities = [e for e in result.entities if "doe" in e.text.lower()]
        assert len(person_entities) == 1

    async def test_deduplication_disabled(self):
        config = FilteringConfig(dedup_enabled=False)
        entities = [
            _entity("john doe", "PERSON"),
            _entity("John Doe", "PERSON"),
        ]
        extraction = _extraction_result(entities)
        workflow = FilteringWorkflow(config=config)
        result = await workflow.process(extraction)

        # Without dedup, normalizer may still merge if they share a key,
        # but deduplicator step is skipped.
        assert len(result.merged_entity_groups) == 0


class TestWorkflowMetadata:
    async def test_metadata_includes_filtering_statistics(self):
        entities = [
            _entity("Alice", "PERSON"),
            _entity("Bob", "PERSON"),
            _entity("---"),
        ]
        relations = [
            _relation("Alice", "Bob", "KNOWS"),
        ]
        extraction = _extraction_result(
            entities, relations, metadata={"source": "test"}
        )
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        assert "filtering" in result.metadata
        stats = result.metadata["filtering"]
        assert stats["input_entities"] == 3
        assert stats["input_relations"] == 1
        assert "output_entities" in stats
        assert "output_relations" in stats
        assert "removed_count" in stats
        assert "merge_groups" in stats
        assert "predicted_edges" in stats

    async def test_original_metadata_preserved(self):
        extraction = _extraction_result(
            metadata={"source": "test_doc", "version": "1.0"}
        )
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        assert result.metadata["source"] == "test_doc"
        assert result.metadata["version"] == "1.0"
        assert "filtering" in result.metadata
