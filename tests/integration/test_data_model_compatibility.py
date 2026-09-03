"""
Cross-boundary Pydantic model compatibility tests.

Validates that models used as contracts between packages serialize/deserialize
correctly and maintain Liskov substitution where expected.
"""

import json

import pytest

from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    FilteredResult,
)
from shared.models.file_tracking import PipelineCacheEntry, SourceFolder

from tests.integration.conftest import make_cache_entry, make_source_folder

pytestmark = pytest.mark.integration


class TestExtractionResultRoundtrip:
    """ExtractionResult JSON serialization round-trips."""

    def test_extraction_result_json_roundtrip(self, realistic_extraction_result):
        """model_dump_json() -> model_validate_json() preserves all fields."""
        json_str = realistic_extraction_result.model_dump_json()
        restored = ExtractionResult.model_validate_json(json_str)

        assert restored.entity_count == realistic_extraction_result.entity_count
        assert restored.relation_count == realistic_extraction_result.relation_count
        assert restored.metadata == realistic_extraction_result.metadata
        assert len(restored.entities) == len(realistic_extraction_result.entities)
        assert len(restored.relations) == len(realistic_extraction_result.relations)

        # Spot-check first entity
        assert restored.entities[0].text == "Albert Einstein"
        assert restored.entities[0].label == "PERSON"
        assert restored.entities[0].confidence == 0.95

    def test_filtered_result_json_roundtrip(self, realistic_extraction_result):
        """FilteredResult with all fields populated round-trips correctly."""
        filtered = FilteredResult(
            entities=realistic_extraction_result.entities[:5],
            relations=realistic_extraction_result.relations[:3],
            removed_entities=realistic_extraction_result.entities[17:],
            merged_entity_groups=[["Albert Einstein", "albert einstein", "Einstein"]],
            predicted_edges=[
                ExtractedRelation(
                    source_entity="Marie Curie",
                    target_entity="Solvay Conference",
                    relation_type="PARTICIPATED_IN",
                    confidence=0.7,
                )
            ],
            validation_report={"ontology": {"valid_entities": 5}},
            kg_resolution_report={"matched_count": 2, "new_count": 3},
            metadata={"filtering": {"input_entities": 20, "output_entities": 5}},
        )

        json_str = filtered.model_dump_json()
        restored = FilteredResult.model_validate_json(json_str)

        assert restored.entity_count == 5
        assert restored.relation_count == 3
        assert len(restored.removed_entities) == 3
        assert len(restored.merged_entity_groups) == 1
        assert len(restored.predicted_edges) == 1
        assert restored.validation_report is not None
        assert restored.kg_resolution_report is not None

    def test_filtered_result_is_valid_extraction_result(self):
        """FilteredResult passes isinstance check for ExtractionResult (Liskov)."""
        filtered = FilteredResult(
            entities=[ExtractedEntity(text="Test", label="PERSON")],
            relations=[],
        )
        assert isinstance(filtered, ExtractionResult)
        assert filtered.entity_count == 1

    def test_extraction_with_all_optional_fields_empty(self):
        """Empty ExtractionResult round-trips cleanly."""
        empty = ExtractionResult()
        json_str = empty.model_dump_json()
        restored = ExtractionResult.model_validate_json(json_str)

        assert restored.entity_count == 0
        assert restored.relation_count == 0
        assert restored.metadata == {}
        assert restored.entities == []
        assert restored.relations == []


class TestExtractionCacheCompatibility:
    """Models are storable as JSON strings (cache-compatible)."""

    def test_extraction_result_storable_as_json_string(self, realistic_extraction_result):
        """json.dumps(result.model_dump()) succeeds."""
        dumped = realistic_extraction_result.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)

        assert len(parsed["entities"]) == 20
        assert len(parsed["relations"]) == 15

    def test_filtered_result_storable_as_json_string(self):
        """json.dumps on FilteredResult with all extra fields succeeds."""
        filtered = FilteredResult(
            entities=[ExtractedEntity(text="Test", label="ORG", confidence=0.9)],
            relations=[],
            removed_entities=[ExtractedEntity(text="noise", label="MISC", confidence=0.1)],
            merged_entity_groups=[["A", "a"]],
            predicted_edges=[],
            validation_report={"test": True},
            linked_entities={"Test": {"uri": "http://example.com"}},
        )
        dumped = filtered.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)

        assert parsed["removed_entities"][0]["text"] == "noise"
        assert parsed["merged_entity_groups"] == [["A", "a"]]


class TestFileTrackingModelRoundtrips:
    """SourceFolder and PipelineCacheEntry dict round-trips."""

    def test_source_folder_dict_roundtrip(self):
        """SourceFolder -> dict -> SourceFolder preserves fields."""
        folder = make_source_folder(
            title="My Research Paper",
            source_type="document",
            original_filename="paper.pdf",
            file_size_bytes=12345,
            mime_type="application/pdf",
        )

        data = folder.model_dump()
        restored = SourceFolder(**data)

        assert restored.title == "My Research Paper"
        assert restored.source_id == folder.source_id
        assert restored.folder_path == folder.folder_path
        assert restored.original_filename == "paper.pdf"
        assert restored.file_size_bytes == 12345
        assert restored.mime_type == "application/pdf"
        assert restored.source_type == "document"

    def test_pipeline_cache_entry_dict_roundtrip(self):
        """PipelineCacheEntry -> dict -> PipelineCacheEntry preserves fields."""
        entry = make_cache_entry(
            pipeline_step="entity_extraction",
            version=3,
            config_hash="abc123def456",
            processing_time_seconds=12.5,
        )

        data = entry.model_dump()
        restored = PipelineCacheEntry(**data)

        assert restored.pipeline_step == "entity_extraction"
        assert restored.version == 3
        assert restored.config_hash == "abc123def456"
        assert restored.processing_time_seconds == 12.5
        assert restored.is_current is True
        assert restored.source_id == entry.source_id
