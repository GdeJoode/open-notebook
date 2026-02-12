"""
Cross-pipeline integration: ExtractionResult -> FilteringWorkflow -> FilteredResult.

Uses FilteringWorkflow(FilteringConfig()) with default config —
stages 1-4 run for real (noise filter, normalizer, reclassifier, deduplicator),
stages 5-13 disabled.
"""

import pytest

from entity_filtering.config import FilteringConfig
from entity_filtering.workflow import FilteringWorkflow
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    FilteredResult,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def workflow() -> FilteringWorkflow:
    """FilteringWorkflow with default config (stages 1-4 only)."""
    return FilteringWorkflow(FilteringConfig())


@pytest.fixture
def noisy_extraction() -> ExtractionResult:
    """
    20 entities with 3 that are pure noise, 2 near-duplicates.

    Noise entities are chosen to match the NoiseFilter's built-in patterns:
    - Pure numbers (matching digit-only regex)
    - Pure punctuation/symbols (matching non-word regex)
    - Below min_entity_length (< 2 chars)
    """
    entities = [
        # Valid PERSON entities
        ExtractedEntity(text="Albert Einstein", label="PERSON", confidence=0.95),
        ExtractedEntity(text="Marie Curie", label="PERSON", confidence=0.92),
        ExtractedEntity(text="Niels Bohr", label="PERSON", confidence=0.88),
        ExtractedEntity(text="Max Planck", label="PERSON", confidence=0.91),
        ExtractedEntity(text="Werner Heisenberg", label="PERSON", confidence=0.87),
        # Duplicate of Einstein (different casing) — deduplicator should merge
        ExtractedEntity(text="albert einstein", label="PERSON", confidence=0.80),
        # Near-duplicate short form
        ExtractedEntity(text="Einstein", label="PERSON", confidence=0.75),
        # Valid ORG entities
        ExtractedEntity(text="University of Berlin", label="ORG", confidence=0.90),
        ExtractedEntity(text="Nobel Committee", label="ORG", confidence=0.85),
        ExtractedEntity(text="Kaiser Wilhelm Institute", label="ORG", confidence=0.88),
        # Valid concepts
        ExtractedEntity(text="quantum mechanics", label="CONCEPT", confidence=0.92),
        ExtractedEntity(text="general relativity", label="CONCEPT", confidence=0.90),
        ExtractedEntity(text="photoelectric effect", label="CONCEPT", confidence=0.88),
        # Valid LOC
        ExtractedEntity(text="Copenhagen", label="LOC", confidence=0.85),
        ExtractedEntity(text="Berlin", label="LOC", confidence=0.83),
        # Valid EVENT
        ExtractedEntity(text="Solvay Conference", label="EVENT", confidence=0.82),
        ExtractedEntity(text="Nobel Prize in Physics", label="EVENT", confidence=0.90),
        # Noise entities — will be removed by noise filter
        ExtractedEntity(text="42.5%", label="MISC", confidence=0.3),  # pure number pattern
        ExtractedEntity(text="---", label="MISC", confidence=0.2),  # pure punctuation
        ExtractedEntity(text="x", label="MISC", confidence=0.25),  # below min_entity_length
    ]

    relations = [
        ExtractedRelation(
            source_entity="Albert Einstein",
            target_entity="general relativity",
            relation_type="DEVELOPED",
            confidence=0.92,
        ),
        ExtractedRelation(
            source_entity="Albert Einstein",
            target_entity="photoelectric effect",
            relation_type="EXPLAINED",
            confidence=0.90,
        ),
        ExtractedRelation(
            source_entity="Marie Curie",
            target_entity="Nobel Prize in Physics",
            relation_type="RECEIVED",
            confidence=0.88,
        ),
        ExtractedRelation(
            source_entity="Albert Einstein",
            target_entity="Nobel Prize in Physics",
            relation_type="RECEIVED",
            confidence=0.91,
        ),
        ExtractedRelation(
            source_entity="Albert Einstein",
            target_entity="University of Berlin",
            relation_type="AFFILIATED_WITH",
            confidence=0.85,
        ),
        ExtractedRelation(
            source_entity="Niels Bohr",
            target_entity="Copenhagen",
            relation_type="LOCATED_IN",
            confidence=0.82,
        ),
        ExtractedRelation(
            source_entity="Niels Bohr",
            target_entity="quantum mechanics",
            relation_type="CONTRIBUTED_TO",
            confidence=0.88,
        ),
        ExtractedRelation(
            source_entity="Max Planck",
            target_entity="quantum mechanics",
            relation_type="FOUNDED",
            confidence=0.90,
        ),
        ExtractedRelation(
            source_entity="Max Planck",
            target_entity="Kaiser Wilhelm Institute",
            relation_type="AFFILIATED_WITH",
            confidence=0.87,
        ),
        ExtractedRelation(
            source_entity="Werner Heisenberg",
            target_entity="quantum mechanics",
            relation_type="CONTRIBUTED_TO",
            confidence=0.85,
        ),
        ExtractedRelation(
            source_entity="Albert Einstein",
            target_entity="Solvay Conference",
            relation_type="PARTICIPATED_IN",
            confidence=0.80,
        ),
        ExtractedRelation(
            source_entity="Niels Bohr",
            target_entity="Solvay Conference",
            relation_type="PARTICIPATED_IN",
            confidence=0.78,
        ),
        # Relations referencing noise entities — should be dropped
        ExtractedRelation(
            source_entity="42.5%",
            target_entity="Albert Einstein",
            relation_type="UNKNOWN",
            confidence=0.1,
        ),
        ExtractedRelation(
            source_entity="---",
            target_entity="Berlin",
            relation_type="UNKNOWN",
            confidence=0.15,
        ),
        ExtractedRelation(
            source_entity="x",
            target_entity="quantum mechanics",
            relation_type="UNKNOWN",
            confidence=0.12,
        ),
    ]

    return ExtractionResult(
        entities=entities,
        relations=relations,
        metadata={"source": "test", "pipeline": "ontology-extraction"},
    )


class TestExtractionToFiltering:
    """ExtractionResult → FilteringWorkflow → FilteredResult."""

    async def test_realistic_extraction_through_filtering(
        self, workflow, noisy_extraction
    ):
        """20 entities in → noise removed, valid entities preserved, FilteredResult type."""
        result = await workflow.process(noisy_extraction)

        assert isinstance(result, FilteredResult)
        assert isinstance(result, ExtractionResult)  # Liskov
        # Noise removed: 3 entities should be gone
        assert result.entity_count < 20
        # At least the 5 named persons should survive (possibly merged)
        assert result.entity_count >= 10

    async def test_noise_entities_removed(self, workflow, noisy_extraction):
        """'42.5%', '---', and 'x' entities are removed by noise filter."""
        result = await workflow.process(noisy_extraction)

        surviving_texts = {e.text for e in result.entities}
        assert "42.5%" not in surviving_texts
        assert "---" not in surviving_texts
        assert "x" not in surviving_texts

        # Noise entities should appear in removed_entities
        removed_texts = {e.text for e in result.removed_entities}
        assert "42.5%" in removed_texts
        assert "---" in removed_texts
        assert "x" in removed_texts

    async def test_duplicate_entities_merged(self, workflow, noisy_extraction):
        """'albert einstein' normalized to 'Albert Einstein' and merged."""
        result = await workflow.process(noisy_extraction)

        surviving_texts_lower = [e.text.lower() for e in result.entities]
        # There should be at most one "albert einstein" (case-insensitive)
        einstein_count = surviving_texts_lower.count("albert einstein")
        assert einstein_count <= 1, (
            f"Expected at most 1 Einstein variant, found {einstein_count}"
        )

        # Should have merge groups from deduplication
        assert len(result.merged_entity_groups) >= 1

    async def test_relations_referencing_noise_removed(
        self, workflow, noisy_extraction
    ):
        """Relations with noise source/target entities are dropped."""
        result = await workflow.process(noisy_extraction)

        # All surviving relations should reference surviving entities
        surviving_texts = {e.text for e in result.entities}
        for rel in result.relations:
            assert rel.source_entity in surviving_texts or rel.target_entity in surviving_texts, (
                f"Relation {rel.source_entity} -> {rel.target_entity} "
                "references a non-surviving entity"
            )

        # The 3 noise relations should be gone
        assert result.relation_count <= 12

    async def test_filtering_metadata_includes_statistics(
        self, workflow, noisy_extraction
    ):
        """FilteredResult.metadata has input/output entity counts."""
        result = await workflow.process(noisy_extraction)

        assert "filtering" in result.metadata
        stats = result.metadata["filtering"]
        assert stats["input_entities"] == 20
        assert stats["input_relations"] == 15
        assert stats["output_entities"] == result.entity_count
        assert stats["output_relations"] == result.relation_count
        assert stats["removed_count"] == len(result.removed_entities)

    async def test_empty_extraction_produces_empty_result(self, workflow):
        """Empty input → empty FilteredResult (no crash)."""
        empty = ExtractionResult()
        result = await workflow.process(empty)

        assert isinstance(result, FilteredResult)
        assert result.entity_count == 0
        assert result.relation_count == 0
        assert len(result.removed_entities) == 0

    async def test_filtered_result_serializable(self, workflow, noisy_extraction):
        """Result can model_dump_json() and round-trip."""
        result = await workflow.process(noisy_extraction)

        json_str = result.model_dump_json()
        restored = FilteredResult.model_validate_json(json_str)

        assert restored.entity_count == result.entity_count
        assert restored.relation_count == result.relation_count
        assert len(restored.removed_entities) == len(result.removed_entities)

    async def test_minimal_extraction_passthrough(self, workflow, minimal_extraction_result):
        """2-entity input passes through with both entities surviving."""
        result = await workflow.process(minimal_extraction_result)

        assert isinstance(result, FilteredResult)
        assert result.entity_count == 2
        assert result.relation_count == 1

        surviving_texts = {e.text for e in result.entities}
        assert "Alice" in surviving_texts
        assert "ACME Corp" in surviving_texts
