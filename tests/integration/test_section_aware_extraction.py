"""
Tests for section-aware entity extraction end-to-end flow.

Verifies that structural metadata (section_path, page_number, element_type)
flows from chunks through ExtractionWorkflow -> FilteringWorkflow -> persistence,
and that the incremental resolver and match candidate persistence work correctly.
"""

import pytest
from unittest.mock import AsyncMock, patch

from entity_filtering.config import FilteringConfig, IncrementalResolutionConfig
from entity_filtering.resolution.incremental_resolver import (
    EntityCluster,
    IncrementalResolver,
)
from entity_filtering.workflow import FilteringWorkflow
from shared.models.extraction import (
    ExtractionContext,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    FilteredResult,
    MatchCandidate,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def section_aware_extraction() -> ExtractionResult:
    """Extraction result with section context on entities and relations."""
    ctx_intro = ExtractionContext(
        section_heading="Inleiding",
        section_path=["Hoofdstuk 1", "Inleiding"],
        section_level=2,
        page_number=1,
        element_type="paragraph",
        surrounding_text="Het ministerie van BZK is verantwoordelijk voor...",
        source_document="source:doc1",
    )
    ctx_beleid = ExtractionContext(
        section_heading="Beleidskader",
        section_path=["Hoofdstuk 2", "Beleidskader"],
        section_level=2,
        page_number=5,
        element_type="paragraph",
        surrounding_text="De Europese Commissie heeft in samenwerking met...",
        source_document="source:doc1",
    )
    ctx_table = ExtractionContext(
        section_heading="Financieel overzicht",
        section_path=["Hoofdstuk 3", "Financieel overzicht"],
        section_level=2,
        page_number=8,
        element_type="table",
        surrounding_text="Begrotingspost | Bedrag | Toelichting",
        source_document="source:doc1",
    )

    entities = [
        ExtractedEntity(
            text="BZK",
            label="ORG",
            confidence=0.95,
            source_chunk_id="chunk:1",
            extraction_context=ctx_intro,
        ),
        ExtractedEntity(
            text="Ministerie van Binnenlandse Zaken",
            label="ORG",
            confidence=0.88,
            source_chunk_id="chunk:2",
            extraction_context=ctx_beleid,
        ),
        ExtractedEntity(
            text="Europese Commissie",
            label="ORG",
            confidence=0.92,
            source_chunk_id="chunk:2",
            extraction_context=ctx_beleid,
        ),
        ExtractedEntity(
            text="Tweede Kamer",
            label="ORG",
            confidence=0.90,
            source_chunk_id="chunk:3",
            extraction_context=ctx_table,
        ),
        ExtractedEntity(
            text="Digitalisering",
            label="CONCEPT",
            confidence=0.85,
            source_chunk_id="chunk:1",
            extraction_context=ctx_intro,
        ),
        ExtractedEntity(
            text="Cybersecurity",
            label="CONCEPT",
            confidence=0.87,
            source_chunk_id="chunk:2",
            extraction_context=ctx_beleid,
        ),
    ]

    relations = [
        ExtractedRelation(
            source_entity="BZK",
            target_entity="Digitalisering",
            relation_type="VERANTWOORDELIJK_VOOR",
            confidence=0.88,
            source_chunk_id="chunk:1",
            extraction_context=ctx_intro,
        ),
        ExtractedRelation(
            source_entity="Europese Commissie",
            target_entity="Cybersecurity",
            relation_type="BEHEERT",
            confidence=0.82,
            source_chunk_id="chunk:2",
            extraction_context=ctx_beleid,
        ),
    ]

    return ExtractionResult(
        entities=entities,
        relations=relations,
        metadata={
            "source": "test_section_aware",
            "ontology_name": "dutch_policy",
        },
    )


@pytest.fixture
def workflow() -> FilteringWorkflow:
    """FilteringWorkflow with default config (stages 1-4 only)."""
    return FilteringWorkflow(FilteringConfig())


# ---------------------------------------------------------------------------
# Section context flow tests
# ---------------------------------------------------------------------------


class TestSectionContextFlow:
    """Section metadata flows through filtering unchanged."""

    async def test_extraction_context_preserved_through_filtering(
        self, workflow, section_aware_extraction
    ):
        """ExtractionContext survives noise filter, normalization, dedup."""
        result = await workflow.process(section_aware_extraction)

        entities_with_context = [
            e for e in result.entities if e.extraction_context is not None
        ]
        assert len(entities_with_context) >= 4, (
            f"Expected at least 4 entities with context, got {len(entities_with_context)}"
        )

    async def test_section_heading_preserved(
        self, workflow, section_aware_extraction
    ):
        """Section headings survive filtering."""
        result = await workflow.process(section_aware_extraction)

        headings = {
            e.extraction_context.section_heading
            for e in result.entities
            if e.extraction_context
        }
        assert "Inleiding" in headings
        assert "Beleidskader" in headings

    async def test_page_numbers_preserved(
        self, workflow, section_aware_extraction
    ):
        """Page numbers survive filtering."""
        result = await workflow.process(section_aware_extraction)

        pages = {
            e.extraction_context.page_number
            for e in result.entities
            if e.extraction_context and e.extraction_context.page_number is not None
        }
        assert 1 in pages
        assert 5 in pages

    async def test_element_type_preserved(
        self, workflow, section_aware_extraction
    ):
        """Element types survive filtering."""
        result = await workflow.process(section_aware_extraction)

        types = {
            e.extraction_context.element_type
            for e in result.entities
            if e.extraction_context and e.extraction_context.element_type
        }
        assert "paragraph" in types
        assert "table" in types

    async def test_section_path_preserved(
        self, workflow, section_aware_extraction
    ):
        """Full section paths survive filtering."""
        result = await workflow.process(section_aware_extraction)

        paths = [
            e.extraction_context.section_path
            for e in result.entities
            if e.extraction_context and e.extraction_context.section_path
        ]
        flat = [item for sublist in paths for item in sublist]
        assert "Hoofdstuk 1" in flat
        assert "Hoofdstuk 2" in flat

    async def test_relation_context_preserved(
        self, workflow, section_aware_extraction
    ):
        """Relations retain their extraction context."""
        result = await workflow.process(section_aware_extraction)

        rels_with_context = [
            r for r in result.relations if r.extraction_context is not None
        ]
        assert len(rels_with_context) >= 1

    async def test_context_survives_serialization(
        self, workflow, section_aware_extraction
    ):
        """Round-trip through JSON preserves context."""
        result = await workflow.process(section_aware_extraction)

        json_str = result.model_dump_json()
        restored = FilteredResult.model_validate_json(json_str)

        original_contexts = [
            e.extraction_context for e in result.entities if e.extraction_context
        ]
        restored_contexts = [
            e.extraction_context for e in restored.entities if e.extraction_context
        ]
        assert len(original_contexts) == len(restored_contexts)

        for orig, rest in zip(original_contexts, restored_contexts):
            assert orig.section_heading == rest.section_heading
            assert orig.page_number == rest.page_number


# ---------------------------------------------------------------------------
# Incremental resolver tests
# ---------------------------------------------------------------------------


class TestIncrementalResolver:
    """IncrementalResolver assigns entities to clusters correctly."""

    def test_new_entities_create_clusters(self):
        """Entities without existing clusters create singletons."""
        resolver = IncrementalResolver(similarity_threshold=0.85)
        entities = [
            {"text": "BZK", "embedding": [1.0, 0.0, 0.0]},
            {"text": "EZK", "embedding": [0.0, 1.0, 0.0]},
            {"text": "VWS", "embedding": [0.0, 0.0, 1.0]},
        ]
        clusters, assignments = resolver.resolve_incremental(entities, [])

        assert len(clusters) == 3
        assert all(a["action"] == "new" for a in assignments)

    def test_similar_entities_assigned_to_cluster(self):
        """Entity with high similarity to existing cluster gets assigned."""
        resolver = IncrementalResolver(similarity_threshold=0.85)

        existing = [
            EntityCluster(
                cluster_id="cluster_bzk",
                members=[{"text": "BZK", "embedding": [1.0, 0.0, 0.0]}],
            )
        ]
        new_entities = [
            {"text": "Binnenlandse Zaken", "embedding": [0.99, 0.05, 0.0]},
        ]

        clusters, assignments = resolver.resolve_incremental(new_entities, existing)

        assert assignments[0]["action"] == "assigned"
        assert assignments[0]["cluster_id"] == "cluster_bzk"

    def test_dissimilar_entity_creates_new_cluster(self):
        """Entity dissimilar to all clusters creates a new one."""
        resolver = IncrementalResolver(similarity_threshold=0.85)

        existing = [
            EntityCluster(
                cluster_id="cluster_bzk",
                members=[{"text": "BZK", "embedding": [1.0, 0.0, 0.0]}],
            )
        ]
        new_entities = [
            {"text": "Cybersecurity", "embedding": [0.0, 0.0, 1.0]},
        ]

        clusters, assignments = resolver.resolve_incremental(new_entities, existing)

        assert assignments[0]["action"] == "new"
        assert len(clusters) == 2

    def test_entity_without_embedding_creates_singleton(self):
        """Entity with no embedding gets a singleton cluster."""
        resolver = IncrementalResolver()
        entities = [{"text": "NoEmbed"}]

        clusters, assignments = resolver.resolve_incremental(entities, [])

        assert len(clusters) == 1
        assert assignments[0]["action"] == "new"

    def test_repair_splits_low_coherence_cluster(self):
        """Cluster with low internal coherence gets split."""
        resolver = IncrementalResolver(coherence_threshold=0.95)

        cluster = EntityCluster(
            cluster_id="mixed",
            members=[
                {"text": "A", "embedding": [1.0, 0.0, 0.0]},
                {"text": "B", "embedding": [0.0, 1.0, 0.0]},
            ],
        )

        repaired, report = resolver.repair_clusters([cluster])

        assert report["splits"] >= 1
        assert len(repaired) >= 2

    def test_repair_merges_similar_clusters(self):
        """Two clusters with nearly identical centroids get merged."""
        resolver = IncrementalResolver(merge_threshold=0.90)

        c1 = EntityCluster(
            cluster_id="c1",
            members=[{"text": "A", "embedding": [1.0, 0.0, 0.0]}],
        )
        c2 = EntityCluster(
            cluster_id="c2",
            members=[{"text": "B", "embedding": [0.99, 0.05, 0.0]}],
        )

        repaired, report = resolver.repair_clusters([c1, c2])

        assert report["merges"] >= 1
        assert len(repaired) == 1


# ---------------------------------------------------------------------------
# Incremental resolver in workflow
# ---------------------------------------------------------------------------


class TestIncrementalResolverInWorkflow:
    """Incremental resolver wired into FilteringWorkflow."""

    async def test_workflow_with_incremental_resolution(
        self, section_aware_extraction
    ):
        """Workflow runs incremental resolution when enabled."""
        config = FilteringConfig(
            incremental_resolution=IncrementalResolutionConfig(
                enabled=True,
                similarity_threshold=0.85,
                repair_enabled=False,
            ),
        )
        wf = FilteringWorkflow(config)
        result = await wf.process(section_aware_extraction)

        assert "filtering" in result.metadata
        filtering = result.metadata["filtering"]
        assert "incremental_resolution" in filtering

        ir = filtering["incremental_resolution"]
        assert ir["assignments"] > 0
        # All entities should be new (no existing clusters)
        assert ir["new_clusters"] == ir["assignments"]

    async def test_workflow_incremental_tags_entities(
        self, section_aware_extraction
    ):
        """Entities get cluster_id in properties after incremental resolution."""
        config = FilteringConfig(
            incremental_resolution=IncrementalResolutionConfig(
                enabled=True,
                repair_enabled=False,
            ),
        )
        wf = FilteringWorkflow(config)
        result = await wf.process(section_aware_extraction)

        entities_with_cluster = [
            e for e in result.entities
            if e.properties.get("cluster_id")
        ]
        assert len(entities_with_cluster) > 0

    async def test_workflow_incremental_with_existing_clusters(
        self, section_aware_extraction
    ):
        """Entities assigned to pre-existing clusters when similar."""
        config = FilteringConfig(
            incremental_resolution=IncrementalResolutionConfig(
                enabled=True,
                repair_enabled=False,
                similarity_threshold=0.1,  # very low to force assignments
            ),
        )
        wf = FilteringWorkflow(config)

        # Pre-populate with a cluster
        existing = EntityCluster(
            cluster_id="cluster_existing",
            members=[{"text": "Existing", "embedding": [1.0] * 100}],
        )
        wf.set_existing_clusters([existing])

        # Add embeddings to entities so they can match
        for e in section_aware_extraction.entities:
            e.properties["embedding"] = [1.0] * 100

        result = await wf.process(section_aware_extraction)

        ir = result.metadata["filtering"]["incremental_resolution"]
        # With threshold=0.1, all entities with embeddings should be assigned
        assert ir["assigned"] > 0


# ---------------------------------------------------------------------------
# Match candidate persistence tests
# ---------------------------------------------------------------------------


class TestMatchCandidatePersistence:
    """Match candidates are stored to resolution_log correctly."""

    async def test_persist_match_candidates(self):
        """Match candidates stored with all provenance fields."""
        from app_main.services.entity_persistence_service import EntityPersistenceService

        svc = EntityPersistenceService()

        candidates = [
            {
                "entity_a_text": "BZK",
                "entity_b_text": "Binnenlandse Zaken",
                "entity_a_label": "ORG",
                "entity_b_label": "ORG",
                "match": True,
                "confidence": 0.92,
                "match_method": "llm_match",
                "match_reasoning": "Abbreviation for Ministerie van Binnenlandse Zaken",
                "iterations": 1,
                "source_section_a": "Inleiding",
                "source_section_b": "Beleidskader",
                "source_document_a": "source:doc1",
                "source_document_b": "source:doc1",
                "matched_by_model": "qwen3.5:35b-a3b",
                "status": "auto_accepted",
            },
            {
                "entity_a_text": "EC",
                "entity_b_text": "Europese Commissie",
                "entity_a_label": "ORG",
                "entity_b_label": "ORG",
                "match": True,
                "confidence": 0.55,
                "match_method": "llm_match",
                "match_reasoning": "Uncertain match - may refer to different EC",
                "iterations": 2,
                "status": "pending",
            },
        ]

        queries = []

        async def capture(query, params=None, config=None):
            queries.append((query, params))
            return []

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            side_effect=capture,
        ):
            stored = await svc.persist_match_candidates("source:doc1", candidates)

        assert stored == 2
        assert len(queries) == 2

        # Verify first candidate has all provenance fields
        p = queries[0][1]
        assert p["entity_a_text"] == "BZK"
        assert p["entity_b_text"] == "Binnenlandse Zaken"
        assert p["match"] is True
        assert p["confidence"] == 0.92
        assert p["match_method"] == "llm_match"
        assert p["source_section_a"] == "Inleiding"
        assert p["matched_by_model"] == "qwen3.5:35b-a3b"
        assert p["status"] == "auto_accepted"

    async def test_persist_filtered_result_with_candidates(self):
        """persist_filtered_result also stores match candidates."""
        from app_main.services.entity_persistence_service import EntityPersistenceService

        svc = EntityPersistenceService()

        async def mock_query(query, params=None, config=None):
            return []

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            side_effect=mock_query,
        ):
            result = await svc.persist_filtered_result(
                source_id="source:doc1",
                entities=[{"text": "BZK", "label": "ORG", "confidence": 0.9, "properties": {}}],
                relations=[],
                match_candidates=[
                    {
                        "entity_a_text": "BZK",
                        "entity_b_text": "Binnenlandse Zaken",
                        "match": True,
                        "confidence": 0.9,
                        "match_method": "llm_match",
                        "status": "auto_accepted",
                    },
                ],
            )

        assert result["candidates_stored"] == 1


# ---------------------------------------------------------------------------
# Resolution log service tests
# ---------------------------------------------------------------------------


class TestResolutionLogService:
    """ResolutionLogService query methods work correctly."""

    async def test_list_candidates_with_filters(self):
        """List candidates respects status filter."""
        from app_main.services.resolution_log_service import ResolutionLogService

        svc = ResolutionLogService()

        async def mock_query(query, params=None, config=None):
            if "SELECT *" in query:
                return [[{"id": "rl:1", "status": "pending"}]]
            if "count()" in query:
                return [[{"total": 1}]]
            return [[]]

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            side_effect=mock_query,
        ) as mock:
            items, total = await svc.list_candidates(status="pending")

        assert total == 1
        assert len(items) == 1
        # Check that WHERE clause includes status filter
        call_args = mock.call_args_list[0]
        assert "status = $status" in call_args[0][0]

    async def test_update_status(self):
        """Update status sets reviewed_at timestamp."""
        from app_main.services.resolution_log_service import ResolutionLogService

        svc = ResolutionLogService()

        async def mock_query(query, params=None, config=None):
            return [[{"id": "rl:1", "status": "accepted", "reviewed_at": "2026-04-07T12:00:00Z"}]]

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            side_effect=mock_query,
        ):
            result = await svc.update_status("rl:1", "accepted")

        assert result["status"] == "accepted"
        assert result["reviewed_at"] is not None

    async def test_get_stats(self):
        """Stats returns aggregated counts."""
        from app_main.services.resolution_log_service import ResolutionLogService

        svc = ResolutionLogService()

        async def mock_query(query, params=None, config=None):
            return [[{
                "total": 10,
                "matches": 6,
                "pending": 3,
                "accepted": 4,
                "rejected": 1,
                "auto_accepted": 2,
            }]]

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            side_effect=mock_query,
        ):
            stats = await svc.get_stats()

        assert stats["total"] == 10
        assert stats["pending"] == 3

    async def test_bulk_update_status(self):
        """Bulk update processes all IDs."""
        from app_main.services.resolution_log_service import ResolutionLogService

        svc = ResolutionLogService()

        async def mock_query(query, params=None, config=None):
            return [[{"id": params.get("id"), "status": params.get("status")}]]

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            side_effect=mock_query,
        ):
            updated = await svc.bulk_update_status(["rl:1", "rl:2", "rl:3"], "accepted")

        assert updated == 3
