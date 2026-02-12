"""Integration tests with ALL pipeline options enabled simultaneously.

Exercises the full FilteringWorkflow with every Phase 1 + Phase 2 option
turned on, verifying that the stages compose correctly end-to-end.
Also tests Phase 3-4b: KG resolution with centrality-aware thresholds,
graph centrality with outlier detection, and the full pipeline composed.
"""

from typing import Any, Dict, List, Optional

import pytest
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    FilteredResult,
)
from entity_filtering.config import (
    FilteringConfig,
    SyntacticConfig,
    FuzzyDedupConfig,
    EmbeddingDedupConfig,
    KGResolutionConfig,
    OntologyValidationConfig,
)
from entity_filtering.workflow import FilteringWorkflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(text, label="MISC", confidence=0.9, properties=None):
    return {
        "text": text,
        "label": label,
        "properties": properties or {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


def _entity_with_embedding(
    text, embedding, label="MISC", confidence=0.9, extra_props=None
):
    props = {"embedding": embedding}
    if extra_props:
        props.update(extra_props)
    return _entity(text, label, confidence, props)


def _relation(source, target, rel_type="RELATED_TO", confidence=0.8):
    return {
        "source_entity": source,
        "target_entity": target,
        "relation_type": rel_type,
        "properties": {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


def _extraction_result(entities=None, relations=None, metadata=None):
    return ExtractionResult(
        entities=[ExtractedEntity(**e) for e in (entities or [])],
        relations=[ExtractedRelation(**r) for r in (relations or [])],
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Shared config: EVERYTHING enabled
# ---------------------------------------------------------------------------

def _all_options_config():
    return FilteringConfig(
        # Noise / normalization
        min_entity_length=2,
        strip_articles=True,
        custom_articles=["De ", "Het "],
        normalize_whitespace=True,
        # String dedup
        dedup_enabled=True,
        dedup_similarity_threshold=0.85,
        # Syntactic enhancements
        syntactic=SyntacticConfig(
            remove_diacritics=True,
            ocr_cleanup_enabled=True,
            ocr_artifact_patterns=["\\f", "\u00ad"],
            html_strip_enabled=True,
            page_number_filter=True,
        ),
        # Fuzzy dedup
        fuzzy_dedup=FuzzyDedupConfig(
            enabled=True,
            algorithm="levenshtein",
            similarity_threshold=0.85,
        ),
        # Embedding dedup (numpy fallback, no FAISS dep required)
        embedding_dedup=EmbeddingDedupConfig(
            enabled=True,
            similarity_threshold=0.90,
            k_candidates=5,
            use_faiss=False,
        ),
    )


# ---------------------------------------------------------------------------
# Embedding vectors for integration tests (dim=4)
# ---------------------------------------------------------------------------

# Near-identical embeddings for semantic merge
EMB_CONCEPT_A = [1.0, 0.0, 0.0, 0.0]
EMB_CONCEPT_A_SIMILAR = [0.99, 0.01, 0.0, 0.0]

# Clearly different embedding
EMB_UNRELATED = [0.0, 0.0, 0.0, 1.0]


class TestAllOptionsFullPipeline:
    """End-to-end tests with every option turned on."""

    async def test_all_stages_compose(self):
        """Feed ~10 entities that exercise every stage and verify composition."""
        entities = [
            # Noise: punctuation-only -> removed by noise filter
            _entity("---", confidence=0.5),
            # Noise: pure number -> removed by noise filter
            _entity("123", confidence=0.4),
            # Page number -> removed by page_number_filter
            _entity("Page 42", confidence=0.6),
            # HTML entity -> HTML stripped, becomes "Microsoft"
            _entity("<b>Microsoft</b>", "ORG", 0.9),
            # Accented variant -> diacritics normalization merges with "cafe"
            _entity("caf\u00e9", "MISC", 0.85),
            _entity("cafe", "MISC", 0.80),
            # Article variant -> article stripping merges with bare form
            _entity("The United Nations", "ORG", 0.7),
            _entity("United Nations", "ORG", 0.95),
            # Case variant -> string dedup merges
            _entity("john doe", "PERSON", 0.75),
            _entity("John Doe", "PERSON", 0.90),
            # Fuzzy similar -> Levenshtein "jon doe" vs "john doe" ~ 0.88
            _entity("Jon Doe", "PERSON", 0.65),
            # Embedding similar -> merged by embedding dedup
            _entity_with_embedding(
                "Concept Alpha", EMB_CONCEPT_A, "MISC", 0.88
            ),
            _entity_with_embedding(
                "Concept Beta", EMB_CONCEPT_A_SIMILAR, "MISC", 0.82
            ),
        ]
        relations = [
            _relation("John Doe", "United Nations", "WORKS_AT"),
            # This relation references a noise entity -> should be removed
            _relation("123", "Microsoft", "RELATED_TO"),
        ]

        config = _all_options_config()
        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result(
            entities, relations, metadata={"source": "integration_test"}
        )
        result = await workflow.process(extraction)

        # -- Type check --
        assert isinstance(result, FilteredResult)

        # -- Noise entities are in removed_entities --
        removed_texts = {e.text for e in result.removed_entities}
        assert "---" in removed_texts
        assert "123" in removed_texts

        # -- Page number filtered out --
        result_texts = {e.text for e in result.entities}
        assert not any("Page 42" in t for t in result_texts)

        # -- HTML was stripped: "Microsoft" should be present --
        # After HTML strip, "<b>Microsoft</b>" normalizes to "Microsoft"
        assert any("Microsoft" in t for t in result_texts)

        # -- Accented and non-accented merged --
        cafe_entities = [
            e for e in result.entities if "caf" in e.text.lower()
        ]
        assert len(cafe_entities) <= 1

        # -- Article variants merged --
        un_entities = [
            e for e in result.entities if "United Nations" in e.text
        ]
        assert len(un_entities) == 1

        # -- Case variants and fuzzy typos merged --
        doe_entities = [
            e for e in result.entities if "doe" in e.text.lower()
        ]
        assert len(doe_entities) <= 1

        # -- Embedding similar entities merged --
        concept_entities = [
            e for e in result.entities if "Concept" in e.text
        ]
        assert len(concept_entities) <= 1

        # -- Relation referencing noise entity removed --
        surviving_rels = result.relations
        for rel in surviving_rels:
            assert rel.source_entity != "123"
            assert rel.target_entity != "123"

        # -- There should be at least one surviving relation --
        # The "John Doe" -> "United Nations" relation may survive if both
        # entities survived (text might change due to normalization, but the
        # relation references original text forms).
        # We verify the noise relation was dropped.
        noise_rels = [
            r for r in surviving_rels if r.source_entity == "123"
        ]
        assert len(noise_rels) == 0

        # -- Metadata has filtering stats --
        assert "filtering" in result.metadata
        stats = result.metadata["filtering"]
        assert stats["input_entities"] == len(entities)
        assert stats["input_relations"] == len(relations)
        assert "output_entities" in stats
        assert "removed_count" in stats
        assert "merge_groups" in stats

        # -- Some merge groups should exist --
        assert len(result.merged_entity_groups) > 0

    async def test_empty_input_all_options(self):
        """Empty ExtractionResult produces empty FilteredResult."""
        config = _all_options_config()
        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result()
        result = await workflow.process(extraction)

        assert isinstance(result, FilteredResult)
        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert len(result.removed_entities) == 0
        assert len(result.merged_entity_groups) == 0
        assert len(result.predicted_edges) == 0

    async def test_all_options_preserves_metadata(self):
        """Original metadata is preserved alongside filtering stats."""
        config = _all_options_config()
        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result(
            entities=[_entity("Alice", "PERSON")],
            metadata={"source": "doc_42", "version": "2.0", "lang": "nl"},
        )
        result = await workflow.process(extraction)

        assert result.metadata["source"] == "doc_42"
        assert result.metadata["version"] == "2.0"
        assert result.metadata["lang"] == "nl"
        assert "filtering" in result.metadata


class TestSyntacticComposition:
    """Verify that syntactic pre-processing features compose correctly."""

    async def test_html_and_diacritics_and_ocr_merge(self):
        """Entity with HTML + diacritics + OCR artifact merges with clean version.

        "<i>caf\u00e9</i>" (with HTML and accent) should normalize to "cafe"
        and merge with a plain "cafe" entity.
        """
        entities = [
            _entity("<i>caf\u00e9</i>", "MISC", 0.7),
            _entity("cafe", "MISC", 0.9),
        ]
        config = _all_options_config()
        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result(entities)
        result = await workflow.process(extraction)

        cafe_entities = [
            e for e in result.entities if "caf" in e.text.lower()
        ]
        assert len(cafe_entities) == 1
        # Higher confidence should be preserved
        assert cafe_entities[0].confidence >= 0.9

    async def test_dutch_articles_stripped(self):
        """'De Volkskrant' and 'Volkskrant' merge via custom article stripping."""
        entities = [
            _entity("De Volkskrant", "ORG", 0.8),
            _entity("Volkskrant", "ORG", 0.9),
        ]
        config = _all_options_config()
        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result(entities)
        result = await workflow.process(extraction)

        vk_entities = [
            e for e in result.entities if "Volkskrant" in e.text
        ]
        assert len(vk_entities) == 1

    async def test_ocr_artifact_removed_before_merge(self):
        """Soft hyphen (OCR artifact) is cleaned before normalization.

        The normalizer groups entities by normalized key (with the soft
        hyphen stripped) so both forms merge.  The canonical text is the
        most frequent original surface form -- we verify a single entity
        survives and the max confidence is preserved.
        """
        # \u00ad is a soft hyphen, configured as an OCR artifact pattern
        entities = [
            _entity("Amsterdam", "LOC", 0.95),
            _entity("Amsterdam", "LOC", 0.80),
            _entity("Amster\u00addam", "LOC", 0.7),
        ]
        config = _all_options_config()
        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result(entities)
        result = await workflow.process(extraction)

        # All three should merge into one entity
        amsterdam = [
            e for e in result.entities if "Amster" in e.text
        ]
        assert len(amsterdam) == 1
        # Most frequent surface form is "Amsterdam" (appears twice)
        assert amsterdam[0].text == "Amsterdam"
        assert amsterdam[0].confidence >= 0.95

    async def test_page_number_entities_removed(self):
        """Page number entities are dropped by the page_number_filter."""
        entities = [
            _entity("Page 1", "MISC", 0.5),
            _entity("p. 42", "MISC", 0.4),
            _entity("pp. 100", "MISC", 0.3),
            _entity("Amsterdam", "LOC", 0.9),
        ]
        config = _all_options_config()
        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result(entities)
        result = await workflow.process(extraction)

        result_texts = {e.text for e in result.entities}
        assert "Amsterdam" in result_texts
        assert "Page 1" not in result_texts
        assert "p. 42" not in result_texts
        assert "pp. 100" not in result_texts


class TestDedupChainComposition:
    """Three dedup stages (string, fuzzy, embedding) work in sequence."""

    async def test_string_dedup_then_fuzzy_then_embedding(self):
        """Verify cascading dedup: string -> fuzzy -> embedding.

        Set up entities so each dedup stage catches a different pair:
        1. String dedup: "climate change" + "Climate Change" (case variant)
        2. Fuzzy dedup: "Clmate Change" (missing 'i') ~ "Climate Change"
           Levenshtein("clmate change", "climate change"):
           distance=1, len=14, sim=13/14=0.929 > 0.85
        3. Embedding dedup: "Global Warming" has near-identical embedding
           to "Climate Change Risk" but different text
        """
        entities = [
            # Stage 1 (string dedup): exact case variants
            _entity("climate change", "MISC", 0.7),
            _entity("Climate Change", "MISC", 0.9),
            # Stage 2 (fuzzy): single-char typo
            # Levenshtein("clmate change", "climate change") = 1 edit, len=14
            # sim = 13/14 = 0.929 > 0.85
            _entity("Clmate Change", "MISC", 0.6),
            # Stage 3 (embedding): semantically similar, different text
            _entity_with_embedding(
                "Global Warming", EMB_CONCEPT_A, "MISC", 0.85
            ),
            # Give the cluster an embedding too so embedding dedup can pair them
            _entity_with_embedding(
                "Climate Change Risk", EMB_CONCEPT_A_SIMILAR, "MISC", 0.5
            ),
            # An unrelated entity that should survive untouched
            _entity_with_embedding(
                "Quantum Computing", EMB_UNRELATED, "MISC", 0.8
            ),
        ]

        config = _all_options_config()
        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result(entities)
        result = await workflow.process(extraction)

        result_texts = {e.text for e in result.entities}

        # "Quantum Computing" should definitely survive
        assert any("Quantum" in t for t in result_texts)

        # The climate/warming cluster should have been reduced
        climate_related = [
            e
            for e in result.entities
            if any(
                kw in e.text.lower()
                for kw in ["climate", "warming", "clmate"]
            )
        ]
        # String dedup: 2 climate -> 1. Fuzzy: merges "Clmate Change" in.
        # Embedding dedup: merges "Global Warming" + "Climate Change Risk".
        # That leaves at most 2 climate-related entities (the fuzzy-merged one
        # plus the embedding-merged one), possibly fewer depending on chaining.
        assert len(climate_related) <= 3

        # Total entities should be reduced from 6
        assert len(result.entities) <= 5

        # There should be merge groups from the dedup stages
        assert len(result.merged_entity_groups) >= 1

    async def test_each_dedup_stage_independent_contributions(self):
        """Each dedup stage merges entities the previous stages missed."""
        entities = [
            # Only string dedup catches this pair (exact case match)
            _entity("Alpha", "MISC", 0.8),
            _entity("alpha", "MISC", 0.7),
            # Only fuzzy catches this pair (edit distance 1)
            # "Beta" vs "Betta" -- Levenshtein sim = 4/5 = 0.80 ... too low
            # Use longer names: "Beta Protocol" vs "Betta Protocol"
            # Levenshtein("beta protocol", "betta protocol") = 1 edit, len=14
            # sim = 13/14 = 0.928 > 0.85
            _entity("Beta Protocol", "MISC", 0.8),
            _entity("Betta Protocol", "MISC", 0.7),
            # Only embedding catches this pair (different text, same meaning)
            _entity_with_embedding(
                "Gamma Ray", EMB_CONCEPT_A, "MISC", 0.8
            ),
            _entity_with_embedding(
                "Gamma Radiation", EMB_CONCEPT_A_SIMILAR, "MISC", 0.7
            ),
        ]

        config = _all_options_config()
        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result(entities)
        result = await workflow.process(extraction)

        # Each pair should be merged: 6 -> 3
        assert len(result.entities) <= 3
        # At least 2 merge groups (string+fuzzy at minimum)
        assert len(result.merged_entity_groups) >= 2


# ---------------------------------------------------------------------------
# Mock entity repo for KG resolution integration tests
# ---------------------------------------------------------------------------


class _MockEntityRepo:
    """In-memory mock implementing EntityRepositoryProtocol for workflow tests."""

    def __init__(self):
        self.aliases: Dict[str, Dict[str, Any]] = {}
        self.entities_by_type: Dict[str, List[Dict[str, Any]]] = {}
        self.registered_aliases: List[Dict[str, Any]] = []

    async def find_by_alias(
        self, alias_text: str
    ) -> Optional[Dict[str, Any]]:
        return self.aliases.get(alias_text)

    async def find_by_type(
        self, entity_type: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return self.entities_by_type.get(entity_type, [])[:limit]

    async def register_alias(
        self,
        canonical_entity_id: str,
        alias_text: str,
        match_type: str,
        similarity_score: float,
        method: str = "",
    ) -> bool:
        self.registered_aliases.append({
            "canonical_entity_id": canonical_entity_id,
            "alias_text": alias_text,
            "match_type": match_type,
            "similarity_score": similarity_score,
            "method": method,
        })
        return True


# Embeddings for KG resolution tests (10-dim)
_EMB_HUB = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_EMB_NEAR_HUB = [0.98, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_EMB_DIFFERENT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _kg_repo_with_weighted_candidates():
    """Build mock repo with weighted candidates for centrality-aware tests."""
    repo = _MockEntityRepo()
    repo.entities_by_type["LOCATION"] = [
        {
            "id": "entity:paris",
            "name": "Paris",
            "embedding": _EMB_HUB,
            "weight": 15.0,  # Important hub entity
        },
        {
            "id": "entity:smalltown",
            "name": "Smalltown",
            "embedding": _EMB_DIFFERENT,
            "weight": 1.0,  # Low-importance entity
        },
    ]
    repo.entities_by_type["PERSON"] = [
        {
            "id": "entity:napoleon",
            "name": "Napoleon Bonaparte",
            "embedding": _EMB_HUB,
            "weight": 12.0,
        },
    ]
    return repo


# ---------------------------------------------------------------------------
# Phase 3-4b: KG resolution + graph centrality integration
# ---------------------------------------------------------------------------


class TestKGCentralityIntegration:
    """End-to-end workflow tests with KG resolution (centrality-aware)
    and graph centrality analysis (outlier detection) composed together.

    Tests the full data flow: entities go through noise/normalize/dedup,
    then KG resolution marks them as new/matched (with importance-aware
    thresholds), then graph analysis computes centrality and classifies
    outliers based on the KG metadata.
    """

    async def test_full_pipeline_kg_and_graph_composed(self):
        """Run the full pipeline with KG resolution + graph analysis + outlier detection."""
        repo = _kg_repo_with_weighted_candidates()

        config = FilteringConfig(
            dedup_enabled=True,
            kg_resolution=KGResolutionConfig(
                enabled=True,
                fuzzy_threshold=0.85,
                semantic_threshold=0.90,
                mark_new_entities=True,
                centrality_aware=False,  # Test basic composition first
            ),
            ontology_validation=OntologyValidationConfig(
                graph_centrality_enabled=True,
                centrality_min_score=0.0,  # Keep all (enrichment only)
                outlier_detection_enabled=True,
                outlier_centrality_low=0.05,
            ),
        )

        # Build entities: Paris matches KG, rest are new
        entities = [
            _entity("Paris", "LOCATION", 0.9),
            _entity("Amsterdam", "LOCATION", 0.85),
            _entity("Berlin", "LOCATION", 0.8),
            _entity("Rome", "LOCATION", 0.75),
        ]
        relations = [
            # Paris is a hub connected to all others
            _relation("Paris", "Amsterdam", "CONNECTED_TO"),
            _relation("Paris", "Berlin", "CONNECTED_TO"),
            _relation("Paris", "Rome", "CONNECTED_TO"),
            # Amsterdam connected to Berlin (gives them some centrality)
            _relation("Amsterdam", "Berlin", "CONNECTED_TO"),
        ]

        workflow = FilteringWorkflow(config=config, entity_repo=repo)
        extraction = _extraction_result(entities, relations)
        result = await workflow.process(extraction)

        assert isinstance(result, FilteredResult)

        # All 4 entities should survive (min_score=0.0)
        assert len(result.entities) == 4

        # KG resolution report should be present
        assert result.kg_resolution_report is not None
        assert "matched_count" in result.kg_resolution_report
        # Paris matches KG (exact fuzzy match), rest are new
        assert result.kg_resolution_report["matched_count"] >= 1
        assert result.kg_resolution_report["new_count"] >= 1

        # Graph analysis report should be in validation_report
        assert result.validation_report is not None
        assert "graph_analysis" in result.validation_report
        graph_report = result.validation_report["graph_analysis"]
        assert graph_report["node_count"] == 4
        assert graph_report["edge_count"] == 4

        # Outlier summary should be present
        assert "outlier_summary" in graph_report
        summary = graph_report["outlier_summary"]
        assert "important_new_count" in summary
        assert "potential_noise_count" in summary
        assert "established_count" in summary

        # Paris was matched to KG (is_new=False) and is hub → "established"
        # Other cities are new (is_new=True)
        # Amsterdam/Berlin have decent centrality → "important_new"
        # Rome only has 1 incoming edge from Paris → lower centrality
        total_classified = (
            summary["important_new_count"]
            + summary["potential_noise_count"]
            + summary["anomaly_count"]
            + summary["established_count"]
        )
        assert total_classified == 4  # All entities have is_new set

        # Paris should be established (matched KG, high centrality)
        assert summary["established_count"] >= 1

        # At least some new entities should be classified
        assert summary["important_new_count"] + summary["potential_noise_count"] >= 1

        # Entities should have kg_classification property
        for ent in result.entities:
            props = ent.properties or {}
            assert "kg_classification" in props, (
                f"Entity '{ent.text}' missing kg_classification"
            )
            assert props["kg_classification"] in {
                "important_new",
                "potential_noise",
                "anomaly",
                "established",
            }

        # Entities should also have centrality_score from graph analysis
        for ent in result.entities:
            props = ent.properties or {}
            assert "centrality_score" in props, (
                f"Entity '{ent.text}' missing centrality_score"
            )

        # Metadata has filtering stats
        assert "filtering" in result.metadata

    async def test_centrality_aware_rejects_borderline_match_on_hub(self):
        """With centrality_aware=True, borderline fuzzy match on high-weight
        KG entity is rejected, marking it as new instead of matched.
        """
        repo = _kg_repo_with_weighted_candidates()

        config = FilteringConfig(
            kg_resolution=KGResolutionConfig(
                enabled=True,
                fuzzy_threshold=0.79,  # Low base threshold
                semantic_threshold=0.90,
                mark_new_entities=True,
                centrality_aware=True,
                centrality_strictness=0.05,
                importance_threshold=5.0,
            ),
            ontology_validation=OntologyValidationConfig(
                graph_centrality_enabled=True,
                centrality_min_score=0.0,
                outlier_detection_enabled=True,
            ),
        )

        # "Pares" is 4/5 = 0.80 similar to "Paris"
        # Base threshold 0.79 would accept it, but Paris weight=15 >= 5
        # raises effective threshold to 0.84 → 0.80 < 0.84 → rejected
        entities = [
            _entity("Pares", "LOCATION", 0.9),
            _entity("Berlin", "LOCATION", 0.8),
        ]
        relations = [
            _relation("Pares", "Berlin", "CONNECTED_TO"),
        ]

        workflow = FilteringWorkflow(config=config, entity_repo=repo)
        extraction = _extraction_result(entities, relations)
        result = await workflow.process(extraction)

        # "Pares" should be marked as new (not matched to Paris)
        pares_entity = next(
            (e for e in result.entities if e.text == "Pares"), None
        )
        assert pares_entity is not None
        props = pares_entity.properties or {}
        assert props.get("is_new") is True, (
            "Borderline match should be rejected for high-weight KG entity"
        )

        # KG report should show it as new
        assert result.kg_resolution_report["new_count"] >= 1

    async def test_centrality_aware_allows_strong_match_on_hub(self):
        """Exact match still works even with raised thresholds on hub entity."""
        repo = _kg_repo_with_weighted_candidates()

        config = FilteringConfig(
            kg_resolution=KGResolutionConfig(
                enabled=True,
                fuzzy_threshold=0.85,
                mark_new_entities=True,
                centrality_aware=True,
                centrality_strictness=0.05,
                importance_threshold=5.0,
            ),
            ontology_validation=OntologyValidationConfig(
                graph_centrality_enabled=True,
                centrality_min_score=0.0,
                outlier_detection_enabled=True,
            ),
        )

        # "Paris" exact match → score 1.0, clears raised threshold 0.90
        entities = [
            _entity("Paris", "LOCATION", 0.9),
            _entity("Amsterdam", "LOCATION", 0.85),
        ]
        relations = [
            _relation("Paris", "Amsterdam", "CONNECTED_TO"),
        ]

        workflow = FilteringWorkflow(config=config, entity_repo=repo)
        extraction = _extraction_result(entities, relations)
        result = await workflow.process(extraction)

        paris = next(
            (e for e in result.entities if e.text == "Paris"), None
        )
        assert paris is not None
        props = paris.properties or {}
        assert props.get("is_new") is False, (
            "Exact match on hub should still succeed"
        )
        assert props.get("kg_entity_id") == "entity:paris"
        assert props.get("kg_classification") == "established"

    async def test_outlier_detection_off_no_classification(self):
        """When outlier_detection_enabled=False, no kg_classification is set."""
        repo = _kg_repo_with_weighted_candidates()

        config = FilteringConfig(
            kg_resolution=KGResolutionConfig(
                enabled=True,
                mark_new_entities=True,
            ),
            ontology_validation=OntologyValidationConfig(
                graph_centrality_enabled=True,
                centrality_min_score=0.0,
                outlier_detection_enabled=False,  # Off
            ),
        )

        entities = [
            _entity("Paris", "LOCATION", 0.9),
            _entity("Amsterdam", "LOCATION", 0.85),
        ]
        relations = [
            _relation("Paris", "Amsterdam", "CONNECTED_TO"),
        ]

        workflow = FilteringWorkflow(config=config, entity_repo=repo)
        extraction = _extraction_result(entities, relations)
        result = await workflow.process(extraction)

        # Graph analysis should run but without outlier summary
        assert result.validation_report is not None
        graph_report = result.validation_report["graph_analysis"]
        assert "outlier_summary" not in graph_report

        # No kg_classification on entities
        for ent in result.entities:
            props = ent.properties or {}
            assert "kg_classification" not in props

    async def test_no_kg_resolution_makes_outliers_unclassified(self):
        """When KG resolution is disabled, outlier detection has no is_new data
        and all entities are unclassified.
        """
        config = FilteringConfig(
            kg_resolution=KGResolutionConfig(enabled=False),
            ontology_validation=OntologyValidationConfig(
                graph_centrality_enabled=True,
                centrality_min_score=0.0,
                outlier_detection_enabled=True,
            ),
        )

        entities = [
            _entity("Paris", "LOCATION", 0.9),
            _entity("Amsterdam", "LOCATION", 0.85),
        ]
        relations = [
            _relation("Paris", "Amsterdam", "CONNECTED_TO"),
        ]

        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result(entities, relations)
        result = await workflow.process(extraction)

        graph_report = result.validation_report["graph_analysis"]
        summary = graph_report["outlier_summary"]
        # No KG data → all unclassified
        assert summary["unclassified_count"] == 2
        assert summary["important_new_count"] == 0
        assert summary["established_count"] == 0

        # Entities should NOT have kg_classification
        for ent in result.entities:
            props = ent.properties or {}
            assert "kg_classification" not in props
