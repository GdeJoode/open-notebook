"""
Main filtering workflow orchestrator.

Composes the individual filter stages (noise removal, normalization,
reclassification, deduplication, fuzzy resolution, embedding dedup,
semantic enrichment, KG resolution, ontology validation, graph analysis,
edge prediction) into a single pipeline that transforms an
ExtractionResult into a FilteredResult.
"""

from typing import Any, Optional

from loguru import logger
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    FilteredResult,
)

from entity_filtering.config import FilteringConfig
from entity_filtering.deduplication.embedding_deduplicator import (
    EmbeddingDeduplicator,
)
from entity_filtering.deduplication.entity_deduplicator import EntityDeduplicator
from entity_filtering.deduplication.fuzzy_resolver import FuzzyResolver
from entity_filtering.filters.noise_filter import NoiseFilter
from entity_filtering.filters.normalizer import EntityNormalizer
from entity_filtering.filters.reclassifier import EntityReclassifier
from entity_filtering.resolution.contextual_clusterer import ContextualClusterer
from entity_filtering.resolution.embedding_resolver import EmbeddingResolver
from entity_filtering.resolution.entity_linker import (
    DBpediaSpotlightLinker,
    EntityLinker,
)
from entity_filtering.resolution.kg_resolver import (
    EntityRepositoryProtocol,
    KGResolver,
)
from entity_filtering.deduplication.semantic_blocker import SemanticBlocker
from entity_filtering.resolution.llm_matcher import LLMMatcher
from entity_filtering.scoring.edge_predictor import EdgePredictor
from entity_filtering.validation.graph_analyzer import GraphAnalyzer
from entity_filtering.validation.ontology_constraint_filter import (
    OntologyConstraintFilter,
)


class FilteringWorkflow:
    """Orchestrates the entity filtering pipeline.

    Stages (in order):
    1.  Noise filtering -- remove invalid / artifact entities
    2.  Normalization -- canonical text forms, merge equivalents
    3.  Reclassification -- fix entity labels via heuristic rules
    4.  Deduplication -- merge entities with identical normalized text
    5.  Fuzzy resolution (optional) -- merge near-duplicate names
    6.  Embedding dedup (optional) -- merge semantically similar entities
    7.  Embedding resolution (optional) -- semantic match enrichment
    8.  Entity linking (optional) -- link to external KBs (DBpedia)
    9.  Contextual clustering (optional) -- cluster by co-occurrence
    10. KG resolution (optional) -- match against existing KG entities
    11. Ontology constraint filter (optional) -- validate against ontology
    12. Graph centrality analysis (optional) -- filter low-centrality entities
    13. Edge prediction (optional) -- discover implicit relations

    Args:
        config: Pipeline configuration. Uses defaults when None.
        entity_repo: Optional repository implementing
            ``EntityRepositoryProtocol`` for KG resolution.
        entity_linker: Optional entity linker instance. When ``None``
            and entity linking is enabled, a ``DBpediaSpotlightLinker``
            is created with config defaults.
        ontology: Optional ontology definition for constraint validation.
    """

    def __init__(
        self,
        config: Optional[FilteringConfig] = None,
        entity_repo: Optional[EntityRepositoryProtocol] = None,
        entity_linker: Optional[EntityLinker] = None,
        ontology: Optional[Any] = None,
    ) -> None:
        self._config = config or FilteringConfig()

        self._noise_filter = NoiseFilter(
            custom_patterns=self._config.custom_noise_patterns,
            min_entity_length=self._config.min_entity_length,
        )
        self._normalizer = EntityNormalizer(
            strip_articles=self._config.strip_articles,
            custom_articles=self._config.custom_articles,
            normalize_whitespace=self._config.normalize_whitespace,
            remove_diacritics=self._config.syntactic.remove_diacritics,
            ocr_cleanup_enabled=self._config.syntactic.ocr_cleanup_enabled,
            ocr_artifact_patterns=self._config.syntactic.ocr_artifact_patterns
            or None,
            html_strip_enabled=self._config.syntactic.html_strip_enabled,
            page_number_filter=self._config.syntactic.page_number_filter,
        )
        self._reclassifier = EntityReclassifier(
            custom_rules=self._config.custom_reclassification_rules,
        )
        self._deduplicator = EntityDeduplicator(
            similarity_threshold=self._config.dedup_similarity_threshold,
        )
        self._fuzzy_resolver: Optional[FuzzyResolver] = None
        if self._config.fuzzy_dedup.enabled:
            self._fuzzy_resolver = FuzzyResolver(
                algorithm=self._config.fuzzy_dedup.algorithm,
                similarity_threshold=self._config.fuzzy_dedup.similarity_threshold,
                phonetic_algorithm=self._config.fuzzy_dedup.phonetic_algorithm,
                phonetic_weight=self._config.fuzzy_dedup.phonetic_weight,
                max_candidates_per_entity=self._config.fuzzy_dedup.max_candidates_per_entity,
            )
        self._embedding_deduplicator: Optional[EmbeddingDeduplicator] = None
        if self._config.embedding_dedup.enabled:
            self._embedding_deduplicator = EmbeddingDeduplicator(
                similarity_threshold=self._config.embedding_dedup.similarity_threshold,
                k_candidates=self._config.embedding_dedup.k_candidates,
                use_faiss=self._config.embedding_dedup.use_faiss,
            )

        # Semantic enrichment stages
        self._embedding_resolver: Optional[EmbeddingResolver] = None
        if self._config.semantic.contextual_clustering_enabled or self._config.semantic.entity_linking_enabled:
            self._embedding_resolver = EmbeddingResolver(
                similarity_threshold=self._config.embedding_dedup.similarity_threshold,
            )

        self._entity_linker: Optional[EntityLinker] = None
        if self._config.semantic.entity_linking_enabled:
            if entity_linker is not None:
                self._entity_linker = entity_linker
            elif self._config.semantic.linking_provider == "dbpedia_spotlight":
                self._entity_linker = DBpediaSpotlightLinker(
                    confidence=self._config.semantic.linking_confidence_threshold,
                )

        self._contextual_clusterer: Optional[ContextualClusterer] = None
        if self._config.semantic.contextual_clustering_enabled:
            self._contextual_clusterer = ContextualClusterer()

        # KG resolution
        self._kg_resolver: Optional[KGResolver] = None
        if self._config.kg_resolution.enabled:
            self._kg_resolver = KGResolver(
                entity_repo=entity_repo,
                fuzzy_threshold=self._config.kg_resolution.fuzzy_threshold,
                semantic_threshold=self._config.kg_resolution.semantic_threshold,
                max_candidates=self._config.kg_resolution.max_candidates,
                register_aliases=self._config.kg_resolution.register_aliases,
                mark_new_entities=self._config.kg_resolution.mark_new_entities,
                use_alias_table=self._config.kg_resolution.use_alias_table,
                centrality_aware=self._config.kg_resolution.centrality_aware,
                centrality_strictness=self._config.kg_resolution.centrality_strictness,
                importance_threshold=self._config.kg_resolution.importance_threshold,
            )

        # Ontology validation
        self._ontology_filter: Optional[OntologyConstraintFilter] = None
        if self._config.ontology_validation.enabled:
            self._ontology_filter = OntologyConstraintFilter(
                ontology=ontology,
                strict=self._config.ontology_validation.strict_mode,
                filter_invalid_entities=self._config.ontology_validation.filter_invalid_entities,
                filter_invalid_relations=self._config.ontology_validation.filter_invalid_relations,
            )

        # Graph centrality analysis
        self._graph_analyzer: Optional[GraphAnalyzer] = None
        if self._config.ontology_validation.graph_centrality_enabled:
            self._graph_analyzer = GraphAnalyzer(
                min_score=self._config.ontology_validation.centrality_min_score,
                classify_outliers=self._config.ontology_validation.outlier_detection_enabled,
                outlier_centrality_low=self._config.ontology_validation.outlier_centrality_low,
            )

        # Semantic blocking
        self._semantic_blocker: Optional[SemanticBlocker] = None
        if self._config.semantic_blocking.enabled:
            self._semantic_blocker = SemanticBlocker(
                umap_n_components=self._config.semantic_blocking.umap_n_components,
                min_cluster_size=self._config.semantic_blocking.min_cluster_size,
                min_samples=self._config.semantic_blocking.min_samples,
            )

        # LLM-based matcher
        self._llm_matcher: Optional[LLMMatcher] = None
        if self._config.llm_matcher.enabled:
            self._llm_matcher = LLMMatcher(
                model=self._config.llm_matcher.model,
                base_url=self._config.llm_matcher.base_url,
                confidence_threshold=self._config.llm_matcher.confidence_threshold,
                timeout=self._config.llm_matcher.timeout,
            )

        self._edge_predictor = EdgePredictor()

    async def process(
        self, extraction_result: ExtractionResult
    ) -> FilteredResult:
        """Run the full filtering pipeline.

        Args:
            extraction_result: Output from the ontology-extraction
                pipeline.

        Returns:
            A FilteredResult containing the cleaned entities,
            surviving relations, removed entities, merge groups,
            and any predicted edges.
        """
        input_entity_count = len(extraction_result.entities)
        input_relation_count = len(extraction_result.relations)

        logger.info(
            "Starting filtering pipeline: {} entities, {} relations",
            input_entity_count,
            input_relation_count,
        )

        entities = [e.model_dump() for e in extraction_result.entities]
        relations = [r.model_dump() for r in extraction_result.relations]

        # ------------------------------------------------------------------
        # Stage 1: Noise filter
        # ------------------------------------------------------------------
        filtered_entities = self._noise_filter.filter_entities(entities)
        removed = [e for e in entities if e not in filtered_entities]
        filtered_relations = self._noise_filter.filter_relations(
            relations, filtered_entities
        )

        logger.debug(
            "After noise filter: {} entities ({} removed), {} relations",
            len(filtered_entities),
            len(removed),
            len(filtered_relations),
        )

        # ------------------------------------------------------------------
        # Stage 2: Normalize
        # ------------------------------------------------------------------
        normalized_entities = self._normalizer.normalize(filtered_entities)
        logger.debug(
            "After normalization: {} entities", len(normalized_entities)
        )

        # ------------------------------------------------------------------
        # Stage 3: Reclassify
        # ------------------------------------------------------------------
        reclassified_entities = self._reclassifier.reclassify(
            normalized_entities
        )

        # ------------------------------------------------------------------
        # Stage 4: Deduplicate (string-based)
        # ------------------------------------------------------------------
        merge_groups: list[list[str]] = []
        if self._config.dedup_enabled:
            deduped_entities, merge_groups = self._deduplicator.deduplicate(
                reclassified_entities
            )
            logger.debug(
                "After deduplication: {} entities, {} merge groups",
                len(deduped_entities),
                len(merge_groups),
            )
        else:
            deduped_entities = reclassified_entities

        # ------------------------------------------------------------------
        # Stage 5: Fuzzy resolution (optional)
        # ------------------------------------------------------------------
        if self._fuzzy_resolver is not None:
            deduped_entities, fuzzy_groups = self._fuzzy_resolver.resolve(
                deduped_entities
            )
            merge_groups.extend(fuzzy_groups)
            logger.debug(
                "After fuzzy resolution: {} entities, {} new merge groups",
                len(deduped_entities),
                len(fuzzy_groups),
            )

        # ------------------------------------------------------------------
        # Stage 6: Embedding dedup (optional)
        # ------------------------------------------------------------------
        if self._embedding_deduplicator is not None:
            deduped_entities, emb_groups = (
                self._embedding_deduplicator.deduplicate(deduped_entities)
            )
            merge_groups.extend(emb_groups)
            logger.debug(
                "After embedding dedup: {} entities, {} new merge groups",
                len(deduped_entities),
                len(emb_groups),
            )

        # ------------------------------------------------------------------
        # Stage 6b: LLM matching (optional)
        # ------------------------------------------------------------------
        if self._llm_matcher is not None and len(deduped_entities) >= 2:
            llm_merge_groups = await self._run_llm_matching(deduped_entities)
            if llm_merge_groups:
                merge_groups.extend(llm_merge_groups)
                # Re-deduplicate with merge decisions applied
                deduped_entities, extra_groups = self._deduplicator.deduplicate(
                    deduped_entities
                )
                merge_groups.extend(extra_groups)
                logger.debug(
                    "After LLM matching: {} entities, {} new merge groups",
                    len(deduped_entities),
                    len(llm_merge_groups),
                )

        # ------------------------------------------------------------------
        # Stage 7: Embedding resolution (optional enrichment)
        # ------------------------------------------------------------------
        if self._embedding_resolver is not None:
            deduped_entities = self._embedding_resolver.resolve(
                deduped_entities
            )

        # ------------------------------------------------------------------
        # Stage 8: Entity linking (optional enrichment)
        # ------------------------------------------------------------------
        linked_entities: dict[str, dict[str, Any]] = {}
        if self._entity_linker is not None:
            deduped_entities = await self._entity_linker.link(
                deduped_entities
            )
            # Collect linked entity metadata for the result
            for ent in deduped_entities:
                props = ent.get("properties", {})
                if props.get("dbpedia_uri") or props.get("wikidata_id"):
                    linked_entities[ent.get("text", "")] = {
                        k: v
                        for k, v in props.items()
                        if k
                        in (
                            "dbpedia_uri",
                            "wikipedia_url",
                            "wikidata_id",
                        )
                    }

        # ------------------------------------------------------------------
        # Stage 9: Contextual clustering (optional enrichment)
        # ------------------------------------------------------------------
        if self._contextual_clusterer is not None:
            deduped_entities = self._contextual_clusterer.cluster(
                deduped_entities
            )

        # ------------------------------------------------------------------
        # Stage 10: KG resolution (optional enrichment)
        # ------------------------------------------------------------------
        kg_resolution_report: Optional[dict[str, Any]] = None
        if self._kg_resolver is not None:
            deduped_entities, kg_resolution_report = (
                await self._kg_resolver.resolve(deduped_entities)
            )
            logger.debug(
                "After KG resolution: {} matched, {} new",
                kg_resolution_report.get("matched_count", 0),
                kg_resolution_report.get("new_count", 0),
            )

        # ------------------------------------------------------------------
        # Stage 11: Ontology constraint filter (optional)
        # ------------------------------------------------------------------
        validation_report: Optional[dict[str, Any]] = None
        if self._ontology_filter is not None:
            deduped_entities, filtered_relations, ontology_report = (
                self._ontology_filter.filter(deduped_entities, filtered_relations)
            )
            validation_report = {"ontology": ontology_report}
            logger.debug(
                "After ontology filter: {}/{} entities valid, {}/{} relations valid",
                ontology_report.get("valid_entities", 0),
                ontology_report.get("total_entities", 0),
                ontology_report.get("valid_relations", 0),
                ontology_report.get("total_relations", 0),
            )

        # ------------------------------------------------------------------
        # Stage 12: Graph centrality analysis (optional)
        # ------------------------------------------------------------------
        if self._graph_analyzer is not None:
            deduped_entities, graph_report = self._graph_analyzer.analyze(
                deduped_entities, filtered_relations
            )
            if validation_report is None:
                validation_report = {}
            validation_report["graph_analysis"] = graph_report
            logger.debug(
                "After graph analysis: {} entities ({} removed below {:.4f})",
                len(deduped_entities),
                graph_report.get("removed_count", 0),
                graph_report.get("min_score_threshold", 0.0),
            )

        # ------------------------------------------------------------------
        # Stage 13: Edge prediction (optional)
        # ------------------------------------------------------------------
        predicted_edges: list[dict[str, Any]] = []
        if self._config.edge_prediction_enabled:
            predicted_edges = self._edge_predictor.predict(
                deduped_entities, filtered_relations
            )
            logger.debug(
                "Edge predictor produced {} predicted edges",
                len(predicted_edges),
            )

        # ------------------------------------------------------------------
        # Build result
        # ------------------------------------------------------------------
        result_entities = [ExtractedEntity(**e) for e in deduped_entities]
        result_relations = [ExtractedRelation(**r) for r in filtered_relations]
        removed_entities = [ExtractedEntity(**e) for e in removed]
        predicted_relations = [
            ExtractedRelation(**e) for e in predicted_edges
        ]

        result = FilteredResult(
            entities=result_entities,
            relations=result_relations,
            removed_entities=removed_entities,
            merged_entity_groups=merge_groups,
            predicted_edges=predicted_relations,
            linked_entities=linked_entities,
            kg_resolution_report=kg_resolution_report,
            validation_report=validation_report,
            metadata={
                **extraction_result.metadata,
                "filtering": {
                    "input_entities": input_entity_count,
                    "input_relations": input_relation_count,
                    "output_entities": len(result_entities),
                    "output_relations": len(result_relations),
                    "removed_count": len(removed_entities),
                    "merge_groups": len(merge_groups),
                    "predicted_edges": len(predicted_relations),
                },
            },
        )

        logger.info(
            "Filtering complete: {} -> {} entities, {} -> {} relations, "
            "{} removed, {} merge groups",
            input_entity_count,
            len(result_entities),
            input_relation_count,
            len(result_relations),
            len(removed_entities),
            len(merge_groups),
        )

        return result

    async def _run_llm_matching(
        self, entities: list[dict[str, Any]]
    ) -> list[list[str]]:
        """Generate candidate pairs and run LLM matching.

        Uses semantic blocking if available, otherwise falls back to
        pairwise comparison on entities with embeddings.

        Returns merge groups (list of [entity_a_text, entity_b_text] pairs).
        """
        from entity_filtering.deduplication.union_find import UnionFind

        assert self._llm_matcher is not None

        # Generate candidate pairs using semantic blocking or brute force
        if self._semantic_blocker is not None:
            blocks = self._semantic_blocker.block(entities)
        else:
            blocks = [entities]  # Single block = all pairs

        # Collect candidate pairs from within each block
        candidate_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for block in blocks:
            for i in range(len(block)):
                for j in range(i + 1, len(block)):
                    # Skip pairs with identical text (already deduped by string dedup)
                    if block[i].get("text", "").strip().lower() != block[j].get("text", "").strip().lower():
                        candidate_pairs.append((block[i], block[j]))

        if not candidate_pairs:
            return []

        logger.info(
            f"LLM matcher: {len(candidate_pairs)} candidate pairs "
            f"from {len(blocks)} blocks"
        )

        # Run LLM matching
        results = await self._llm_matcher.match_batch(candidate_pairs)
        match_indices = self._llm_matcher.filter_matches(results)

        if not match_indices:
            return []

        # Build merge groups via UnionFind
        uf = UnionFind()
        for idx in match_indices:
            a_text = candidate_pairs[idx][0].get("text", "")
            b_text = candidate_pairs[idx][1].get("text", "")
            uf.union(a_text, b_text)

            # Normalize the matched entity's text to canonical form
            # (the LLM decided they're the same, so mark for merging)
            a_entity = candidate_pairs[idx][0]
            b_entity = candidate_pairs[idx][1]
            # Store match info in properties for provenance
            a_entity.setdefault("properties", {})["llm_match"] = b_text
            b_entity.setdefault("properties", {})["llm_match"] = a_text

        all_texts = set()
        for idx in match_indices:
            all_texts.add(candidate_pairs[idx][0].get("text", ""))
            all_texts.add(candidate_pairs[idx][1].get("text", ""))

        groups = uf.get_groups(list(all_texts))
        merge_groups = [members for members in groups.values() if len(members) > 1]

        logger.info(
            f"LLM matcher: {len(match_indices)} matches → "
            f"{len(merge_groups)} merge groups"
        )
        return merge_groups
