"""
Main filtering workflow orchestrator.

Composes the individual filter stages (noise removal, normalization,
reclassification, deduplication, fuzzy resolution, embedding dedup,
semantic enrichment, KG resolution, ontology validation, graph analysis,
edge prediction) into a single pipeline that transforms an
ExtractionResult into a FilteredResult.
"""

from typing import Any, Dict, Optional

from loguru import logger
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    FilteredResult,
    MatchCandidate,
)

from entity_filtering.config import FilteringConfig
from entity_filtering.deduplication.embedding_deduplicator import (
    EmbeddingDeduplicator,
)
from entity_filtering.deduplication.entity_deduplicator import EntityDeduplicator
from entity_filtering.deduplication.fuzzy_resolver import FuzzyResolver
from entity_filtering.deduplication.semantic_blocker import SemanticBlocker
from entity_filtering.filters.noise_filter import NoiseFilter
from entity_filtering.filters.normalizer import EntityNormalizer
from entity_filtering.filters.reclassifier import EntityReclassifier
from entity_filtering.resolution import orphan_connector as _orphan_connector
from entity_filtering.resolution.concept_alignment import (
    ConceptAligner,
    build_is_a_seeds,
)
from entity_filtering.resolution.contextual_clusterer import ContextualClusterer
from entity_filtering.resolution.embedding_resolver import EmbeddingResolver
from entity_filtering.resolution.entity_linker import (
    DBpediaSpotlightLinker,
    EntityLinker,
)
from entity_filtering.resolution.incremental_resolver import (
    EntityCluster,
    IncrementalResolver,
)
from entity_filtering.resolution.kg_resolver import (
    EntityRepositoryProtocol,
    KGResolver,
)
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
    6b. LLM matching (optional) -- LLM-based entity pair evaluation
    7.  Embedding resolution (optional) -- semantic match enrichment
    8.  Entity linking (optional) -- link to external KBs (DBpedia)
    9.  Contextual clustering (optional) -- cluster by co-occurrence
    10. KG resolution (optional) -- match against existing KG entities
    10b.Incremental resolution (optional) -- assign to existing clusters
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
        # Kept for Stage 15 (Track N.4 concept alignment), which needs the
        # same repository Stage 10 uses plus the applied ontology for the
        # canonical type resolution.
        self._entity_repo = entity_repo
        self._ontology = ontology

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
                agentic_enabled=self._config.llm_matcher.agentic_enabled,
                agentic_lower_threshold=self._config.llm_matcher.agentic_lower_threshold,
                agentic_upper_threshold=self._config.llm_matcher.agentic_upper_threshold,
                agentic_max_iterations=self._config.llm_matcher.agentic_max_iterations,
                context_provider=self._fetch_agentic_context if self._config.llm_matcher.agentic_enabled else None,
            )

        # Incremental cluster resolution
        self._incremental_resolver: Optional[IncrementalResolver] = None
        if self._config.incremental_resolution.enabled:
            self._incremental_resolver = IncrementalResolver(
                similarity_threshold=self._config.incremental_resolution.similarity_threshold,
                coherence_threshold=self._config.incremental_resolution.coherence_threshold,
                merge_threshold=self._config.incremental_resolution.merge_threshold,
            )

        self._edge_predictor = EdgePredictor()

        # Cache for agentic context — populated during processing
        self._current_entities: list[dict[str, Any]] = []

        # Existing clusters for incremental resolution (injected externally)
        self._existing_clusters: list[EntityCluster] = []

    def set_existing_clusters(self, clusters: list[EntityCluster]) -> None:
        """Inject existing KG clusters for incremental resolution."""
        self._existing_clusters = clusters

    async def _fetch_agentic_context(
        self,
        entity_a: Dict[str, Any],
        entity_b: Dict[str, Any],
    ) -> str:
        """Fetch additional context for uncertain LLM matches (stap 2E).

        Provides:
        - Co-occurring entities (same source chunk)
        - Surrounding text from extraction context
        - Existing cluster/match info from properties
        """
        parts: list[str] = []

        text_a = entity_a.get("text", "")
        text_b = entity_b.get("text", "")
        chunk_a = entity_a.get("source_chunk_id")
        chunk_b = entity_b.get("source_chunk_id")

        # 1. Co-occurring entities in the same chunks
        for label, chunk_id, text in [("A", chunk_a, text_a), ("B", chunk_b, text_b)]:
            if not chunk_id:
                continue
            co_entities = [
                e.get("text", "")
                for e in self._current_entities
                if e.get("source_chunk_id") == chunk_id and e.get("text", "") != text
            ]
            if co_entities:
                parts.append(
                    f"Entiteiten in hetzelfde fragment als {label}: "
                    + ", ".join(co_entities[:10])
                )

        # 2. Surrounding text from extraction context
        for label, entity in [("A", entity_a), ("B", entity_b)]:
            ctx = entity.get("extraction_context") or {}
            if isinstance(ctx, dict) and ctx.get("surrounding_text"):
                parts.append(f"Omringende tekst {label}: {ctx['surrounding_text'][:200]}")

        # 3. Existing match info (if entity was already matched by embedding/fuzzy)
        for label, entity in [("A", entity_a), ("B", entity_b)]:
            props = entity.get("properties", {})
            if props.get("llm_match"):
                parts.append(f"{label} eerder gematcht met: {props['llm_match']}")
            if props.get("cluster_id"):
                parts.append(f"{label} cluster: {props['cluster_id']}")

        return "\n".join(parts) if parts else ""

    async def process(
        self,
        extraction_result: ExtractionResult,
        *,
        source_id: Optional[str] = None,
        chunks: Optional[list[dict[str, Any]]] = None,
        orphan_entity_repo: Optional[
            "_orphan_connector.OrphanEntityRepoProtocol"
        ] = None,
        orphan_llm_caller: Optional[Any] = None,
        alignment_llm_caller: Optional[Any] = None,
    ) -> FilteredResult:
        """Run the full filtering pipeline.

        Args:
            extraction_result: Output from the ontology-extraction
                pipeline.
            source_id: Source record ID. Required for the orphan-
                connector stage; omitting it (or omitting any of the
                other orphan-stage prerequisites) skips that stage.
            chunks: Chunk dicts with ``id``, ``text`` and ``entities``
                fields. Required for the orphan-connector stage.
            orphan_entity_repo: Repository implementing
                :class:`orphan_connector.OrphanEntityRepoProtocol`.
                Required for the orphan-connector stage.
            orphan_llm_caller: Sync or async LLM caller — same
                callable shape as Pass-1/Pass-2. Required for the
                orphan-connector stage.

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
        # Populate entity cache for agentic context provider (stap 2E)
        self._current_entities = deduped_entities
        all_match_candidates: list[MatchCandidate] = []
        if self._llm_matcher is not None and len(deduped_entities) >= 2:
            llm_merge_groups, llm_candidates = await self._run_llm_matching(deduped_entities)
            all_match_candidates.extend(llm_candidates)
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
        # Stage 10b: Incremental cluster resolution (optional)
        # ------------------------------------------------------------------
        incremental_report: Optional[dict[str, Any]] = None
        if self._incremental_resolver is not None:
            clusters, assignments = self._incremental_resolver.resolve_incremental(
                deduped_entities, list(self._existing_clusters)
            )
            # Tag entities with their cluster assignment
            assignment_map = {a["entity_text"]: a for a in assignments}
            for ent in deduped_entities:
                info = assignment_map.get(ent.get("text", ""))
                if info:
                    ent.setdefault("properties", {})["cluster_id"] = info["cluster_id"]
                    ent["properties"]["cluster_action"] = info["action"]
                    ent["properties"]["cluster_score"] = info["score"]

            # Optionally repair clusters
            if self._config.incremental_resolution.repair_enabled:
                clusters, repair_report = self._incremental_resolver.repair_clusters(clusters)
                incremental_report = {
                    "assignments": len(assignments),
                    "assigned": sum(1 for a in assignments if a["action"] == "assigned"),
                    "new_clusters": sum(1 for a in assignments if a["action"] == "new"),
                    "repair": repair_report,
                }
            else:
                incremental_report = {
                    "assignments": len(assignments),
                    "assigned": sum(1 for a in assignments if a["action"] == "assigned"),
                    "new_clusters": sum(1 for a in assignments if a["action"] == "new"),
                }

            # Store updated clusters back for potential re-use
            self._existing_clusters = clusters
            logger.debug(
                "After incremental resolution: {} assignments, {} clusters",
                len(assignments),
                len(clusters),
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
        # Stage 14: Orphan-connector (B.5a)
        # ------------------------------------------------------------------
        # Runs after dedup, before persistence, on entities that have
        # already been persisted by an earlier source-import. The stage
        # is a strict ADD-ONLY: it appends LLM-confirmed relations to
        # ``filtered_relations`` and never touches entities or removes
        # rows. Skipped silently when any of its DI inputs are missing
        # (e.g. when tests call ``process(extraction_result)`` without
        # an entity repo or LLM caller).
        #
        # Ontology validation bypass (decision, B.5a attempt 2):
        # Stage 14 runs AFTER Stage 11 (ontology constraint filter), so
        # orphan-confirmed relations BYPASS ontology validation. The LLM
        # is prompted with the ontology context in the confirm step, so
        # validating again here risks dropping legitimate connections
        # that the LLM reasoned through. Trade-off: LLM-invented
        # ``relation_type`` values (e.g. ``"KNOWS_SECRETLY"``) can slip
        # past the constraint filter. B.4 telemetry tracks orphan-
        # relation types so drift surfaces early. See
        # ``orphan_connector.run`` for the same decision documented at
        # the module entry point.
        orphan_cfg = self._config.orphan_connector

        # Minor-1 fix (review attempt 1): when the operator enabled the
        # stage but the caller forgot to pass any DI input, log a
        # WARNING so the silent-skip doesn't hide a misconfiguration.
        # The skip predicate below mixes truthiness (``source_id``) and
        # None-checks (``chunks``, repos, caller); mirror that here so
        # the WARNING fires exactly when the stage would skip.
        if orphan_cfg.enabled:
            missing: list[str] = []
            if not source_id:
                missing.append("source_id")
            if chunks is None:
                missing.append("chunks")
            if orphan_entity_repo is None:
                missing.append("orphan_entity_repo")
            if orphan_llm_caller is None:
                missing.append("orphan_llm_caller")
            if missing:
                logger.warning(
                    "Orphan-connector enabled but skipped: missing DI "
                    "input(s): {missing}",
                    missing=missing,
                )

        if (
            orphan_cfg.enabled
            and source_id
            and chunks is not None
            and orphan_entity_repo is not None
            and orphan_llm_caller is not None
        ):
            try:
                new_relations = await _orphan_connector.run(
                    source_id=source_id,
                    chunks=chunks,
                    entity_repo=orphan_entity_repo,
                    llm_caller=orphan_llm_caller,
                    max_proposals_per_orphan=(
                        orphan_cfg.max_proposals_per_orphan
                    ),
                    min_confidence=orphan_cfg.min_confidence,
                )
            except _orphan_connector.OrphanTokenBudgetExceeded as exc:
                # Token budget breach is loud-by-design — log and
                # continue rather than aborting the whole filtering
                # pipeline. Budget is a per-call concern; the rest of
                # the result is still useful to the caller.
                logger.warning(
                    "Orphan-connector aborted on token-budget breach: "
                    "{exc}. Continuing without orphan relations.",
                    exc=exc,
                )
                new_relations = []
            for rel in new_relations:
                filtered_relations.append(rel.model_dump())
            logger.info(
                "Orphan-connector contributed {n} new relations",
                n=len(new_relations),
            )

        # ------------------------------------------------------------------
        # Stage 15: Concept alignment (Track N.4)
        # ------------------------------------------------------------------
        # Places the entities Stage 10 marked ``is_new`` relative to the graph
        # (NARROWER_THAN / BROADER_THAN / RELATED_TO / NOVEL) instead of leaving
        # them floating, and seeds an ``is_a`` edge for a NARROWER verdict that
        # has a materialised target. Like Stage 14 this is strictly ADD-ONLY: it
        # enriches ``properties`` and appends relations, and never merges,
        # re-types or removes an entity.
        #
        # Placement (D-N4-4) — this is the fix for the two blockers that sank the
        # first attempt at this phase, so the ordering is load-bearing:
        #
        #   * AFTER Stage 11 (ontology constraint filter). That filter drops any
        #     relation whose endpoints are not among the batch's entities, and a
        #     seeded edge points at an EXISTING graph node by construction — so
        #     running before it discarded 100% of the seeds silently while the
        #     report still counted them. Same bypass decision as Stage 14, with
        #     the same trade-off: a seeded ``relation_type`` is not re-validated.
        #     Here the exposure is far smaller than for the orphan-connector,
        #     because the type is the fixed constant ``is_a`` (the same one N.2's
        #     Hearst miner seeds) rather than an LLM-chosen string. It is not
        #     ZERO, though: the filter also checks the predicate's declaration
        #     and its domain/range, so under ``strict_mode=True`` a Hearst-mined
        #     ``is_a`` (which enters BEFORE Stage 11) can be dropped while an
        #     alignment-seeded one survives — the same predicate, a different
        #     fate. Harmless at the shipped default (``strict_mode=False``, which
        #     only warns), but worth knowing before that default changes.
        #   * AFTER Stage 12 (graph centrality). ``_build_graph`` auto-creates a
        #     node for an unknown edge endpoint, and PageRank is normalised over
        #     all nodes — so an edge added earlier would shift every entity's
        #     ``centrality_score`` and could change which entities Stage 12
        #     REMOVES. That would make this non-destructive pass destructive.
        #   * AFTER Stages 13-14 as well, purely so their inputs stay identical
        #     whether or not alignment runs.
        concept_alignment_report: Optional[dict[str, Any]] = None
        align_cfg = self._config.concept_alignment
        if align_cfg.enabled:
            # Mirror Stage 14's misconfiguration WARNING, but say what ACTUALLY
            # happens per branch. "Will not classify anything" is only true when
            # KG resolution is off (nothing is marked ``is_new``); with a missing
            # repo or ontology the stage still RUNS and writes NOVEL verdicts —
            # which N.4c turns into ontology gaps. Over-claiming in the log is the
            # same defect class this track keeps having to fix in its evidence.
            if not self._config.kg_resolution.enabled:
                logger.warning(
                    "Concept alignment enabled but nothing will be classified: "
                    "kg_resolution is disabled, so no entity is marked is_new."
                )
            degraded: list[str] = []
            if self._entity_repo is None:
                degraded.append("entity_repo (the graph is never queried)")
            if self._ontology is None:
                degraded.append("ontology (no canonical type resolves)")
            if align_cfg.judge_enabled and alignment_llm_caller is None:
                degraded.append("alignment_llm_caller (the judge tier cannot run)")
            if degraded:
                logger.warning(
                    "Concept alignment enabled but DEGRADED — it will still run "
                    "and record NOVEL verdicts. Missing: {missing}",
                    missing=degraded,
                )

            aligner = ConceptAligner(
                self._entity_repo,
                schemas=[self._ontology] if self._ontology is not None else None,
                llm_caller=alignment_llm_caller,
                judge_enabled=align_cfg.judge_enabled,
                type_chain_enabled=align_cfg.type_chain_enabled,
                related_floor=align_cfg.related_floor,
                match_ceiling=align_cfg.match_ceiling,
                max_candidates=align_cfg.max_candidates,
                min_inner_tokens=align_cfg.min_inner_tokens,
            )
            deduped_entities, concept_alignment_report = await aligner.align(
                deduped_entities
            )
            seeded = (
                build_is_a_seeds(deduped_entities) if align_cfg.seed_is_a else []
            )
            filtered_relations.extend(seeded)
            # Report what SURVIVES, not what was produced: the count must not
            # outlive the edges it describes (the attempt-1 report lied here).
            concept_alignment_report["seeded_is_a"] = len(seeded)
            logger.debug(
                "After concept alignment: {} aligned, {} is_a seeded",
                concept_alignment_report.get("aligned_count", 0),
                len(seeded),
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
            match_candidates=all_match_candidates,
            predicted_edges=predicted_relations,
            linked_entities=linked_entities,
            kg_resolution_report=kg_resolution_report,
            concept_alignment_report=concept_alignment_report,
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
                    **({"incremental_resolution": incremental_report} if incremental_report else {}),
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
    ) -> tuple[list[list[str]], list[MatchCandidate]]:
        """Generate candidate pairs and run LLM matching.

        Uses semantic blocking if available, otherwise falls back to
        pairwise comparison on entities with embeddings.

        Returns:
            (merge_groups, match_candidates) — merge groups for dedup,
            and all match decisions with provenance for the resolution log.
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
            return [], []

        logger.info(
            f"LLM matcher: {len(candidate_pairs)} candidate pairs "
            f"from {len(blocks)} blocks"
        )

        # Run LLM matching
        results = await self._llm_matcher.match_batch(candidate_pairs)

        # Build MatchCandidate records for all decisions (stap 2C/2B)
        match_candidates: list[MatchCandidate] = []
        for (a, b), res in zip(candidate_pairs, results):
            is_match = res.get("match", False)
            conf = res.get("confidence", 0.0)
            candidate = MatchCandidate(
                entity_a_text=a.get("text", ""),
                entity_b_text=b.get("text", ""),
                entity_a_label=a.get("label", "UNKNOWN"),
                entity_b_label=b.get("label", "UNKNOWN"),
                match=is_match,
                confidence=conf,
                match_method=res.get("match_method", "llm_match"),
                match_reasoning=res.get("reasoning", ""),
                iterations=res.get("iterations", 1),
                source_section_a=res.get("source_section_a"),
                source_section_b=res.get("source_section_b"),
                source_document_a=res.get("source_document_a"),
                source_document_b=res.get("source_document_b"),
                matched_by_model=res.get("matched_by_model"),
                # Auto-accept high-confidence matches; flag uncertain ones for review
                status="auto_accepted" if (is_match and conf >= self._llm_matcher._confidence_threshold) else "pending",
            )
            match_candidates.append(candidate)

        match_indices = self._llm_matcher.filter_matches(results)

        if not match_indices:
            return [], match_candidates

        # Build merge groups via UnionFind
        uf = UnionFind()
        for idx in match_indices:
            a_text = candidate_pairs[idx][0].get("text", "")
            b_text = candidate_pairs[idx][1].get("text", "")
            uf.union(a_text, b_text)

            a_entity = candidate_pairs[idx][0]
            b_entity = candidate_pairs[idx][1]
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
            f"{len(merge_groups)} merge groups, "
            f"{len(match_candidates)} candidates logged"
        )
        return merge_groups, match_candidates
