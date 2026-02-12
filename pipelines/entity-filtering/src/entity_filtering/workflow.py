"""
Main filtering workflow orchestrator.

Composes the individual filter stages (noise removal, normalization,
reclassification, deduplication, edge prediction) into a single
pipeline that transforms an ExtractionResult into a FilteredResult.
"""

from typing import Optional

from loguru import logger
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    FilteredResult,
)

from entity_filtering.config import FilteringConfig
from entity_filtering.deduplication.entity_deduplicator import EntityDeduplicator
from entity_filtering.filters.noise_filter import NoiseFilter
from entity_filtering.filters.normalizer import EntityNormalizer
from entity_filtering.filters.reclassifier import EntityReclassifier
from entity_filtering.scoring.edge_predictor import EdgePredictor


class FilteringWorkflow:
    """Orchestrates the entity filtering pipeline.

    Stages (in order):
    1. Noise filtering -- remove invalid / artifact entities
    2. Normalization -- canonical text forms, merge equivalents
    3. Reclassification -- fix entity labels via heuristic rules
    4. Deduplication -- merge entities with identical normalized text
    5. Edge prediction (optional) -- discover implicit relations

    Args:
        config: Pipeline configuration. Uses defaults when None.
    """

    def __init__(self, config: Optional[FilteringConfig] = None) -> None:
        self._config = config or FilteringConfig()

        self._noise_filter = NoiseFilter(
            custom_patterns=self._config.custom_noise_patterns,
            min_entity_length=self._config.min_entity_length,
        )
        self._normalizer = EntityNormalizer(
            strip_articles=self._config.strip_articles,
            custom_articles=self._config.custom_articles,
            normalize_whitespace=self._config.normalize_whitespace,
        )
        self._reclassifier = EntityReclassifier(
            custom_rules=self._config.custom_reclassification_rules,
        )
        self._deduplicator = EntityDeduplicator(
            similarity_threshold=self._config.dedup_similarity_threshold,
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
        # Stage 4: Deduplicate
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
        # Stage 5: Edge prediction (optional)
        # ------------------------------------------------------------------
        predicted_edges: list[dict] = []
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
