"""
Service bridging the app layer to the ontology-extraction and entity-filtering pipelines.

Fetches source chunks, runs ExtractionWorkflow, optionally runs
FilteringWorkflow for deduplication, and persists results to SurrealDB.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories import SourceRepository

from ontology_extraction.config import ExtractionConfig
from ontology_extraction.workflow import ExtractionWorkflow

from entity_filtering.config import FilteringConfig
from entity_filtering.workflow import FilteringWorkflow

from app_main.services.entity_persistence_service import EntityPersistenceService


class EntityExtractionService:
    """Runs ontology-guided entity extraction on a source's chunks."""

    def __init__(self, source_repo: SourceRepository):
        self._source_repo = source_repo
        self._persistence = EntityPersistenceService()

    async def _embed_entities(
        self, result: "ExtractionResult"
    ) -> None:
        """Embed entity texts and store vectors in entity properties.

        This enables embedding-based deduplication in the FilteringWorkflow.
        Runs in-place — modifies entity properties directly.
        """
        texts = [e.text for e in result.entities if e.text.strip()]
        if not texts:
            return

        try:
            from app_main.dependencies import get_embedding_service

            embed_svc = await get_embedding_service()
            # Batch embed all entity texts
            vectors = await embed_svc.embedding_model.aembed(texts)

            # Map vectors back to entities
            text_to_vec = dict(zip(texts, vectors))
            embedded = 0
            for entity in result.entities:
                vec = text_to_vec.get(entity.text)
                if vec:
                    entity.properties["embedding"] = vec
                    embedded += 1

            logger.info(f"Embedded {embedded}/{len(result.entities)} entities")
        except Exception as e:
            logger.warning(
                f"Entity embedding failed (dedup will use string-only): {e}"
            )

    async def run_extraction(
        self,
        source_id: str,
        ontology_name: str = "general",
        extractor_type: str = "llm",
        config_overrides: Dict[str, Any] | None = None,
        run_filtering: bool = True,
        filtering_config: Optional[FilteringConfig] = None,
    ) -> Dict[str, Any]:
        """
        Run entity extraction and optional filtering for a source.

        1. Fetch chunks via SourceRepository.
        2. Build ExtractionConfig and ExtractionWorkflow.
        3. Run extraction.
        4. Optionally run FilteringWorkflow for dedup/enrichment.
        5. Persist raw results to ``extraction_result`` table.
        6. Persist filtered entities to KG tables (entity, relation).
        7. Return summary dict.
        """
        # TODO(B.1f): Insert Pass-1 schema validation here. The
        # ``Pass1SchemaValidator`` module (B.1c) is implemented and
        # importable; B.1f will sample chunks, run it, and persist
        # the result via ``Pass1ResultRepository.record(...)`` before
        # the typed-extraction call below.
        logger.info(f"Starting entity extraction for source: {source_id}")

        # 1. Fetch chunks
        chunks = await self._source_repo.get_chunks(source_id)
        if not chunks:
            logger.warning(f"No chunks found for source {source_id}")
            return {
                "source_id": source_id,
                "entity_count": 0,
                "relation_count": 0,
            }

        # 2. Convert to workflow format — include structural metadata for
        #    section-aware entity extraction (stap 2D).
        chunk_dicts = []
        for c in chunks:
            if not c.text:
                continue
            d: Dict[str, Any] = {"text": c.text, "id": str(c.id)}
            # Carry document structure through to extraction
            d["section_path"] = c.section_path or []
            d["section_level"] = c.section_level
            d["physical_page"] = c.physical_page
            d["element_type"] = c.element_type
            d["source_id"] = source_id
            if c.section_path:
                d["section_heading"] = c.section_path[-1]
            chunk_dicts.append(d)

        # 3. Build config and workflow
        config_kwargs: Dict[str, Any] = {
            "ontology_name": ontology_name,
            "extractor_type": extractor_type,
        }
        if config_overrides:
            config_kwargs.update(config_overrides)
        config = ExtractionConfig(**config_kwargs)
        workflow = ExtractionWorkflow(config)

        # 4. Run extraction
        result = await workflow.extract(chunk_dicts)

        # Store extractor_type in metadata
        if not hasattr(result, "metadata") or result.metadata is None:
            result.metadata = {}
        result.metadata["extractor_type"] = extractor_type

        # 4b. Embed entity texts for semantic dedup
        if result.entities:
            await self._embed_entities(result)

        # 5. Optionally run filtering/deduplication
        filtered_entities = result.entities
        filtered_relations = result.relations
        merge_groups = None
        filtering_stats = {}

        if run_filtering and (result.entities or result.relations):
            try:
                if filtering_config:
                    f_config = filtering_config
                else:
                    # Default config: string dedup + fuzzy + embedding
                    from entity_filtering.config import (
                        EmbeddingDedupConfig,
                        FuzzyDedupConfig,
                    )

                    f_config = FilteringConfig(
                        dedup_enabled=True,
                        fuzzy_dedup=FuzzyDedupConfig(
                            enabled=True,
                            algorithm="levenshtein",
                            similarity_threshold=0.85,
                        ),
                        embedding_dedup=EmbeddingDedupConfig(
                            enabled=True,
                            similarity_threshold=0.90,
                        ),
                        edge_prediction_enabled=True,
                    )
                f_workflow = FilteringWorkflow(config=f_config)

                filtered = await f_workflow.process(result)

                merge_groups = filtered.merged_entity_groups
                all_relations = [
                    r.model_dump() for r in filtered.relations
                ] + [
                    r.model_dump() for r in filtered.predicted_edges
                ]

                filtering_stats = {
                    "entities_before": len(result.entities),
                    "entities_after": len(filtered.entities),
                    "entities_removed": len(filtered.removed_entities),
                    "merge_groups": len(merge_groups) if merge_groups else 0,
                    "predicted_edges": len(filtered.predicted_edges),
                }
                result.metadata["filtering"] = filtering_stats

                logger.info(
                    f"Filtering complete for source {source_id}: "
                    f"{filtering_stats}"
                )

                # 6. Persist filtered entities to KG
                await self._persistence.persist_filtered_result(
                    source_id=source_id,
                    entities=[e.model_dump() for e in filtered.entities],
                    relations=all_relations,
                    merge_groups=merge_groups,
                    match_candidates=[c.model_dump() for c in filtered.match_candidates] if filtered.match_candidates else None,
                )

            except Exception as e:
                logger.error(f"Filtering failed for source {source_id}: {e}")
                # Fall through — raw results will still be saved

        # 7. Persist raw extraction results
        await self._save_result(source_id, result)

        summary = {
            "source_id": source_id,
            "entity_count": result.entity_count,
            "relation_count": result.relation_count,
            **filtering_stats,
        }
        logger.info(
            f"Entity extraction completed for source {source_id}: "
            f"{result.entity_count} entities, {result.relation_count} relations"
        )
        return summary

    async def run_filtering_only(
        self,
        source_id: str,
        filtering_config: Optional[FilteringConfig] = None,
    ) -> Dict[str, Any]:
        """Run filtering on an existing extraction result without re-extracting.

        Fetches the raw extraction_result, runs FilteringWorkflow, and
        persists filtered entities to the KG tables.
        """
        # Fetch existing extraction result
        rows = await execute_query(
            "SELECT * FROM extraction_result WHERE source_id = $source_id LIMIT 1",
            {"source_id": source_id},
        )
        if not rows:
            raise ValueError(f"No extraction result found for source {source_id}")

        row = rows[0]
        entity_dicts = row.get("entities", [])
        relation_dicts = row.get("relations", [])

        if not entity_dicts:
            return {
                "source_id": source_id,
                "entities_before": 0,
                "entities_after": 0,
                "entities_removed": 0,
                "merge_groups": 0,
                "predicted_edges": 0,
            }

        # Reconstruct ExtractionResult for the FilteringWorkflow
        from shared.models.extraction import (
            ExtractedEntity,
            ExtractedRelation,
            ExtractionResult,
        )

        extraction = ExtractionResult(
            entities=[ExtractedEntity(**e) for e in entity_dicts],
            relations=[ExtractedRelation(**r) for r in relation_dicts],
            metadata=row.get("metadata", {}),
        )

        # Embed entities for semantic dedup (if not already embedded)
        has_embeddings = any(
            e.properties.get("embedding") for e in extraction.entities
        )
        if not has_embeddings:
            await self._embed_entities(extraction)

        # Run filtering
        f_config = filtering_config or FilteringConfig()
        f_workflow = FilteringWorkflow(config=f_config)
        filtered = await f_workflow.process(extraction)

        # Persist to KG
        all_relations = [
            r.model_dump() for r in filtered.relations
        ] + [
            r.model_dump() for r in filtered.predicted_edges
        ]
        await self._persistence.persist_filtered_result(
            source_id=source_id,
            entities=[e.model_dump() for e in filtered.entities],
            relations=all_relations,
            merge_groups=filtered.merged_entity_groups,
            match_candidates=[c.model_dump() for c in filtered.match_candidates] if filtered.match_candidates else None,
        )

        stats = {
            "source_id": source_id,
            "entities_before": len(entity_dicts),
            "entities_after": len(filtered.entities),
            "entities_removed": len(filtered.removed_entities),
            "merge_groups": len(filtered.merged_entity_groups)
            if filtered.merged_entity_groups
            else 0,
            "predicted_edges": len(filtered.predicted_edges),
        }

        # Update extraction_result metadata with filtering stats
        metadata = row.get("metadata", {})
        metadata["filtering"] = stats
        await execute_query(
            "UPDATE extraction_result SET metadata = $metadata "
            "WHERE source_id = $source_id",
            {"source_id": source_id, "metadata": metadata},
        )

        logger.info(f"Filtering-only completed for source {source_id}: {stats}")
        return stats

    async def _save_result(self, source_id: str, result) -> None:
        """Persist extraction result to SurrealDB."""
        try:
            await execute_query(
                "DELETE FROM extraction_result WHERE source_id = $source_id",
                {"source_id": source_id},
            )
            await execute_query(
                "CREATE extraction_result SET "
                "source_id = $source_id, "
                "entities = $entities, "
                "relations = $relations, "
                "metadata = $metadata, "
                "entity_count = $entity_count, "
                "relation_count = $relation_count, "
                "created = time::now()",
                {
                    "source_id": source_id,
                    "entities": [e.model_dump() for e in result.entities],
                    "relations": [r.model_dump() for r in result.relations],
                    "metadata": result.metadata,
                    "entity_count": result.entity_count,
                    "relation_count": result.relation_count,
                },
            )
            logger.info(
                f"Saved extraction result for source {source_id}"
            )
        except Exception as e:
            logger.error(f"Failed to save extraction result: {e}")
            raise
