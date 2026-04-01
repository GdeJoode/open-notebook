"""
Service bridging the app layer to the ontology-extraction pipeline.

Fetches source chunks, runs ExtractionWorkflow, and persists results
to the ``extraction_result`` SurrealDB table.
"""

from typing import Any, Dict

from loguru import logger
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories import SourceRepository

from ontology_extraction.config import ExtractionConfig
from ontology_extraction.workflow import ExtractionWorkflow


class EntityExtractionService:
    """Runs ontology-guided entity extraction on a source's chunks."""

    def __init__(self, source_repo: SourceRepository):
        self._source_repo = source_repo

    async def run_extraction(
        self,
        source_id: str,
        ontology_name: str = "general",
        extractor_type: str = "llm",
        config_overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Run entity extraction for a source.

        1. Fetch chunks via SourceRepository.
        2. Build ExtractionConfig and ExtractionWorkflow.
        3. Run extraction.
        4. Persist results to ``extraction_result`` table.
        5. Return summary dict.
        """
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

        # 2. Convert to workflow format
        chunk_dicts = [
            {"text": c.text, "id": str(c.id)} for c in chunks if c.text
        ]

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

        # 5. Persist to SurrealDB (upsert pattern from preprocessing_service)
        # Store extractor_type in metadata so the frontend can choose visualization
        if not hasattr(result, "metadata") or result.metadata is None:
            result.metadata = {}
        result.metadata["extractor_type"] = extractor_type
        await self._save_result(source_id, result)

        summary = {
            "source_id": source_id,
            "entity_count": result.entity_count,
            "relation_count": result.relation_count,
        }
        logger.info(
            f"Entity extraction completed for source {source_id}: "
            f"{result.entity_count} entities, {result.relation_count} relations"
        )
        return summary

    async def _save_result(self, source_id: str, result) -> None:
        """Persist extraction result to SurrealDB."""
        try:
            # Upsert: delete existing, then create
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
