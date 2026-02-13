"""Orchestrator that processes batches of text chunks for extraction."""

from typing import Any, Dict, List, Optional

from loguru import logger
from ontology_manager import get_ontology_manager
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)

from .config import ExtractionConfig
from .extractors.base import ExtractorBase
from .extractors.llm_extractor import LLMExtractor


class ExtractionWorkflow:
    """Orchestrates ontology-guided extraction across text chunks."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self._config = config or ExtractionConfig()
        self._extractor: Optional[ExtractorBase] = None

    def _get_extractor(self) -> ExtractorBase:
        if self._extractor is None:
            self._extractor = LLMExtractor(
                llm_model=self._config.llm_model,
                confidence_threshold=self._config.confidence_threshold,
            )
        return self._extractor

    async def extract(self, chunks: List[Dict[str, Any]]) -> ExtractionResult:
        """
        Extract entities and relations from a list of text chunks.

        Args:
            chunks: List of dicts with at least a "text" key.
                   Optional "id" key for chunk tracking.

        Returns:
            Combined ExtractionResult from all chunks.
        """
        manager = get_ontology_manager()
        ontology = await manager.get_ontology(self._config.ontology_name)
        if not ontology:
            logger.error(f"Ontology '{self._config.ontology_name}' not found")
            return ExtractionResult(
                metadata={
                    "error": f"Ontology not found: {self._config.ontology_name}"
                }
            )

        extractor = self._get_extractor()
        all_entities: List[ExtractedEntity] = []
        all_relations: List[ExtractedRelation] = []

        # Process chunks in batches
        for i in range(0, len(chunks), self._config.batch_size):
            batch = chunks[i : i + self._config.batch_size]
            for chunk in batch:
                text = chunk.get("text", "")
                chunk_id = chunk.get("id")

                if not text.strip():
                    continue

                result = await extractor.extract(text, ontology)

                # Tag entities with chunk_id
                for entity in result.entities:
                    entity.source_chunk_id = chunk_id
                    all_entities.append(entity)

                for relation in result.relations:
                    relation.source_chunk_id = chunk_id
                    all_relations.append(relation)

            logger.info(
                f"Processed batch {i // self._config.batch_size + 1}: "
                f"{len(all_entities)} entities, {len(all_relations)} relations so far"
            )

        return ExtractionResult(
            entities=all_entities,
            relations=all_relations,
            metadata={
                "ontology_name": self._config.ontology_name,
                "chunk_count": len(chunks),
                "total_entities": len(all_entities),
                "total_relations": len(all_relations),
            },
        )

    async def extract_single(
        self, text: str, chunk_id: Optional[str] = None
    ) -> ExtractionResult:
        """Extract from a single text. Convenience wrapper around extract()."""
        return await self.extract([{"text": text, "id": chunk_id}])
