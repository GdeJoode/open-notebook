"""Orchestrator that processes batches of text chunks for extraction."""

import asyncio
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
            if self._config.extractor_type == "langextract":
                from pathlib import Path

                from .extractors.langextract_extractor import LangExtractExtractor

                self._extractor = LangExtractExtractor(
                    model_id=self._config.langextract_model_id,
                    model_url=self._config.langextract_model_url,
                    confidence_threshold=self._config.confidence_threshold,
                    extraction_passes=self._config.langextract_extraction_passes,
                    max_workers=self._config.langextract_max_workers,
                    max_char_buffer=self._config.langextract_max_char_buffer,
                    examples_dir=Path(self._config.langextract_examples_dir)
                    if self._config.langextract_examples_dir
                    else None,
                    batch_length=self._config.langextract_batch_length,
                    temperature=self._config.langextract_temperature,
                    max_output_tokens=self._config.langextract_max_output_tokens,
                    top_p=self._config.langextract_top_p,
                    top_k=self._config.langextract_top_k,
                    use_schema_constraints=self._config.langextract_use_schema_constraints,
                    fence_output=self._config.langextract_fence_output,
                    api_key=self._config.langextract_api_key,
                    provider=self._config.langextract_provider,
                    provider_kwargs=self._config.langextract_provider_kwargs,
                    language_model_params=self._config.langextract_language_model_params,
                    save_jsonl=self._config.langextract_save_jsonl,
                    jsonl_output_dir=self._config.langextract_jsonl_output_dir,
                    visualize=self._config.langextract_visualize,
                    visualize_output_dir=self._config.langextract_visualize_output_dir,
                )
            else:
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

                # Pass chunk_id and additional_context for Document support
                extra_kwargs: Dict[str, Any] = {}
                if chunk_id is not None:
                    extra_kwargs["chunk_id"] = chunk_id
                additional_context = chunk.get("additional_context")
                if additional_context is not None:
                    extra_kwargs["additional_context"] = additional_context

                try:
                    coro = extractor.extract(text, ontology, **extra_kwargs)
                    if self._config.extraction_timeout > 0:
                        result = await asyncio.wait_for(
                            coro, timeout=self._config.extraction_timeout
                        )
                    else:
                        result = await coro
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Extraction timed out for chunk {chunk_id} "
                        f"after {self._config.extraction_timeout}s, skipping"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Extraction failed for chunk {chunk_id}: {e}"
                    )
                    continue

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
