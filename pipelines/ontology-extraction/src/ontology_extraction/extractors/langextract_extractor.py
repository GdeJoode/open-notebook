"""LangExtract-based extractor using few-shot examples for entity extraction."""

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger
from ontology_manager.schema import Ontology
from shared.models.extraction import (
    ExtractedEntity,
    ExtractionResult,
)

from .base import ExtractorBase
from .example_builder import ExampleBuilder

try:
    import langextract as lx
except ImportError:
    lx = None  # type: ignore[assignment]


class LangExtractExtractor(ExtractorBase):
    """Extract entities using LangExtract's few-shot example-driven approach.

    LangExtract uses example-based extraction patterns instead of ontology-guided
    system prompts, and returns results with precise source grounding (character
    offsets mapping extractions back to exact text positions).

    Note: LangExtract extracts flat entities, not relations. Relation inference
    is handled by the downstream entity-filtering pipeline's EdgePredictor.
    """

    def __init__(
        self,
        model_id: str = "qwen2.5:latest",
        model_url: str | None = "http://localhost:11434",
        confidence_threshold: float = 0.5,
        extraction_passes: int = 1,
        max_workers: int = 4,
        max_char_buffer: int = 5000,
        examples_dir: Path | None = None,
        # Performance tuning
        batch_length: int | None = None,
        # Model behavior
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        # Schema/output control
        use_schema_constraints: bool = True,
        fence_output: bool = True,
        # Provider configuration
        api_key: str | None = None,
        provider: str | None = None,
        provider_kwargs: dict | None = None,
        language_model_params: dict | None = None,
        # I/O features
        save_jsonl: bool = False,
        jsonl_output_dir: str | None = None,
        visualize: bool = False,
        visualize_output_dir: str | None = None,
    ):
        if lx is None:
            raise ImportError(
                "langextract is required for LangExtractExtractor. "
                "Install it with: pip install langextract"
            )
        self._model_id = model_id
        self._model_url = model_url
        self._confidence_threshold = confidence_threshold
        self._extraction_passes = extraction_passes
        self._max_workers = max_workers
        self._max_char_buffer = max_char_buffer
        self._example_builder = ExampleBuilder(examples_dir=examples_dir)

        # New parameters
        self._batch_length = batch_length
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._top_p = top_p
        self._top_k = top_k
        self._use_schema_constraints = use_schema_constraints
        self._fence_output = fence_output
        self._api_key = api_key
        self._provider = provider
        self._provider_kwargs = provider_kwargs
        self._language_model_params = language_model_params
        self._save_jsonl = save_jsonl
        self._jsonl_output_dir = jsonl_output_dir
        self._visualize = visualize
        self._visualize_output_dir = visualize_output_dir

    async def extract(
        self,
        text: str,
        ontology: Ontology,
        chunk_id: str | None = None,
        additional_context: str | None = None,
    ) -> ExtractionResult:
        """Extract entities from text using LangExtract.

        1. Build ExampleData from ontology + YAML overrides
        2. Construct lx.data.Document if chunk_id is provided
        3. Call lx.extract() with configured model (sync, wrapped in to_thread)
        4. Optionally save JSONL and/or generate visualization
        5. Map Extraction objects -> ExtractedEntity
        6. Return ExtractionResult with source grounding in metadata
        """
        try:
            prompt, examples = self._example_builder.build_examples(ontology)

            if not examples:
                logger.warning(
                    "No examples available for LangExtract extraction. "
                    "Provide YAML overrides or ensure ontology entity types have examples."
                )
                return ExtractionResult(
                    metadata={"warning": "no examples available for langextract"}
                )

            # Build input: Document object (if chunk_id provided) or plain string
            if chunk_id is not None:
                doc_kwargs: dict[str, Any] = {
                    "text": text,
                    "document_id": chunk_id,
                }
                if additional_context is not None:
                    doc_kwargs["additional_context"] = additional_context
                text_or_documents = lx.data.Document(**doc_kwargs)
            else:
                text_or_documents = text

            # lx.extract() is synchronous — run in a thread
            extract_kwargs: dict[str, Any] = {
                "text_or_documents": text_or_documents,
                "prompt_description": prompt,
                "examples": examples,
                "model_id": self._model_id,
                "extraction_passes": self._extraction_passes,
                "max_workers": self._max_workers,
                "max_char_buffer": self._max_char_buffer,
                "use_schema_constraints": self._use_schema_constraints,
                "fence_output": self._fence_output,
            }
            if self._model_url is not None:
                extract_kwargs["model_url"] = self._model_url
            if self._batch_length is not None:
                extract_kwargs["batch_length"] = self._batch_length
            if self._provider is not None:
                extract_kwargs["provider"] = self._provider
            if self._language_model_params is not None:
                extract_kwargs["language_model_params"] = self._language_model_params

            # Build provider kwargs from model behavior params + explicit provider_kwargs
            prov_kwargs: dict[str, Any] = {}
            if self._temperature is not None:
                prov_kwargs["temperature"] = self._temperature
            if self._max_output_tokens is not None:
                prov_kwargs["max_output_tokens"] = self._max_output_tokens
            if self._top_p is not None:
                prov_kwargs["top_p"] = self._top_p
            if self._top_k is not None:
                prov_kwargs["top_k"] = self._top_k
            if self._api_key is not None:
                prov_kwargs["api_key"] = self._api_key
            if self._provider_kwargs:
                prov_kwargs.update(self._provider_kwargs)

            result = await asyncio.to_thread(
                lx.extract, **extract_kwargs, **prov_kwargs
            )

            # Post-extraction I/O
            if self._save_jsonl:
                try:
                    save_kwargs: dict[str, Any] = {}
                    if self._jsonl_output_dir is not None:
                        save_kwargs["output_dir"] = self._jsonl_output_dir
                    docs = result if isinstance(result, list) else [result]
                    await asyncio.to_thread(
                        lx.io.save_annotated_documents, docs, **save_kwargs
                    )
                except Exception as e:
                    logger.warning(f"Failed to save JSONL output: {e}")

            if self._visualize:
                try:
                    viz_kwargs: dict[str, Any] = {}
                    if self._visualize_output_dir is not None:
                        viz_kwargs["output_dir"] = self._visualize_output_dir
                    docs = result if isinstance(result, list) else [result]
                    await asyncio.to_thread(lx.visualize, docs, **viz_kwargs)
                except Exception as e:
                    logger.warning(f"Failed to generate visualization: {e}")

            return self._map_result(result)

        except Exception as e:
            logger.error(f"LangExtract extraction failed: {e}")
            return ExtractionResult(metadata={"error": str(e)})

    def _map_result(self, result: Any) -> ExtractionResult:
        """Map LangExtract AnnotatedDocument(s) -> ExtractionResult.

        Converts each lx.data.Extraction to an ExtractedEntity with
        source grounding metadata. Accepts a single AnnotatedDocument
        or a list of them.
        """
        if isinstance(result, list):
            all_entities: list[ExtractedEntity] = []
            for doc in result:
                sub_result = self._map_single_document(doc)
                all_entities.extend(sub_result.entities)
            return ExtractionResult(
                entities=all_entities,
                relations=[],
                metadata={
                    "extractor": "langextract",
                    "model_id": self._model_id,
                    "entity_count": len(all_entities),
                    "extraction_passes": self._extraction_passes,
                    "document_count": len(result),
                },
            )
        return self._map_single_document(result)

    def _map_single_document(self, annotated_doc: Any) -> ExtractionResult:
        """Map a single LangExtract AnnotatedDocument -> ExtractionResult."""
        entities: list[ExtractedEntity] = []

        # Capture document_id if available
        document_id = getattr(annotated_doc, "document_id", None)

        for extraction in annotated_doc.extractions:
            confidence = self._map_confidence(extraction)
            if confidence < self._confidence_threshold:
                continue

            # Build source grounding from char_interval
            grounding: dict[str, Any] | None = None
            if extraction.char_interval:
                grounding = {
                    "start_pos": extraction.char_interval.start_pos,
                    "end_pos": extraction.char_interval.end_pos,
                }

            entity = ExtractedEntity(
                text=extraction.extraction_text or "",
                label=extraction.extraction_class or "UNKNOWN",
                properties=dict(extraction.attributes) if extraction.attributes else {},
                confidence=confidence,
                source_grounding=grounding,
            )
            entities.append(entity)

        metadata: dict[str, Any] = {
            "extractor": "langextract",
            "model_id": self._model_id,
            "entity_count": len(entities),
            "extraction_passes": self._extraction_passes,
        }
        if document_id is not None:
            metadata["document_id"] = document_id

        return ExtractionResult(
            entities=entities,
            relations=[],  # LangExtract does not extract relations
            metadata=metadata,
        )

    @staticmethod
    def _map_confidence(extraction: Any) -> float:
        """Map LangExtract alignment status to a numeric confidence score.

        LangExtract uses char_interval presence and text matching rather than
        numeric confidence. We use a simple heuristic:
        - Has char_interval (exact match found in text) -> 1.0
        - No char_interval (text not found verbatim) -> 0.6
        """
        if extraction.char_interval is not None:
            return 1.0
        return 0.6
