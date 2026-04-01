"""LLM-based extractor using ontology-guided prompts."""

import json

from loguru import logger
from ontology_manager import OntologyPromptGenerator
from ontology_manager.schema import Ontology
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)

from .base import ExtractorBase


class LLMExtractor(ExtractorBase):
    """Extract entities and relations using LLM with ontology-guided prompts."""

    def __init__(
        self, llm_model: str = "default", confidence_threshold: float = 0.5
    ):
        self._llm_model = llm_model
        self._confidence_threshold = confidence_threshold

    async def extract(self, text: str, ontology: Ontology, **kwargs) -> ExtractionResult:
        """Extract entities and relations using LLM with ontology-guided prompts."""
        generator = OntologyPromptGenerator(ontology)
        system_prompt = generator.generate_combined_extraction_prompt(
            include_concepts=True, include_claims=True
        )
        user_prompt = f"Extract knowledge from the following text:\n\n{text}"

        try:
            # Call LLM via llm-manager (import lazily)
            from llm_manager.manager import LLMManager

            manager = LLMManager()
            response = await manager.generate(
                model_name=self._llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            return self._parse_response(response)
        except ImportError:
            logger.warning("llm-manager not available, returning empty result")
            return ExtractionResult()
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return ExtractionResult(metadata={"error": str(e)})

    def _parse_response(self, response: str) -> ExtractionResult:
        """Parse LLM JSON response into ExtractionResult."""
        try:
            # Try to extract JSON from response
            text = response.strip()
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)

            entities = []
            for e in data.get("entities", []):
                entity = ExtractedEntity(
                    text=e.get("name", ""),
                    label=e.get("entity_type", "UNKNOWN"),
                    properties=e.get("properties", {}),
                    confidence=float(e.get("confidence", 1.0)),
                )
                if entity.confidence >= self._confidence_threshold:
                    entities.append(entity)

            relations = []
            for r in data.get("relationships", []):
                relation = ExtractedRelation(
                    source_entity=r.get("subject", ""),
                    target_entity=r.get("object", ""),
                    relation_type=r.get("predicate", "RELATED_TO"),
                    properties=r.get("properties", {}),
                    confidence=float(r.get("confidence", 1.0)),
                )
                if relation.confidence >= self._confidence_threshold:
                    relations.append(relation)

            return ExtractionResult(
                entities=entities,
                relations=relations,
                metadata={
                    "concept_count": len(data.get("concepts", [])),
                    "claim_count": len(data.get("claims", [])),
                },
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return ExtractionResult(metadata={"parse_error": str(e)})
