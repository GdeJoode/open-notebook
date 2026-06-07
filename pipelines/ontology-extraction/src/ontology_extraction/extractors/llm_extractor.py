"""LLM-based extractor using ontology-guided prompts.

Back-compat shim. New code should call
``ontology_extraction.run_pass2`` (B.1d) directly — it adds
accepted-extension injection and enforces the B4 confidence-
everywhere invariant. This module stays in place so existing
wiring (``workflow.py`` + tests) keeps working unchanged.

Phase B.1f changes
==================

The pre-B.1f code attempted ``from llm_manager.manager import LLMManager``
and called ``manager.generate(...)``. Both symbols are wrong: the class
is :class:`llm_manager.manager.ModelManager` and there is no
``generate`` method — the post-rename API is
``ModelManager().get_model_from_config(model).achat_complete([...])``
returning an ``esperanto`` ``ChatCompletion``. Until B.1f the import
raised ``ImportError`` and the extractor silently produced an empty
result (a silent-failure footgun).

This module now switches to **dependency injection**: callers pass an
async ``llm_caller`` with the same signature as Pass-2's
:type:`LLMCaller` alias (``(system_prompt, user_prompt, model) -> str``).
Defaulting to ``None`` preserves the legacy "no LLM available → empty
result + WARNING" behaviour, but in production
:class:`app_main.services.entity_extraction_service.EntityExtractionService`
wires :func:`_make_default_llm_caller` which goes through
:class:`ModelManager` properly.
"""

from __future__ import annotations

import json
from typing import Awaitable, Callable, Optional, Union

from loguru import logger
from ontology_manager import OntologyPromptGenerator
from ontology_manager.schema import Ontology
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)

from .base import ExtractorBase

# Mirrors ``ontology_extraction.pass2_typed_extraction.LLMCaller`` so
# the same caller can be passed to either module. We re-declare here
# rather than import to avoid a circular dependency (pass2 imports
# from the package root and from the extractors subpackage).
SyncLLMCaller = Callable[[str, str, str], str]
AsyncLLMCaller = Callable[[str, str, str], Awaitable[str]]
LLMCaller = Union[SyncLLMCaller, AsyncLLMCaller]


class LLMExtractor(ExtractorBase):
    """Extract entities and relations using LLM with ontology-guided prompts.

    Args:
        llm_model: Model identifier forwarded to the injected caller.
        confidence_threshold: Per-element confidence floor; the parser
            drops entities/relations below this value.
        llm_caller: Optional sync or async callable
            ``(system_prompt, user_prompt, model) -> str``. ``None``
            triggers the legacy "no caller available → empty result +
            WARNING" path. Production wires
            :func:`make_default_llm_caller` from
            ``app_main.services.entity_extraction_service``.
    """

    def __init__(
        self,
        llm_model: str = "default",
        confidence_threshold: float = 0.5,
        llm_caller: Optional[LLMCaller] = None,
    ):
        self._llm_model = llm_model
        self._confidence_threshold = confidence_threshold
        self._llm_caller = llm_caller

    async def extract(self, text: str, ontology: Ontology, **kwargs) -> ExtractionResult:
        """Extract entities and relations using LLM with ontology-guided prompts."""
        generator = OntologyPromptGenerator(ontology)
        system_prompt = generator.generate_combined_extraction_prompt(
            include_concepts=True, include_claims=True
        )
        user_prompt = f"Extract knowledge from the following text:\n\n{text}"

        # No caller injected → log canary and return empty (matches the
        # pre-B.1f silent-empty behaviour but with an explicit WARNING).
        if self._llm_caller is None:
            logger.warning(
                "LLMExtractor invoked without an llm_caller; returning "
                "empty result. Wire one via the constructor for "
                "production use (B.1f)."
            )
            return ExtractionResult()

        try:
            # Both sync and async callers supported — the result of the
            # call is awaited iff it has an ``__await__`` attribute.
            raw = self._llm_caller(system_prompt, user_prompt, self._llm_model)
            if hasattr(raw, "__await__"):
                raw = await raw  # type: ignore[misc]
            response = str(raw)
            return self._parse_response(response)
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
