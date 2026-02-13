"""Extractors for ontology-guided knowledge extraction."""

from ontology_extraction.extractors.base import ExtractorBase
from ontology_extraction.extractors.llm_extractor import LLMExtractor

__all__ = [
    "ExtractorBase",
    "LLMExtractor",
]
