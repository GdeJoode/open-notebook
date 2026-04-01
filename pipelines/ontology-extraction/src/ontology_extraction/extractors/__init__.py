"""Extractors for ontology-guided knowledge extraction."""

from ontology_extraction.extractors.base import ExtractorBase
from ontology_extraction.extractors.example_builder import ExampleBuilder
from ontology_extraction.extractors.llm_extractor import LLMExtractor

# LangExtractExtractor is imported lazily to avoid requiring langextract
# when only using the LLM extractor.
__all__ = [
    "ExtractorBase",
    "ExampleBuilder",
    "LLMExtractor",
]
