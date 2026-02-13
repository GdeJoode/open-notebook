"""
Ontology-guided entity and relation extraction via LLM.

This pipeline takes text chunks and an ontology schema, sends them to an LLM
(via llm-manager), and returns structured entities and relations.
"""

from ontology_extraction.config import ExtractionConfig
from ontology_extraction.extractors.base import ExtractorBase
from ontology_extraction.extractors.llm_extractor import LLMExtractor
from ontology_extraction.workflow import ExtractionWorkflow

__all__ = [
    "ExtractionConfig",
    "ExtractionWorkflow",
    "ExtractorBase",
    "LLMExtractor",
]
