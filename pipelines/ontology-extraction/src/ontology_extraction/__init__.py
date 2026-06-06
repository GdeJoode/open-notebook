"""
Ontology-guided entity and relation extraction via LLM.

This pipeline takes text chunks and an ontology schema, sends them to an LLM
(via llm-manager), and returns structured entities and relations.
"""

from ontology_extraction.config import ExtractionConfig
from ontology_extraction.extractors.base import ExtractorBase
from ontology_extraction.extractors.llm_extractor import LLMExtractor
from ontology_extraction.pass1_schema_validation import (
    Pass1Output,
    Pass1ParseError,
    Pass1SchemaValidator,
    TokenBudgetExceeded,
)
from ontology_extraction.pass2_typed_extraction import (
    Pass2ParseError,
    Pass2TokenBudgetExceeded,
    run_pass2,
)
from ontology_extraction.multi_schema_orchestrator import (
    MIN_APPLICABLE_CONFIDENCE,
    SOFT_NUDGE_COVERAGE_HIGH,
    SOFT_NUDGE_COVERAGE_LOW,
    SoftNudgeDecision,
    detect_applicable_schemas,
    run_multi_schema,
)
from ontology_extraction.workflow import ExtractionWorkflow

__all__ = [
    "ExtractionConfig",
    "ExtractionWorkflow",
    "ExtractorBase",
    "LLMExtractor",
    "MIN_APPLICABLE_CONFIDENCE",
    "Pass1Output",
    "Pass1ParseError",
    "Pass1SchemaValidator",
    "Pass2ParseError",
    "Pass2TokenBudgetExceeded",
    "SOFT_NUDGE_COVERAGE_HIGH",
    "SOFT_NUDGE_COVERAGE_LOW",
    "SoftNudgeDecision",
    "TokenBudgetExceeded",
    "detect_applicable_schemas",
    "run_multi_schema",
    "run_pass2",
]
