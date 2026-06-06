"""Prompt templates for ontology-extraction pipeline.

Kept in code (not YAML) so they can be unit-tested as plain Python
strings. Mirrors the ``ontology_manager.prompts`` pattern.
"""

from ontology_extraction.prompts.pass1 import (
    CANDIDATE_SCHEMA_NAMES,
    COMPACT_SUMMARY_THRESHOLD,
    PASS1_SYSTEM_PROMPT,
    build_pass1_prompt,
    build_schema_summary,
)
from ontology_extraction.prompts.pass2 import (
    COMPACT_RELATION_THRESHOLD,
    COMPACT_TYPE_THRESHOLD,
    PASS2_SYSTEM_PROMPT,
    build_pass2_prompt,
)

__all__ = [
    "CANDIDATE_SCHEMA_NAMES",
    "COMPACT_RELATION_THRESHOLD",
    "COMPACT_SUMMARY_THRESHOLD",
    "COMPACT_TYPE_THRESHOLD",
    "PASS1_SYSTEM_PROMPT",
    "PASS2_SYSTEM_PROMPT",
    "build_pass1_prompt",
    "build_pass2_prompt",
    "build_schema_summary",
]
