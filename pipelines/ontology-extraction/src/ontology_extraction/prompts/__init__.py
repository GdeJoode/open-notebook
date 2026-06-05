"""Prompt templates for ontology-extraction pipeline.

Kept in code (not YAML) so they can be unit-tested as plain Python
strings. Mirrors the ``ontology_manager.prompts`` pattern.
"""

from ontology_extraction.prompts.pass1 import (
    build_pass1_prompt,
    build_schema_summary,
)

__all__ = [
    "build_pass1_prompt",
    "build_schema_summary",
]
