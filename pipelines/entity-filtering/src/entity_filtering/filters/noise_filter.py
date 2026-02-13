"""
Noise filter for removing noisy or invalid entities.

Applies a set of built-in patterns (citations, single characters,
pure numbers, URLs, emails) plus any user-supplied custom patterns.
Relations referencing removed entities are also removed.
"""

import re
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from entity_filtering.filters.base import FilterBase

# Default patterns that are almost universally noise in entity extraction.
# These are intentionally generic and domain-agnostic.
_DEFAULT_NOISE_PATTERNS: List[str] = [
    # Citation artifacts
    r"\bet\s+al\.?\b",
    r"\bibid\.?\b",
    r"\bop\.?\s*cit\.?\b",
    r"\bloc\.?\s*cit\.?\b",
    r"\bcf\.?\b",
    # Pure numeric strings (integers, decimals, percentages)
    r"^\d[\d.,:%]*$",
    # URLs
    r"^https?://\S+$",
    # Email addresses
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
    # File paths
    r"^[/\\]?(?:\w+[/\\])+\w+(?:\.\w+)?$",
    # Isolated punctuation or symbols
    r"^[\W_]+$",
]


class NoiseFilter(FilterBase):
    """Removes noisy entities using regex patterns.

    Built-in patterns cover common extraction artifacts.
    Additional patterns can be supplied via ``custom_patterns``.

    Args:
        custom_patterns: Extra regex patterns to treat as noise.
        min_entity_length: Minimum character count for an entity
            to be kept.
    """

    def __init__(
        self,
        custom_patterns: Optional[List[str]] = None,
        min_entity_length: int = 2,
    ) -> None:
        self._min_length = min_entity_length
        raw_patterns = list(_DEFAULT_NOISE_PATTERNS)
        if custom_patterns:
            raw_patterns.extend(custom_patterns)
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in raw_patterns
        ]

    # ------------------------------------------------------------------
    # FilterBase implementation
    # ------------------------------------------------------------------

    def filter_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove entities matching any noise pattern or below min length."""
        kept: List[Dict[str, Any]] = []
        removed_count = 0
        for entity in entities:
            text = entity.get("text", "").strip()
            if self._is_noise(text):
                removed_count += 1
                continue
            kept.append(entity)
        if removed_count:
            logger.debug("NoiseFilter removed {} entities", removed_count)
        return kept

    def filter_relations(
        self,
        relations: List[Dict[str, Any]],
        valid_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep only relations whose source and target are in valid_entities."""
        valid_texts: Set[str] = {e.get("text", "") for e in valid_entities}
        kept: List[Dict[str, Any]] = []
        for rel in relations:
            src = rel.get("source_entity", "")
            tgt = rel.get("target_entity", "")
            if src in valid_texts and tgt in valid_texts:
                kept.append(rel)
        return kept

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_noise(self, text: str) -> bool:
        """Return True if the text matches any noise criterion."""
        if len(text) < self._min_length:
            return True
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                return True
        return False
