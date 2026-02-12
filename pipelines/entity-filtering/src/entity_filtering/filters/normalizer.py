"""
Entity normalizer.

Handles text normalization: article stripping, whitespace collapsing,
unicode normalization. Merges entities that normalize to the same form,
keeping the most frequent surface form as canonical.
"""

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional

from loguru import logger

# Default leading articles to strip (common across many languages).
_DEFAULT_ARTICLES: List[str] = [
    "The ",
    "the ",
    "A ",
    "a ",
    "An ",
    "an ",
]


class EntityNormalizer:
    """Normalizes entity text and merges equivalent surface forms.

    Args:
        strip_articles: Whether to remove leading articles.
        custom_articles: Additional leading article strings to strip.
        normalize_whitespace: Whether to collapse whitespace.
    """

    def __init__(
        self,
        strip_articles: bool = True,
        custom_articles: Optional[List[str]] = None,
        normalize_whitespace: bool = True,
    ) -> None:
        self._strip_articles = strip_articles
        self._normalize_ws = normalize_whitespace
        self._articles = list(_DEFAULT_ARTICLES)
        if custom_articles:
            self._articles.extend(custom_articles)

    def normalize(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize entity texts and merge duplicates.

        Entities that map to the same normalized key are merged.
        The most common original surface form is kept as canonical.
        The highest confidence value across duplicates is preserved.

        Args:
            entities: List of entity dicts.

        Returns:
            Deduplicated list after normalization.
        """
        # Group entities by normalized key
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for entity in entities:
            key = self._normalize_text(entity.get("text", ""))
            groups.setdefault(key, []).append(entity)

        result: List[Dict[str, Any]] = []
        merge_count = 0
        for key, group in groups.items():
            if not key:
                continue
            canonical = self._select_canonical(group)
            result.append(canonical)
            if len(group) > 1:
                merge_count += 1

        if merge_count:
            logger.debug(
                "Normalizer merged {} groups ({} entities -> {})",
                merge_count,
                len(entities),
                len(result),
            )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        """Produce a normalized key from raw entity text."""
        # Unicode normalization (NFKC)
        text = unicodedata.normalize("NFKC", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Strip leading articles
        if self._strip_articles:
            for article in self._articles:
                if text.startswith(article):
                    text = text[len(article) :]
                    break

        # Collapse whitespace
        if self._normalize_ws:
            text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def _select_canonical(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pick the best representative from a group of equivalent entities.

        Strategy:
        - Use the most frequent surface form.
        - Take the highest confidence value.
        - Merge properties from all variants.
        """
        text_counts: Counter[str] = Counter(e.get("text", "") for e in group)
        canonical_text = text_counts.most_common(1)[0][0]

        # Start from the first entity with canonical text
        base = next(e for e in group if e.get("text") == canonical_text).copy()

        # Merge properties and take max confidence
        merged_props: Dict[str, Any] = {}
        max_confidence = 0.0
        for entity in group:
            merged_props.update(entity.get("properties", {}))
            conf = entity.get("confidence", 0.0)
            if conf > max_confidence:
                max_confidence = conf

        base["properties"] = merged_props
        base["confidence"] = max_confidence
        return base
