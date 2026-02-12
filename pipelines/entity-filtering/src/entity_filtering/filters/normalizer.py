"""
Entity normalizer.

Handles text normalization: article stripping, whitespace collapsing,
unicode normalization, diacritics removal, OCR artifact cleanup, HTML tag
stripping, and page-number filtering.  Merges entities that normalize to
the same form, keeping the most frequent surface form as canonical.
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


_PAGE_NUMBER_RE = re.compile(r"^(page|p\.?|pp\.?)\s*\d+", re.IGNORECASE)


class EntityNormalizer:
    """Normalizes entity text and merges equivalent surface forms.

    Args:
        strip_articles: Whether to remove leading articles.
        custom_articles: Additional leading article strings to strip.
        normalize_whitespace: Whether to collapse whitespace.
        remove_diacritics: Whether to strip diacritics/accents (e.g. e->e).
        ocr_cleanup_enabled: Whether to remove common OCR artifacts.
        ocr_artifact_patterns: Custom regex patterns for OCR artifact removal.
            Each pattern is applied via ``re.sub(pattern, "", text)``.
        html_strip_enabled: Whether to strip HTML tags from entity text.
        page_number_filter: Whether to drop entities that are just page
            numbers (e.g. "Page 1", "p. 42").
    """

    def __init__(
        self,
        strip_articles: bool = True,
        custom_articles: Optional[List[str]] = None,
        normalize_whitespace: bool = True,
        remove_diacritics: bool = False,
        ocr_cleanup_enabled: bool = False,
        ocr_artifact_patterns: Optional[List[str]] = None,
        html_strip_enabled: bool = False,
        page_number_filter: bool = False,
    ) -> None:
        self._strip_articles = strip_articles
        self._normalize_ws = normalize_whitespace
        self._articles = list(_DEFAULT_ARTICLES)
        if custom_articles:
            self._articles.extend(custom_articles)

        self._remove_diacritics = remove_diacritics
        self._ocr_cleanup = ocr_cleanup_enabled
        self._ocr_patterns: List[re.Pattern[str]] = []
        if ocr_artifact_patterns:
            for pat in ocr_artifact_patterns:
                self._ocr_patterns.append(re.compile(pat))
        self._html_strip = html_strip_enabled
        self._page_number_filter = page_number_filter

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
        # Filter out page-number-only entities before grouping
        if self._page_number_filter:
            filtered: List[Dict[str, Any]] = []
            page_dropped = 0
            for entity in entities:
                raw = entity.get("text", "").strip()
                if _PAGE_NUMBER_RE.match(raw):
                    page_dropped += 1
                else:
                    filtered.append(entity)
            if page_dropped:
                logger.debug(
                    "Normalizer dropped {} page-number entities",
                    page_dropped,
                )
            entities = filtered

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

        # HTML tag stripping (before any whitespace handling)
        if self._html_strip:
            text = re.sub(r"<[^>]+>", "", text)

        # OCR artifact cleanup
        if self._ocr_cleanup:
            for pattern in self._ocr_patterns:
                text = pattern.sub("", text)

        # Diacritics removal (NFD decompose, strip combining marks)
        if self._remove_diacritics:
            text = unicodedata.normalize("NFD", text)
            text = "".join(
                ch for ch in text if unicodedata.category(ch) != "Mn"
            )

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
