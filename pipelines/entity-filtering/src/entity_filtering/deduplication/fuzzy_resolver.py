"""
Fuzzy entity resolver.

Merges entities with similar names using string distance algorithms:
- Levenshtein similarity (built-in, no external deps)
- Jaro-Winkler similarity (requires jellyfish)
- Phonetic codes: Soundex, Metaphone (requires jellyfish)

Uses UnionFind for transitive merging of entity groups.
"""

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Tuple

from loguru import logger
from shared.utils.text_folding import fold_for_comparison

from entity_filtering.deduplication.union_find import UnionFind

# Optional jellyfish for advanced algorithms
try:
    import jellyfish

    _JELLYFISH_AVAILABLE = True
except ImportError:
    _JELLYFISH_AVAILABLE = False


class FuzzyResolver:
    """Merge entities with similar names using fuzzy string matching.

    Args:
        algorithm: Distance algorithm -- ``"levenshtein"`` or ``"jaro_winkler"``.
        similarity_threshold: Minimum similarity (0-1) to consider a match.
        phonetic_algorithm: Phonetic code -- ``"soundex"``, ``"metaphone"``,
            or ``"none"``.
        phonetic_weight: Weight for phonetic similarity (0-1).
            Final score = ``(1 - phonetic_weight) * string_sim +
            phonetic_weight * phonetic_match``.
        max_candidates_per_entity: Max comparisons per entity (performance limit).
    """

    def __init__(
        self,
        algorithm: str = "levenshtein",
        similarity_threshold: float = 0.85,
        phonetic_algorithm: str = "none",
        phonetic_weight: float = 0.2,
        max_candidates_per_entity: int = 10,
    ) -> None:
        self._algorithm = algorithm
        self._threshold = similarity_threshold
        self._phonetic_algo = phonetic_algorithm
        self._phonetic_weight = phonetic_weight
        self._max_candidates = max_candidates_per_entity

        if algorithm == "jaro_winkler" and not _JELLYFISH_AVAILABLE:
            logger.warning(
                "jellyfish not available, falling back to levenshtein"
            )
            self._algorithm = "levenshtein"

        if phonetic_algorithm != "none" and not _JELLYFISH_AVAILABLE:
            logger.warning(
                "jellyfish not available, disabling phonetic matching"
            )
            self._phonetic_algo = "none"

    def resolve(
        self, entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
        """Merge entities with fuzzy-similar names.

        Args:
            entities: List of entity dicts with at least a ``"text"`` key.

        Returns:
            Tuple of (merged entities, merge groups).
            Merge groups only contain groups where more than one distinct
            surface form was merged.
        """
        if len(entities) <= 1:
            return list(entities), []

        uf = UnionFind()
        texts = [e.get("text", "") for e in entities]
        normalized = [self._normalize(t) for t in texts]

        # Compare all pairs (bounded by max_candidates)
        for i in range(len(entities)):
            if not normalized[i]:
                continue
            candidates_found = 0
            for j in range(i + 1, len(entities)):
                if not normalized[j]:
                    continue
                if candidates_found >= self._max_candidates:
                    break

                score = self._compute_similarity(normalized[i], normalized[j])
                if score >= self._threshold:
                    uf.union(texts[i], texts[j])
                    candidates_found += 1

        # Group by union-find root
        groups = uf.get_groups(texts)

        result: List[Dict[str, Any]] = []
        merge_groups: List[List[str]] = []

        for root, group_texts in groups.items():
            group_entities = [
                e for e in entities if e.get("text", "") in group_texts
            ]
            if not group_entities:
                continue

            canonical = self._select_canonical(group_entities)
            result.append(canonical)

            unique_forms = sorted(set(group_texts))
            if len(unique_forms) > 1:
                merge_groups.append(unique_forms)

        if merge_groups:
            logger.debug(
                "FuzzyResolver merged {} groups ({} -> {} entities)",
                len(merge_groups),
                len(entities),
                len(result),
            )

        return result, merge_groups

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _compute_similarity(self, a: str, b: str) -> float:
        """Compute blended similarity between two normalized strings."""
        string_sim = self._string_similarity(a, b)

        if self._phonetic_algo == "none" or self._phonetic_weight <= 0:
            return string_sim

        phonetic_match = self._phonetic_similarity(a, b)
        w = self._phonetic_weight
        return (1 - w) * string_sim + w * phonetic_match

    def _string_similarity(self, a: str, b: str) -> float:
        """Compute string distance similarity using the configured algorithm."""
        if self._algorithm == "jaro_winkler" and _JELLYFISH_AVAILABLE:
            return jellyfish.jaro_winkler_similarity(a, b)
        return self._levenshtein_similarity(a, b)

    def _phonetic_similarity(self, a: str, b: str) -> float:
        """Compare phonetic codes (1.0 if same, 0.0 if different)."""
        if not _JELLYFISH_AVAILABLE:
            return 0.0

        if self._phonetic_algo == "soundex":
            return 1.0 if jellyfish.soundex(a) == jellyfish.soundex(b) else 0.0
        elif self._phonetic_algo == "metaphone":
            return (
                1.0
                if jellyfish.metaphone(a) == jellyfish.metaphone(b)
                else 0.0
            )
        return 0.0

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison (lowercase, strip, collapse ws).

        PC.2: one shared fold, so a sixth copy cannot drift from the other four.
        Kept as a thin shim rather than deleted because this method is called from
        several places and the class has its own suite; the guard in
        `tests/test_one_comparison_fold.py` sees through it.
        """
        return fold_for_comparison(text)

    @staticmethod
    def _levenshtein_similarity(s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity (0-1, where 1 = identical).

        Uses a single-row dynamic programming approach for O(min(m, n))
        space complexity.
        """
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        current_row = list(range(len1 + 1))
        for i in range(1, len2 + 1):
            previous_row, current_row = current_row, [i] + [0] * len1
            for j in range(1, len1 + 1):
                add = previous_row[j] + 1
                delete = current_row[j - 1] + 1
                change = previous_row[j - 1]
                if s1[j - 1] != s2[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        distance = current_row[len1]
        max_len = max(len1, len2)
        return 1 - (distance / max_len) if max_len > 0 else 1.0

    # ------------------------------------------------------------------
    # Canonical selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_canonical(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pick the best representative from a merge group.

        Strategy:
        - Most frequent surface form wins the ``text`` field.
        - Highest confidence score is kept.
        - Most common label is kept.
        - Properties are merged (later entities overwrite earlier keys).
        """
        text_counts: Counter[str] = Counter(
            e.get("text", "") for e in group
        )
        canonical_text = text_counts.most_common(1)[0][0]

        base = next(
            e for e in group if e.get("text") == canonical_text
        ).copy()

        merged_props: Dict[str, Any] = {}
        max_confidence = 0.0
        label_counts: Counter[str] = Counter()

        for entity in group:
            merged_props.update(entity.get("properties", {}))
            conf = entity.get("confidence", 0.0)
            if conf > max_confidence:
                max_confidence = conf
            label_counts[entity.get("label", "MISC")] += 1

        base["properties"] = merged_props
        base["confidence"] = max_confidence
        base["label"] = label_counts.most_common(1)[0][0]
        return base
