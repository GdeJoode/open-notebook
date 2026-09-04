"""
String-based entity deduplicator.

Groups entities by normalized text form and keeps the most common
surface form as canonical. Tracks merge groups for provenance.

A FAISS/embedding-based deduplicator will be added as a separate
module in a future migration.
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

from loguru import logger
from shared.utils.text_folding import fold_for_comparison


class EntityDeduplicator:
    """Deduplicates entities by normalized string matching.

    Two entities are considered duplicates if they share the same
    normalized key (lowercased, whitespace-collapsed, unicode-normalized).

    Args:
        similarity_threshold: Reserved for future embedding-based
            deduplication. Currently unused by string matching.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._similarity_threshold = similarity_threshold

    def deduplicate(
        self, entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
        """Deduplicate entities by normalized text.

        Args:
            entities: List of entity dicts.

        Returns:
            A tuple of (deduplicated entities, merge groups).
            Each merge group is a list of original surface forms
            that were merged into a single canonical entity.
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for entity in entities:
            key = self._normalize_key(entity.get("text", ""))
            if key:
                groups.setdefault(key, []).append(entity)

        deduped: List[Dict[str, Any]] = []
        merge_groups: List[List[str]] = []

        for _key, group in groups.items():
            canonical = self._select_canonical(group)
            deduped.append(canonical)

            if len(group) > 1:
                surface_forms = list(
                    {e.get("text", "") for e in group}
                )
                if len(surface_forms) > 1:
                    merge_groups.append(sorted(surface_forms))

        if merge_groups:
            logger.debug(
                "Deduplicator merged {} groups ({} -> {} entities)",
                len(merge_groups),
                len(entities),
                len(deduped),
            )

        return deduped, merge_groups

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_key(text: str) -> str:
        """Create a deduplication key from entity text.

        PC.2: one shared fold, so a fifth copy cannot drift from the
        other four. Kept as a thin shim rather than deleted because this method is
        called from several places and the class has its own suite; the guard in
        `tests/test_one_comparison_fold.py` sees through it.
        """
        return fold_for_comparison(text)

    @staticmethod
    def _select_canonical(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best representative from a duplicate group.

        Strategy:
        - Most frequent surface form wins.
        - Highest confidence is kept.
        - Properties are merged across all variants.
        - Most common label is kept.
        """
        text_counts: Counter[str] = Counter(
            e.get("text", "") for e in group
        )
        canonical_text = text_counts.most_common(1)[0][0]

        label_counts: Counter[str] = Counter(
            e.get("label", "") for e in group
        )
        canonical_label = label_counts.most_common(1)[0][0]

        base = next(
            e for e in group if e.get("text") == canonical_text
        ).copy()
        base["label"] = canonical_label

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
