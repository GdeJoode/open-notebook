"""
Entity reclassifier.

Applies generic heuristic rules and user-supplied custom rules to
relabel entities. No domain-specific rules are hardcoded -- all
domain knowledge is injected via the ``custom_rules`` mapping.
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger


# Generic heuristic rules that apply across domains.
# Each rule is a tuple of (compiled_regex, new_label, description).
_GENERIC_RULES = [
    # Hyphenated multi-word strings are likely person names
    # (e.g. "Jean-Pierre", "Mary-Ann")
    (
        re.compile(r"^[A-Z][a-z]+-[A-Z][a-z]+$"),
        "PERSON",
        "hyphenated proper name",
    ),
    # Short all-caps strings (2-6 chars) are likely abbreviations
    (
        re.compile(r"^[A-Z]{2,6}$"),
        "ABBREVIATION",
        "short all-caps string",
    ),
]


class EntityReclassifier:
    """Relabels entities based on pattern-matching rules.

    Generic heuristic rules are applied first, then any custom rules
    from configuration. Custom rules take precedence when both match.

    Args:
        custom_rules: Mapping of regex pattern string to the new label
            that should be assigned when the pattern matches the entity
            text.
    """

    def __init__(
        self, custom_rules: Optional[Dict[str, str]] = None
    ) -> None:
        self._custom_rules = []
        if custom_rules:
            for pattern, label in custom_rules.items():
                self._custom_rules.append(
                    (re.compile(pattern, re.IGNORECASE), label)
                )

    def reclassify(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply reclassification rules to entity labels.

        Args:
            entities: List of entity dicts.

        Returns:
            List of entity dicts with potentially updated labels.
        """
        reclassified_count = 0
        result: List[Dict[str, Any]] = []

        for entity in entities:
            updated = entity.copy()
            text = updated.get("text", "")
            original_label = updated.get("label", "")
            new_label = self._match_rules(text)

            if new_label and new_label != original_label:
                updated["label"] = new_label
                updated.setdefault("properties", {})["original_label"] = (
                    original_label
                )
                reclassified_count += 1

            result.append(updated)

        if reclassified_count:
            logger.debug(
                "Reclassifier changed labels on {} entities",
                reclassified_count,
            )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match_rules(self, text: str) -> Optional[str]:
        """Return the new label if any rule matches, else None.

        Custom rules are checked first so they can override generic
        heuristics.
        """
        # Custom rules have priority
        for pattern, label in self._custom_rules:
            if pattern.search(text):
                return label

        # Generic heuristic rules
        for pattern, label, _desc in _GENERIC_RULES:
            if pattern.search(text):
                return label

        return None
