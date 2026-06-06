"""
Entity name normalization stub for cross-source matching.

V1 stub introduced in B.1c: lowercases, collapses interior whitespace,
and strips trailing punctuation. Track M4 Q9 will replace this with a
fuller pipeline (TOOI lookup, Crossref reconciliation, alias tables,
etc.). The single import point ``shared.utils.name_normalizer`` is
intentional — downstream callers should depend only on the public
``normalize_entity_name`` function so the eventual swap is a drop-in.

Why a stub at this point in the timeline
========================================
Pass-1 schema validation (B.1c) needs to compare LLM-emitted surface
forms against ontology entity-type names. Even a conservative
lowercase+strip is enough to deduplicate cosmetic variants like
``"Apple Inc."`` vs ``"apple inc"`` while we wait for the full
normalizer (Q9). Anything richer here would be premature.
"""

from __future__ import annotations

import re

# Collapse all runs of any whitespace (spaces, tabs, newlines) to a
# single space. Compiled at module load so callers pay the cost once.
_WHITESPACE_RE = re.compile(r"\s+")

# Strip trailing whitespace + punctuation. Conservative set on purpose —
# we only remove sentence-end / clause-end glyphs that commonly leak
# from sentence-segmented surface forms. Q9 will handle Unicode
# punctuation, brackets, footnote markers, etc.
_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:!\?]+$")


def normalize_entity_name(name: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation.

    V1 stub. Track M4 Q9 replaces this with TOOI + Crossref lookups
    plus alias resolution; until then, keep callers funnelled through
    this single function so the upgrade is invisible.

    Args:
        name: Raw surface form as emitted by the LLM or upstream
            extractor. Leading whitespace is also stripped as a
            side-effect of the final ``.strip()``.

    Returns:
        Normalized name. Empty input produces an empty string rather
        than raising — callers that need to reject empties should
        check ``bool(normalize_entity_name(x))``.

    Examples:
        >>> normalize_entity_name("  Apple Inc.  ")
        'apple inc'
        >>> normalize_entity_name("Foo\\n\\tBar")
        'foo bar'
        >>> normalize_entity_name("")
        ''
    """
    if not name:
        return ""
    name = name.lower()
    name = _WHITESPACE_RE.sub(" ", name)
    name = _TRAILING_PUNCT_RE.sub("", name)
    return name.strip()
