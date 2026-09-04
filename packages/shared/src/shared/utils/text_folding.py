"""One comparison fold, shared by every stage that compares two entity names.

Track PC.2. Four stages carried a byte-identical private copy of this four-line
transform — ``EntityDeduplicator._normalize_key``, ``FuzzyResolver._normalize``,
``KGResolver._normalize``, ``concept_alignment._normalize`` — and a fifth call
site reached one of them through a private attribute across a package boundary.
Verified identical by execution over adversarial input (NBSP, full-width forms,
ligatures, Turkish dotted I, ideographic space, empty and whitespace-only), so
consolidating is a pure refactor with no behavioural delta at any of them.

**Why this is not called ``normalize_*``.** ``normalize_entity_name`` already
lives one file away and does something materially different: no NFKC, plus
trailing-punctuation stripping, Dutch article stripping, spelling canonicalisation
and curated org-alias expansion. At a call site the two would be indistinguishable
by name, and choosing the wrong one inside concept alignment would compare
post-alias-expansion strings — pre-merging exactly the identities D-N4-9 says must
not be merged without a decision. ``fold_for_comparison`` cannot be misread as the
entity-name canonicaliser.

**Use this** to decide whether two surface forms are the same string for
comparison purposes. **Use** :func:`shared.utils.name_normalizer.normalize_entity_name`
to derive an entity's canonical name or its ``hash_id``; that one is the identity
rule and it is deliberately more opinionated.

**Not folded here, deliberately**: ``EntityNormalizer._normalize_text``
(``entity_filtering/filters/normalizer.py``) trips a similar shape but is a
configurable *merging transform inside a pipeline stage* rather than a comparison
key — it does not lowercase, and it strips English articles. Folding it in would
change what the Normalizer stage merges, on by default, in a phase whose remit is
a refactor.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def fold_for_comparison(text: str) -> str:
    """NFKC-fold, lowercase, strip, and collapse interior whitespace.

    Exactly what the four private copies did, in the order they did it. Empty and
    whitespace-only input fold to ``""``.

    What it deliberately does NOT do, because no caller did it and adding any of
    them would change what four stages consider equal: strip punctuation, fold
    diacritics, remove articles, reorder tokens, or expand aliases.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_RE.sub(" ", folded.lower().strip())


__all__ = ["fold_for_comparison"]
