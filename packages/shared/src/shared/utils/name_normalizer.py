"""
Entity name normalization for cross-source matching.

The single import point ``shared.utils.name_normalizer.normalize_entity_name``
is intentional — every dedup/matching path (persistence dedup key, notebook
merge, ontology-extraction merge, exporters) funnels through it, so behaviour
changes here propagate everywhere at once.

Pipeline
========
``normalize_entity_name`` composes three stages:

1. **V1 content normalization** (B.1c): lowercase, collapse interior
   whitespace, strip trailing punctuation. Deduplicates cosmetic variants like
   ``"Apple Inc."`` vs ``"apple inc"``.
2. **NL leading-noise strip** (K.1): remove a leading article
   (``de``/``het``/``een``) and one role/org leader phrase (``ministerie van``,
   ``minister van``, ``gemeente``, …) when a discriminating tail survives. This
   collapses ``Ministerie van BZK`` and ``BZK`` onto ``bzk``.
3. **NL spelling canonicalization** (K.1): map documented spelling variants
   (``koninkrijkrelaties`` → ``koninkrijksrelaties``) onto one form.

Precision over recall
======================
Over-merging two distinct real entities is silently-wrong data; fragmentation
is merely annoying. The NL stages are therefore *tail-preserving*:
``Minister van BZK`` and ``Minister van Financiën`` stay distinct because their
tails differ, and ``Gemeente`` on its own is left unchanged rather than stripped
to nothing. See :mod:`shared.utils.nl_normalization` for the rule details and
guards.

The B.8 dedup-key contract is unaffected: callers derive
``hash_id = md5(f"{normalize_entity_name(name)}|{entity_type}")`` and the upsert
lookup is by ``(canonical_name, entity_type)``. Only the *string* this function
returns changes; the derive-rule is unchanged.
"""

from __future__ import annotations

import re

from shared.utils.nl_normalization import (
    canonicalize_spelling,
    strip_leading_noise,
)

# Collapse all runs of any whitespace (spaces, tabs, newlines) to a
# single space. Compiled at module load so callers pay the cost once.
_WHITESPACE_RE = re.compile(r"\s+")

# Strip trailing whitespace + punctuation. Conservative set on purpose —
# we only remove sentence-end / clause-end glyphs that commonly leak
# from sentence-segmented surface forms.
_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:!\?]+$")


def normalize_entity_name(name: str) -> str:
    """Normalize an entity surface form to a canonical comparison key.

    Composes V1 content normalization (lowercase, collapse whitespace, strip
    trailing punctuation) with the K.1 NL rules (leading article + role/org
    prefix strip, documented spelling-variant canonicalization). The NL rules
    are tail-preserving: a prefix is only stripped when a discriminating tail
    survives, so distinct entities sharing a leader phrase stay distinct.

    Args:
        name: Raw surface form as emitted by the LLM or upstream extractor.
            Leading whitespace is stripped as a side-effect.

    Returns:
        Normalized name. Empty input produces an empty string rather than
        raising — callers that need to reject empties should check
        ``bool(normalize_entity_name(x))``.

    Examples:
        >>> normalize_entity_name("  Apple Inc.  ")
        'apple inc'
        >>> normalize_entity_name("Foo\\n\\tBar")
        'foo bar'
        >>> normalize_entity_name("De Regio Deal")
        'regio deal'
        >>> normalize_entity_name("Ministerie van BZK")
        'bzk'
        >>> normalize_entity_name("Binnenlandse Zaken en Koninkrijkrelaties")
        'binnenlandse zaken en koninkrijksrelaties'
        >>> normalize_entity_name("")
        ''
    """
    if not name:
        return ""
    name = name.lower()
    name = _WHITESPACE_RE.sub(" ", name)
    name = _TRAILING_PUNCT_RE.sub("", name)
    name = name.strip()
    # NL-aware stages run on the post-V1 string.
    name = strip_leading_noise(name)
    name = canonicalize_spelling(name)
    return name
