"""
Entity name normalization for cross-source matching.

The single import point ``shared.utils.name_normalizer.normalize_entity_name``
is intentional — every dedup/matching path (persistence dedup key, notebook
merge, ontology-extraction merge, exporters) funnels through it, so behaviour
changes here propagate everywhere at once.

Pipeline
========
``normalize_entity_name`` composes three stages:

1. **Content normalization** (B.1c): lowercase, collapse interior whitespace,
   strip trailing punctuation. Deduplicates cosmetic variants like
   ``"Apple Inc."`` vs ``"apple inc"``.
2. **NL leading-article strip** (K.1): remove a leading article
   (``de``/``het``/``een``) when a non-empty remainder survives. This collapses
   ``De Regio Deal`` and ``Regio Deal`` onto one key. Only articles are
   stripped — no content-bearing prefix (``ministerie van``, ``gemeente``,
   ``minister van`` …) is touched, because each of those can collide an org/role
   surface form onto a bare concept token another entity owns.
3. **NL spelling canonicalization** (K.1): map a small curated set of documented
   spelling variants (``koninkrijkrelaties`` → ``koninkrijksrelaties``) onto one
   form.
4. **Curated org-alias expansion** (K.2): resolve a known NL government-org
   surface form (``bzk`` / ``ministerie van bzk`` / the bare full-form) to one
   canonical full-form via an exact-match closed allow-list (built-in floor +
   operator overrides). Runs LAST, on the post-K.1 string, because its keys are
   already-normalized forms. See :mod:`shared.utils.org_aliases`.

Precision over recall
======================
Over-merging two distinct real entities is silently-wrong data; fragmentation
is merely annoying. Relations resolve their endpoints by name ALONE
(``WHERE canonical_name = $name``, no entity_type filter), so a normalized name
shared by two different-typed real entities corrupts the graph. The K.1 NL
stages are restricted to *collision-safe* edits: stripping a (meaning-free)
article and rewriting curated spelling variants — neither can fold a content
token onto another entity's name. K.2's org-alias expansion stays collision-safe
a different way: it is **exact-match against a curated allow-list** whose values
are distinct org canonicals (``ocw`` → ``onderwijs cultuur en wetenschap``, never
the bare ``onderwijs``), and it never keys the ``minister van …`` person-role
forms — so it cannot fold a ministry onto a bare different-typed token.
``Ministerie van Onderwijs`` and ``Onderwijs`` deliberately stay distinct
(``onderwijs cultuur en wetenschap`` ≠ ``onderwijs``).

The B.8 dedup-key contract is unaffected: callers derive
``hash_id = md5(f"{normalize_entity_name(name)}|{entity_type}")`` and the upsert
lookup is by ``(canonical_name, entity_type)``. Only the *string* this function
returns changes; the derive-rule is unchanged.
"""

from __future__ import annotations

import re

from shared.config.alias_overrides import get_resolved_aliases
from shared.utils.nl_normalization import (
    canonicalize_spelling,
    strip_leading_noise,
)
from shared.utils.org_aliases import expand_org_alias

# Collapse all runs of any whitespace (spaces, tabs, newlines) to a
# single space. Compiled at module load so callers pay the cost once.
_WHITESPACE_RE = re.compile(r"\s+")

# Strip trailing whitespace + punctuation. Conservative set on purpose —
# we only remove sentence-end / clause-end glyphs that commonly leak
# from sentence-segmented surface forms.
_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:!\?]+$")


def normalize_entity_name(name: str) -> str:
    """Normalize an entity surface form to a canonical comparison key.

    Composes content normalization (lowercase, collapse whitespace, strip
    trailing punctuation) with the K.1 NL rules (leading-article strip,
    documented spelling-variant canonicalization) and the K.2 curated org-alias
    expansion (``bzk`` / ``ministerie van bzk`` → one full-form). Every stage is
    collision-safe: distinct entities never end up sharing a name through
    normalization. ``Ministerie van Onderwijs`` and ``Onderwijs`` stay distinct
    (the OCW canonical is ``onderwijs cultuur en wetenschap``, not the bare
    concept), and ``BZK`` ≠ ``EZK``.

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
        >>> normalize_entity_name("Ministerie van Onderwijs")
        'ministerie van onderwijs'
        >>> normalize_entity_name("BZK")
        'binnenlandse zaken en koninkrijksrelaties'
        >>> normalize_entity_name("Ministerie van BZK")
        'binnenlandse zaken en koninkrijksrelaties'
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
    # Curated org-alias expansion (K.2) runs LAST, after article-strip and
    # spelling canonicalization, because its keys are post-K.1-normalized forms
    # (e.g. ``ministerie van bzk``, ``binnenlandse zaken en koninkrijksrelaties``)
    # — the spelling fix must already have rewritten ``koninkrijkrelaties`` for
    # the full-form key to match. Exact-match only against the built-in+override
    # allow-list; unknown forms pass through untouched.
    name = expand_org_alias(name, aliases=get_resolved_aliases())
    return name
