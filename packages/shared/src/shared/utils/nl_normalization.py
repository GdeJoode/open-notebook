"""Dutch (NL) entity-name normalization rules for the resolution swap-point.

This module is the NL-aware ruleset that ``name_normalizer.normalize_entity_name``
composes on top of the V1 content normalizer (lowercase / whitespace / trailing
punct). It is kept separate so ``name_normalizer.py`` stays a thin orchestrator
and K.2's abbreviation-alias layer can be composed in cleanly.

Two transformations live here:

``strip_leading_noise``
    Strips a leading article (``de``/``het``/``een``) and at most one role/org
    leader phrase (``ministerie van``, ``minister van``, ``gemeente``, …). This
    collapses ``Ministerie van BZK`` and ``BZK`` onto the same tail ``bzk``.

``canonicalize_spelling``
    Maps a small **curated** set of documented spelling variants onto one
    canonical form (``koninkrijkrelaties`` → ``koninkrijksrelaties``). It is a
    dict, not a fuzzy edit-distance matcher — fuzzy matching is K.5's job and
    carries over-merge risk a closed allow-list does not.

The central design constraint is **precision over recall**: over-merging two
distinct real entities is silently-wrong data the user cannot detect, whereas
fragmentation is merely annoying. So both functions are *tail-preserving*:

- ``strip_leading_noise`` only removes a prefix when a discriminating tail
  survives. ``Minister van BZK`` → ``bzk`` and ``Minister van Financiën`` →
  ``financiën`` stay distinct because their tails differ; ``Gemeente`` on its
  own is returned unchanged rather than stripped to an empty string.
- ``canonicalize_spelling`` only rewrites whole-token variants it knows about;
  everything else passes through untouched.

All inputs to these functions are assumed to be post-V1 (already lowercased and
whitespace-collapsed); they do not re-run that step.
"""

from __future__ import annotations

# Leading articles, matched after V1 lowercasing. Order is irrelevant (mutually
# exclusive at the head of a string), but the trailing space is part of the
# match so we never strip the ``de`` inside ``deal``.
_LEADING_ARTICLES: tuple[str, ...] = ("de ", "het ", "een ")

# Role/org leader phrases stripped when a discriminating tail remains. Ordered
# longest-first so ``ministerie van`` wins over a (hypothetical) shorter overlap
# and so ``staatssecretaris van`` is tried before any prefix it contains. The
# article-bearing variants (``de minister van``) are redundant with the
# article strip running first, but are listed so a single pass over a string
# that was *not* article-stripped (e.g. a direct caller) still works.
_ROLE_ORG_PREFIXES: tuple[str, ...] = (
    "het ministerie van ",
    "de staatssecretaris van ",
    "de minister van ",
    "staatssecretaris van ",
    "ministerie van ",
    "minister van ",
    "provincie ",
    "gemeente ",
)

# Minimum surviving tail length (in characters) for a prefix strip to be
# accepted. ``bzk`` (3) must survive; a 1-2 char leftover almost certainly
# means we stripped the entire meaningful content (e.g. an abbreviation that
# *is* the org), so we keep the original instead.
_MIN_TAIL_LEN = 2

# Curated spelling-variant map for the documented variant classes. Keys/values
# are post-V1 (lowercased) whole strings or substrings replaced token-wise.
# This is deliberately tiny and auditable; edit-distance generalization is K.5.
_SPELLING_VARIANTS: dict[str, str] = {
    "koninkrijkrelaties": "koninkrijksrelaties",
}


def strip_leading_noise(name: str) -> str:
    """Strip a leading article + one role/org leader, preserving the tail.

    Removes at most one article (``de``/``het``/``een``) and then at most one
    role/org leader phrase (longest-match-first), but only when a discriminating
    tail of at least :data:`_MIN_TAIL_LEN` characters survives. If stripping a
    role/org prefix would leave an empty or too-short tail, the prefix is *not*
    stripped and the (article-stripped) string is returned instead.

    Args:
        name: A post-V1 (lowercased, whitespace-collapsed) surface form.

    Returns:
        The name with leading noise removed where safe, else unchanged.

    Examples:
        >>> strip_leading_noise("de regio deal")
        'regio deal'
        >>> strip_leading_noise("ministerie van bzk")
        'bzk'
        >>> strip_leading_noise("minister van financiën")
        'financiën'
        >>> strip_leading_noise("gemeente")
        'gemeente'
    """
    if not name:
        return name

    # 1. Strip at most one leading article.
    article_stripped = name
    for article in _LEADING_ARTICLES:
        if article_stripped.startswith(article):
            candidate = article_stripped[len(article) :].strip()
            # An article on its own ("de") collapsing to "" is never useful.
            if candidate:
                article_stripped = candidate
            break

    # 2. Strip at most one role/org leader phrase, longest-match-first, but only
    #    when a discriminating tail survives (the precision guard).
    for prefix in _ROLE_ORG_PREFIXES:
        if article_stripped.startswith(prefix):
            tail = article_stripped[len(prefix) :].strip()
            if len(tail) >= _MIN_TAIL_LEN:
                return tail
            # Tail too short → stripping would erase the entity. Keep as-is.
            return article_stripped

    return article_stripped


def canonicalize_spelling(name: str) -> str:
    """Map documented spelling variants onto a single canonical form.

    A curated, closed allow-list — not a fuzzy matcher. Each known variant
    substring is replaced by its canonical equivalent. Unknown strings pass
    through untouched so an unrecognized variant fragments (safe) rather than
    being guessed into the wrong cluster (unsafe).

    Args:
        name: A post-V1 (lowercased) surface form.

    Returns:
        The name with known spelling variants canonicalized.

    Examples:
        >>> canonicalize_spelling("binnenlandse zaken en koninkrijkrelaties")
        'binnenlandse zaken en koninkrijksrelaties'
        >>> canonicalize_spelling("regio deal")
        'regio deal'
    """
    if not name:
        return name
    for variant, canonical in _SPELLING_VARIANTS.items():
        if variant in name:
            name = name.replace(variant, canonical)
    return name
