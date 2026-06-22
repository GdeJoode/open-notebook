"""Dutch (NL) entity-name normalization rules for the resolution swap-point.

This module is the NL-aware ruleset that ``name_normalizer.normalize_entity_name``
composes on top of the V1 content normalizer (lowercase / whitespace / trailing
punct). It is kept separate so ``name_normalizer.py`` stays a thin orchestrator
and K.2's curated abbreviation/alias layer can be composed in cleanly.

Two transformations live here, both deliberately **collision-safe** — neither
strips a content token that could collide with a bare token elsewhere:

``strip_leading_noise``
    Strips ONLY a leading article (``de``/``het``/``een``). Articles carry no
    discriminating meaning, so removing them can never make two distinct
    entities share a name. No content-bearing prefix (``ministerie van``,
    ``gemeente``, ``minister van`` …) is stripped here — every one of them can
    collapse an org/role surface form onto a bare concept token that another
    real entity already owns (``Ministerie van Onderwijs`` → ``onderwijs`` ==
    the bare ``onderwijs`` concept), which corrupts name-only relation
    resolution. Type-aware org-form / abbreviation merging (``Ministerie van
    BZK`` ↔ ``BZK``) is therefore deferred to Track K.2's curated alias table.

``canonicalize_spelling``
    Maps a small **curated** set of documented spelling variants onto one
    canonical form (``koninkrijkrelaties`` → ``koninkrijksrelaties``). It is a
    dict, not a fuzzy edit-distance matcher — fuzzy matching is a later phase
    and carries over-merge risk a closed allow-list does not.

The central design constraint is **precision over recall**: over-merging two
distinct real entities is silently-wrong data the user cannot detect, whereas
fragmentation is merely annoying. Because relations resolve their endpoints by
name ALONE (``WHERE canonical_name = $name``, no entity_type filter), a
normalized name shared by two different-typed real entities corrupts the graph
regardless of the dedup key. So both functions only touch material that cannot
introduce such a collision:

- ``strip_leading_noise`` removes an article only when a non-empty remainder
  survives (``de`` on its own is returned unchanged), and never touches a
  content token.
- ``canonicalize_spelling`` only rewrites whole-token variants it knows about;
  everything else passes through untouched.

All inputs to these functions are assumed to be post-V1 (already lowercased and
whitespace-collapsed); they do not re-run that step.
"""

from __future__ import annotations

import re

# Shared with ``name_normalizer`` so alias-config keys/values can be normalized
# through the *same* pre-alias pipeline (lowercase / whitespace / trailing punct
# / article-strip / spelling) without importing ``name_normalizer`` (which would
# create an import cycle: name_normalizer -> alias_overrides -> name_normalizer).
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:!\?]+$")

# Leading articles, matched after V1 lowercasing. Order is irrelevant (mutually
# exclusive at the head of a string), but the trailing space is part of the
# match so we never strip the ``de`` inside ``deal``.
#
# Articles are the ONLY prefix stripped here. They are collision-safe by
# construction: an article carries no discriminating content, so removing it
# can never make two distinct entities share a name. Content-bearing leaders
# (``ministerie van``, ``gemeente``, ``minister van`` …) were all REMOVED
# because each can collapse an org/role surface form onto a bare concept token
# another real entity owns — e.g. ``Ministerie van Onderwijs`` → ``onderwijs``
# collides with the bare ``onderwijs`` concept, and since relations resolve by
# name alone that corrupts the graph. Curated, type-aware org-form merging
# lives in Track K.2's alias table, not here.
_LEADING_ARTICLES: tuple[str, ...] = ("de ", "het ", "een ")

# Curated spelling-variant map for the documented variant classes. Keys/values
# are post-V1 (lowercased) tokens; ``canonicalize_spelling`` matches them on
# word boundaries so only whole tokens are rewritten, never substrings of a
# longer compound. Deliberately tiny + auditable; edit-distance is a later phase.
_SPELLING_VARIANTS: dict[str, str] = {
    "koninkrijkrelaties": "koninkrijksrelaties",
}


def strip_leading_noise(name: str) -> str:
    """Strip a leading article (``de``/``het``/``een``), preserving the rest.

    Removes at most one leading article, but only when a non-empty remainder
    survives — an article on its own (``de``) is returned unchanged rather than
    collapsed to an empty string. No content-bearing prefix is stripped:
    article removal is the only transformation, because it is the only one that
    cannot make two distinct entities share a normalized name.

    Org/role leader phrases (``ministerie van``, ``gemeente``, ``minister van``
    …) are intentionally NOT handled here. Each of them can collapse a surface
    form onto a bare concept token another real entity owns (``Ministerie van
    Onderwijs`` → ``onderwijs`` collides with the bare ``onderwijs``), and
    relations resolve endpoints by name alone, so such a strip corrupts the
    graph. Type-aware org-form / abbreviation merging is Track K.2's job.

    Args:
        name: A post-V1 (lowercased, whitespace-collapsed) surface form.

    Returns:
        The name with a leading article removed where safe, else unchanged.

    Examples:
        >>> strip_leading_noise("de regio deal")
        'regio deal'
        >>> strip_leading_noise("ministerie van onderwijs")
        'ministerie van onderwijs'
        >>> strip_leading_noise("gemeente groningen")
        'gemeente groningen'
        >>> strip_leading_noise("de")
        'de'
    """
    if not name:
        return name

    for article in _LEADING_ARTICLES:
        if name.startswith(article):
            candidate = name[len(article) :].strip()
            # An article on its own ("de") collapsing to "" is never useful.
            if candidate:
                return candidate
            return name

    return name


def normalize_alias_key(name: str) -> str:
    """Run the full pre-alias normalization pipeline on an alias-config string.

    Operator/user-supplied alias entries (``{"Min AZ": "Algemene Zaken"}``) must
    be keyed in the *same* normalized space the resolver sees at lookup time, or
    they would never match. This applies the exact stages
    ``normalize_entity_name`` runs *before* alias expansion — lowercase,
    whitespace-collapse, trailing-punct strip, leading-article strip, spelling
    canonicalization — so a raw config key lands on the post-K.1 form that
    :func:`shared.utils.org_aliases.expand_org_alias` is asked to resolve.

    Kept here (not in ``name_normalizer``) purely to avoid an import cycle, since
    the alias-config loader needs it.

    Args:
        name: A raw alias-config key or value.

    Returns:
        The post-K.1 normalized form (empty string for empty/blank input).

    Examples:
        >>> normalize_alias_key("Min AZ")
        'min az'
        >>> normalize_alias_key("De Regio Deal")
        'regio deal'
    """
    if not name:
        return ""
    name = name.lower()
    name = _WHITESPACE_RE.sub(" ", name)
    name = _TRAILING_PUNCT_RE.sub("", name)
    name = name.strip()
    name = strip_leading_noise(name)
    name = canonicalize_spelling(name)
    return name


def canonicalize_spelling(name: str) -> str:
    """Map documented spelling variants onto a single canonical form.

    A curated, closed allow-list — not a fuzzy matcher. Each known variant
    is replaced by its canonical equivalent on a **whole-token** (word
    boundary) basis: ``koninkrijkrelaties`` is rewritten, but the same
    fragment inside a longer compound (``koninkrijkrelatiesbeleid``) is
    left untouched. Unknown strings pass through untouched so an
    unrecognized variant fragments (safe) rather than being guessed into
    the wrong cluster (unsafe).

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
        # ``\b`` anchors the match to token boundaries so the curated map
        # only rewrites whole tokens, never a substring inside a longer
        # compound word. ``re.escape`` keeps the variant a literal.
        pattern = rf"\b{re.escape(variant)}\b"
        name = re.sub(pattern, canonical, name)
    return name
