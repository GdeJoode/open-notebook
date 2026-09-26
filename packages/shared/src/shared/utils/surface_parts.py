"""Decompose an entity's surface form into the names it asserts — without deleting any.

Track PC, decision D1. The extractor decorates names: `Economische Zaken
(Ministerie)`, `VRO (Volkshuisvesting en Ruimtelijke Ordening)`, `Ministerie van
Volksgezondheid, Welzijn en Sport`. PC.3 measured the ministries alone as 20 rows
under 20 keys, and every mechanism downstream — exact lookup, fuzzy, embeddings,
an authority — failed on the same decoration.

**Split, never strip.** This module does not produce an identity key and nothing
here may be written as one. `normalize_entity_name` stays the identity rule
(owner decision, 2026-09-26): a normalisation merges silently and irreversibly,
which is the objection `nl_normalization` and `org_affixes` already state. What
this produces is the set of NAMES one surface form asserts for one referent, for
consumers that either check them against an authority or put them in front of a
human:

* ``TOOIProvider.lookup`` — each name is looked up; the hits must agree;
* ``CandidateDedupService`` — two rows sharing a name become a REVIEW proposal.

**The shapes, and why each is treated as it is.** Measured on the 531 active rows
of the working corpus (2026-09-26); the decision per shape is the point, the
counts are the evidence:

* ``X (Ministerie)`` — a TYPE SUFFIX (5 rows). The word repeats what
  ``entity_type`` already says, so ``X`` is a name of the same referent. Only for
  the types in :data:`TYPE_SUFFIXES`.
* ``ABBR (gloss)`` or ``long form (ABBR)`` — an ABBREVIATION PAIR (10 rows). Both
  sides are names of one referent. The extractor invents glosses (`VRO (Ministerie
  van Volksgezondheid, Welzijn en Sport)` sits on a line that names VRO), so the
  two are offered as separate names and a consumer that finds them disagreeing
  must refuse, never pick.
* any other ``X (Y)`` — a QUALIFIER (~30 rows): `sterk regionaal netwerk (op
  onderwerpen)`, `Regio (Zuidwest-Friesland)`. The parenthesis narrows or renames;
  dropping it changes the referent. Kept whole.
* ``(De) Staatssecretaris van …``, ``Minister van …`` and the other organs in
  :data:`ORGAN_PREFIXES` — BLOCKED. An office or body OF X is not X; this is
  ``org_affixes``'s rule, and a D1 simulation that ignored it linked
  `De Staatssecretaris van Infrastructuur en Waterstaat (IenW)` to the ministry.
* ``Ministerie van X`` — a TYPE PREFIX for ``government_organization`` only, the
  form TOOI's ``naamInclSoort`` takes. Not stripped for any other type: `Ministerie
  van Onderwijs` and the topic `Onderwijs` are different things.
* ``Gemeente X`` / ``Provincie X`` — recorded as a type word, NOT split:
  `Gemeente Utrecht` and `Provincie Utrecht` are both ``administrative_area``, and
  offering `Utrecht` as a name of each would propose merging them.
* ``Label: X`` — kept whole. `Programmalijn 4: Economie` is one programme line of
  one deal; the number is identity. Only the curated plural category prefixes in
  :data:`CATEGORY_PREFIXES` are removed, and a comma list after one
  (`ministeries van het Rijk: A, B, C`) is BLOCKED as several referents.

Extend a list here when a curator keeps confirming the same shape, not from a
corpus frequency alone — the rule ``org_affixes`` learned by listing governance
affixes that turned out to name organs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple

# Shapes. Named because consumers print them as evidence.
PLAIN = "plain"
TYPE_SUFFIX = "type_suffix"
ABBREVIATION = "abbreviation"
QUALIFIER = "qualifier"
TYPE_PREFIX = "type_prefix"
CATEGORY_PREFIX = "category_prefix"
ORGAN = "organ"
MULTIPLE = "multiple"

#: Parenthesised words that restate the entity type, per type. Lowercase.
TYPE_SUFFIXES: Dict[str, FrozenSet[str]] = {
    "government_organization": frozenset({"ministerie"}),
}

#: Leading words that restate the entity type, per type, and may be split off.
TYPE_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "government_organization": ("het ministerie van ", "ministerie van "),
}

#: Leading words recorded as a type hint but never split (see module docstring).
TYPE_HINTS: Tuple[str, ...] = ("gemeente ", "provincie ", "waterschap ")

#: An office, body or office-holder OF something — a different entity from it.
#: The governance runs `org_affixes` removed from its merge list, plus the
#: leading article the corpus writes before them.
ORGAN_PREFIXES: Tuple[str, ...] = (
    "minister van ", "minister voor ", "staatssecretaris van ", "staatssecretaris voor ",
    "raad van ", "college van ", "gemeenteraad van ", "dagelijks bestuur van ",
    "gedeputeerde staten van ", "burgemeester van ", "commissaris van de koning",
)

#: Plural category labels whose tail names the referent(s) themselves.
CATEGORY_PREFIXES: Tuple[str, ...] = ("ministeries van het rijk:",)

_PAREN_TAIL = re.compile(r"^(?P<head>.*\S)\s*\((?P<inner>[^()]+)\)\s*$")
# An abbreviation: one token, 2-10 characters, at least two capitals
# (`VRO`, `IenW`, `HG/BZK`, `WOZO`). `Regio` (one capital) is a word.
_ABBR = re.compile(r"^[A-Za-z&/.\-]{2,10}$")


def _is_abbreviation(token: str) -> bool:
    return bool(_ABBR.match(token)) and sum(c.isupper() for c in token) >= 2


def _strip_prefix(text: str, prefixes: Tuple[str, ...]) -> Tuple[str, str]:
    """(prefix found, remainder) for the first matching prefix, case-insensitive."""
    low = text.casefold()
    for p in prefixes:
        if low.startswith(p):
            return p.strip(), text[len(p):].strip()
    return "", text


@dataclass(frozen=True)
class SurfaceParts:
    """What one surface form asserts.

    ``names`` holds every name the surface gives for ONE referent, the raw form
    first, in the original casing. It is empty when the form is ``blocked``: an
    organ of something, or several referents in one string. A blocked form must
    not be resolved or merged on any of its parts.
    """

    raw: str
    shape: str
    names: Tuple[str, ...] = ()
    type_word: str = ""
    qualifier: str = ""
    blocked: bool = False
    detail: Dict[str, str] = field(default_factory=dict)


def parse_surface(name: str, entity_type: str = "") -> SurfaceParts:
    """Decompose ``name`` (see the module docstring for the shape rules)."""
    raw = " ".join((name or "").split())
    if not raw:
        return SurfaceParts(raw="", shape=PLAIN)
    low = raw.casefold()

    # Organs first: `De Staatssecretaris van IenW (IenW)` must never reach the
    # parenthesis rule, whose gloss would name the ministry.
    body = low[len("de "):] if low.startswith("de ") else low
    for p in ORGAN_PREFIXES:
        if body.startswith(p):
            return SurfaceParts(raw=raw, shape=ORGAN, blocked=True,
                                detail={"organ": p.strip()})

    rest = raw
    shape = PLAIN
    type_word = ""
    category, after = _strip_prefix(rest, CATEGORY_PREFIXES)
    if category:
        if "," in after:
            return SurfaceParts(raw=raw, shape=MULTIPLE, blocked=True,
                                detail={"category": category})
        rest, shape = after, CATEGORY_PREFIX

    prefix, after = _strip_prefix(rest, TYPE_PREFIXES.get(entity_type, ()))
    if prefix:
        rest, shape, type_word = after, TYPE_PREFIX, "ministerie"

    names = [raw]
    qualifier = ""
    detail: Dict[str, str] = {}
    m = _PAREN_TAIL.match(rest)
    if m:
        head, inner = m.group("head").strip(), m.group("inner").strip()
        if inner.casefold() in TYPE_SUFFIXES.get(entity_type, frozenset()):
            shape, type_word = TYPE_SUFFIX, inner.casefold()
            names.append(head)
        elif _is_abbreviation(head) and not _is_abbreviation(inner):
            shape = ABBREVIATION
            _, gloss = _strip_prefix(inner, TYPE_PREFIXES.get(entity_type, ()))
            names += [head, gloss]
            detail = {"abbreviation": head, "gloss": gloss}
        elif _is_abbreviation(inner) and not _is_abbreviation(head):
            shape = ABBREVIATION
            names += [head, inner]
            detail = {"abbreviation": inner, "gloss": head}
        else:
            # A qualifier: kept whole, recorded, never split into names.
            shape, qualifier = QUALIFIER, inner
            if rest != raw:
                names.append(rest)
    elif rest != raw:
        names.append(rest)

    if not type_word:
        hint, _ = _strip_prefix(raw, TYPE_HINTS)
        type_word = hint

    seen, ordered = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    return SurfaceParts(raw=raw, shape=shape, names=tuple(ordered), type_word=type_word,
                        qualifier=qualifier, detail=detail)


__all__ = [
    "SurfaceParts", "parse_surface", "TYPE_SUFFIXES", "TYPE_PREFIXES", "TYPE_HINTS",
    "ORGAN_PREFIXES", "CATEGORY_PREFIXES", "PLAIN", "TYPE_SUFFIX", "ABBREVIATION",
    "QUALIFIER", "TYPE_PREFIX", "CATEGORY_PREFIX", "ORGAN", "MULTIPLE",
]
