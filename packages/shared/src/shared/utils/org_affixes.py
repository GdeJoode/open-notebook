"""Curated head runs that qualify a name without changing which entity it names.

Track PC.2. `Gemeente Leudal` and `Leudal` are the same place; `Regio Deal
Groningen` and `Regio Deal` are not the same thing. Both look like containment,
and two facts about Dutch naming separate them:

* a qualifier that does NOT change the referent sits at the **head** —
  `Gemeente Leudal`, `Minister van BZK`, `de heer Rob Opdam`;
* a discriminator that DOES change the referent sits at the **tail** —
  `Regio Deal Groningen`, `Regio Deal Drenthe`.

So the shorter name must be a token SUFFIX of the longer, and the removed head
run must be a **class or role noun** rather than a proper name.

**Both halves are load-bearing, measured on 5000 live entities.** Unanchored
containment (`concept_alignment._is_token_subsequence`) manufactures exactly the
merge the dedup config refuses — it pairs `Regio Deal` with both
`Regio Deal Groningen` and `Regio Deal Drenthe`. Head-anchored containment with a
free length guard (inner ≥ 2 tokens) drops those but produces 315 candidates, and
its new failure mode is a place name in tail position: `Het Hogeland` pairs with
seven organisations that merely operate there — `Ondernemersplatform Het Hogeland`,
`Mensenwerk Het Hogeland`, `Regio Deal Het Hogeland`. Head-anchored **and curated**
yields 82, with the place-name noise gone.

A curated list is therefore not a shortcut around a general rule; on this data it
IS the general rule, and it has the property a length threshold does not: every
candidate can be explained to a curator by naming the run that was removed.

**Why a merge rule may not do this and a review rule may.**
`nl_normalization.strip_leading_noise` refuses to strip exactly these, in writing,
because *"each can collapse a surface form onto a bare concept token another real
entity owns (`Ministerie van Onderwijs` → `onderwijs`) … Type-aware org-form merging
is Track K.2's job"*. That objection is against a NORMALISATION rule, which merges
silently and irreversibly. It does not carry to a REVIEW proposal, where both forms
are put in front of a human who decides. `Onderwijs` / `Ministerie van Onderwijs` is
in the 82 and belongs there.

Seeded from those docstrings and from the shapes present in the live graph. Extend
it when a curator keeps approving the same shape — not in anticipation.
"""

from __future__ import annotations

from typing import FrozenSet, Sequence, Tuple

_AFFIX_SOURCE: Tuple[str, ...] = (
    # Municipalities and provinces
    "gemeente",
    "de gemeente",
    "provincie",
    "de provincie",
    "waterschap",
    "het waterschap",
    # Legal forms
    "stichting",
    "vereniging",
    "coöperatie",
    # Ministries and their office-holders
    "ministerie van",
    "het ministerie van",
    "minister van",
    "de minister van",
    "staatssecretaris van",
    "staatssecretaris voor",
    "de staatssecretaris van",
    "de staatssecretaris voor",
    # Governing bodies
    "college van",
    "het college van",
    "college van burgemeester en wethouders van",
    "het college van burgemeester en wethouders van",
    "college van burgemeester en wethouders van de gemeente",
    "het college van burgemeester en wethouders van de gemeente",
    "dagelijks bestuur van",
    "het dagelijks bestuur van",
    "algemeen bestuur van",
    "het algemeen bestuur van",
    "gedeputeerde staten van",
    "de gedeputeerde staten van",
    "gemeenteraad van",
    "de gemeenteraad van",
    "raad van",
    "de raad van",
    "burgemeester van",
    "wethouder van",
    # Personal honorifics
    "heer",
    "de heer",
    "mevrouw",
)

#: Token runs that may be removed from the HEAD of a name to reach the same
#: entity. Lowercase and whitespace-folded, i.e. in the form
#: :func:`shared.utils.text_folding.fold_for_comparison` produces, because that is
#: what the caller tokenises with.
ORG_HEAD_AFFIXES: FrozenSet[Tuple[str, ...]] = frozenset(
    tuple(a.split()) for a in _AFFIX_SOURCE
)


def head_affix(outer: Sequence[str], inner: Sequence[str]) -> Tuple[str, ...] | None:
    """Return the curated run removed to get from ``outer`` to ``inner``, or None.

    Both arguments are token lists from the comparison fold. Returns None when the
    two are equal (an exact match is not a containment proposal), when ``inner`` is
    not a suffix of ``outer``, or when the removed run is not curated.

    Returning the run rather than a bool is deliberate: a curator card that says
    *"differs by the head run `gemeente`"* is reviewable, and one that says
    *"containment"* is not.
    """
    outer, inner = list(outer), list(inner)
    if not inner or len(outer) <= len(inner):
        return None
    if outer[len(outer) - len(inner) :] != inner:
        return None
    removed = tuple(outer[: len(outer) - len(inner)])
    return removed if removed in ORG_HEAD_AFFIXES else None


__all__ = ["ORG_HEAD_AFFIXES", "head_affix"]
