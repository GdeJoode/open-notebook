"""Curated head runs that qualify a name without changing WHICH entity it names.

Track PC.2. `Gemeente Leudal` and `Leudal` are one actor; `Regio Deal Groningen`
and `Regio Deal` are not. Both look like containment, and two facts about Dutch
naming separate them:

* a qualifier that does NOT change the referent sits at the **head** —
  `Gemeente Leudal`, `Stichting X`, `de heer Rob Opdam`;
* a discriminator that DOES change the referent sits at the **tail** —
  `Regio Deal Groningen`, `Regio Deal Drenthe`.

So the shorter name must be a token SUFFIX of the longer, and the removed head
run must be **curated**.

Both halves are load-bearing, measured on 5000 entity rows of the working corpus
(2026-09-03; that database has since been emptied, so the counts below are a
record, not something a later reader can re-run):

* Unanchored containment — which is what `concept_alignment._is_token_subsequence`
  did — manufactures exactly the merge the dedup config refuses: it pairs
  `Regio Deal` with both `Regio Deal Groningen` and `Regio Deal Drenthe`.
* Head-anchored with a free length guard (inner ≥ 2 tokens) drops those but yields
  315 candidates, and pairs a place with every organisation named after it:
  `Het Hogeland` with `Mensenwerk Het Hogeland`, `Ondernemersplatform Het
  Hogeland`, five more.

**An organ OF X is not X — the rule this list learned the hard way.** An earlier
version of this list carried the governance affixes seen in that corpus:
`raad van`, `college van`, `gemeenteraad van`, `dagelijks bestuur van`,
`gedeputeerde staten van`, `burgemeester van`, `ministerie van`, `minister van`,
`staatssecretaris van`. Every one of them names a body, an office or an
office-holder OF something, which is a *different entity* from that something.
Adversarial review produced these, all of which the rule then proposed as merges:

    'Raad van Toezicht'                 ~ 'Toezicht'
    'Raad van Advies'                   ~ 'Advies'
    'College van Beroep'                ~ 'Beroep'
    'Gemeenteraad van Amsterdam'        ~ 'Amsterdam'
    'Burgemeester van Rotterdam'        ~ 'Rotterdam'
    'Gedeputeerde Staten van Drenthe'   ~ 'Drenthe'
    'Dagelijks Bestuur van Wetterskip Fryslân' ~ 'Wetterskip Fryslân'
    'Ministerie van Onderwijs'          ~ 'Onderwijs'

The first three are verbatim the objection `nl_normalization.strip_leading_noise`
states in writing — collapsing a named body onto a bare concept token another
entity owns. "It is only a review proposal, a human decides" answers that one, but
it does not answer the rest: the mayor is not the city, and asking a curator to
arbitrate between two identities the rule has declared to be one is not review, it
is a leading question in front of a destructive button.

**What that costs, stated plainly.** The PC.2 plan named
`Minister van Binnenlandse Zaken en Koninkrijksrelaties` beside
`Binnenlandse Zaken en Koninkrijksrelaties` as a pair to surface. It is an
office-of pair, so this list no longer produces it. That class is real and worth
showing — but as an **organ-of relation**, not as a merge, and proposing a merge
asserts something stronger than the evidence supports. Filed for a later phase
rather than smuggled in here.

**Why a merge rule may not strip these and a review rule may — for what remains.**
`nl_normalization` refuses to strip `gemeente`/`provincie` because a normalisation
merges silently and irreversibly. A review proposal puts both forms in front of a
human. That distinction earns `Gemeente Leudal` / `Leudal`; it does not earn
`Burgemeester van Rotterdam` / `Rotterdam`, because there the two forms are not
two spellings of one identity for a human to arbitrate.

Extend this list when a curator keeps approving the same shape — not in
anticipation, and not from a corpus frequency alone. Frequency is what put the
governance affixes here.
"""

from __future__ import annotations

from typing import FrozenSet, Sequence, Tuple

_AFFIX_SOURCE: Tuple[str, ...] = (
    # Municipalities and provinces. The body and the area it governs are not
    # literally the same thing, and this is the one place the list bends: in Dutch
    # policy prose `Gemeente Leudal` and `Leudal` denote one actor and appear
    # interchangeably in the same paragraph, because there is no separate "Leudal"
    # for the municipality to be distinguished FROM. That is the pair the PC.2
    # plan named as the case to solve. A judgement about this corpus, not a
    # linguistic universal.
    #
    # `waterschap` was here and was REMOVED. It fails the test the other two pass:
    # no Dutch text uses "Limburg" to mean "Waterschap Limburg". A water board is
    # a distinct legal actor whose management area is named after a province that
    # is itself a separate entity, so `Waterschap Limburg` / `Limburg` is a
    # body-versus-territory pair — the class this list was just cut to remove. It
    # also contradicted the sibling test: `Dagelijks Bestuur van Wetterskip
    # Fryslân` / `Wetterskip Fryslân` is rejected because a water board's
    # executive is not the water board, while `Waterschap Limburg` / `Limburg`
    # asserted that a water board IS the province. Both cannot be the rule.
    "gemeente",
    "de gemeente",
    # `provincie` survives on a DIFFERENT argument, and review corrected me here:
    # the sentence above does not cover it, because there IS a separate Groningen
    # and a separate Utrecht — the cities. It survives on the same argument as
    # `stichting` below: "Groningen" genuinely is used to mean the province in
    # policy prose, so the pair is a homonym rather than a reference error, and a
    # homonym is a question a curator can answer.
    #
    # It does mean the door emits `Gemeente Groningen` ~ `Groningen` AND
    # `Provincie Groningen` ~ `Groningen` — two mutually exclusive proposals for
    # one short form, at most one right. That is why `head_affix` returns the run
    # it removed and why `MergeCandidate.evidence` carries it to the card: without
    # it the two proposals are indistinguishable.
    "provincie",
    "de provincie",
    # A legal form in front of the organisation's OWN name, so the two denote one
    # entity. The residual risk is a homonym rather than a reference error —
    # `Stichting Lezen` / `Lezen`, `Stichting Vluchteling` / `Vluchteling` — where
    # the foundation's name is also a common noun another entity may own. That is
    # what review is for, and the `entity_type` bucket suppresses the usual case
    # (organization against topic). It does NOT suppress it where PC.4 found 38%
    # of entities land, in `concept`/`other`, so this is a known cost.
    "stichting",
    "vereniging",
    # Both spellings: NFKC does not strip the diaeresis, so the curated form must
    # carry the one people actually type as well.
    "coöperatie",
    "cooperatie",
    # Personal honorifics — unambiguously the same person.
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
