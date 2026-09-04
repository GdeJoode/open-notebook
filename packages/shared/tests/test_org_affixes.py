"""Curated head runs (PC.2).

The rule has two halves — head-anchored AND curated — and each was chosen over a
weaker variant on measured live data. These tests pin both halves, including the
shapes that made the weaker variants fail.
"""

from __future__ import annotations

import pytest

from shared.utils.org_affixes import ORG_HEAD_AFFIXES, head_affix
from shared.utils.text_folding import fold_for_comparison


def _t(text: str) -> list[str]:
    return [t for t in fold_for_comparison(text).split(" ") if t]


@pytest.mark.parametrize(
    ("outer", "inner", "removed"),
    [
        ("Gemeente Leudal", "Leudal", ("gemeente",)),
        ("de gemeente Roermond", "Roermond", ("de", "gemeente")),
        ("Provincie Groningen", "Groningen", ("provincie",)),
        ("Stichting Mensenwerk", "Mensenwerk", ("stichting",)),
        # Both spellings: NFKC does not strip the diaeresis, so the ASCII form
        # people actually type must be curated separately.
        ("Coöperatie Zorgverlening", "Zorgverlening", ("coöperatie",)),
        ("Cooperatie Zorgverlening", "Zorgverlening", ("cooperatie",)),
        ("de heer Rob Opdam", "Rob Opdam", ("de", "heer")),
        ("mevrouw Ans de Vries", "Ans de Vries", ("mevrouw",)),
    ],
)
def test_short_forms_are_recognised(outer: str, inner: str, removed: tuple) -> None:
    assert head_affix(_t(outer), _t(inner)) == removed


@pytest.mark.parametrize(
    ("outer", "inner"),
    [
        # An organ, office or office-holder OF X is a different entity from X.
        # Every line here was produced by the list's first version and proposed
        # as a merge; adversarial review is what surfaced them. A rule that asks
        # a curator whether the mayor is the city is not asking a review question.
        ("Raad van Toezicht", "Toezicht"),
        ("Raad van Advies", "Advies"),
        ("Raad van State", "State"),
        ("College van Beroep", "Beroep"),
        ("Gemeenteraad van Amsterdam", "Amsterdam"),
        ("Burgemeester van Rotterdam", "Rotterdam"),
        ("Wethouder van Utrecht", "Utrecht"),
        ("Gedeputeerde Staten van Drenthe", "Drenthe"),
        ("Dagelijks Bestuur van Wetterskip Fryslân", "Wetterskip Fryslân"),
        ("College van Burgemeester en Wethouders van Leudal", "Leudal"),
        # `nl_normalization.strip_leading_noise` names this one in writing:
        # collapsing a named body onto a bare concept token another entity owns.
        ("Ministerie van Onderwijs", "Onderwijs"),
        # `waterschap` was in this list's first version and is removed. It fails
        # the test `gemeente` and `provincie` pass: no Dutch text uses "Limburg"
        # to mean "Waterschap Limburg". A water board is a distinct legal actor
        # whose management area is named after a province that is ITSELF a
        # separate entity, so this is a body-versus-territory pair.
        #
        # It also contradicted the case below: a water board's executive is not
        # the water board, while `Waterschap Limburg` / `Limburg` asserted that a
        # water board IS the province. Both cannot be the rule.
        ("Waterschap Limburg", "Limburg"),
        ("Waterschap Rivierenland", "Rivierenland"),
        # The PC.2 plan asked for this pair. It is an office-of pair, so it is
        # deliberately NOT produced — see the module docstring. The class is real
        # and belongs in a later phase as an organ-of RELATION, not a merge.
        ("Minister van Binnenlandse Zaken", "Binnenlandse Zaken"),
    ],
)
def test_an_organ_of_x_is_not_x(outer: str, inner: str) -> None:
    assert head_affix(_t(outer), _t(inner)) is None


@pytest.mark.parametrize(
    ("outer", "inner", "why"),
    [
        ("Regio Deal Groningen", "Regio Deal",
         "tail qualifier: a sibling, not a short form"),
        ("Mensenwerk Het Hogeland", "Het Hogeland",
         "an organisation named after a place is not that place"),
        ("Gemeente Amsterdam", "Utrecht",
         "curated head run, but the remainder does not match"),
        ("Gemeente Leudal", "Gemeente Leudal", "equal is not containment"),
        ("Leudal", "Gemeente Leudal", "arguments the wrong way round"),
        ("Gemeente Leudal", "", "empty inner"),
    ],
)
def test_rejected(outer: str, inner: str, why: str) -> None:
    assert head_affix(_t(outer), _t(inner)) is None, why


def test_no_curated_run_ends_in_a_relational_van() -> None:
    """A structural guard against re-adding the organ-of class.

    Every affix removed in review had the shape `<body> van` — the `van` is what
    makes it relational, i.e. an organ OF something rather than a qualifier on
    something. `de gemeente` and `de heer` are the counter-shape: the article
    binds forward to the same referent.

    This cannot catch every bad addition, and it is not meant to. It catches the
    one that already happened, which is the addition most likely to happen again.
    """
    for affix in ORG_HEAD_AFFIXES:
        assert affix[-1] != "van", (
            f"{' '.join(affix)!r} names an organ/office OF the remainder, not a "
            "qualifier on it — see the module docstring"
        )


def test_affixes_are_stored_in_folded_form() -> None:
    """Every curated run must be in the form the caller tokenises with.

    A single stray capital or double space in the list would make that entry
    silently unmatchable, and nothing else would fail.
    """
    for affix in ORG_HEAD_AFFIXES:
        assert affix, "empty affix"
        for token in affix:
            assert token == fold_for_comparison(token), f"unfolded token: {token!r}"
            assert " " not in token
