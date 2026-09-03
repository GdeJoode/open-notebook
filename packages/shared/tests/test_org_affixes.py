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
        ("Provincie Groningen", "Groningen", ("provincie",)),
        ("Ministerie van Onderwijs", "Onderwijs", ("ministerie", "van")),
        ("de heer Rob Opdam", "Rob Opdam", ("de", "heer")),
        ("Dagelijks Bestuur van Wetterskip Fryslân", "Wetterskip Fryslân",
         ("dagelijks", "bestuur", "van")),
        ("Het college van burgemeester en wethouders van de gemeente Leudal",
         "Leudal",
         ("het", "college", "van", "burgemeester", "en", "wethouders", "van",
          "de", "gemeente")),
    ],
)
def test_short_forms_are_recognised(outer: str, inner: str, removed: tuple) -> None:
    assert head_affix(_t(outer), _t(inner)) == removed


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
