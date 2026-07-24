"""Tests for the footnote classifier (Track V, GF.3).

Routes each footnote to government (a kamerstuk id), academic (a scholarly cue —
author-year / DOI / journal shape; GF-D2), or prose (neither, dropped). A mixed
fixture — the NPVR footnotes plus synthetic academic ones — covers all three.
"""

from __future__ import annotations

import pytest
from shared.references.footnote_classifier import classify_footnote

# --------------------------------------------------------------------------
# government — a kamerstuk identifier (GF-D1)
# --------------------------------------------------------------------------

_GOVERNMENT = [
    "Kamerstuk 31305-489",
    "Motie 36410-111",
    "Tweede Kamer, vergaderjaar 24/25, 12345-VII, blg I",
    # embedded in explanatory prose — still a government reference
    "Bij beleids- en investeringslogica … (Kamerstuk 29697-158).",
]


@pytest.mark.parametrize("text", _GOVERNMENT)
def test_government_footnotes(text: str) -> None:
    assert classify_footnote(text) == "government"


# --------------------------------------------------------------------------
# academic — a scholarly cue, no kamerstuk id (GF-D2)
# --------------------------------------------------------------------------

_ACADEMIC = [
    # author-year
    "Acemoglu, D. & Robinson, J. A. (2012). Why nations fail. Crown.",
    "See Card (2001) on native outflows.",
    # DOI
    "Card, D., Immigrant inflows, doi:10.1257/aer.91.5.1369",
    "Bare DOI 10.1080/13662710220123653 in a note.",
    # journal shape
    "Alesina et al., Fractionalization, Journal of Economic Growth, Vol. 8, pp. 155.",
]


@pytest.mark.parametrize("text", _ACADEMIC)
def test_academic_footnotes(text: str) -> None:
    assert classify_footnote(text) == "academic"


# --------------------------------------------------------------------------
# prose — neither (dropped)
# --------------------------------------------------------------------------

_PROSE = [
    "Een overzicht van deze verdeling vindt u in de bijlage.",
    "Zoals onderwijs-en zorgspartijen en vertegenwoordigers van arbeidsmarkten",
    "Samenwerkagenda met Nedersaksen, de Grenslandagenda en de grensgovernance "
    "Nederland-Vlaanderen.",
    "Het NPVR heeft drie overkoepelende doelstellingen: 1. Veilige regio's; "
    "2. Een duurzaam voorzieningenniveau; 3. Een gezonde toekomst.",
]


@pytest.mark.parametrize("text", _PROSE)
def test_prose_footnotes(text: str) -> None:
    assert classify_footnote(text) == "prose"


def test_three_way_split_is_exhaustive() -> None:
    # Every classification is one of the three labels — no fourth bucket.
    labels = {classify_footnote(t) for t in (_GOVERNMENT + _ACADEMIC + _PROSE)}
    assert labels == {"government", "academic", "prose"}
