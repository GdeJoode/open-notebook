"""The one comparison fold (PC.2).

These cases are the ones that were executed against all four private copies
before they were replaced, to establish that consolidating was a pure refactor.
They are kept as the fold's own contract, so a later "improvement" to it has to
argue with the four stages it silently changes.
"""

from __future__ import annotations

import pytest

from shared.utils.text_folding import fold_for_comparison


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Regio Deal", "regio deal"),
        ("  Regio   Deal  ", "regio deal"),
        ("REGIO\tDEAL", "regio deal"),
        ("Regio Deal", "regio deal"),  # NBSP, folded by NFKC
        ("Regio　Deal", "regio deal"),  # ideographic space
        ("Ｒｅｇｉｏ Ｄｅａｌ", "regio deal"),  # full-width forms
        ("ﬁnanciering", "financiering"),  # ligature
        ("Regio\nDeal", "regio deal"),
        ("", ""),
        ("   ", ""),
        ("\n\t ", ""),
    ],
)
def test_fold(raw: str, expected: str) -> None:
    assert fold_for_comparison(raw) == expected


def test_idempotent() -> None:
    """Folding a folded string changes nothing.

    Matters because folded values are stored as dict keys and compared against
    freshly folded input; a non-idempotent fold would make lookups miss.
    """
    for raw in ("  Ｒｅｇｉｏ Deal ", "ﬁnanciering", "A", ""):
        once = fold_for_comparison(raw)
        assert fold_for_comparison(once) == once


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Regio Deal", "Regio-Deal"),  # punctuation is NOT stripped
        ("Fryslân", "Fryslan"),  # diacritics are NOT folded
        ("de gemeente", "gemeente"),  # articles are NOT removed
        ("Deal Regio", "Regio Deal"),  # tokens are NOT reordered
    ],
)
def test_what_the_fold_deliberately_does_not_do(a: str, b: str) -> None:
    """Each of these would change what four shipped stages consider equal.

    Pinned as non-behaviour because every one of them is a plausible-looking
    addition, and the cost of adding one is paid in a different package.
    """
    assert fold_for_comparison(a) != fold_for_comparison(b)


def test_turkish_dotted_i_lowercases_ascii_not_locale() -> None:
    """`I` folds to `i`, not to the Turkish dotless `ı`.

    Python's `str.lower()` is locale-independent; recorded so that a later switch
    to a locale-aware casefold is visible as a behaviour change rather than a
    tidy-up.
    """
    assert fold_for_comparison("IJSSEL") == "ijssel"
    assert fold_for_comparison("İstanbul") != "istanbul"
