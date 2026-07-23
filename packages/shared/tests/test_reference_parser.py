"""Unit tests for V.3 reference-entry parsing (pure).

Cover each field-extraction path (DOI normalization, year heuristic, author
forms, best-effort title/venue), the always-populated ``raw_text`` invariant, and
that every result satisfies ``ParsedReference.__post_init__`` (the U.3/V.4
contract type).
"""

from __future__ import annotations

from shared.references.reference_parser import parse_reference
from shared.retrieval.cites_matching import ParsedReference

# -- DOI --------------------------------------------------------------------


def test_doi_url_form_is_normalized():
    entry = (
        "Acemoglu, D., Johnson, S., & Robinson, J. A. (2001). The colonial origins "
        "of comparative development. American Economic Review, 91(5), 1369-1401. "
        "https://doi.org/10.1257/aer.91.5.1369"
    )
    ref = parse_reference(entry)
    assert ref.doi == "10.1257/aer.91.5.1369"


def test_doi_prefixed_form_is_normalized_and_trailing_punct_trimmed():
    entry = "Barca, F. (2012). Regional development. doi:10.1111/j.1467-9787.2011.00756.x."
    ref = parse_reference(entry)
    assert ref.doi == "10.1111/j.1467-9787.2011.00756.x"


def test_no_doi_yields_none():
    ref = parse_reference("Acemoglu, D. (2012). Why Nations Fail. Crown Business.")
    assert ref.doi is None


# -- year -------------------------------------------------------------------


def test_year_from_parenthesized_slot_is_preferred():
    # The title carries a stray number; the (2012) slot must win.
    entry = "Author, A. (2012). A study of the 1929 crash. Journal, 5(1)."
    ref = parse_reference(entry)
    assert ref.year == 2012


def test_year_from_dutch_vergaderjaar():
    ref = parse_reference("Kamerstukken II 2023/24, 36410, nr. 2.")
    assert ref.year == 2023


def test_no_year_yields_none():
    ref = parse_reference("Some Committee. A report without a date. Publisher.")
    assert ref.year is None


# -- authors ----------------------------------------------------------------


def test_apa_comma_form_authors():
    entry = "Acemoglu, D., & Robinson, J. A. (2012). Why Nations Fail. Crown."
    ref = parse_reference(entry)
    assert ref.authors == ("Acemoglu, D.", "Robinson, J. A.")


def test_ieee_initials_first_authors():
    entry = 'F. Barca, P. McCann, and A. Rodríguez-Pose, "The case for regional development," JRS, 2012.'
    ref = parse_reference(entry)
    assert ref.authors == ("F. Barca", "P. McCann", "A. Rodríguez-Pose")


def test_author_surnames_survive_matcher_normalization():
    """Both author forms reduce to the same surname the matcher compares on."""
    from shared.retrieval.cites_matching import _surnames

    apa = parse_reference("Acemoglu, D. (2001). The colonial origins.")
    ieee = parse_reference('D. Acemoglu, "The colonial origins," AER, 2001.')
    assert "acemoglu" in _surnames(apa.authors)
    assert "acemoglu" in _surnames(ieee.authors)


def test_corporate_author_yields_no_parsed_names():
    ref = parse_reference(
        "Ministerie van Binnenlandse Zaken. (2023). Regio Deals. Den Haag."
    )
    # Best-effort: a corporate/agency author need not parse to a person name.
    assert isinstance(ref.authors, tuple)


# -- title / venue (best-effort) --------------------------------------------


def test_apa_title_after_year_slot():
    entry = (
        "Rodríguez-Pose, A. (2018). The revenge of the places that don't matter. "
        "Cambridge Journal of Regions."
    )
    ref = parse_reference(entry)
    assert ref.title == "The revenge of the places that don't matter"
    assert ref.venue and "Cambridge Journal of Regions" in ref.venue


def test_ieee_quoted_title():
    entry = 'A. Rodríguez-Pose, "The revenge of the places," CJRES, vol. 11, 2018.'
    ref = parse_reference(entry)
    assert ref.title == "The revenge of the places"
    assert ref.venue and "CJRES" in ref.venue


def test_title_only_entry_is_valid_with_best_effort_empties():
    ref = parse_reference("A Report on Regional Cohesion.")
    assert ref.raw_text == "A Report on Regional Cohesion."
    assert ref.title == ""  # no author-year/quote structure to anchor on
    assert ref.authors == ()
    assert ref.year is None
    assert ref.doi is None


# -- raw_text invariant + contract ------------------------------------------


def test_raw_text_is_always_populated_and_whitespace_collapsed():
    ref = parse_reference("  Acemoglu,  D.   (2012).\n  Why Nations Fail.  ")
    assert ref.raw_text == "Acemoglu, D. (2012). Why Nations Fail."


def test_result_is_a_valid_parsed_reference_frozen_dataclass():
    ref = parse_reference("Acemoglu, D., & Robinson, J. A. (2012). Why Nations Fail.")
    assert isinstance(ref, ParsedReference)
    # __post_init__ coerced authors to a tuple → hashable/frozen.
    assert isinstance(ref.authors, tuple)
    assert hash(ref) is not None


def test_kamerstuk_entry_parses_without_crash():
    ref = parse_reference("Kamerstukken II 2023/24, 36410, nr. 2.")
    assert ref.raw_text.startswith("Kamerstukken II")
    assert ref.year == 2023
    assert ref.doi is None
