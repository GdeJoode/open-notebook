"""D1: `parse_surface` splits a surface form into names, and never over-splits.

Every string below is from the working corpus (staging, 2026-09-26; "wider" means
a non-active row) unless marked "constructed". The traps are the point: each is a
shape a cruder rule got wrong.
"""

from __future__ import annotations

import pytest
from shared.utils.surface_parts import (
    ABBREVIATION,
    CATEGORY_PREFIX,
    MULTIPLE,
    ORGAN,
    PLAIN,
    QUALIFIER,
    TYPE_PREFIX,
    TYPE_SUFFIX,
    parse_surface,
)

GOV = "government_organization"


@pytest.mark.parametrize(
    "name, etype, shape, names",
    [
        # Type suffix: the word restates entity_type.
        ("Economische Zaken (Ministerie)", GOV, TYPE_SUFFIX,
         ("Economische Zaken (Ministerie)", "Economische Zaken")),
        # Type prefix, TOOI's naamInclSoort form.
        ("Ministerie van Volksgezondheid, Welzijn en Sport", GOV, TYPE_PREFIX,
         ("Ministerie van Volksgezondheid, Welzijn en Sport",
          "Volksgezondheid, Welzijn en Sport")),
        # Abbreviation + gloss; a gloss's own "ministerie van" is the type prefix.
        ("VRO (ministerie van Volkshuisvesting en Ruimtelijke Ordening)", GOV, ABBREVIATION,
         ("VRO (ministerie van Volkshuisvesting en Ruimtelijke Ordening)", "VRO",
          "Volkshuisvesting en Ruimtelijke Ordening")),
        ("IenW (Infrastructuur en Waterstaat)", GOV, ABBREVIATION,
         ("IenW (Infrastructuur en Waterstaat)", "IenW", "Infrastructuur en Waterstaat")),
        # Long form first, abbreviation in the parenthesis.
        ("Nationaal Programma Landelijk Gebied (NPLG)", "topic", ABBREVIATION,
         ("Nationaal Programma Landelijk Gebied (NPLG)",
          "Nationaal Programma Landelijk Gebied", "NPLG")),
        # Initials over every word, function words included.
        ("Spil in de Wijk (SIDW)", "programme", ABBREVIATION,
         ("Spil in de Wijk (SIDW)", "Spil in de Wijk", "SIDW")),
        # Category prefix with ONE referent after it.
        ("ministeries van het Rijk: Volkshuisvesting en Ruimtelijke Ordening", GOV,
         CATEGORY_PREFIX,
         ("ministeries van het Rijk: Volkshuisvesting en Ruimtelijke Ordening",
          "Volkshuisvesting en Ruimtelijke Ordening")),
        ("IenW", GOV, PLAIN, ("IenW",)),
    ],
)
def test_the_names_a_surface_asserts(name, etype, shape, names):
    parts = parse_surface(name, etype)
    assert parts.shape == shape
    assert parts.names == names
    assert not parts.blocked
    assert parts.names[0] == parts.raw  # lossless: the raw form is always kept


# --- the traps ---------------------------------------------------------------


def test_an_office_of_a_ministry_is_blocked_not_resolved_through_its_gloss():
    """The D1 simulation linked this row to IenW through "(IenW)"."""
    parts = parse_surface("De Staatssecretaris van Infrastructuur en Waterstaat (IenW)", GOV)
    assert parts.shape == ORGAN and parts.blocked and parts.names == ()


@pytest.mark.parametrize("name", [
    # constructed: org_affixes's review findings, one per organ run
    "Minister van Binnenlandse Zaken en Koninkrijksrelaties",  # PC.2's office-of pair
    "Gedeputeerde Staten van Drenthe",
    "Burgemeester van Rotterdam",
    "Raad van Toezicht",
])
def test_every_organ_prefix_blocks(name):
    assert parse_surface(name, GOV).blocked


def test_a_list_of_ministries_is_several_referents_and_blocked():
    parts = parse_surface(
        "ministeries van het Rijk: Volkshuisvesting en Ruimtelijke Ordening, Binnenlandse "
        "Zaken en Koninkrijkrelaties/Herstel Groningen, Sociale Zaken en Werkgelegenheid, "
        "Justitie en Veiligheid", GOV)
    assert parts.shape == MULTIPLE and parts.blocked and parts.names == ()


@pytest.mark.parametrize("name, etype", [
    ("sterk regionaal netwerk (op onderwerpen)", "topic"),
    ("Regio (Zuidwest-Friesland)", "administrative_area"),
    ("Bereikbare en duurzame bedrijventerreinen (Beter Werken)", "topic"),
])
def test_a_qualifier_is_kept_whole(name, etype):
    """Dropping the parenthesis would change the referent."""
    parts = parse_surface(name, etype)
    assert parts.shape == QUALIFIER
    assert parts.names == (name,)


def test_a_numbered_label_is_identity_and_kept_whole():
    parts = parse_surface("Programmalijn 4: Economie", "topic")
    assert parts.names == ("Programmalijn 4: Economie",)


@pytest.mark.parametrize("name", ["Gemeente Utrecht", "Provincie Utrecht"])  # constructed
def test_a_municipal_or_provincial_prefix_is_a_hint_never_a_split(name):
    """Offering `Utrecht` as a name of both would propose merging them."""
    parts = parse_surface(name, "administrative_area")
    assert parts.names == (name,)
    assert parts.type_word in {"gemeente", "provincie"}


def test_the_type_prefix_is_scoped_to_its_type():
    """`Ministerie van Onderwijs` and the topic `Onderwijs` are different things."""
    assert parse_surface("Ministerie van Onderwijs", "topic").names == ("Ministerie van Onderwijs",)
    # constructed
    assert parse_surface("Onderwijs (Ministerie)", "topic").names == ("Onderwijs (Ministerie)",)


@pytest.mark.parametrize("name, etype", [
    # The extractor's gloss is not VRO's initials (it names VWS).
    ("VRO (Ministerie van Volksgezondheid, Welzijn en Sport)", GOV),
    ("VRO (VROM, Volkshuisvesting, Ruimtelijke Ordening en Milieubeheer)", GOV),
    # D1 review, wider graph: an X of the abbreviation, and different expansions.
    ("cultuur van Leven Lang Ontwikkelen (LLO)", "topic"),
    ("€ 975.000 voor Uitvoeringskosten (VAT)", "topic"),
    ("Hoge Veluwe National Park (NHN)", "other"),
    ("Hollandse Hoogte-Nieuwkoop (NHN)", "other"),
    ("Urban Population (PU)", "concept"),
    ("Institutional Quality Index (EQI)", "concept"),
    ("Stuurgroep Lokaal Educatieve Agenda (LEA)", "organization"),
    ("VRO (Vereniging van Regio's)", "organization"),
    # A leading generic word is not in the initials: kept whole (recall loss).
    ("Programma Wonen, Ondersteuning en Zorg Ouderen (WOZO)", "topic"),
])
def test_a_parenthesis_that_is_not_the_initials_is_kept_whole(name, etype):
    parts = parse_surface(name, etype)
    assert parts.shape == QUALIFIER
    assert parts.names == (name,)


@pytest.mark.parametrize("name", [
    "Het college van burgemeester en wethouders van de gemeente De Fryske Marren",
    "Algemeen Bestuur van de gemeenteraad",
    "Staatssecretaris Herstel Groningen",
])
def test_organ_forms_found_one_boundary_wider_are_blocked(name):
    """D1 review: the article `het`, `algemeen bestuur van`, and an office with
    no `van` all escaped the first organ list."""
    assert parse_surface(name, GOV).blocked


def test_the_type_rules_also_hold_for_plain_organization():
    """The wider graph types ministries as `organization` too."""
    parts = parse_surface("Ministerie van Infrastructuur en Waterstaat", "organization")
    assert parts.shape == TYPE_PREFIX
    assert parts.names[1:] == ("Infrastructuur en Waterstaat",)


def test_empty_and_whitespace():
    assert parse_surface("", GOV).names == ()
    assert parse_surface("  IenW \n", GOV).names == ("IenW",)
