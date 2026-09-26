"""D1: `parse_surface` splits a surface form into names, and never over-splits.

Every string below is from the working corpus (active rows, 2026-09-26) unless
marked. The traps are the point: each is a shape a cruder rule got wrong.
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
        ("Programma Wonen, Ondersteuning en Zorg Ouderen (WOZO)", "topic", ABBREVIATION,
         ("Programma Wonen, Ondersteuning en Zorg Ouderen (WOZO)",
          "Programma Wonen, Ondersteuning en Zorg Ouderen", "WOZO")),
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
    "Minister van Binnenlandse Zaken en Koninkrijksrelaties",  # PC.2's office-of pair
    "Gedeputeerde Staten van Drenthe",
    "Burgemeester van Rotterdam",
    "Raad van Toezicht",
])
def test_every_organ_prefix_blocks(name):
    """Not from the corpus: org_affixes's review findings, one per organ run."""
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


@pytest.mark.parametrize("name", ["Gemeente Utrecht", "Provincie Utrecht"])
def test_a_municipal_or_provincial_prefix_is_a_hint_never_a_split(name):
    """Offering `Utrecht` as a name of both would propose merging them."""
    parts = parse_surface(name, "administrative_area")
    assert parts.names == (name,)
    assert parts.type_word in {"gemeente", "provincie"}


def test_the_type_prefix_is_scoped_to_its_type():
    """`Ministerie van Onderwijs` and the topic `Onderwijs` are different things."""
    assert parse_surface("Ministerie van Onderwijs", "topic").names == ("Ministerie van Onderwijs",)
    assert parse_surface("Onderwijs (Ministerie)", "topic").names == ("Onderwijs (Ministerie)",)


def test_an_invented_gloss_is_offered_separately_so_a_consumer_can_see_it_disagree():
    """The text says VRO; the extractor glossed it as VWS. Both names are offered —
    choosing between them is the consumer's refusal to make, not this module's."""
    parts = parse_surface("VRO (Ministerie van Volksgezondheid, Welzijn en Sport)", GOV)
    assert parts.shape == ABBREVIATION
    assert parts.names[1:] == ("VRO", "Volksgezondheid, Welzijn en Sport")


def test_empty_and_whitespace():
    assert parse_surface("", GOV).names == ()
    assert parse_surface("  IenW \n", GOV).names == ("IenW",)
