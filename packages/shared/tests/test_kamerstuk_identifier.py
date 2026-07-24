"""Tests for the Kamerstuk identifier normalizer (Track V, GF.2).

The normalizer is a bounded-identifier parser (like a DOI parser): it recognizes a
Dutch government-document identifier across its surface variants and normalizes it
to one structured id, so two strings denoting the same document compare equal. The
detector fires only on a dossiernummer PLUS a government cue (GF-D1); a bare number
(page / year / € amount) must not fire.
"""

from __future__ import annotations

import pytest
from shared.references.kamerstuk import (
    KamerstukIdentifier,
    parse_kamerstuk_identifier,
)

# --------------------------------------------------------------------------
# The five surface variants → normalized id (the GF.2 variant table)
# --------------------------------------------------------------------------

_VARIANTS = [
    (
        "Tweede Kamer, vergaderjaar 24/25, 12345-VII, blg I",
        KamerstukIdentifier("12345", "VII", "blg", "1", "kamerstuk"),
    ),
    (
        "12345-VII-blg-1",
        KamerstukIdentifier("12345", "VII", "blg", "1", "kamerstuk"),
    ),
    (
        "Kamerstukken II 2024/25, 36410, nr. 111",
        KamerstukIdentifier("36410", None, None, "111", "kamerstuk"),
    ),
    (
        "Kamerstuk 31305-489",
        KamerstukIdentifier("31305", None, None, "489", "kamerstuk"),
    ),
    (
        "Motie 36410-111",
        KamerstukIdentifier("36410", None, None, "111", "motie"),
    ),
]


@pytest.mark.parametrize("text, expected", _VARIANTS)
def test_variants_normalize(text: str, expected: KamerstukIdentifier) -> None:
    assert parse_kamerstuk_identifier(text) == expected


def test_surface_variants_of_same_document_are_equal() -> None:
    # The equality contract: "12345-VII, blg I" ≡ "12345-VII-blg-1" (Roman I ≡ 1,
    # comma/space ≡ dash).
    a = parse_kamerstuk_identifier("12345-VII, blg I")
    b = parse_kamerstuk_identifier("12345-VII-blg-1")
    assert a is not None and b is not None
    assert a == b


def test_full_lead_words_match_the_bare_core() -> None:
    # The verbose "Tweede Kamer, vergaderjaar …" form normalizes to the same id as
    # the bare "12345-VII, blg I" core.
    verbose = parse_kamerstuk_identifier(
        "Tweede Kamer, vergaderjaar 24/25, 12345-VII, blg I"
    )
    bare = parse_kamerstuk_identifier("12345-VII, blg I")
    assert verbose == bare


def test_motie_kind_is_detected() -> None:
    ident = parse_kamerstuk_identifier("Motie 36410-111")
    assert ident is not None
    assert ident.kind == "motie"
    assert ident.dossier == "36410"
    assert ident.nummer == "111"


def test_vergaderjaar_year_is_not_read_as_dossier() -> None:
    # "2024" (a session year) must not win over "36410" (the dossiernummer).
    ident = parse_kamerstuk_identifier("Kamerstukken II 2024/25, 36410, nr. 111")
    assert ident is not None
    assert ident.dossier == "36410"


# --------------------------------------------------------------------------
# Negatives — a bare number never fires (GF-D1: number + cue)
# --------------------------------------------------------------------------

_NEGATIVES = [
    "2026",  # a bare year
    "blz. 12",  # a page number
    "€ 3,5 miljoen",  # a monetary amount
    "Een overzicht van deze verdeling vindt u in de bijlage.",  # plain prose
    "Zoals onderwijs-en zorgspartijen en vertegenwoordigers van arbeidsmarkten",
    "",  # empty
    "   ",  # whitespace
]


@pytest.mark.parametrize("text", _NEGATIVES)
def test_non_government_text_does_not_fire(text: str) -> None:
    assert parse_kamerstuk_identifier(text) is None


def test_number_without_cue_does_not_fire() -> None:
    # 5 digits but no government cue → not a dossiernummer.
    assert parse_kamerstuk_identifier("De begroting bedraagt 36410 euro") is None


def test_embedded_identifier_in_prose_fires() -> None:
    # A real reference wrapped in explanatory prose is still a government id.
    ident = parse_kamerstuk_identifier(
        "Bij beleids- en investeringslogica ligt de focus op … (Kamerstuk 29697-158)."
    )
    assert ident is not None
    assert ident.dossier == "29697"
    assert ident.nummer == "158"
    assert ident.kind == "kamerstuk"
