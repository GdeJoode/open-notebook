"""Kamerstuk identifier normalizer (Track V, GF.2) — a bounded-identifier parser.

A Dutch government-document reference (Kamerstuk / Motie / begroting / bijlage)
carries a **bounded, stable identifier** — like a DOI or an arXiv id — not an
open-ended citation style. This module recognizes that identifier across its
surface variants and normalizes it to one structured :class:`KamerstukIdentifier`,
so two strings that denote the same document compare equal.

Why a normalizer, not a citation-style enumerator
==================================================
The academic-reference path (G.*) delegates to GROBID precisely because scholarly
citation styles are unbounded. A kamerstuk identifier is the opposite: a fixed
scheme — **dossiernummer** (4–6 digits) + optional **begrotings-hoofdstuk** (a
Roman numeral, e.g. ``VII``) + an optional **sub-document** (``blg`` = bijlage) +
a **nummer**. Only the SURFACE varies (separators, lead words, Roman↔arabic), so
the tolerance lives in this one normalizer, never in per-format branches.

Detector tolerance (GF-D1)
==========================
The parser fires only on a dossiernummer **plus** a government cue
(``Kamerstuk`` / ``Motie`` / ``blg`` / ``nr.`` / a ``-<Roman>`` hoofdstuk /
``vergaderjaar`` / ``Tweede Kamer``). A bare 4–6 digit number (a page, a year, a
count, a € amount) must NOT fire — precision over recall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# A vergaderjaar span ("2024/25" or "24/25"). Removed BEFORE the dossier search so
# a session year is never mistaken for the dossiernummer.
_VERGADERJAAR = re.compile(r"\b\d{2}(?:\d{2})?/\d{2}\b")

# The dossiernummer: a 4–6 digit run.
_DOSSIER = re.compile(r"\b(\d{4,6})\b")

# A Roman numeral immediately following the dossier is the begrotings-hoofdstuk
# (e.g. "12345-VII"). Uppercase only, so lowercase prose never matches; a trailing
# separator/end guards against gluing onto the next token.
_HOOFDSTUK = re.compile(r"^[\s,.\-]*([IVXLCDM]+)(?=[\s,.\-]|$)")

# The sub-document cue: "blg" (bijlage). "nr." is only a nummer marker, not a
# sub-document type, so it is not a subtype here.
_BLG = re.compile(r"\bblg\b", re.IGNORECASE)

# A token that is a whole UPPERCASE Roman numeral (so "blg I" ≡ "blg 1"). Uppercase
# only, so a lowercase prose word made of Roman letters (e.g. "mix", "did") is never
# read as a number.
_ROMAN_TOKEN = re.compile(r"^[IVXLCDM]+$")

# Government cues (GF-D1). Any one of these, together with a dossiernummer, fires
# the detector; without a cue a bare number is left alone.
_CUES: tuple[re.Pattern[str], ...] = (
    re.compile(r"kamerstuk", re.IGNORECASE),
    re.compile(r"motie", re.IGNORECASE),
    re.compile(r"\bblg\b", re.IGNORECASE),
    re.compile(r"\bnr\.?", re.IGNORECASE),
    re.compile(r"vergaderjaar", re.IGNORECASE),
    re.compile(r"tweede\s+kamer", re.IGNORECASE),
    # a dossier-attached Roman hoofdstuk, e.g. "12345-VII" — a digit, then a dash,
    # then an UPPERCASE Roman numeral (so "Nederland-Vlaanderen" does not count).
    re.compile(r"\d\s*-\s*[IVXLCDM]+\b"),
)

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


@dataclass(frozen=True)
class KamerstukIdentifier:
    """A normalized Dutch government-document identifier.

    Two surface variants denoting the same document normalize to an equal instance
    (frozen dataclass → value equality), e.g. ``12345-VII, blg I`` ≡
    ``12345-VII-blg-1``.

    Attributes:
        dossier: The dossiernummer (4–6 digit string, verbatim digits).
        hoofdstuk: The begrotings-hoofdstuk as an uppercase Roman numeral (e.g.
            ``"VII"``), or ``None``.
        subtype: The sub-document type (``"blg"`` for a bijlage), or ``None``.
        nummer: The sub-document / document number in canonical ARABIC form (a
            Roman ``I`` normalizes to ``"1"``), or ``None``.
        kind: ``"motie"`` when the citation names a motie, else ``"kamerstuk"``.
    """

    dossier: str
    hoofdstuk: Optional[str]
    subtype: Optional[str]
    nummer: Optional[str]
    kind: str


def _roman_to_int(roman: str) -> int:
    """Classic subtractive Roman-to-arabic (``VII`` → 7, ``IV`` → 4)."""
    total = 0
    prev = 0
    for char in reversed(roman.upper()):
        value = _ROMAN_VALUES[char]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total


def _canonical_number(token: str) -> str:
    """Canonicalize a nummer token to an arabic string (``"I"`` → ``"1"``)."""
    if token.isdigit():
        return str(int(token))
    return str(_roman_to_int(token))


def _first_number(text: str) -> Optional[str]:
    """First arabic-or-Roman number token in ``text``, canonicalized, else ``None``."""
    for token in re.findall(r"[A-Za-z0-9]+", text):
        if token.isdigit():
            return str(int(token))
        if _ROMAN_TOKEN.match(token):
            return str(_roman_to_int(token))
    return None


def _has_cue(text: str) -> bool:
    return any(cue.search(text) for cue in _CUES)


def parse_kamerstuk_identifier(text: str) -> Optional[KamerstukIdentifier]:
    """Normalize a government-document reference to a structured identifier.

    Format-tolerant: the same parser reads every surface variant (comma/space/dash
    separators, ``Kamerstukken II``/``Tweede Kamer``/``vergaderjaar`` lead words,
    Roman-or-arabic sub-numbers). Fires only on a dossiernummer **plus** a
    government cue (GF-D1); returns ``None`` for a bare number, a page, a year, a €
    amount, or plain prose.

    Args:
        text: A footnote (or any) string that may hold a kamerstuk identifier.

    Returns:
        The normalized :class:`KamerstukIdentifier`, or ``None`` when the text does
        not denote a government document.
    """
    if not text or not text.strip():
        return None
    # GF-D1: a government cue is required — a bare number never fires.
    if not _has_cue(text):
        return None

    # Strip vergaderjaar spans so a session year is not read as the dossiernummer.
    cleaned = _VERGADERJAAR.sub(" ", text)
    dossier_match = _DOSSIER.search(cleaned)
    if dossier_match is None:
        return None
    dossier = dossier_match.group(1)

    kind = "motie" if re.search(r"motie", text, re.IGNORECASE) else "kamerstuk"

    tail = cleaned[dossier_match.end() :]

    # Optional begrotings-hoofdstuk: a Roman numeral directly after the dossier.
    hoofdstuk: Optional[str] = None
    hoofdstuk_match = _HOOFDSTUK.match(tail)
    if hoofdstuk_match is not None:
        hoofdstuk = hoofdstuk_match.group(1).upper()
        tail = tail[hoofdstuk_match.end() :]

    # Optional sub-document (bijlage) + its nummer; else the first trailing number.
    subtype: Optional[str] = None
    blg_match = _BLG.search(tail)
    if blg_match is not None:
        subtype = "blg"
        nummer = _first_number(tail[blg_match.end() :])
    else:
        nummer = _first_number(tail)

    return KamerstukIdentifier(
        dossier=dossier,
        hoofdstuk=hoofdstuk,
        subtype=subtype,
        nummer=nummer,
        kind=kind,
    )


__all__ = ["KamerstukIdentifier", "parse_kamerstuk_identifier"]
