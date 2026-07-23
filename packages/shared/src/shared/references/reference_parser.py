"""Reference-entry parsing (Track V.3) — one entry string → ``ParsedReference``.

The V.2 segmenter yields the verbatim text of each reference; this module turns
each into a structured :class:`shared.retrieval.cites_matching.ParsedReference` —
the exact type V.4's ``WorkResolver`` and U.3's matcher consume. It is the last
pure stage of the producer pipeline.

Best-effort, but ``raw_text`` is always set
==========================================
Bibliographic entries are wildly heterogeneous (APA, Chicago, IEEE-numbered, a
Dutch Kamerstuk, a bare title). Rather than a brittle full grammar, this parser
extracts the fields it can with transparent regex heuristics and leaves the rest
empty — every field except ``raw_text`` is best-effort. ``raw_text`` is ALWAYS
the verbatim entry, so even an entry we can only partially parse still produces a
valid ``ParsedReference`` (the matcher/resolver fall back to whatever is present,
and a reviewer always sees the original string).

Extraction discipline
=====================
* **DOI** — regex for a ``10.xxxx/...`` (bare, ``doi:``-prefixed, or a
  ``doi.org`` URL), canonicalized through the shared
  :func:`shared.retrieval.cites_matching.normalize_doi` (the same normalization
  U.3/V.4 compare on).
* **Year** — a 4-digit 1500–2099 token; a parenthesized ``(2012)`` is preferred
  (the author-year slot) over a stray in-title number, and a Dutch vergaderjaar
  ``2023/24`` yields the first year.
* **Authors** — the leading ``Surname, I.`` author list (APA/Chicago), split on
  ``&`` / ``;`` / ``and`` / the comma between authors. The matcher normalizes to
  surnames later, so the parsed form need only preserve the surname.
* **Title / venue** — best-effort: a quoted title (IEEE), or the sentence after
  the ``(year).`` slot (APA), with the remainder as the venue.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from shared.retrieval.cites_matching import ParsedReference, normalize_doi

# --------------------------------------------------------------------------
# DOI
# --------------------------------------------------------------------------

# A DOI as it appears in text: optional ``doi:`` / ``https://doi.org/`` prefix,
# then the ``10.<registrant>/<suffix>``. The suffix stops before whitespace and
# trailing sentence punctuation is trimmed afterwards.
_DOI_RE = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?"
    r"(10\.\d{4,9}/[^\s]+)",
    re.IGNORECASE,
)


def _extract_doi(text: str) -> Optional[str]:
    """Return the normalized DOI in ``text``, or ``None``.

    Trims trailing sentence punctuation (a DOI at the end of an entry often has a
    closing ``.`` or ``)`` that is not part of the identifier), then normalizes.
    """
    m = _DOI_RE.search(text)
    if not m:
        return None
    raw = m.group(1).rstrip(".,;)]}")
    normalized = normalize_doi(raw)
    return normalized or None


# --------------------------------------------------------------------------
# Year
# --------------------------------------------------------------------------

_YEAR_PAREN = re.compile(r"\((?:[^)]*?\b)?((?:1[5-9]|20)\d{2})[a-z]?\b[^)]*\)")
_YEAR_ANY = re.compile(r"\b((?:1[5-9]|20)\d{2})\b")


def _extract_year(text: str) -> Optional[int]:
    """Return a publication year (1500–2099), preferring the author-year slot.

    A parenthesized year — ``(2012)`` / ``(2012a)`` — is the author-year slot and
    is preferred over any other 4-digit run (e.g. a number inside a title or a
    page range). Falls back to the first plausible 4-digit year anywhere.
    """
    m = _YEAR_PAREN.search(text)
    if m:
        return int(m.group(1))
    m2 = _YEAR_ANY.search(text)
    if m2:
        return int(m2.group(1))
    return None


# --------------------------------------------------------------------------
# Authors
# --------------------------------------------------------------------------

# The author-block terminator: the author list ends at the year slot ``(2012).``,
# a bare ``2001.`` year (Chicago), or the first quoted title. Whichever comes
# first bounds the author segment.
_YEAR_SLOT = re.compile(r"\((?:1[5-9]|20)\d{2}[a-z]?\)|(?<!\d)(?:1[5-9]|20)\d{2}\.")
_QUOTE_OPEN = re.compile(r"[\"“]")

# One APA/Chicago author: "Surname, D." / "Surname, Daron" / "van Reenen, J."
_AUTHOR_TOKEN = re.compile(
    r"(?:(?:van|de|der|von|el|di|la|le)\s+)?"
    r"[A-Z][A-Za-z'À-ſ-]+,\s*(?:[A-Z]\.?(?:\s*[A-Z]\.?)*|[A-Z][a-z]+)",
)

# One IEEE-style initials-first author: "D. Acemoglu" / "J. A. Robinson". Used
# only as a fallback when no comma-form author is present (the matcher takes the
# last whitespace token as the surname, so "D. Acemoglu" → "acemoglu").
_AUTHOR_TOKEN_INITIALS = re.compile(
    r"(?:[A-Z]\.\s*)+(?:(?:van|de|der|von|el|di|la|le)\s+)?[A-Z][A-Za-z'À-ſ-]+"
)

# The author segment opens with an initial ("F. Barca …") → IEEE initials-first.
_LEADING_INITIALS = re.compile(r"[A-Z]\.\s")


def _author_segment(text: str) -> str:
    """Return the leading substring that holds the author list.

    Bounded by the first year slot or the first opening quote (a title), so the
    author extractor never runs past the authors into the title/venue.
    """
    ends: List[int] = []
    ym = _YEAR_SLOT.search(text)
    if ym:
        ends.append(ym.start())
    qm = _QUOTE_OPEN.search(text)
    if qm:
        ends.append(qm.start())
    cut = min(ends) if ends else len(text)
    return text[:cut]


def _extract_authors(text: str) -> Tuple[str, ...]:
    """Extract author names as ``("Surname, I.", ...)`` (best-effort, may be empty).

    Finds every ``Surname, Initials`` token in the bounded author segment. A
    corporate/agency author or a bare title yields no token → an empty tuple
    (valid: the reference simply carries no authors). "et al." contributes no
    extra name (the named leads are already captured).
    """
    segment = _author_segment(text)
    if _LEADING_INITIALS.match(segment.lstrip()):
        # IEEE initials-first list ("F. Barca, P. McCann, and A. Rodríguez-Pose"):
        # the comma-form regex mis-reads these, so parse initials-first directly.
        matches = [m.group(0).strip() for m in _AUTHOR_TOKEN_INITIALS.finditer(segment)]
    else:
        matches = [
            m.group(0).strip().rstrip(",") for m in _AUTHOR_TOKEN.finditer(segment)
        ]
        if not matches:
            # No comma-form author — try the initials-first form as a fallback.
            matches = [
                m.group(0).strip() for m in _AUTHOR_TOKEN_INITIALS.finditer(segment)
            ]
    # De-duplicate while preserving order (a malformed segment can double-match).
    seen: set = set()
    authors: List[str] = []
    for name in matches:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            authors.append(name)
    return tuple(authors)


# --------------------------------------------------------------------------
# Title / venue (best-effort)
# --------------------------------------------------------------------------

_QUOTED_TITLE = re.compile(r"[\"“]([^\"”]+)[\"”]")
# APA: after the ``(year).`` slot, the title runs to the next period.
_APA_TITLE = re.compile(
    r"\((?:1[5-9]|20)\d{2}[a-z]?\)\.\s*(.+?)(?:\.\s|\.$)",
)


def _extract_title_venue(text: str) -> Tuple[str, str]:
    """Best-effort ``(title, venue)`` from an entry.

    Two cheap paths (empty strings when neither applies — a title-only or
    Kamerstuk entry may yield no structured title, which is valid):

    * **Quoted** (IEEE-ish) — the first ``"..."`` is the title; the text after the
      closing quote is the venue.
    * **APA** — the sentence right after the ``(year).`` slot is the title; the
      remainder is the venue.
    """
    qm = _QUOTED_TITLE.search(text)
    if qm:
        title = qm.group(1).strip().rstrip(",.")
        venue = text[qm.end() :].strip().lstrip(",. ")
        return title, _clean_venue(venue)

    am = _APA_TITLE.search(text)
    if am:
        title = am.group(1).strip()
        venue = text[am.end() :].strip()
        return title, _clean_venue(venue)

    return "", ""


def _clean_venue(venue: str) -> str:
    """Trim a venue tail: drop a trailing DOI/URL and surrounding punctuation."""
    venue = _DOI_RE.sub("", venue)
    venue = re.sub(r"https?://[^\s]+", "", venue)
    return venue.strip().strip(".,;:- ")


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def parse_reference(entry: str) -> ParsedReference:
    """Parse ONE reference entry string into a :class:`ParsedReference` (pure).

    Best-effort field extraction; ``raw_text`` is always the verbatim (whitespace-
    collapsed) entry so the result is valid even when only some fields parse. The
    returned object satisfies :meth:`ParsedReference.__post_init__` (authors are
    coerced to a tuple).

    Args:
        entry: A single reference entry (the V.2 segmenter output unit).

    Returns:
        A :class:`ParsedReference` with whatever fields could be extracted; an
        empty/whitespace entry still returns a valid object with ``raw_text``
        preserved (the collapsed original).
    """
    raw = re.sub(r"\s+", " ", entry).strip()

    doi = _extract_doi(entry)
    year = _extract_year(entry)
    authors = _extract_authors(entry)
    title, venue = _extract_title_venue(entry)

    return ParsedReference(
        raw_text=raw,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        venue=venue or None,
    )


__all__ = [
    "parse_reference",
]
