"""Footnote classifier + router (Track V, GF.3) — government | academic | prose.

A policy-document footnote is not always a reference. This pure classifier routes
each footnote so the extractor (GF.4) can send it to the right resolver:

* **government** — the footnote holds a Dutch government-document identifier
  (:func:`parse_kamerstuk_identifier` hits). Routed to the ``OverheidResolver``.
* **academic** — no kamerstuk id, but a scholarly cue is present (author-year, a
  DOI, or a journal shape). Routed to GROBID ``processCitation`` (GF-D2).
* **prose** — neither. An explanatory note; dropped.

Precision over recall (GF-D2): a footnote is only called academic on a real
scholarly cue, never guessed. Everything else is prose.
"""

from __future__ import annotations

import re

from shared.references.kamerstuk import parse_kamerstuk_identifier

# A DOI, bare (``10.1257/aer.91.5.1369``) or prefixed (``doi:…``).
_DOI = re.compile(r"\b10\.\d{4,9}/\S+", re.IGNORECASE)
_DOI_PREFIX = re.compile(r"\bdoi\b", re.IGNORECASE)

# Author-year: a capitalized surname (optionally with diacritics/apostrophe/hyphen)
# followed, within a short span, by a parenthesized 4-digit year (1800–2099). This
# is the signature of an academic citation ("Card (2001)", "Acemoglu et al. (2012)").
_AUTHOR_YEAR = re.compile(
    r"[A-ZÀ-Þ][A-Za-zÀ-ÿ'’\-]+.{0,80}?\(\s*(?:1[89]\d{2}|20\d{2})\s*\)"
)

# A journal / volume shape ("et al.", "Vol. 12", "pp. 1369", "no. 5").
_JOURNAL_SHAPE = re.compile(
    r"\bet\s+al\.|\bvol\.?\s*\d+|\bpp\.\s*\d+|\bno\.\s*\d+", re.IGNORECASE
)


def _has_scholarly_cue(text: str) -> bool:
    """True when the text shows an academic citation cue (GF-D2)."""
    if _DOI.search(text) or _DOI_PREFIX.search(text):
        return True
    if _AUTHOR_YEAR.search(text):
        return True
    return bool(_JOURNAL_SHAPE.search(text))


def classify_footnote(text: str) -> str:
    """Classify a footnote as ``"government"``, ``"academic"``, or ``"prose"``.

    Government is checked first: a kamerstuk identifier is an unambiguous
    government-document cue, so a footnote carrying one is a government reference
    even when wrapped in explanatory prose.

    Args:
        text: The footnote text.

    Returns:
        ``"government"`` (a kamerstuk id), ``"academic"`` (a scholarly cue), or
        ``"prose"`` (neither).
    """
    if parse_kamerstuk_identifier(text) is not None:
        return "government"
    if _has_scholarly_cue(text):
        return "academic"
    return "prose"


__all__ = ["classify_footnote"]
