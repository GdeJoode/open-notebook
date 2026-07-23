"""Reference-region segmentation (Track V.2) — split a region into entries.

The V.1 locator returns the bibliography region as one text blob; this module
splits it into the individual reference entries that V.3 then parses. It is the
"reference-FORM classifier" of the Track V plan: it recognizes the *shape* of the
region (a numbered list, an author-year block, blank-line-separated paragraphs)
and segments accordingly.

Cheap-first, LLM-on-the-margin
==============================
Per the design's cheap-first / LLM-on-the-margin cascade
(``design-thematic-classification``), this phase is **deterministic and offline**:
regex/whitespace heuristics only, no LLM call. A seam is exposed
(``ambiguity_resolver``) where an LLM classifier COULD later re-segment a block
the heuristics flag as ambiguous, but the default path never calls it — so the
segmenter stays pure and unit-testable.

Bounded over-segmentation
=========================
The main failure mode of a naive splitter is over-segmentation: a single wrapped
multi-line entry (the second/third line is a continuation, not a new reference)
gets split into several. Every strategy here therefore *merges* continuation
lines into the entry they belong to — a line only starts a new entry when it
carries an explicit start signal (a list marker, or an author-year opener).
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

#: Optional seam: given the region text and the heuristic segmentation, an
#: external (e.g. LLM) classifier MAY return a refined list of entries. The
#: default pipeline passes ``None`` and never invokes it.
AmbiguityResolver = Callable[[str, Sequence[str]], Sequence[str]]


# --------------------------------------------------------------------------
# Start-signal patterns
# --------------------------------------------------------------------------

# A numbered list marker at the start of a line: "[1]", "1.", "1)", "(1)".
_NUMBER_MARKER = re.compile(r"^\s*(?:\[(\d{1,3})\]|\((\d{1,3})\)|(\d{1,3})[.)])\s+")

# An author-year opener: a capitalized surname followed by an initial or a
# co-author/year cue near the line start. Matches the common APA/Chicago forms
#   "Acemoglu, D., & Robinson, J. (2012). ..."
#   "Acemoglu, Daron. 2001. ..."
#   "Van Reenen, J. (2011). ..."   (Dutch/particle surnames)
# It deliberately requires the "Surname," + capital cue so a wrapped continuation
# line (usually lowercase/URL/venue) does NOT read as a new entry.
_AUTHOR_YEAR_OPENER = re.compile(
    r"^\s*(?:van\s+|de\s+|der\s+|von\s+|el\s+)?"
    r"[A-Z][A-Za-z'À-ſ-]+,\s+"
    r"(?:[A-Z]\.|[A-Z][a-z]+)"
)

# A wrapped-continuation line that must never start a new entry: it opens with a
# URL / DOI, or with a lowercase letter (venue/continuation prose). NOT
# case-insensitive — the lowercase-start test is the whole point (an uppercase
# ``Surname,`` line is a genuine new-entry candidate, a lowercase one is not).
_CONTINUATION_CUE = re.compile(r"^\s*(?:https?://|doi:|10\.\d{4,9}/|[a-z])")


def _clean_lines(text: str) -> List[str]:
    """Split into right-stripped lines, preserving blank lines as ``""``."""
    return [ln.rstrip() for ln in text.splitlines()]


def _strip_number_marker(line: str) -> str:
    """Remove a leading ``[1]`` / ``1.`` / ``1)`` marker from an entry's first line."""
    return _NUMBER_MARKER.sub("", line, count=1).strip()


# --------------------------------------------------------------------------
# Strategy detection
# --------------------------------------------------------------------------


def _looks_numbered(lines: Sequence[str]) -> bool:
    """True when ≥2 lines begin with a list marker (a numbered bibliography)."""
    marked = sum(1 for ln in lines if _NUMBER_MARKER.match(ln))
    return marked >= 2


def _has_blank_separators(lines: Sequence[str]) -> bool:
    """True when blank lines separate ≥2 non-empty blocks (paragraph form)."""
    blocks = _blank_line_blocks(lines)
    return len(blocks) >= 2


def _blank_line_blocks(lines: Sequence[str]) -> List[List[str]]:
    """Group lines into blocks separated by one-or-more blank lines."""
    blocks: List[List[str]] = []
    current: List[str] = []
    for ln in lines:
        if ln.strip():
            current.append(ln)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


# --------------------------------------------------------------------------
# Segmentation strategies (each merges continuations → bounded over-segmentation)
# --------------------------------------------------------------------------


def _segment_numbered(lines: Sequence[str]) -> List[str]:
    """Segment a numbered list: a new entry starts on each list marker.

    Lines between markers (continuations of a wrapped entry) are joined into the
    current entry, so a reference spanning several physical lines stays one entry.
    """
    entries: List[str] = []
    current: List[str] = []
    for ln in lines:
        if not ln.strip():
            continue
        if _NUMBER_MARKER.match(ln):
            if current:
                entries.append(" ".join(current).strip())
            current = [_strip_number_marker(ln)]
        elif current:
            current.append(ln.strip())
        else:
            # Leading noise before the first marker (rare) — keep as its own entry.
            current = [ln.strip()]
    if current:
        entries.append(" ".join(current).strip())
    return [e for e in entries if e]


def _segment_blank_separated(lines: Sequence[str]) -> List[str]:
    """Segment paragraph-form entries separated by blank lines.

    Each blank-line-delimited block is ONE entry; its wrapped lines are joined.
    This is the most reliable author-year path when the region preserves the
    inter-entry blank lines (the natural shape of a ``full_text`` bibliography).
    """
    entries: List[str] = []
    for block in _blank_line_blocks(lines):
        joined = " ".join(ln.strip() for ln in block).strip()
        if joined:
            entries.append(joined)
    return entries


def _segment_author_year(lines: Sequence[str]) -> List[str]:
    """Segment author-year entries with NO blank separators (start-detection).

    A new entry begins on a line that matches the author-year opener; every other
    non-empty line is a continuation of the current entry. This keeps a wrapped
    multi-line entry (whose continuation lines are venue/URL/lowercase text)
    intact rather than over-segmenting it.
    """
    entries: List[str] = []
    current: List[str] = []
    for ln in lines:
        if not ln.strip():
            continue
        starts_entry = bool(_AUTHOR_YEAR_OPENER.match(ln)) and not _CONTINUATION_CUE.match(ln)
        if starts_entry and current:
            entries.append(" ".join(current).strip())
            current = [ln.strip()]
        elif starts_entry:
            current = [ln.strip()]
        elif current:
            current.append(ln.strip())
        else:
            current = [ln.strip()]
    if current:
        entries.append(" ".join(current).strip())
    return [e for e in entries if e]


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def segment_region(
    region_text: str,
    *,
    ambiguity_resolver: Optional[AmbiguityResolver] = None,
) -> List[str]:
    """Split a reference region into individual entry strings (pure, offline).

    Strategy selection (cheap-first):

    1. **Numbered** — if ≥2 lines carry a ``[1]`` / ``1.`` marker, split on the
       markers (continuations merged).
    2. **Blank-separated** — else, if blank lines separate ≥2 blocks, each block
       is one entry (wrapped lines merged).
    3. **Author-year** — else, start-detection on author-year openers.

    Args:
        region_text: The located region text (the V.1 output ``LocatedRegion.text``).
        ambiguity_resolver: Optional seam for a later LLM classifier. When
            provided it is called ONCE with ``(region_text, heuristic_entries)``
            and its result (coerced to a clean list) is returned instead. The
            default pipeline passes ``None`` — the segmenter stays deterministic
            and never calls out.

    Returns:
        The list of entry strings (each a single reference), or ``[]`` for an
        empty / whitespace-only region.
    """
    if not region_text or not region_text.strip():
        return []

    lines = _clean_lines(region_text)

    if _looks_numbered(lines):
        entries = _segment_numbered(lines)
    elif _has_blank_separators(lines):
        entries = _segment_blank_separated(lines)
    else:
        entries = _segment_author_year(lines)

    if ambiguity_resolver is not None:
        refined = ambiguity_resolver(region_text, entries)
        return [e.strip() for e in refined if e and e.strip()]

    return entries


__all__ = [
    "AmbiguityResolver",
    "segment_region",
]
