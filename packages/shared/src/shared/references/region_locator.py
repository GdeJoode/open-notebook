"""Reference-region location (Track V.1) — the producer's first stage.

Track V turns a source's persisted document structure into a
``List[ParsedReference]`` (the Track V → U.3 boundary,
:class:`shared.retrieval.cites_matching.ParsedReference`). Before a bibliography
can be segmented and parsed it must be *located* inside the document. This module
is that locator: given a source's chunks (plus its optional ``full_text``), it
returns the text span of the reference / bibliography region.

Why chunks, not raw PDF / docling JSON
======================================
The ``docling_document_json`` is transient and never persisted; the durable
document structure is the persisted **chunks**. Each chunk carries a
``section_path`` breadcrumb, an ``element_type`` (Docling emits only generic
``heading`` / ``section_header`` / ``list_item`` / ``text`` — never a
``reference`` / ``bibliography`` type; measured on staging in Track U.1), page
info and text. So the region cannot be found by a dedicated element type — it is
found by matching a *heading* against the reference-section vocabulary.

Structure-first, full_text fallback
===================================
1. **Structure** — a chunk whose ``section_path`` tail (or whose own heading
   text) matches a reference-section header (English + Dutch: References,
   Referenties, Bibliography, Literatuur, Bronnen, Works Cited, …). This is the
   robust path: it uses the document's own hierarchy.
2. **Full-text fallback** — when the chunk structure carries no such header (a
   flat document), a multiline regex locates the bibliography block in
   ``full_text`` from a heading-only line to the end of the text.
3. **None** — neither path finds a region → an empty :class:`LocatedRegion` with
   ``located_via == "none"`` (never a crash).

Everything here is pure and deterministic (no I/O, no DB, no network), so the
region heuristics are unit-testable against small committed fixtures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Reference-section vocabulary (the heading keywords we recognize)
# --------------------------------------------------------------------------

#: Heading labels that mark the start of a reference / bibliography section.
#: English + Dutch (the live corpus is economics papers + Regio Deal
#: convenanten). Matched case-insensitively against a normalized heading; a
#: heading "matches" when, after stripping punctuation/numbering, it equals or
#: starts with one of these.
_REFERENCE_HEADINGS: Tuple[str, ...] = (
    "references",
    "reference list",
    "referenties",
    "bibliography",
    "bibliografie",
    "literatuur",
    "literatuurlijst",
    "geraadpleegde literatuur",
    "bronnen",
    "bronvermelding",
    "works cited",
    "cited references",
    "literature",
    "literature cited",
)

#: ``element_type`` values (normalized) that denote a heading-like chunk. Docling
#: emits ``heading`` / ``section_header`` / ``title``; we also accept a few
#: spelling variants so the locator is robust to element-type drift.
_HEADING_ELEMENT_TYPES: Tuple[str, ...] = (
    "heading",
    "section_header",
    "section-header",
    "sectionheader",
    "title",
    "header",
)


def _normalize_heading(text: str) -> str:
    """Lowercase, drop leading numbering and surrounding punctuation.

    "  7. References " → "references", "Bibliography:" → "bibliography". Keeps
    inner spaces so "works cited" stays intact. Used to compare a heading against
    :data:`_REFERENCE_HEADINGS`.
    """
    if not text:
        return ""
    t = text.strip().lower()
    # Strip a leading section number like "7", "7.", "7.1", "II." plus separators.
    t = re.sub(r"^[\divxlc]+[.)]?\s+", "", t)
    # Strip surrounding non-word punctuation.
    t = t.strip(" \t\r\n.:;-–—*#")
    return re.sub(r"\s+", " ", t).strip()


def _is_reference_heading(text: str) -> bool:
    """True when a heading label denotes a reference / bibliography section.

    A heading matches when its normalized form equals a known label or begins
    with one (so "References and notes" or "Bibliography (selected)" still
    match), but NOT when the label is merely a substring mid-heading (so a body
    heading like "Data sources" does not false-match "sources").
    """
    norm = _normalize_heading(text)
    if not norm:
        return False
    for label in _REFERENCE_HEADINGS:
        if norm == label or norm.startswith(label + " ") or norm.startswith(label + ":"):
            return True
    return False


# --------------------------------------------------------------------------
# DB-free chunk projection — the input shape V operates on
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceChunk:
    """A minimal, DB-free projection of a persisted ``chunk`` row.

    Mirrors the fields Track V actually needs from
    :class:`shared.models.source.Chunk` (``text`` / ``section_path`` /
    ``element_type`` / ``order`` / page), kept as a plain dataclass so the
    reference pipeline is unit-testable without a database — the same DB-free
    discipline as :class:`shared.references.work_resolver.ResolvedWork`.

    Attributes:
        text: The chunk's text content.
        section_path: The heading breadcrumb, e.g. ``("References",)`` for an
            entry that lives under a References heading.
        element_type: The Docling element type (``heading`` / ``list_item`` /
            ``text`` …); heading-like types drive structure detection.
        order: The chunk's 0-indexed position in the document (defines reading
            order for region assembly).
        page: The physical page number, when known (carried for provenance).
    """

    text: str
    section_path: Tuple[str, ...] = ()
    element_type: str = "text"
    order: int = 0
    page: Optional[int] = None

    def __post_init__(self) -> None:
        # Accept a list ``section_path`` (the natural shape off a Chunk) and
        # freeze it to a tuple so the dataclass stays hashable.
        if not isinstance(self.section_path, tuple):
            object.__setattr__(self, "section_path", tuple(self.section_path or ()))


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LocatedRegion:
    """The located reference region (text + provenance + span).

    Attributes:
        text: The region's text (the bibliography body — the heading line itself
            is excluded so the segmenter sees only entries). Empty when no region
            was found.
        located_via: How the region was found — ``"structure"`` (a heading in the
            chunk hierarchy), ``"full_text"`` (the regex fallback), or ``"none"``
            (no region; ``text`` is empty).
        span: ``(start, end)`` character offsets of ``text`` within the source's
            ``full_text`` when they can be established, else ``None`` (e.g. a
            structure hit whose assembled text is not a verbatim ``full_text``
            substring, or when no ``full_text`` was supplied).
    """

    text: str = ""
    located_via: str = "none"
    span: Optional[Tuple[int, int]] = None

    @property
    def found(self) -> bool:
        """True when a non-empty region was located."""
        return self.located_via != "none" and bool(self.text.strip())


_EMPTY_REGION = LocatedRegion(text="", located_via="none", span=None)


# --------------------------------------------------------------------------
# Structure-first locator
# --------------------------------------------------------------------------


def _heading_of_chunk(chunk: ReferenceChunk) -> Optional[str]:
    """Return the chunk's own heading label if it IS a heading chunk, else None.

    A chunk is heading-like when its ``element_type`` is a heading type; its own
    ``text`` is then the candidate label.
    """
    if chunk.element_type and chunk.element_type.strip().lower() in _HEADING_ELEMENT_TYPES:
        return chunk.text
    return None


def _section_tail_is_reference(chunk: ReferenceChunk) -> bool:
    """True when the chunk's ``section_path`` tail is a reference heading.

    An entry chunk under a References heading carries ``section_path`` ending in
    that heading, so its membership in the region is read directly off the path
    (independent of element_type).
    """
    if not chunk.section_path:
        return False
    return _is_reference_heading(chunk.section_path[-1])


def _locate_via_structure(chunks: Sequence[ReferenceChunk]) -> Optional[List[ReferenceChunk]]:
    """Collect the chunks belonging to the reference section, or ``None``.

    Two complementary signals (a document may carry either):

    * **section_path membership** — every chunk whose ``section_path`` tail is a
      reference heading is a region member. This is the primary signal: it needs
      no assumption about sibling ordering.
    * **heading + following body** — if the reference heading appears only as its
      own heading chunk (its following entries carry no matching
      ``section_path``), the region is the run of chunks AFTER that heading up to
      the next heading chunk.

    Returns the ordered member chunks (heading excluded), or ``None`` if no
    reference heading exists in the structure at all.
    """
    ordered = sorted(chunks, key=lambda c: c.order)

    # Signal 1: section_path membership.
    by_path = [c for c in ordered if _section_tail_is_reference(c)]
    if by_path:
        # Exclude a chunk that is itself the heading (its text is the label).
        return [c for c in by_path if not _is_reference_heading(c.text)]

    # Signal 2: a standalone heading chunk followed by body until the next heading.
    heading_index: Optional[int] = None
    for idx, chunk in enumerate(ordered):
        label = _heading_of_chunk(chunk)
        if label is not None and _is_reference_heading(label):
            heading_index = idx
            break
    if heading_index is None:
        return None

    body: List[ReferenceChunk] = []
    for chunk in ordered[heading_index + 1 :]:
        if _heading_of_chunk(chunk) is not None:
            break  # next section starts — stop the region
        body.append(chunk)
    return body


# --------------------------------------------------------------------------
# Full-text fallback locator
# --------------------------------------------------------------------------

# A heading-only line: the reference label alone on its own line (optionally
# numbered / punctuated), anchored multiline. Captures nothing but marks where
# the bibliography body begins.
_FULLTEXT_HEADING = re.compile(
    r"(?im)^[ \t]*(?:[\divxlc]+[.)]?[ \t]+)?"
    r"(?:references|reference list|referenties|bibliography|bibliografie|"
    r"literatuurlijst|literatuur|geraadpleegde\ literatuur|bronvermelding|bronnen|"
    r"works\ cited|cited\ references|literature\ cited|literature)"
    r"[ \t]*:?[ \t]*$"
)

# A following section heading that should terminate the bibliography block (so an
# appendix after the references is not swallowed). These tail-section headers may
# carry a short trailing label ("Appendix A. Data sources"), so they match the
# keyword at the start of a heading line and allow the rest of that line.
_FULLTEXT_TERMINATOR = re.compile(
    r"(?im)^[ \t]*(?:[\divxlc]+[.)]?[ \t]+)?"
    r"(?:appendix|appendices|annex(?:es)?|bijlage(?:n)?|acknowledg(?:e)?ments?|"
    r"dankwoord|about\ the\ authors?|author\ biographies?|supplementary(?:\ material)?)"
    r"\b[^\n]*$"
)


def _locate_via_full_text(full_text: str) -> Optional[Tuple[str, Tuple[int, int]]]:
    """Find the bibliography block in ``full_text`` → ``(text, span)`` or ``None``.

    Scans for the LAST heading-only reference line (the real bibliography is at
    the document tail; an in-body mention of "references" as running prose is not
    a heading-only line and so is ignored), then takes from just after that
    heading to either the next terminator heading (appendix / acknowledgements /
    …) or the end of the text.
    """
    if not full_text:
        return None

    matches = list(_FULLTEXT_HEADING.finditer(full_text))
    if not matches:
        return None
    heading = matches[-1]  # the bibliography sits at the tail of the document
    body_start = heading.end()
    # Skip the newline(s) immediately after the heading line.
    while body_start < len(full_text) and full_text[body_start] in "\r\n":
        body_start += 1

    terminator = _FULLTEXT_TERMINATOR.search(full_text, body_start)
    body_end = terminator.start() if terminator else len(full_text)

    text = full_text[body_start:body_end].strip()
    if not text:
        return None
    # Recompute the span against the stripped text so ``full_text[start:end]``
    # round-trips to ``text``.
    start = full_text.index(text, body_start)
    return text, (start, start + len(text))


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def _span_in_full_text(text: str, full_text: str) -> Optional[Tuple[int, int]]:
    """Return the char span of ``text`` in ``full_text`` if verbatim, else ``None``."""
    if not full_text or not text:
        return None
    idx = full_text.find(text)
    if idx < 0:
        return None
    return idx, idx + len(text)


def locate_reference_region(
    chunks: Sequence[ReferenceChunk],
    full_text: str = "",
) -> LocatedRegion:
    """Locate the reference / bibliography region for a source (pure).

    Structure-first: if the chunk hierarchy carries a reference-section heading,
    the region is assembled from the chunks under / after it. Otherwise, if a
    ``full_text`` was supplied, a heading-only regex locates the bibliography
    block. If neither finds a region, an empty ``located_via == "none"`` result
    is returned (never a crash).

    Args:
        chunks: The source's persisted chunks (DB-free projections). Order is
            taken from each chunk's ``order`` field.
        full_text: The source's concatenated text, used only for the fallback and
            to compute the returned ``span``.

    Returns:
        A :class:`LocatedRegion`. ``located_via`` is ``"structure"`` /
        ``"full_text"`` / ``"none"``; ``text`` is the region body (heading
        excluded); ``span`` is the char offset into ``full_text`` when known.
    """
    body = _locate_via_structure(chunks) if chunks else None
    if body:
        text = "\n".join(c.text.strip() for c in body if c.text and c.text.strip())
        if text.strip():
            return LocatedRegion(
                text=text,
                located_via="structure",
                span=_span_in_full_text(text, full_text),
            )

    fallback = _locate_via_full_text(full_text)
    if fallback is not None:
        text, span = fallback
        return LocatedRegion(text=text, located_via="full_text", span=span)

    return _EMPTY_REGION


__all__ = [
    "ReferenceChunk",
    "LocatedRegion",
    "locate_reference_region",
]
