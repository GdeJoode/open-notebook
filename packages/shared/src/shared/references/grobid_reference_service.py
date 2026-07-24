"""GROBID-backed reference extraction (Track V, phase G.2).

GROBID's CRF models turn a source PDF's bibliography into structured TEI
``<biblStruct>`` records. This module is the boundary adapter: it maps that TEI
onto the :class:`~shared.retrieval.cites_matching.ParsedReference` contract that
Track U.3 consumes, and wraps the HTTP call in a fail-soft service.

Why GROBID replaces the V.2/V.3 heuristic
=========================================
The earlier region-segmenter + entry-parser had to *understand* how a given
bibliography formats its entries (bracketing conventions, name/date ordering,
footnote numbering, …) to slice it into entries and pull fields out. That
formatting knowledge is exactly what GROBID's trained models already own. So this
module contains **no formatting-specific logic at all**: the SAME code path parses
both recorded fixtures — an economics bibliography and a political-science one —
which differ only in their input, never in how we read them. GROBID owns the
formatting; we own the field mapping.

The two halves
==============
* :func:`parse_grobid_tei` is PURE — TEI string in, ``List[ParsedReference]`` out.
  It is offline-testable against recorded ``/api/processReferences`` responses and
  carries the whole field mapping.
* :class:`GrobidReferenceService` is the thin HTTP wrapper. It is **best-effort**:
  GROBID being unreachable, slow, or returning garbage yields ``[]`` (logged), never
  an exception — mirroring the guarded post-ingest hook, so a down GROBID can never
  break ingest.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree as ET

import httpx
from loguru import logger

from datetime import datetime, timezone

from shared.retrieval.cites_matching import ParsedReference, normalize_doi

#: A publication year past this is implausible (a future date, i.e. not a citation).
_MAX_PLAUSIBLE_YEAR = datetime.now(timezone.utc).year + 1

# TEI namespace every element in a GROBID response lives under.
_TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": _TEI_NS}

# A GROBID ``@when`` may be a bare year ("2004") or a fuller stamp ("1986-03");
# either way the publication year is the leading 4-digit run.
_YEAR_RE = re.compile(r"(\d{4})")

# Default location of the GROBID service (the compose service added in G.1).
_GROBID_URL_ENV = "GROBID_URL"
_DEFAULT_GROBID_URL = "http://localhost:8070"


def _text(el: Optional[ET.Element]) -> str:
    """Whitespace-collapsed text of an element (``""`` when missing/empty)."""
    if el is None or el.text is None:
        return ""
    return " ".join(el.text.split())


def _text_key(text: str) -> str:
    """Normalized title key for dedup: lowercased, whitespace-collapsed."""
    return " ".join((text or "").lower().split())


def _first_year(raw: str) -> Optional[int]:
    """First 4-digit year in a string, or ``None``."""
    m = _YEAR_RE.search(raw or "")
    return int(m.group(1)) if m else None


def _extract_title(bibl: ET.Element) -> str:
    """Title of a ``<biblStruct>``: article title first, else book (monograph).

    An article's title lives at ``analytic/title[@level="a"]``; a book has no
    analytic level and its title is ``monogr/title[@level="m"]``. Journal titles
    (``level="j"``) are the venue, not the work title, so they are never used here.
    """
    article = bibl.find('.//tei:analytic/tei:title[@level="a"]', _NS)
    if _text(article):
        return _text(article)
    book = bibl.find('.//tei:monogr/tei:title[@level="m"]', _NS)
    return _text(book)


def _extract_authors(bibl: ET.Element) -> tuple[str, ...]:
    """Surnames of every ``persName`` in the record (analytic and monograph).

    Track U.3 matches on surnames, so the surname is the only author field we
    carry. Both the analytic authors (article) and monograph authors (book /
    editors) contribute, in document order.
    """
    surnames: List[str] = []
    for surname in bibl.findall(".//tei:author/tei:persName/tei:surname", _NS):
        name = _text(surname)
        if name:
            surnames.append(name)
    return tuple(surnames)


def _extract_doi(bibl: ET.Element) -> Optional[str]:
    """Normalized DOI (the gold matching signal), or ``None`` when absent."""
    idno = bibl.find('.//tei:idno[@type="DOI"]', _NS)
    doi = normalize_doi(_text(idno))
    return doi or None


def _extract_year(bibl: ET.Element) -> Optional[int]:
    """Publication year from ``monogr/imprint/date`` — ``@when`` first, else text."""
    date = bibl.find(".//tei:monogr/tei:imprint/tei:date", _NS)
    if date is None:
        return None
    return _first_year(date.get("when", "")) or _first_year(_text(date))


def _extract_venue(bibl: ET.Element) -> Optional[str]:
    """Venue: journal title (``level="j"``) else the monograph title (``level="m"``)."""
    journal = bibl.find('.//tei:monogr/tei:title[@level="j"]', _NS)
    if _text(journal):
        return _text(journal)
    book = bibl.find('.//tei:monogr/tei:title[@level="m"]', _NS)
    return _text(book) or None


def _synthesize_raw_text(
    authors: tuple[str, ...],
    year: Optional[int],
    title: str,
    venue: Optional[str],
    doi: Optional[str],
) -> str:
    """Build a human-readable citation string from the parsed fields.

    GROBID's ``<biblStruct>`` has no verbatim reference string, but
    :class:`ParsedReference` requires a non-empty ``raw_text`` (it is what a
    created ``cites`` edge records so a reviewer can read what matched). So we
    synthesize one — ``"Surnames (year). Title. Venue. doi:…"`` — from whatever
    fields are present, skipping the empty ones.
    """
    parts: List[str] = []
    surnames = ", ".join(authors)
    if surnames and year is not None:
        parts.append(f"{surnames} ({year})")
    elif surnames:
        parts.append(surnames)
    elif year is not None:
        parts.append(f"({year})")
    if title:
        parts.append(title)
    if venue:
        parts.append(venue)
    text = ". ".join(parts)
    if text:
        text += "."
    if doi:
        text = f"{text} doi:{doi}".strip()
    return text.strip()


def parse_footnotes_tei(tei_xml: str) -> List[str]:
    """Pull the text of every ``<note place="foot">`` from a fulltext TEI (GF.1).

    PURE and offline-testable against a recorded ``/api/processFulltextDocument``
    response. Policy documents (Kamerbrieven, convenanten) carry their
    cross-references in footnotes, not a scholarly bibliography, so the footnote
    texts are the raw material the GF.* path classifies and routes.

    Each note's full text (including any nested ``<p>``) is flattened and
    whitespace-collapsed; empty notes are skipped. Order is document order.

    Args:
        tei_xml: The raw TEI XML body returned by ``processFulltextDocument``.

    Returns:
        One string per non-empty footnote, in document order (``[]`` on empty or
        unparseable input).
    """
    if not tei_xml or not tei_xml.strip():
        return []
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as exc:
        logger.warning("GROBID fulltext TEI parse failed: {}", exc)
        return []

    footnotes: List[str] = []
    for note in root.iter(f"{{{_TEI_NS}}}note"):
        if note.get("place") != "foot":
            continue
        text = " ".join("".join(note.itertext()).split())
        if text:
            footnotes.append(text)
    return footnotes


def parse_grobid_tei(tei_xml: str, *, guard: bool = True) -> List[ParsedReference]:
    """Map a GROBID TEI document to parsed references.

    PURE and offline-testable. Each ``<biblStruct>`` becomes one
    :class:`ParsedReference`, always **deduplicated** (repeats collapse on DOI, else
    normalized title + year).

    With ``guard=True`` (the ``/api/processReferences`` bibliography path) a
    **precision guard** also applies: a real citation carries a DOI, or a year plus
    something to identify it (a title or authors). This drops GROBID's mis-parses of
    non-bibliography structure in policy documents — signatory blocks
    ("… hierna te noemen"), party lists, tables — into year-less "references".
    ``guard=False`` (a single ``/api/processCitation`` result, already classified as
    an academic footnote) keeps any structured parse (a title or a DOI), since a
    lone scholarly citation may legitimately lack a parseable year.

    Args:
        tei_xml: The raw TEI XML body returned by GROBID.
        guard: Apply the precision guard (bibliography path). Default ``True``.

    Returns:
        One :class:`ParsedReference` per kept ``<biblStruct>``, in document order.
    """
    if not tei_xml or not tei_xml.strip():
        return []
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as exc:
        logger.warning("GROBID TEI parse failed: {}", exc)
        return []

    references: List[ParsedReference] = []
    seen: set = set()
    for bibl in root.iter(f"{{{_TEI_NS}}}biblStruct"):
        title = _extract_title(bibl)
        doi = _extract_doi(bibl)
        year = _extract_year(bibl)
        authors = _extract_authors(bibl)
        # A publication year cannot be in the future; a future "year" is a policy-doc
        # artifact (e.g. a Regio Deal effectuation clause "… treedt in werking … 2028").
        year_ok = year is not None and year <= _MAX_PLAUSIBLE_YEAR
        if guard:
            # Bibliography: a DOI, or a plausible year + an identifier (title/authors).
            keep = doi is not None or (year_ok and (bool(title) or bool(authors)))
        else:
            # Single already-classified citation: keep any structured parse.
            keep = bool(title) or doi is not None
        if not keep:
            continue
        # Dedup: collapse repeats (a work cited twice, or a mis-parse duplicated).
        # Key on the DOI when present, else the normalized title + year.
        key = ("doi", doi) if doi else ("tv", _text_key(title), year)
        if key in seen:
            continue
        seen.add(key)
        venue = _extract_venue(bibl)
        raw_text = _synthesize_raw_text(authors, year, title, venue, doi)
        references.append(
            ParsedReference(
                raw_text=raw_text,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                venue=venue,
            )
        )
    return references


class GrobidReferenceService:
    """Best-effort HTTP wrapper around GROBID's ``/api/processReferences``.

    Turns a source PDF into a ``List[ParsedReference]`` by delegating all
    bibliography parsing to GROBID and mapping the TEI with :func:`parse_grobid_tei`.
    Every failure mode (transport error, non-200, unparseable body) returns ``[]``
    and is logged — extraction is a best-effort enrichment and must never break
    ingest.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout: float = 60.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Args:

        base_url: GROBID base URL. When ``None`` it is read from the ``GROBID_URL``
            env var, defaulting to ``http://localhost:8070``.
        timeout: Per-request timeout in seconds (bounded so a hung GROBID cannot
            stall ingest indefinitely).
        client: Inject an ``httpx.AsyncClient`` (tests pass a mock transport).
        """
        if base_url is None:
            base_url = os.environ.get(_GROBID_URL_ENV) or _DEFAULT_GROBID_URL
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @property
    def base_url(self) -> str:
        return self._base_url

    async def extract_references(self, pdf: bytes | str | Path) -> List[ParsedReference]:
        """Extract references from a PDF via GROBID; ``[]`` on any failure.

        Args:
            pdf: The PDF as raw bytes, or a path (str/Path) to read them from.

        Returns:
            The parsed references, or ``[]`` if GROBID is unreachable, errors, or
            returns nothing usable. NEVER raises.
        """
        try:
            data = pdf if isinstance(pdf, bytes) else Path(pdf).read_bytes()
        except OSError as exc:
            logger.warning("GROBID: cannot read PDF {}: {}", pdf, exc)
            return []

        url = f"{self._base_url}/api/processReferences"
        try:
            response = await self._client.post(
                url,
                files={"input": ("document.pdf", data, "application/pdf")},
                data={"consolidateCitations": "0"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("GROBID processReferences failed ({}): {}", url, exc)
            return []

        return parse_grobid_tei(response.text)

    async def fulltext_footnotes(self, pdf: bytes | str | Path) -> List[str]:
        """Extract the footnote texts from a PDF via GROBID fulltext; ``[]`` on failure.

        POSTs the PDF to ``/api/processFulltextDocument`` and returns the text of
        each ``<note place="foot">`` (:func:`parse_footnotes_tei`). This is the GF.1
        entry point for the policy-document footnote path — the cross-references
        that GROBID's bibliography model (``/api/processReferences``) misses.

        Best-effort like :meth:`extract_references`: an unreadable file, transport
        error, non-200, or unparseable body yields ``[]`` (logged), NEVER raises.

        Args:
            pdf: The PDF as raw bytes, or a path (str/Path) to read them from.

        Returns:
            The footnote texts in document order, or ``[]`` on any failure.
        """
        try:
            data = pdf if isinstance(pdf, bytes) else Path(pdf).read_bytes()
        except OSError as exc:
            logger.warning("GROBID fulltext: cannot read PDF {}: {}", pdf, exc)
            return []

        url = f"{self._base_url}/api/processFulltextDocument"
        try:
            response = await self._client.post(
                url,
                files={"input": ("document.pdf", data, "application/pdf")},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("GROBID processFulltextDocument failed ({}): {}", url, exc)
            return []

        return parse_footnotes_tei(response.text)

    async def parse_citation(self, text: str) -> Optional[ParsedReference]:
        """Parse a single raw citation string via GROBID; ``None`` on failure (GF.4).

        A thin wrapper over ``/api/processCitation`` used by the footnote router for
        the rare scholarly footnote (GF-D2): GROBID returns a single ``<biblStruct>``
        which :func:`parse_grobid_tei` maps like any bibliography entry. Best-effort:
        empty input, transport error, non-200, or no usable parse yields ``None``.

        Args:
            text: The raw citation string (one footnote's text).

        Returns:
            The parsed reference, or ``None`` when GROBID cannot structure it.
        """
        if not text or not text.strip():
            return None

        url = f"{self._base_url}/api/processCitation"
        try:
            response = await self._client.post(
                url,
                data={"citations": text, "consolidateCitations": "0"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("GROBID processCitation failed ({}): {}", url, exc)
            return None

        # A single already-classified academic footnote citation: keep the parse
        # even if GROBID could not extract a year (guard is for the bibliography).
        refs = parse_grobid_tei(response.text, guard=False)
        return refs[0] if refs else None

    async def aclose(self) -> None:
        """Close the underlying client if this service created it."""
        if self._owns_client:
            await self._client.aclose()
