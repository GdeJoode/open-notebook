"""Footnote reference extractor + router (Track V, GF.4).

Wires the GF.1–GF.3 halves into one producer: a PDF's footnotes → classified →
routed → a ``List[ParsedReference]``. This is the policy-document counterpart to
the G.* GROBID bibliography producer; both emit the same ``ParsedReference``
contract (the Track V → U.3 boundary), so the extraction service can merge them.

Routing (per footnote)
======================
* **government** → a :class:`ParsedReference` is always RECORDED (raw_text = the
  footnote text, venue = ``Kamerstuk``/``Motie``, the identifier living in
  raw_text for the resolver). When an ``OverheidResolver`` is supplied it is asked,
  best-effort, to enrich the reference with the resolved title/year/venue —
  resolution is OPTIONAL and never required (GF-D3: an unresolved government ref is
  just an external ref with no ``cites`` edge; resolution only attaches metadata).
* **academic** → the footnote text is sent to GROBID ``processCitation``; the
  parsed reference (if any) is kept.
* **prose** → dropped.

Fail-soft throughout: footnote extraction, resolution, and citation parsing each
degrade to "no reference" on error, mirroring the guarded ingest hook.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from loguru import logger

from shared.references.footnote_classifier import classify_footnote
from shared.references.grobid_reference_service import GrobidReferenceService
from shared.references.kamerstuk import parse_kamerstuk_identifier
from shared.references.overheid_resolver import OverheidResolver
from shared.retrieval.cites_matching import ParsedReference


def _normalize_footnote_text(text: str) -> str:
    """Collapse whitespace to single spaces and strip the ends.

    Footnote text arrives from docling/GROBID with the stray newlines and runs of
    spaces a note picks up when it wraps across lines in the source PDF. Normalizing
    here does double duty: a citation split over a line break — a Kamerstuk
    identifier, a DOI — is only matchable once the break is gone, and the value
    stored as ``ParsedReference.raw_text`` stays clean for dedup and display. Empty
    or whitespace-only input collapses to ``""`` (skipped by the caller).
    """
    return " ".join((text or "").split())


class FootnoteReferenceExtractor:
    """Extract references from a PDF's footnotes, routing each by kind (GF.4)."""

    def __init__(
        self,
        grobid_service: GrobidReferenceService,
        *,
        overheid_resolver: Optional[OverheidResolver] = None,
    ) -> None:
        """Args:

        grobid_service: The GROBID wrapper (GF.1 ``fulltext_footnotes`` for the
            footnotes, GF.4 ``parse_citation`` for the academic path).
        overheid_resolver: Optional KOOP SRU resolver used, best-effort, to enrich
            government references. When ``None`` government refs are recorded with
            their identifier but not enriched (GF-D3).
        """
        self._grobid = grobid_service
        self._resolver = overheid_resolver

    async def extract(self, pdf: bytes | str | Path) -> List[ParsedReference]:
        """Extract + route a PDF's footnotes into references; ``[]`` on no input.

        Best-effort: an unreadable/footnote-less PDF yields ``[]`` (GF.1 already
        fails soft), and each per-footnote route degrades to "no reference" on
        error, never raising.

        Args:
            pdf: The PDF as raw bytes, or a path to read them from.

        Returns:
            The government + academic references from the footnotes, in document
            order; prose footnotes contribute nothing.
        """
        footnotes = await self._grobid.fulltext_footnotes(pdf)
        references: List[ParsedReference] = []
        for raw in footnotes:
            text = _normalize_footnote_text(raw)
            if not text:
                continue
            kind = classify_footnote(text)
            if kind == "government":
                references.append(await self._government_reference(text))
            elif kind == "academic":
                ref = await self._grobid.parse_citation(text)
                if ref is not None:
                    references.append(ref)
            # prose → dropped
        return references

    async def _government_reference(self, text: str) -> ParsedReference:
        """Record a government footnote as a reference, best-effort enriched.

        The reference is ALWAYS recorded from the footnote text (GF-D3); resolution
        only adds metadata. ``venue`` reflects the kind (``Motie``/``Kamerstuk``),
        and the identifier is carried verbatim in ``raw_text`` (which the resolver
        reads for the dossiernummer).
        """
        identifier = parse_kamerstuk_identifier(text)
        venue = "Motie" if identifier and identifier.kind == "motie" else "Kamerstuk"
        reference = ParsedReference(raw_text=text, venue=venue)

        if self._resolver is None:
            return reference

        try:
            work = await self._resolver.resolve(reference)
        except Exception as exc:  # noqa: BLE001 — resolution must fail soft (GF-D3)
            logger.warning(
                "GF.4: government resolution raised (kept unresolved) for {r}: {exc}",
                r=text[:80],
                exc=exc,
            )
            work = None

        if work is None:
            return reference

        # Enrich in place: keep the verbatim footnote as raw_text (the reviewer's
        # citation), overlay the resolved metadata.
        return ParsedReference(
            raw_text=text,
            title=work.title or "",
            year=work.year,
            venue=work.venue or venue,
        )


__all__ = ["FootnoteReferenceExtractor"]
