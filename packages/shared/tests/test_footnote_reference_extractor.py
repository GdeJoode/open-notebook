"""Tests for the FootnoteReferenceExtractor (Track V, GF.4).

Drives GF.1→GF.4 offline: a MOCKED GROBID service returns the recorded NPVR
footnotes (GF.1), the extractor classifies + routes each (GF.2/GF.3), and a
STUBBED ``OverheidResolver`` stands in for KOOP (GF-D3: resolution is optional).
A live end-to-end (NPVR PDF → real GROBID) test is gated.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from shared.references.footnote_reference_extractor import (
    FootnoteReferenceExtractor,
)
from shared.references.grobid_reference_service import (
    GrobidReferenceService,
    parse_footnotes_tei,
)

pytestmark = pytest.mark.asyncio

_FIXTURES = Path(__file__).parent / "fixtures" / "references"
_NPVR_FULLTEXT_TEI = _FIXTURES / "npvr_fulltext_response.xml"
_NPVR_PDF = _FIXTURES / "live_corpus" / "npvr-voortgangsbrief-2026-01.pdf"


def _npvr_footnotes() -> list[str]:
    return parse_footnotes_tei(_NPVR_FULLTEXT_TEI.read_text(encoding="utf-8"))


def _grobid_stub(*, footnotes: list[str], citation=None) -> AsyncMock:
    grobid = AsyncMock(spec=GrobidReferenceService)
    grobid.fulltext_footnotes = AsyncMock(return_value=list(footnotes))
    grobid.parse_citation = AsyncMock(return_value=citation)
    return grobid


# --------------------------------------------------------------------------
# NPVR set → the government footnotes become references
# --------------------------------------------------------------------------


async def test_npvr_footnotes_yield_the_government_references() -> None:
    grobid = _grobid_stub(footnotes=_npvr_footnotes())
    extractor = FootnoteReferenceExtractor(grobid)  # no resolver (GF-D3)

    refs = await extractor.extract(b"%PDF fake")

    raws = [r.raw_text for r in refs]
    assert "Kamerstuk 31305-489" in raws
    assert "Motie 36410-111" in raws
    # The 3 government footnotes (incl. the prose-embedded Kamerstuk 29697-158); the
    # 4 prose notes are dropped, and no academic call was made (none had a cue).
    assert len(refs) == 3
    grobid.parse_citation.assert_not_awaited()


async def test_government_reference_venue_reflects_kind() -> None:
    grobid = _grobid_stub(footnotes=["Kamerstuk 31305-489", "Motie 36410-111"])
    extractor = FootnoteReferenceExtractor(grobid)

    refs = await extractor.extract(b"%PDF fake")

    by_raw = {r.raw_text: r for r in refs}
    assert by_raw["Kamerstuk 31305-489"].venue == "Kamerstuk"
    assert by_raw["Motie 36410-111"].venue == "Motie"


async def test_prose_only_footnotes_yield_no_references() -> None:
    grobid = _grobid_stub(
        footnotes=[
            "Een overzicht van deze verdeling vindt u in de bijlage.",
            "Samenwerkagenda met Nedersaksen en de Grenslandagenda.",
        ]
    )
    extractor = FootnoteReferenceExtractor(grobid)
    assert await extractor.extract(b"%PDF fake") == []


async def test_no_footnotes_yields_no_references() -> None:
    grobid = _grobid_stub(footnotes=[])
    extractor = FootnoteReferenceExtractor(grobid)
    assert await extractor.extract(b"%PDF fake") == []


async def test_wrapped_footnote_whitespace_is_normalized() -> None:
    # A government footnote docling wrapped across lines with trailing/leading
    # whitespace. Normalization stores a clean, single-spaced raw_text and keeps the
    # identifier intact across the (space-boundary) line break.
    grobid = _grobid_stub(footnotes=["  Kamerstuk\n   31305-489  \n"])
    extractor = FootnoteReferenceExtractor(grobid)

    refs = await extractor.extract(b"%PDF fake")

    assert len(refs) == 1
    assert refs[0].raw_text == "Kamerstuk 31305-489"
    assert refs[0].venue == "Kamerstuk"  # classified + routed as government


async def test_whitespace_only_footnotes_are_skipped() -> None:
    grobid = _grobid_stub(footnotes=["   \n\t ", ""])
    extractor = FootnoteReferenceExtractor(grobid)
    assert await extractor.extract(b"%PDF fake") == []


# --------------------------------------------------------------------------
# academic footnote → GROBID processCitation
# --------------------------------------------------------------------------


async def test_academic_footnote_is_routed_to_processCitation() -> None:
    from shared.retrieval.cites_matching import ParsedReference

    citation = ParsedReference(
        raw_text="Card (2001)", title="Immigrant inflows", authors=("Card",)
    )
    grobid = _grobid_stub(
        footnotes=["Card, D. (2001). Immigrant inflows. American Economic Review."],
        citation=citation,
    )
    extractor = FootnoteReferenceExtractor(grobid)

    refs = await extractor.extract(b"%PDF fake")

    grobid.parse_citation.assert_awaited_once()
    assert len(refs) == 1
    assert refs[0].title == "Immigrant inflows"


async def test_academic_footnote_unparseable_by_grobid_is_dropped() -> None:
    grobid = _grobid_stub(
        footnotes=["Card, D. (2001). Immigrant inflows."], citation=None
    )
    extractor = FootnoteReferenceExtractor(grobid)
    assert await extractor.extract(b"%PDF fake") == []


# --------------------------------------------------------------------------
# best-effort resolution (GF-D3) — enrich when KOOP resolves, else keep recorded
# --------------------------------------------------------------------------


async def test_resolver_enriches_government_reference() -> None:
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(
        return_value=SimpleNamespace(
            title="Voortgang NPVR", year=2026, venue="Kamerstuk"
        )
    )
    grobid = _grobid_stub(footnotes=["Kamerstuk 31305-489"])
    extractor = FootnoteReferenceExtractor(grobid, overheid_resolver=resolver)

    refs = await extractor.extract(b"%PDF fake")

    resolver.resolve.assert_awaited_once()
    assert len(refs) == 1
    # Verbatim footnote preserved as raw_text; resolved metadata overlaid.
    assert refs[0].raw_text == "Kamerstuk 31305-489"
    assert refs[0].title == "Voortgang NPVR"
    assert refs[0].year == 2026


async def test_unresolved_government_reference_stays_recorded() -> None:
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(return_value=None)  # KOOP has no record
    grobid = _grobid_stub(footnotes=["Kamerstuk 31305-489"])
    extractor = FootnoteReferenceExtractor(grobid, overheid_resolver=resolver)

    refs = await extractor.extract(b"%PDF fake")

    assert len(refs) == 1
    assert refs[0].raw_text == "Kamerstuk 31305-489"
    assert refs[0].title == ""  # recorded but not enriched


async def test_resolver_failure_is_soft() -> None:
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(side_effect=RuntimeError("KOOP down"))
    grobid = _grobid_stub(footnotes=["Kamerstuk 31305-489"])
    extractor = FootnoteReferenceExtractor(grobid, overheid_resolver=resolver)

    refs = await extractor.extract(b"%PDF fake")  # must not raise

    assert len(refs) == 1
    assert refs[0].raw_text == "Kamerstuk 31305-489"


# --------------------------------------------------------------------------
# LIVE (GF.4) — NPVR PDF → real GROBID → the government references appear
# --------------------------------------------------------------------------

_GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")


def _grobid_alive() -> bool:
    try:
        return httpx.get(f"{_GROBID_URL}/api/isalive", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(
    not _grobid_alive() or not _NPVR_PDF.exists(),
    reason=f"GROBID not reachable at {_GROBID_URL}/api/isalive (or NPVR PDF absent)",
)
async def test_live_npvr_footnote_references() -> None:
    service = GrobidReferenceService(base_url=_GROBID_URL)
    extractor = FootnoteReferenceExtractor(service)  # no resolver (offline enrich)
    try:
        refs = await extractor.extract(_NPVR_PDF)
    finally:
        await service.aclose()
    raws = " || ".join(r.raw_text for r in refs)
    assert "Kamerstuk 31305-489" in raws
    assert "Motie 36410-111" in raws
