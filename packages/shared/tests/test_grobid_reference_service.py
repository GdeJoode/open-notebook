"""Tests for the GROBID reference service (Track V, phase G.2).

Two layers:

* **Offline (the CI core)** — :func:`parse_grobid_tei` against two RECORDED
  ``/api/processReferences`` responses (an economics and a political-science
  bibliography, differently formatted). The same parser reads both: there is no
  format-specific branch anywhere. These assert exact counts and the field mapping
  on named entries.
* **Best-effort wrapper** — :class:`GrobidReferenceService` with a mocked transport
  proves the fail-soft contract (errors → ``[]``, never raise) without a network.
* **Gated integration** — a real POST to a live GROBID, skipped unless
  ``/api/isalive`` answers 200 (mirrors the ``requires_docker`` skip philosophy).
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from shared.references.grobid_reference_service import (
    GrobidReferenceService,
    parse_grobid_tei,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "references"
_ECON_TEI = _FIXTURES / "grobid_econ_response.xml"
_CENTRIFUGAL_TEI = _FIXTURES / "grobid_centrifugal_response.xml"
_LIVE_PDF = _FIXTURES / "live_corpus" / "economics-without-equilibrium.pdf"

_GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")


def _grobid_alive() -> bool:
    """True iff a live GROBID answers ``/api/isalive`` with 200 (the gate)."""
    try:
        resp = httpx.get(f"{_GROBID_URL}/api/isalive", timeout=2.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


# --------------------------------------------------------------------------
# Offline parser — economics fixture (31 biblStruct)
# --------------------------------------------------------------------------


def test_econ_fixture_parses_all_31_references() -> None:
    refs = parse_grobid_tei(_ECON_TEI.read_text(encoding="utf-8"))
    assert len(refs) == 31


def test_econ_fixture_every_reference_has_raw_text() -> None:
    refs = parse_grobid_tei(_ECON_TEI.read_text(encoding="utf-8"))
    assert all(r.raw_text.strip() for r in refs)


def test_econ_fixture_authors_are_surnames() -> None:
    refs = parse_grobid_tei(_ECON_TEI.read_text(encoding="utf-8"))
    all_surnames = {name for r in refs for name in r.authors}
    # Surnames, not initials or full names.
    assert "Baumol" in all_surnames
    assert "Fritsch" in all_surnames
    assert "Schumpeter" in all_surnames
    # A middle initial like "J" must never leak in as an author.
    assert "J" not in all_surnames


def test_econ_fixture_article_two_multi_authors() -> None:
    # The Brakman/Dixit/Dixon article authors all appear across the bibliography.
    refs = parse_grobid_tei(_ECON_TEI.read_text(encoding="utf-8"))
    all_surnames = {name for r in refs for name in r.authors}
    assert {"Brakman", "Dixit", "Dixon"}.issubset(all_surnames)


def test_econ_fixture_years_are_ints() -> None:
    refs = parse_grobid_tei(_ECON_TEI.read_text(encoding="utf-8"))
    years = [r.year for r in refs if r.year is not None]
    assert years, "expected at least some parsed years"
    assert all(isinstance(y, int) for y in years)


def test_econ_fixture_dois_normalized() -> None:
    refs = parse_grobid_tei(_ECON_TEI.read_text(encoding="utf-8"))
    dois = [r.doi for r in refs if r.doi]
    assert dois, "expected at least some DOIs"
    # normalize_doi lowercases and strips prefixes; no residual scheme/prefix.
    for doi in dois:
        assert doi == doi.lower()
        assert not doi.startswith("http")
        assert not doi.startswith("doi:")


def test_econ_fixture_fragmented_schumpeter_is_one_entry_with_doi() -> None:
    # The 2002 "The economy as a whole" Schumpeter entry is ONE reference and
    # carries its DOI 10.1080/13662710220123653 (not split across entries).
    refs = parse_grobid_tei(_ECON_TEI.read_text(encoding="utf-8"))
    carriers = [r for r in refs if r.doi == "10.1080/13662710220123653"]
    assert len(carriers) == 1
    entry = carriers[0]
    assert entry.year == 2002
    assert "Schumpeter" in entry.authors


# --------------------------------------------------------------------------
# Offline parser — centrifugal fixture (62 biblStruct)
# --------------------------------------------------------------------------


def test_centrifugal_fixture_parses_all_62_references() -> None:
    refs = parse_grobid_tei(_CENTRIFUGAL_TEI.read_text(encoding="utf-8"))
    assert len(refs) == 62


def test_centrifugal_fixture_named_authors_present() -> None:
    refs = parse_grobid_tei(_CENTRIFUGAL_TEI.read_text(encoding="utf-8"))
    all_surnames = {name for r in refs for name in r.authors}
    assert {"Alesina", "Arzaghi", "Beramendi", "Blankart"}.issubset(all_surnames)


def test_centrifugal_fixture_two_alesina_entries_are_distinct() -> None:
    # "Fractionalization" (article) and "The size of nations" (book) are two
    # DISTINCT Alesina references, not a merged one.
    refs = parse_grobid_tei(_CENTRIFUGAL_TEI.read_text(encoding="utf-8"))
    alesina = [r for r in refs if "Alesina" in r.authors]
    titles = {r.title for r in alesina}
    assert "Fractionalization" in titles
    assert "The size of nations" in titles
    assert len({t for t in titles}) >= 2


def test_centrifugal_fixture_every_reference_has_raw_text() -> None:
    refs = parse_grobid_tei(_CENTRIFUGAL_TEI.read_text(encoding="utf-8"))
    assert all(r.raw_text.strip() for r in refs)


# --------------------------------------------------------------------------
# Cross-fixture: the SAME code parses both fixtures (no format-specific branch)
# --------------------------------------------------------------------------


def test_same_parser_handles_both_fixtures() -> None:
    econ = parse_grobid_tei(_ECON_TEI.read_text(encoding="utf-8"))
    centrifugal = parse_grobid_tei(_CENTRIFUGAL_TEI.read_text(encoding="utf-8"))
    assert len(econ) == 31
    assert len(centrifugal) == 62


def test_empty_and_garbage_tei_yield_empty_list() -> None:
    assert parse_grobid_tei("") == []
    assert parse_grobid_tei("   ") == []
    assert parse_grobid_tei("<not-xml") == []


def test_biblstruct_without_title_or_doi_is_skipped() -> None:
    tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><back><div><listBibl>
      <biblStruct><monogr><imprint>
        <date type="published" when="1999">1999</date>
      </imprint></monogr></biblStruct>
      <biblStruct><analytic>
        <title level="a" type="main">A real title</title>
      </analytic></biblStruct>
    </listBibl></div></back></text></TEI>"""
    refs = parse_grobid_tei(tei)
    assert len(refs) == 1
    assert refs[0].title == "A real title"


# --------------------------------------------------------------------------
# Best-effort HTTP wrapper (mocked transport — no network)
# --------------------------------------------------------------------------


def _service_with_transport(handler) -> GrobidReferenceService:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GrobidReferenceService(base_url="http://grobid:8070", client=client)


async def test_service_maps_successful_response() -> None:
    body = _ECON_TEI.read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/processReferences"
        return httpx.Response(200, text=body)

    service = _service_with_transport(handler)
    try:
        refs = await service.extract_references(b"%PDF-1.7 fake")
    finally:
        await service.aclose()
    assert len(refs) == 31


async def test_service_returns_empty_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    service = _service_with_transport(handler)
    try:
        refs = await service.extract_references(b"%PDF-1.7 fake")
    finally:
        await service.aclose()
    assert refs == []


async def test_service_returns_empty_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    service = _service_with_transport(handler)
    try:
        refs = await service.extract_references(b"%PDF-1.7 fake")
    finally:
        await service.aclose()
    assert refs == []


async def test_service_returns_empty_on_missing_pdf_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not POST for an unreadable file")

    service = _service_with_transport(handler)
    try:
        refs = await service.extract_references("/no/such/file.pdf")
    finally:
        await service.aclose()
    assert refs == []


def test_service_reads_base_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GROBID_URL", "http://example:9999/")
    service = GrobidReferenceService()
    # Trailing slash stripped so path joins are clean.
    assert service.base_url == "http://example:9999"


def test_service_default_base_url(monkeypatch) -> None:
    monkeypatch.delenv("GROBID_URL", raising=False)
    service = GrobidReferenceService()
    assert service.base_url == "http://localhost:8070"


# --------------------------------------------------------------------------
# Gated integration — real GROBID, skipped cleanly in CI
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not _grobid_alive(),
    reason=f"GROBID not reachable at {_GROBID_URL}/api/isalive",
)
async def test_live_grobid_extracts_references_from_pdf() -> None:
    service = GrobidReferenceService(base_url=_GROBID_URL)
    try:
        refs = await service.extract_references(_LIVE_PDF)
    finally:
        await service.aclose()
    assert len(refs) >= 25
    assert all(r.raw_text.strip() for r in refs)
    # Same shape as the offline path: surnames, int years, normalized DOIs.
    for r in refs:
        assert r.year is None or isinstance(r.year, int)
        assert r.doi is None or r.doi == r.doi.lower()
