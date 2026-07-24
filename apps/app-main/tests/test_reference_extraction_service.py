"""Track V.5 / G.3: the ``ReferenceExtractionService`` orchestration (DB-free unit).

These tests pin the GROBID producer -> U.3 ``cites`` materialization bridge with
the source repo, the GROBID service, and the U.3 materializer all MOCKED (no DB,
no network):

  * a source whose ``asset.file_path`` is a PDF -> references come from GROBID and
    are handed to U.3's ``materialize`` as a ``{source_id: [ParsedReference]}`` map;
  * a source with no asset / no ``file_path`` / a non-PDF path -> ``[]`` (best
    effort) and it never reaches the map;
  * idempotent re-run -> the same map handed to U.3 both times (U.3 is
    clear-before-relate, so re-running yields the same edge set);
  * an empty corpus -> a clean no-op (materialize called with ``{}``);
  * ``resolve_external=True`` -> every extracted reference routed through the V.4
    cascade (mocked), counted into the summary.

The TRUE end-to-end ``cites`` edge count on real documents is a deferred LIVE
acceptance criterion — see ``test_live_grobid_source_extracts_references`` below,
which runs only when a live GROBID is reachable.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock

import httpx
import pytest
from app_main.services.references import ReferenceExtractionService
from shared.retrieval.cites_matching import ParsedReference

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Synthetic references — what a mocked GROBID hands back for a PDF source.
# ---------------------------------------------------------------------------

_REFS = [
    ParsedReference(
        raw_text="Acemoglu, D., Robinson, J. A. (2012). Why nations fail. Crown.",
        title="Why nations fail",
        authors=("Acemoglu", "Robinson"),
        year=2012,
        doi=None,
        venue="Crown",
    ),
    ParsedReference(
        raw_text=(
            "Card, D. (2001). Immigrant inflows, native outflows, and the local "
            "labor market. American Economic Review. doi:10.1257/aer.91.5.1369"
        ),
        title="Immigrant inflows, native outflows, and the local labor market",
        authors=("Card",),
        year=2001,
        doi="10.1257/aer.91.5.1369",
        venue="American Economic Review",
    ),
]


def _source_with_pdf(path: str = "data/uploads/econ.pdf") -> SimpleNamespace:
    """A source row whose original upload is a PDF at ``asset.file_path``."""
    return SimpleNamespace(asset=SimpleNamespace(file_path=path))


def _make_service(
    *,
    records,
    sources_by_id,
    grobid_refs=None,
    materialize_created=0,
    resolver=None,
):
    """Build a service with mocked source_repo, GROBID service, cites_service.

    ``records`` is the ``load_source_records`` return (``[{"id": ...}, ...]``);
    ``sources_by_id`` maps a source id -> the row ``source_repo.get`` returns;
    ``grobid_refs`` is what the mocked GROBID returns for a PDF (default ``_REFS``).
    """
    src_repo = AsyncMock()
    src_repo.load_source_records = AsyncMock(return_value=records)

    async def _get(sid):
        return sources_by_id.get(str(sid))

    src_repo.get = AsyncMock(side_effect=_get)

    grobid = AsyncMock()
    grobid.extract_references = AsyncMock(
        return_value=list(_REFS if grobid_refs is None else grobid_refs)
    )

    cites_service = AsyncMock()
    cites_service.materialize = AsyncMock(
        return_value=SimpleNamespace(created=materialize_created)
    )

    service = ReferenceExtractionService(
        source_repo=src_repo,
        cites_service=cites_service,
        grobid_service=grobid,
        resolver_cascade=resolver,
    )
    return service, src_repo, grobid, cites_service


# ---------------------------------------------------------------------------
# extract_source_references — PDF retrieval + GROBID hand-off
# ---------------------------------------------------------------------------


async def test_extract_source_references_reads_asset_and_calls_grobid() -> None:
    service, _, grobid, _ = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={"source:s1": _source_with_pdf("data/uploads/econ.pdf")},
    )

    refs = await service.extract_source_references("source:s1")

    # GROBID was handed the source's PDF path verbatim.
    grobid.extract_references.assert_awaited_once_with("data/uploads/econ.pdf")
    assert [r.doi for r in refs] == [None, "10.1257/aer.91.5.1369"]


async def test_source_without_asset_yields_no_references() -> None:
    service, _, grobid, _ = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={"source:s1": SimpleNamespace(asset=None)},
    )
    assert await service.extract_source_references("source:s1") == []
    grobid.extract_references.assert_not_awaited()


async def test_source_with_no_file_path_yields_no_references() -> None:
    service, _, grobid, _ = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={
            "source:s1": SimpleNamespace(asset=SimpleNamespace(file_path=None))
        },
    )
    assert await service.extract_source_references("source:s1") == []
    grobid.extract_references.assert_not_awaited()


async def test_non_pdf_source_yields_no_references() -> None:
    # A URL/note-style source with a non-PDF path is skipped (best-effort).
    service, _, grobid, _ = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={
            "source:s1": _source_with_pdf("https://example.com/article.html")
        },
    )
    assert await service.extract_source_references("source:s1") == []
    grobid.extract_references.assert_not_awaited()


async def test_missing_source_row_yields_no_references() -> None:
    service, _, grobid, _ = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={},  # source_repo.get -> None
    )
    assert await service.extract_source_references("source:s1") == []
    grobid.extract_references.assert_not_awaited()


# ---------------------------------------------------------------------------
# materialize_corpus — hand-off to U.3
# ---------------------------------------------------------------------------


async def test_extracts_and_hands_map_to_materialize() -> None:
    service, _, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={"source:s1": _source_with_pdf()},
        materialize_created=1,
    )

    summary = await service.materialize_corpus()

    # U.3's materialize is called exactly once, with THIS source's references.
    cites.materialize.assert_awaited_once()
    payload = cites.materialize.await_args.args[0]
    assert set(payload.keys()) == {"source:s1"}
    refs = payload["source:s1"]
    assert len(refs) == 2
    assert any(r.doi == "10.1257/aer.91.5.1369" for r in refs)

    assert summary.sources_scanned == 1
    assert summary.sources_with_references == 1
    assert summary.refs_extracted == 2
    assert summary.edges_materialized == 1  # passed through from U.3's `created`
    assert summary.external_resolved == 0


async def test_only_sources_with_references_appear_in_map() -> None:
    # One source has a PDF, one is a non-PDF (URL) source — only the former appears.
    service, _, _, cites = _make_service(
        records=[{"id": "source:s1"}, {"id": "source:s2"}],
        sources_by_id={
            "source:s1": _source_with_pdf(),
            "source:s2": _source_with_pdf("https://example.com/x.html"),
        },
    )

    summary = await service.materialize_corpus()

    payload = cites.materialize.await_args.args[0]
    assert set(payload.keys()) == {"source:s1"}
    assert summary.sources_scanned == 2
    assert summary.sources_with_references == 1


# ---------------------------------------------------------------------------
# idempotent re-run
# ---------------------------------------------------------------------------


async def test_idempotent_rerun_hands_same_map() -> None:
    service, _, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={"source:s1": _source_with_pdf()},
        materialize_created=1,
    )

    first = await service.materialize_corpus()
    second = await service.materialize_corpus()

    # U.3 is clear-before-relate: the SAME map is handed both runs, so the
    # (mocked) edge outcome is identical. G.3 adds no second dedup.
    calls = cites.materialize.await_args_list
    assert len(calls) == 2
    assert set(calls[0].args[0].keys()) == set(calls[1].args[0].keys())
    assert first == second


# ---------------------------------------------------------------------------
# no-references corpus -> clean no-op
# ---------------------------------------------------------------------------


async def test_no_references_is_clean_noop() -> None:
    # A source with a PDF that GROBID finds no references in.
    service, _, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={"source:s1": _source_with_pdf()},
        grobid_refs=[],
        materialize_created=0,
    )

    summary = await service.materialize_corpus()

    cites.materialize.assert_awaited_once_with({})
    assert summary.refs_extracted == 0
    assert summary.sources_with_references == 0
    assert summary.edges_materialized == 0


async def test_empty_corpus_is_clean_noop() -> None:
    service, _, _, cites = _make_service(records=[], sources_by_id={})
    summary = await service.materialize_corpus()
    cites.materialize.assert_awaited_once_with({})
    assert summary.sources_scanned == 0
    assert summary.refs_extracted == 0


# ---------------------------------------------------------------------------
# resolve_external -> V.4 cascade invoked (mocked)
# ---------------------------------------------------------------------------


async def test_resolve_external_invokes_cascade_and_counts() -> None:
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(return_value=SimpleNamespace(title="x"))

    service, _, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={"source:s1": _source_with_pdf()},
        resolver=resolver,
    )

    summary = await service.materialize_corpus(resolve_external=True)

    # The cascade saw every extracted reference (before materialization).
    assert resolver.resolve.await_count == 2
    assert summary.external_resolved == 2
    cites.materialize.assert_awaited_once()


async def test_resolve_external_off_never_touches_cascade() -> None:
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(return_value=SimpleNamespace(title="x"))

    service, _, _, _ = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={"source:s1": _source_with_pdf()},
        resolver=resolver,
    )

    summary = await service.materialize_corpus()  # flag OFF (default)

    resolver.resolve.assert_not_awaited()
    assert summary.external_resolved == 0


async def test_resolve_external_flag_without_cascade_is_zero() -> None:
    # Flag ON but no cascade injected -> warn + 0 (no crash).
    service, _, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={"source:s1": _source_with_pdf()},
        resolver=None,
    )
    summary = await service.materialize_corpus(resolve_external=True)
    assert summary.external_resolved == 0
    cites.materialize.assert_awaited_once()


async def test_resolve_external_failing_leg_is_soft() -> None:
    # A cascade that raises must be treated as a no-match, never propagate.
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(side_effect=RuntimeError("network down"))

    service, _, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        sources_by_id={"source:s1": _source_with_pdf()},
        resolver=resolver,
    )

    summary = await service.materialize_corpus(resolve_external=True)  # no raise
    assert summary.external_resolved == 0
    cites.materialize.assert_awaited_once()


# ---------------------------------------------------------------------------
# LIVE acceptance — real PDF source -> real GROBID -> >= 25 references
# ---------------------------------------------------------------------------

_LIVE_PDF = (
    Path(__file__).resolve().parents[3]
    / "packages"
    / "shared"
    / "tests"
    / "fixtures"
    / "references"
    / "live_corpus"
    / "economics-without-equilibrium.pdf"
)
_GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")


def _grobid_alive() -> bool:
    """True iff a live GROBID answers ``/api/isalive`` with 200 (the gate)."""
    try:
        resp = httpx.get(f"{_GROBID_URL}/api/isalive", timeout=2.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


requires_grobid = pytest.mark.skipif(
    not _grobid_alive() or not _LIVE_PDF.exists(),
    reason=f"GROBID not reachable at {_GROBID_URL}/api/isalive (or fixture PDF absent)",
)


@requires_grobid
async def test_live_grobid_source_extracts_references() -> None:
    """LIVE: a real PDF source drives real GROBID through the service seam.

    Uses the REAL :class:`GrobidReferenceService` (not the mock) but a stubbed
    source repo pointing at the committed fixture PDF, so it exercises the
    ``asset.file_path`` -> GROBID path end-to-end without a DB. Skips cleanly when
    GROBID is absent.
    """
    from shared.references.grobid_reference_service import GrobidReferenceService

    src_repo = AsyncMock()
    src_repo.get = AsyncMock(
        return_value=_source_with_pdf(str(_LIVE_PDF))
    )
    cites = AsyncMock()
    grobid = GrobidReferenceService(base_url=_GROBID_URL)
    service = ReferenceExtractionService(
        source_repo=src_repo, cites_service=cites, grobid_service=grobid
    )
    try:
        refs: List[ParsedReference] = await service.extract_source_references(
            "source:live"
        )
    finally:
        await grobid.aclose()

    assert len(refs) >= 25
    assert all(r.raw_text.strip() for r in refs)
