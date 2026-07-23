"""Track V.5: the ``ReferenceExtractionService`` orchestration (DB-free unit).

These tests pin the V.1-V.3 producer -> U.3 ``cites`` materialization bridge with
the repositories + the U.3 materializer MOCKED (no DB, no network):

  * a source with a bibliography -> references extracted and handed to U.3's
    ``materialize`` as a ``{source_id: [ParsedReference]}`` map;
  * idempotent re-run -> the same map handed to U.3 both times (U.3 is
    clear-before-relate, so re-running yields the same edge set);
  * a source with no reference region -> a clean no-op (materialize called with
    ``{}``, 0 refs, 0 edges);
  * ``resolve_external=True`` -> every extracted reference routed through the V.4
    cascade (mocked), counted into the summary.

The TRUE end-to-end ``cites`` edge count on real documents is a deferred LIVE-DB
acceptance criterion — see ``test_end_to_end_cites_edge_count`` (skipped).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock

import pytest
from app_main.services.references import ReferenceExtractionService
from shared.models.source import Chunk

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Synthetic fixtures — a structured References section (mirrors the V.1-V.3
# synthetic shape) as persisted ``Chunk`` rows.
# ---------------------------------------------------------------------------

_ENTRIES = [
    "Acemoglu, D., & Robinson, J. A. (2012). Why nations fail. Crown.",
    (
        "Card, D. (2001). Immigrant inflows, native outflows, and the local "
        "labor market. American Economic Review, 91(5), 1369-1401. "
        "https://doi.org/10.1257/aer.91.5.1369"
    ),
]


def _ref_chunks() -> List[Chunk]:
    """A structured doc: a References heading + one list_item chunk per entry."""
    chunks: List[Chunk] = [
        Chunk(
            text="Conclusion",
            order=0,
            physical_page=1,
            element_type="heading",
            section_path=["Conclusion"],
        ),
        Chunk(
            text="References",
            order=1,
            physical_page=2,
            element_type="section_header",
            section_path=["References"],
        ),
    ]
    for i, entry in enumerate(_ENTRIES):
        chunks.append(
            Chunk(
                text=entry,
                order=2 + i,
                physical_page=2,
                element_type="list_item",
                section_path=["References"],
            )
        )
    return chunks


def _no_ref_chunks() -> List[Chunk]:
    """A document with no bibliography region at all."""
    return [
        Chunk(
            text="Introduction",
            order=0,
            physical_page=1,
            element_type="heading",
            section_path=["Introduction"],
        ),
        Chunk(
            text="Body text with no citations.",
            order=1,
            physical_page=1,
            element_type="text",
            section_path=["Introduction"],
        ),
    ]


def _make_service(
    *,
    records,
    chunks_by_source,
    materialize_created=0,
    resolver=None,
):
    """Build a service with mocked source_repo + cites_service.

    ``records`` is the ``load_source_records`` return (``[{"id": ...}, ...]``);
    ``chunks_by_source`` maps a source id -> its ``get_chunks`` return.
    """

    src_repo = AsyncMock()
    src_repo.load_source_records = AsyncMock(return_value=records)

    async def _get_chunks(sid):
        return chunks_by_source.get(str(sid), [])

    src_repo.get_chunks = AsyncMock(side_effect=_get_chunks)
    # full_text is unused for the structured fixtures; return a source with none.
    src_repo.get = AsyncMock(return_value=SimpleNamespace(full_text=None))

    cites_service = AsyncMock()
    cites_service.materialize = AsyncMock(
        return_value=SimpleNamespace(created=materialize_created)
    )

    service = ReferenceExtractionService(
        source_repo=src_repo,
        cites_service=cites_service,
        resolver_cascade=resolver,
    )
    return service, src_repo, cites_service


# ---------------------------------------------------------------------------
# extract + hand-off to U.3
# ---------------------------------------------------------------------------


async def test_extracts_and_hands_map_to_materialize() -> None:
    service, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        chunks_by_source={"source:s1": _ref_chunks()},
        materialize_created=1,
    )

    summary = await service.materialize_corpus()

    # U.3's materialize is called exactly once, with THIS source's references.
    cites.materialize.assert_awaited_once()
    payload = cites.materialize.await_args.args[0]
    assert set(payload.keys()) == {"source:s1"}
    refs = payload["source:s1"]
    assert len(refs) == 2
    # The DOI-bearing entry parsed its normalized DOI (the V.3 producer ran).
    assert any(r.doi == "10.1257/aer.91.5.1369" for r in refs)

    assert summary.sources_scanned == 1
    assert summary.sources_with_references == 1
    assert summary.refs_extracted == 2
    assert summary.edges_materialized == 1  # passed through from U.3's `created`
    assert summary.external_resolved == 0


async def test_only_sources_with_references_appear_in_map() -> None:
    # One source has a bibliography, one does not — only the former is in the map.
    service, _, cites = _make_service(
        records=[{"id": "source:s1"}, {"id": "source:s2"}],
        chunks_by_source={
            "source:s1": _ref_chunks(),
            "source:s2": _no_ref_chunks(),
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
    service, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        chunks_by_source={"source:s1": _ref_chunks()},
        materialize_created=1,
    )

    first = await service.materialize_corpus()
    second = await service.materialize_corpus()

    # U.3 is clear-before-relate: the SAME map is handed both runs, so the
    # (mocked) edge outcome is identical. V.5 adds no second dedup.
    calls = cites.materialize.await_args_list
    assert len(calls) == 2
    assert set(calls[0].args[0].keys()) == set(calls[1].args[0].keys())
    assert first == second


# ---------------------------------------------------------------------------
# no-references corpus -> clean no-op
# ---------------------------------------------------------------------------


async def test_no_references_is_clean_noop() -> None:
    service, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        chunks_by_source={"source:s1": _no_ref_chunks()},
        materialize_created=0,
    )

    summary = await service.materialize_corpus()

    # materialize is still called (the U.3 no-op clears to 0) — with an empty map.
    cites.materialize.assert_awaited_once_with({})
    assert summary.refs_extracted == 0
    assert summary.sources_with_references == 0
    assert summary.edges_materialized == 0


async def test_empty_corpus_is_clean_noop() -> None:
    service, _, cites = _make_service(records=[], chunks_by_source={})
    summary = await service.materialize_corpus()
    cites.materialize.assert_awaited_once_with({})
    assert summary.sources_scanned == 0
    assert summary.refs_extracted == 0


# ---------------------------------------------------------------------------
# resolve_external -> V.4 cascade invoked (mocked)
# ---------------------------------------------------------------------------


async def test_resolve_external_invokes_cascade_and_counts() -> None:
    # Cascade resolves every reference to a (truthy) work -> external_resolved=2.
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(return_value=SimpleNamespace(title="x"))

    service, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        chunks_by_source={"source:s1": _ref_chunks()},
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

    service, _, _ = _make_service(
        records=[{"id": "source:s1"}],
        chunks_by_source={"source:s1": _ref_chunks()},
        resolver=resolver,
    )

    summary = await service.materialize_corpus()  # flag OFF (default)

    resolver.resolve.assert_not_awaited()
    assert summary.external_resolved == 0


async def test_resolve_external_flag_without_cascade_is_zero() -> None:
    # Flag ON but no cascade injected -> warn + 0 (no crash).
    service, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        chunks_by_source={"source:s1": _ref_chunks()},
        resolver=None,
    )
    summary = await service.materialize_corpus(resolve_external=True)
    assert summary.external_resolved == 0
    cites.materialize.assert_awaited_once()


async def test_resolve_external_failing_leg_is_soft() -> None:
    # A cascade that raises must be treated as a no-match, never propagate.
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(side_effect=RuntimeError("network down"))

    service, _, cites = _make_service(
        records=[{"id": "source:s1"}],
        chunks_by_source={"source:s1": _ref_chunks()},
        resolver=resolver,
    )

    summary = await service.materialize_corpus(resolve_external=True)  # no raise
    assert summary.external_resolved == 0
    cites.materialize.assert_awaited_once()


# ---------------------------------------------------------------------------
# full_text fallback path (no structured chunks)
# ---------------------------------------------------------------------------


async def test_full_text_fallback_extracts_references() -> None:
    # No structured chunks; the bibliography lives in the source's full_text.
    full_text = (
        "Introduction\n\nBody.\n\n"
        "References\n\n"
        "Acemoglu, D. (2012). Why nations fail. Crown.\n\n"
        "Card, D. (2001). Immigrant inflows. AER, 91(5), 1369-1401.\n"
    )
    src_repo = AsyncMock()
    src_repo.load_source_records = AsyncMock(return_value=[{"id": "source:s1"}])
    src_repo.get_chunks = AsyncMock(return_value=[])
    src_repo.get = AsyncMock(return_value=SimpleNamespace(full_text=full_text))

    cites = AsyncMock()
    cites.materialize = AsyncMock(return_value=SimpleNamespace(created=0))

    service = ReferenceExtractionService(
        source_repo=src_repo, cites_service=cites
    )
    summary = await service.materialize_corpus()

    assert summary.refs_extracted >= 2
    payload = cites.materialize.await_args.args[0]
    assert "source:s1" in payload


# ---------------------------------------------------------------------------
# DEFERRED live-DB acceptance
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="TODO(V-live): true end-to-end cites edge count needs a live DB + "
    "real ingested documents; covered offline by the seam tests above."
)
async def test_end_to_end_cites_edge_count() -> None:  # pragma: no cover
    """LIVE-DB smoke: ingest real documents, run the V.5 pass, assert the actual
    ``cites`` edge count. Deferred to the live smoke — not required in CI."""
    raise NotImplementedError
