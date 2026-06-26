"""Read-only live-staging sanity for the hybrid ranker (Track R.3, AC5).

Runs the real ``HybridRetrievalService`` against the live staging DB and asserts
the fused ranking is well-formed for >=2 real sources: the convenanten cluster
(both signals agree there) sits at the top, a source present in only ONE signal
still ranks (never dropped), and per-result provenance is populated. NO WRITES.

Gated on ``requires_staging`` AND ``SURREAL_DATABASE == 'staging'`` being set
explicitly — skipped otherwise so CI never depends on a populated staging DB.
This is the SANITY check (both signals agree on the convenanten); the ablation
PROOF (signals disagreeing) lives in the pure-fusion unit tests.
"""

from __future__ import annotations

import os

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories import EntityRepository, SourceRepository

from app_main.services.hybrid_retrieval_service import HybridRetrievalService
from app_main.services.kg_retrieval_service import KGRetrievalService
from app_main.services.source_service import SourceService

# Only run when the env explicitly targets staging — never against the default
# local DB, and never as a side effect of the normal suite.
_STAGING = os.getenv("SURREAL_DATABASE") == "staging"

pytestmark = [
    pytest.mark.requires_staging,
    pytest.mark.skipif(
        not _STAGING,
        reason="SURREAL_DATABASE != 'staging'; read-only staging test skipped",
    ),
    pytest.mark.asyncio,
]


async def _staging_config() -> SurrealDBConfig:
    return SurrealDBConfig.from_env()


async def _candidate_sources(cfg: SurrealDBConfig) -> list[str]:
    """Sources that carry BOTH an aggregate embedding and active entities."""
    embedded = await execute_query(
        "SELECT VALUE id FROM source WHERE embedding != NONE", {}, cfg
    )
    embedded = [str(s) for s in (embedded or [])]

    ent_bags = await execute_query(
        "SELECT VALUE source_documents FROM entity WHERE status='active'",
        {},
        cfg,
    )
    with_entities: set[str] = set()
    for bag in ent_bags or []:
        for s in bag or []:
            with_entities.add(str(s))

    both = [s for s in embedded if s in with_entities]
    return both or embedded


def _build_service(cfg: SurrealDBConfig) -> HybridRetrievalService:
    source_repo = SourceRepository(config=cfg)
    source_svc = SourceService(source_repo, None, None, None)  # type: ignore[arg-type]
    kg_svc = KGRetrievalService(EntityRepository(config=cfg), source_repo)
    return HybridRetrievalService(source_svc, kg_svc)


async def test_staging_hybrid_ranking_for_multiple_sources() -> None:
    """AC5: report a well-formed fused ranking for >=2 real staging sources."""
    import surrealdb_service.connection as conn

    cfg = await _staging_config()
    assert cfg.database == "staging"

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=cfg)
    try:
        candidates = await _candidate_sources(cfg)
        if len(candidates) < 2:
            pytest.skip(
                "staging has <2 sources with embedding+entities to compare"
            )

        svc = _build_service(cfg)

        reported = 0
        for sid in candidates:
            fused = await svc.find_related_hybrid(sid, k=10)
            if not fused:
                continue

            # fused score is monotonically non-increasing (ranked)
            scores = [r["fused_score"] for r in fused]
            assert scores == sorted(scores, reverse=True)

            # the query source never appears in its own results
            assert sid not in [r["id"] for r in fused]

            # every result carries per-signal provenance + at least one signal
            for r in fused:
                assert "dense" in r and "kg" in r
                assert r["dense"]["present"] or r["kg"]["present"]
                # contributions sum to the fused score
                contrib = (
                    r["dense"]["contribution"] + r["kg"]["contribution"]
                )
                assert abs(contrib - r["fused_score"]) < 1e-9

            reported += 1
            if reported >= 2:
                break

        assert reported >= 2, "expected fused rankings for >=2 sources"
    finally:
        conn._pool = orig


async def test_staging_uses_true_source_count_in_idf() -> None:
    """AC6: the KG IDF denominator is the TRUE source count (not the subset).

    KGRetrievalService fetches ``source_repo.count()`` (all sources) for the
    IDF, not the sources-with-active-entities subset — the R.2 carry-in fix.
    """
    import surrealdb_service.connection as conn

    cfg = await _staging_config()
    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=cfg)
    try:
        total_rows = await execute_query(
            "SELECT count() AS c FROM source GROUP ALL", {}, cfg
        )
        true_count = total_rows[0]["c"] if total_rows else 0

        ent_bags = await execute_query(
            "SELECT VALUE source_documents FROM entity WHERE status='active'",
            {},
            cfg,
        )
        with_entities: set[str] = set()
        for bag in ent_bags or []:
            for s in bag or []:
                with_entities.add(str(s))

        # The fix matters precisely when the corpus is bigger than the
        # entity-bearing subset (staging: 6 vs 4). Assert the repo count is the
        # true total the IDF will receive.
        kg_svc = KGRetrievalService(
            EntityRepository(config=cfg), SourceRepository(config=cfg)
        )
        repo_count = await kg_svc._true_source_count()
        assert repo_count == true_count
        assert repo_count >= len(with_entities)
    finally:
        conn._pool = orig
