"""Container roundtrip for source-level kNN retrieval (Track R.1).

Seeds sources with hand-built aggregate ``source.embedding`` vectors whose
cosine ordering is known by construction, then asserts
``SourceRepository.find_related_by_embedding`` returns them in that order.
This validates the actual SurrealQL ``vector::similarity::cosine`` ranking,
the self-exclusion, the NONE-embedding handling, and the deterministic
``id ASC`` tie-break — which the DB-free router mocks cannot.

Gated on ``requires_docker``; skipped cleanly without Docker.
"""

from __future__ import annotations

import math
import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.source import SourceRepository


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _unit(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


async def _make_source(
    cfg: SurrealDBConfig, embedding=None
) -> str:
    """Create a source, optionally with a populated aggregate embedding."""
    created = await execute_query(
        "CREATE source SET title = $t, full_text = 'body';",
        {"t": _uid("r1-src")},
        config=cfg,
    )
    sid = str(created[0]["id"])
    if embedding is not None:
        await execute_query(
            "UPDATE $id SET embedding = $e;",
            {"id": ensure_record_id(sid), "e": embedding},
            config=cfg,
        )
    return sid


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_ranks_by_cosine_excluding_self(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC1+AC4: nearest sources first, by cosine, query source excluded.

    Build 4-dim vectors with a known angular order relative to the query:
      query  = [1, 0, 0, 0]
      near   = [0.99, 0.14, 0, 0]      (cosine ~0.99 — closest)
      mid    = [0.7, 0.7, 0, 0]        (cosine ~0.707)
      far    = [0, 1, 0, 0]            (cosine 0 — orthogonal)
    Expect order: near, mid, far. Self never appears.
    """
    import surrealdb_service.connection as conn

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)
        q = await _make_source(live_surrealdb, _unit([1.0, 0.0, 0.0, 0.0]))
        near = await _make_source(live_surrealdb, _unit([0.99, 0.14, 0.0, 0.0]))
        mid = await _make_source(live_surrealdb, _unit([0.7, 0.7, 0.0, 0.0]))
        far = await _make_source(live_surrealdb, _unit([0.0, 1.0, 0.0, 0.0]))

        results = await repo.find_related_by_embedding(q, k=10)
        ids = [r["id"] for r in results]

        # Self excluded.
        assert q not in ids
        # The three seeded sources appear in cosine-desc order.
        seeded = [i for i in ids if i in {near, mid, far}]
        assert seeded == [near, mid, far]
        # Scores are descending and the nearest is ~1.0.
        scores = [r["score"] for r in results if r["id"] in {near, mid, far}]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > 0.98
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_k_limits_and_more_than_available(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC2: k bounds the result count; k > available returns all available."""
    import surrealdb_service.connection as conn

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)
        q = await _make_source(live_surrealdb, _unit([1.0, 0.0, 0.0]))
        peers = [
            await _make_source(live_surrealdb, _unit([1.0, 0.1 * i, 0.0]))
            for i in range(1, 4)
        ]

        top1 = await repo.find_related_by_embedding(q, k=1)
        # Only our seeded peers are candidates for this assertion; other tests'
        # sources may coexist in the session DB, so just assert the bound holds.
        assert len(top1) == 1

        # k far larger than the 3 peers returns all of them (plus any other
        # embedded sources from sibling tests) — never crashes, never pads.
        many = await repo.find_related_by_embedding(q, k=1000)
        returned_peers = [r["id"] for r in many if r["id"] in set(peers)]
        assert sorted(returned_peers) == sorted(peers)
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_query_source_without_aggregate_returns_empty(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC3: querying a source whose embedding is NONE returns []."""
    import surrealdb_service.connection as conn

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)
        # A peer with an embedding exists so the candidate set is non-empty.
        await _make_source(live_surrealdb, _unit([1.0, 0.0, 0.0]))
        no_emb = await _make_source(live_surrealdb, embedding=None)

        assert await repo.find_related_by_embedding(no_emb, k=5) == []
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_none_embedding_sources_never_appear_as_results(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC3: a NONE-embedding source is excluded from another source's results."""
    import surrealdb_service.connection as conn

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)
        q = await _make_source(live_surrealdb, _unit([1.0, 0.0, 0.0]))
        embedded = await _make_source(live_surrealdb, _unit([1.0, 0.05, 0.0]))
        no_emb = await _make_source(live_surrealdb, embedding=None)

        ids = [r["id"] for r in await repo.find_related_by_embedding(q, k=1000)]
        assert embedded in ids
        assert no_emb not in ids
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_tie_break_is_deterministic_by_id(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC2: equal-cosine sources come back in a stable ``id ASC`` order."""
    import surrealdb_service.connection as conn

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)
        q = await _make_source(live_surrealdb, _unit([1.0, 0.0, 0.0]))
        # Two peers with the IDENTICAL vector -> identical cosine -> tie.
        same_vec = _unit([0.6, 0.8, 0.0])
        a = await _make_source(live_surrealdb, same_vec)
        b = await _make_source(live_surrealdb, same_vec)

        results = await repo.find_related_by_embedding(q, k=1000)
        tied = [r["id"] for r in results if r["id"] in {a, b}]
        # Same order across repeated calls, and that order is id-ascending.
        assert tied == sorted(tied)
        again = [
            r["id"]
            for r in await repo.find_related_by_embedding(q, k=1000)
            if r["id"] in {a, b}
        ]
        assert tied == again
    finally:
        conn._pool = orig
