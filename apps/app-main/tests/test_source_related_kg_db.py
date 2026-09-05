"""Container roundtrip for KG-proximity source retrieval (Track R.2).

Seeds sources + active entities whose shared-entity overlap is known by
construction, then asserts the repo loaders + ``KGRetrievalService`` rank them
by KG proximity with the named/typed entity outranking the generic ``topic``.
This validates the real SurrealQL projection (``status='active'`` filter,
``source_documents`` string-space ids) and the title hydration that the
DB-free pure-scorer tests cannot.

Gated on ``requires_docker``; skipped cleanly without Docker.
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories import EntityRepository, SourceRepository

from app_main.services.kg_retrieval_service import KGRetrievalService


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def _make_source(cfg: SurrealDBConfig, title: str) -> str:
    created = await execute_query(
        "CREATE source SET title = $t, full_text = 'body';",
        {"t": title},
        config=cfg,
    )
    return str(created[0]["id"])


async def _make_entity(
    cfg: SurrealDBConfig,
    name: str,
    entity_type: str,
    source_ids: list,
    status: str = "active",
) -> str:
    created = await execute_query(
        "CREATE entity SET canonical_name = $n, name_key = string::lowercase(string::trim($n)), name = $n, "
        "entity_type = $et, status = $st, source_documents = $sd, "
        "embedding = [], hash_id = $h;",
        {
            "n": name,
            "et": entity_type,
            "st": status,
            "sd": source_ids,
            "h": _uid("h"),
        },
        config=cfg,
    )
    return str(created[0]["id"])


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_named_entity_outranks_topic_overlap(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC1+AC2: shared named programme outranks shared generic topic, end-to-end.

    Q shares a ``programme`` with B and only a ``topic`` with C ⇒ B ranks
    above C, and the explanation names the driving entity.
    """
    import surrealdb_service.connection as conn

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        q = await _make_source(live_surrealdb, _uid("r2-q"))
        b = await _make_source(live_surrealdb, _uid("r2-b"))
        c = await _make_source(live_surrealdb, _uid("r2-c"))

        # Named programme shared by q & b.
        await _make_entity(
            live_surrealdb, _uid("Regio-Deal"), "programme", [q, b]
        )
        # Generic topic shared by q & c.
        await _make_entity(
            live_surrealdb, _uid("gezondheid"), "topic", [q, c]
        )

        svc = KGRetrievalService(
            EntityRepository(config=live_surrealdb),
            SourceRepository(config=live_surrealdb),
        )
        results = await svc.find_related_by_kg(q, k=10)

        ids = [r["id"] for r in results]
        assert q not in ids
        assert b in ids and c in ids
        assert ids.index(b) < ids.index(c)
        score_b = next(r["score"] for r in results if r["id"] == b)
        score_c = next(r["score"] for r in results if r["id"] == c)
        assert score_b > score_c
        # Explanation lineage present and typed.
        expl_b = next(r["explanation"] for r in results if r["id"] == b)
        assert expl_b and expl_b[0]["type"] == "programme"
        assert all("weight" in e for e in expl_b)
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_archived_entities_excluded(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC3: an archived shared entity does not create a related link."""
    import surrealdb_service.connection as conn

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        q = await _make_source(live_surrealdb, _uid("r2-q2"))
        b = await _make_source(live_surrealdb, _uid("r2-b2"))

        # Only shared entity is ARCHIVED ⇒ must not link q and b.
        await _make_entity(
            live_surrealdb,
            _uid("ArchivedOrg"),
            "organization",
            [q, b],
            status="archived",
        )

        svc = KGRetrievalService(
            EntityRepository(config=live_surrealdb),
            SourceRepository(config=live_surrealdb),
        )
        results = await svc.find_related_by_kg(q, k=10)
        assert b not in [r["id"] for r in results]
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_no_overlap_returns_empty(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A source sharing no active entity with any other returns []."""
    import surrealdb_service.connection as conn

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        q = await _make_source(live_surrealdb, _uid("r2-lonely"))
        await _make_entity(
            live_surrealdb, _uid("solo"), "organization", [q]
        )

        svc = KGRetrievalService(
            EntityRepository(config=live_surrealdb),
            SourceRepository(config=live_surrealdb),
        )
        # q's only entity is q-only ⇒ no related source.
        assert await svc.find_related_by_kg(q, k=10) == []
    finally:
        conn._pool = orig
