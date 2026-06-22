"""Integration roundtrip for the K.3 retroactive merge (testcontainers).

Seeds a fragmented entity set + relations against a real SurrealDB container,
runs ``RecanonicalizationService.plan_merges`` then ``apply_merge``, and asserts
the full merge contract:

* the cluster groups the fragmented org variants (one cluster, one winner);
* losers end up ``status='merged'`` + ``merged_into=<winner>`` (soft, reversible);
* relations re-point from losers onto the winner BY ID — no relation references
  a merged entity afterwards;
* an ``entity_alias`` row exists per loser surface form;
* re-applying the same cluster is a no-op (idempotent);
* migration 54's ``aliases`` field reads back (default ``[]`` on a fresh entity).

Gated behind ``@requires_docker`` like the other roundtrip suites.
"""

from __future__ import annotations

import uuid

import pytest

# The service lives in apps/app-main; it imports cleanly (no app wiring needed).
from app_main.services.entity_resolution import RecanonicalizationService
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def _create_entity(
    config: SurrealDBConfig,
    name: str,
    etype: str,
    *,
    confidence: float = 1.0,
    sources: list | None = None,
) -> str:
    rows = await execute_query(
        "CREATE entity SET canonical_name = $n, name = $n, entity_type = $t, "
        "embedding = [], confidence = $c, source_documents = $s, "
        "status = 'active';",
        {"n": name, "t": etype, "c": confidence, "s": sources or []},
        config=config,
    )
    return str(rows[0]["id"])


async def _relate(config: SurrealDBConfig, src_id: str, tgt_id: str, rel_type: str):
    # RELATE needs bare record-id literals (SurrealDB #4232); ids are DB-issued.
    await execute_query(
        f"RELATE {src_id}->relation->{tgt_id} SET "
        "relation_type = $rt, confidence = 0.9, status = 'active';",
        {"rt": rel_type},
        config=config,
    )


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_aliases_field_reads_back_empty(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Migration 54: a fresh entity reads back ``aliases=[]`` and status active."""
    repo = EntityRepository(config=live_surrealdb)
    eid = await _create_entity(live_surrealdb, _u("alias-default"), "organization")
    ent = await repo.get_entity(eid)
    assert ent is not None
    assert ent.aliases == []
    assert ent.status == "active"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_plan_does_not_write(live_surrealdb: SurrealDBConfig) -> None:
    """plan_merges groups the variants but leaves all members active."""
    bzk = _u("BZK")
    # Two entities that normalize to the same key (article-stripped same tail).
    a = await _create_entity(live_surrealdb, f"De {bzk}", "organization", confidence=0.9)
    b = await _create_entity(live_surrealdb, bzk, "organization", confidence=0.8)

    repo = EntityRepository(config=live_surrealdb)
    svc = RecanonicalizationService(entity_repo=repo)
    plan = await svc.plan_merges()

    # Find our cluster among any other active entities in the container.
    ours = [
        c
        for c in plan.clusters
        if a in [c.winner_id, *c.loser_ids] or b in [c.winner_id, *c.loser_ids]
    ]
    assert len(ours) == 1
    assert set([ours[0].winner_id, *ours[0].loser_ids]) == {a, b}

    # No writes: both members still active.
    for eid in (a, b):
        ent = await repo.get_entity(eid)
        assert ent is not None and ent.status == "active"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_apply_merge_full_contract(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """End-to-end: plan → apply → merged status, relation re-point, aliases."""
    repo = EntityRepository(config=live_surrealdb)
    svc = RecanonicalizationService(entity_repo=repo)

    tail = _u("Koninkrijksrelaties")
    # Winner (highest confidence) + two losers that normalize to the same key.
    winner = await _create_entity(
        live_surrealdb, tail, "organization", confidence=0.95, sources=["src:win"]
    )
    loser1 = await _create_entity(
        live_surrealdb, f"De {tail}", "organization", confidence=0.6, sources=["src:l1"]
    )
    loser2 = await _create_entity(
        live_surrealdb, f"Het {tail}", "organization", confidence=0.5, sources=["src:l2"]
    )

    # A neighbour entity + relations pointing at the losers.
    neighbour = await _create_entity(live_surrealdb, _u("Neighbour"), "person")
    await _relate(live_surrealdb, neighbour, loser1, "MENTIONS")
    await _relate(live_surrealdb, loser2, neighbour, "PART_OF")

    plan = await svc.plan_merges()
    cluster = next(
        c
        for c in plan.clusters
        if c.winner_id == winner and set(c.loser_ids) == {loser1, loser2}
    )

    result = await svc.apply_merge(cluster)
    assert not result.skipped
    assert set(result.merged_loser_ids) == {loser1, loser2}
    assert result.relations_repointed == 2

    # Losers soft-merged, never deleted.
    for lid in (loser1, loser2):
        ent = await repo.get_entity(lid)
        assert ent is not None
        assert ent.status == "merged"
        assert ent.merged_into == winner

    # Winner unioned the losers' source_documents + aliases.
    win_ent = await repo.get_entity(winner)
    assert win_ent is not None
    assert {"src:win", "src:l1", "src:l2"}.issubset(set(win_ent.source_documents))
    assert f"De {tail}" in win_ent.aliases
    assert f"Het {tail}" in win_ent.aliases

    # Relations now reference the winner, never a merged entity.
    dangling = await execute_query(
        "SELECT id FROM relation WHERE in = type::thing($l1) OR out = type::thing($l1) "
        "OR in = type::thing($l2) OR out = type::thing($l2);",
        {"l1": loser1, "l2": loser2},
        config=live_surrealdb,
    )
    assert dangling == [], "no relation may reference a merged loser"

    win_edges = await execute_query(
        "SELECT id FROM relation WHERE in = type::thing($w) OR out = type::thing($w);",
        {"w": winner},
        config=live_surrealdb,
    )
    assert len(win_edges) == 2, "both edges re-pointed onto the winner"

    # entity_alias rows exist for each loser surface form.
    aliases = await execute_query(
        "SELECT alias_text FROM entity_alias "
        "WHERE canonical_entity = type::thing($w);",
        {"w": winner},
        config=live_surrealdb,
    )
    alias_texts = {a["alias_text"] for a in aliases}
    assert f"De {tail}" in alias_texts
    assert f"Het {tail}" in alias_texts


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_apply_merge_idempotent(live_surrealdb: SurrealDBConfig) -> None:
    """Re-applying a cluster whose losers are already merged is a no-op."""
    repo = EntityRepository(config=live_surrealdb)
    svc = RecanonicalizationService(entity_repo=repo)

    tail = _u("Idem")
    winner = await _create_entity(live_surrealdb, tail, "organization", confidence=0.9)
    loser = await _create_entity(live_surrealdb, f"De {tail}", "organization", confidence=0.5)

    plan = await svc.plan_merges()
    cluster = next(c for c in plan.clusters if c.winner_id == winner)

    first = await svc.apply_merge(cluster)
    assert not first.skipped

    # Count alias rows after first apply.
    aliases_after_first = await execute_query(
        "SELECT id FROM entity_alias WHERE canonical_entity = type::thing($w);",
        {"w": winner},
        config=live_surrealdb,
    )

    second = await svc.apply_merge(cluster)
    assert second.skipped is True
    assert second.merged_loser_ids == []

    aliases_after_second = await execute_query(
        "SELECT id FROM entity_alias WHERE canonical_entity = type::thing($w);",
        {"w": winner},
        config=live_surrealdb,
    )
    assert len(aliases_after_second) == len(aliases_after_first), (
        "idempotent re-apply must not create duplicate aliases"
    )
    # Loser still merged into the same winner.
    ent = await repo.get_entity(loser)
    assert ent is not None and ent.status == "merged" and ent.merged_into == winner


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_person_org_homograph_never_merges(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The over-merge guard: a person and an org with the same name stay split."""
    repo = EntityRepository(config=live_surrealdb)
    svc = RecanonicalizationService(entity_repo=repo)

    name = _u("Minister van BZK")
    person = await _create_entity(live_surrealdb, name, "person")
    # Same surface form, different type.
    org = await _create_entity(live_surrealdb, name, "organization")

    plan = await svc.plan_merges()
    for cluster in plan.clusters:
        members = {cluster.winner_id, *cluster.loser_ids}
        assert not (person in members and org in members), (
            "person and org homograph must never share a cluster"
        )
