"""Roundtrip tests for ``EntityRepository.get_all_entities_and_relations``.

These guard the knowledge-graph ``/graph`` endpoint against the regression
that returned **0 edges** on a real KG: the old implementation sliced an
arbitrary ``LIMIT`` of *all* entities (archived included) and then asked for
relations whose both endpoints fell inside that arbitrary slice — which is
~never true on a large graph.

The rewrite is edge-first + active-only, so a mock would hide the very bug we
care about. These tests therefore seed a real mini-KG via the ``live_surrealdb``
container fixture (mirrors ``test_relation_endpoint_resolution_roundtrip.py``)
and assert:

* active relations among active/reference entities are returned as edges,
* every edge endpoint is present in the node set (so the graph renders),
* archived / merged entities and their edges are excluded.
"""

from __future__ import annotations

import uuid

import pytest

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


async def _create_entity(
    config: SurrealDBConfig,
    name: str,
    *,
    entity_type: str = "person",
    status: str = "active",
    weight: float = 1.0,
) -> str:
    """Create an entity and return its record-id string."""
    rows = await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type=$t, "
        "embedding=[], confidence=0.9, source_documents=[], "
        "status=$s, weight=$w;",
        {"n": name, "t": entity_type, "s": status, "w": weight},
        config,
    )
    return str(rows[0]["id"])


async def _relate(
    config: SurrealDBConfig,
    src: str,
    tgt: str,
    *,
    relation_type: str = "knows",
    status: str = "active",
) -> None:
    # Bare record-id literals in the RELATE arrow (SurrealDB issue #4232); the
    # ids come straight from the DB, so interpolation is safe here.
    await execute_query(
        f"RELATE {src}->relation->{tgt} SET relation_type=$rt, "
        "confidence=0.9, status=$s, properties={};",
        {"rt": relation_type, "s": status},
        config,
    )


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_active_relation_returns_edge_with_both_endpoints(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """An active relation among active entities surfaces as an edge whose
    endpoints are both present in the node set."""
    repo = EntityRepository(config=live_surrealdb)
    a = await _create_entity(live_surrealdb, _u("alice"), weight=5.0)
    b = await _create_entity(live_surrealdb, _u("bob"), weight=3.0)
    await _relate(live_surrealdb, a, b, relation_type="works_with")

    result = await repo.get_all_entities_and_relations(limit=500)

    node_ids = {str(n["id"]) for n in result["nodes"]}
    edges = [
        e
        for e in result["edges"]
        if {str(e["source"]), str(e["target"])} == {a, b}
    ]
    assert edges, "active relation must surface as an edge (the 0-edge regression)"
    assert edges[0]["relation_type"] == "works_with"
    # Endpoint guarantee: every edge endpoint is in the node set so it renders.
    for edge in result["edges"]:
        assert str(edge["source"]) in node_ids
        assert str(edge["target"]) in node_ids


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_archived_endpoints_excluded(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A relation touching an archived entity is dropped, and the archived
    node never appears in the node set."""
    repo = EntityRepository(config=live_surrealdb)
    live = await _create_entity(live_surrealdb, _u("live"), weight=4.0)
    archived = await _create_entity(
        live_surrealdb, _u("archived"), status="archived", weight=9.0
    )
    await _relate(live_surrealdb, live, archived, relation_type="cites")

    result = await repo.get_all_entities_and_relations(limit=500)

    node_ids = {str(n["id"]) for n in result["nodes"]}
    assert archived not in node_ids, "archived entity must be excluded from nodes"
    bad = [
        e
        for e in result["edges"]
        if archived in {str(e["source"]), str(e["target"])}
    ]
    assert not bad, "edge touching an archived endpoint must be excluded"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_inactive_relation_excluded(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A relation with status != 'active' (e.g. archived) is not returned even
    when both endpoints are active entities."""
    repo = EntityRepository(config=live_surrealdb)
    a = await _create_entity(live_surrealdb, _u("p1"), weight=2.0)
    b = await _create_entity(live_surrealdb, _u("p2"), weight=2.0)
    await _relate(live_surrealdb, a, b, status="archived")

    result = await repo.get_all_entities_and_relations(limit=500)

    inactive = [
        e
        for e in result["edges"]
        if {str(e["source"]), str(e["target"])} == {a, b}
    ]
    assert not inactive, "non-active relation must be excluded from edges"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_entity_type_filter_constrains_both_endpoints(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """With an entity_type filter, only relations whose BOTH endpoints match
    the type are returned."""
    repo = EntityRepository(config=live_surrealdb)
    person_a = await _create_entity(
        live_surrealdb, _u("pa"), entity_type="person", weight=3.0
    )
    person_b = await _create_entity(
        live_surrealdb, _u("pb"), entity_type="person", weight=3.0
    )
    org = await _create_entity(
        live_surrealdb, _u("org"), entity_type="organization", weight=3.0
    )
    await _relate(live_surrealdb, person_a, person_b)  # person<->person: kept
    await _relate(live_surrealdb, person_a, org)  # mixed: dropped under filter

    result = await repo.get_all_entities_and_relations(
        entity_type="person", limit=500
    )

    kept = [
        e
        for e in result["edges"]
        if {str(e["source"]), str(e["target"])} == {person_a, person_b}
    ]
    mixed = [
        e
        for e in result["edges"]
        if org in {str(e["source"]), str(e["target"])}
    ]
    assert kept, "person<->person edge must survive the type filter"
    assert not mixed, "person<->organization edge must be filtered out"
    assert all(n["entity_type"] == "person" for n in result["nodes"])
