"""Q-B1 part 1: `sources_for_entities` against a real SurrealDB.

Pins what the hover on *Docs* depends on: the entity's own `source_documents`
order, distinct; titles resolved from `source`; a document without a `source`
row still listed (by id); an entity without documents mapping to [].
"""

from __future__ import annotations

import uuid

import pytest
from app_main.services.triage.queue_sources import entity_types, sources_for_entities
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


async def _source(config, title: str) -> str:
    rows = await execute_query("CREATE source SET title = $t RETURN VALUE id;",
                               {"t": title}, config=config)
    return str(rows[0])


async def _entity(config, docs) -> str:
    name = f"Org-{uuid.uuid4().hex[:8]}"
    rows = await execute_query(
        "CREATE entity SET canonical_name=$n, name_key=string::lowercase($n), name=$n, "
        "entity_type='organization', embedding=[], confidence=0.9, status='active', "
        "source_documents = $docs RETURN VALUE id;",
        {"n": name, "docs": docs}, config=config,
    )
    return str(rows[0])


async def test_titles_in_the_entitys_order_distinct(live_surrealdb: SurrealDBConfig):
    s1 = await _source(live_surrealdb, "49b9320d0e1e__Convenant Regio Deal Zuidwest-Friesland")
    s2 = await _source(live_surrealdb, "5753cb08f382__Convenant Regio Deal Het Hogeland")
    ghost = f"source:ghost{uuid.uuid4().hex[:6]}"  # a provenance id with no row
    e1 = await _entity(live_surrealdb, [s2, s1, s2, ghost])
    e2 = await _entity(live_surrealdb, [])

    out = await sources_for_entities([e1, e2], config=live_surrealdb)
    assert [d["id"] for d in out[e1]] == [s2, s1, ghost]
    assert [d["title"] for d in out[e1]] == [
        "5753cb08f382__Convenant Regio Deal Het Hogeland",
        "49b9320d0e1e__Convenant Regio Deal Zuidwest-Friesland",
        "",
    ]
    assert out[e2] == []


async def test_no_entities_no_query(live_surrealdb: SurrealDBConfig):
    assert await sources_for_entities([], config=live_surrealdb) == {}


async def test_entity_types_are_the_canonical_types(live_surrealdb: SurrealDBConfig):
    """WB.3: the triage table's type is the entity's ``entity_type``, not the
    label the row stored; a deleted entity is absent."""
    e1 = await _entity(live_surrealdb, [])
    gone = "entity:gone" + uuid.uuid4().hex[:6]
    out = await entity_types([e1, gone, e1], config=live_surrealdb)
    assert out == {e1: "organization"}
    assert await entity_types([], config=live_surrealdb) == {}
