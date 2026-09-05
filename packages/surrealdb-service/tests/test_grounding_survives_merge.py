"""Mention provenance survives the persist boundary, per source (PC.3 step 5).

`source_chunk_id`, `source_grounding` and `extraction_context` are first-class
fields on `ExtractedEntity`, and `persist_filtered_result` reads only
`entity["properties"]` — so all three were dropped. PC.1b's inventory named that
boundary; this phase makes it matter, because with cross-document resolution on
one row now answers for mentions in several documents and "why is this one
entity" is unanswerable without knowing where each mention was found.

The merge rule is the load-bearing half. A canonical row is merged across
documents while grounding is per mention, so storing it flat would mean the last
document to mention an entity overwrites where every earlier one found it — and
with resolution on that is the normal case, not an edge one.
"""

from __future__ import annotations

import uuid

import pytest
from shared.models import Entity
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository


def _entity(name: str, source: str, page: int) -> Entity:
    return Entity(
        canonical_name=name,
        entity_type="topic",
        confidence=0.8,
        source_documents=[source],
        properties={
            "grounding": {source: {"chunk_id": f"chunk:{page}", "grounding": {"page": page}}}
        },
    )


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_two_documents_both_keep_their_grounding(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The case cross-document resolution creates, and the one a flat key loses."""
    repo = EntityRepository(config=live_surrealdb)
    name = f"Brede Welvaart {uuid.uuid4().hex[:6]}"

    first = await repo.upsert_entity(_entity(name, "source:a", 3))
    second = await repo.upsert_entity(_entity(name, "source:b", 7))
    assert first == second, "setup: the two mentions did not resolve to one row"

    rows = await execute_query(
        f"SELECT properties, source_documents FROM {first};", config=live_surrealdb
    )
    grounding = rows[0]["properties"]["grounding"]
    assert set(grounding) == {"source:a", "source:b"}, (
        "a document's grounding was overwritten by the next one — the merge is "
        "an overlay, not a union by source"
    )
    assert grounding["source:a"]["grounding"]["page"] == 3
    assert grounding["source:b"]["grounding"]["page"] == 7
    assert set(rows[0]["source_documents"]) == {"source:a", "source:b"}


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_a_second_write_from_the_same_source_replaces_its_own_entry(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Re-extracting one document updates that document's entry, not the others.

    The counterweight to the union: without it, a re-run would accumulate stale
    grounding for a source whose chunks have since changed.
    """
    repo = EntityRepository(config=live_surrealdb)
    name = f"Regio Deal {uuid.uuid4().hex[:6]}"

    await repo.upsert_entity(_entity(name, "source:a", 3))
    await repo.upsert_entity(_entity(name, "source:b", 7))
    row_id = await repo.upsert_entity(_entity(name, "source:a", 99))

    rows = await execute_query(
        f"SELECT properties FROM {row_id};", config=live_surrealdb
    )
    grounding = rows[0]["properties"]["grounding"]
    assert grounding["source:a"]["grounding"]["page"] == 99, "own entry not updated"
    assert grounding["source:b"]["grounding"]["page"] == 7, "other source disturbed"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_an_entity_without_grounding_is_unaffected(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The merge rule must not invent a key, or every row grows one.

    Extractors that supply no grounding are normal — the single-schema path does
    not always carry it — and a row that gains an empty `grounding` would make the
    field's presence meaningless as a signal.
    """
    repo = EntityRepository(config=live_surrealdb)
    name = f"Plain {uuid.uuid4().hex[:6]}"
    row_id = await repo.upsert_entity(
        Entity(canonical_name=name, entity_type="topic", confidence=0.8,
               source_documents=["source:a"], properties={"x": 1})
    )
    await repo.upsert_entity(
        Entity(canonical_name=name, entity_type="topic", confidence=0.9,
               source_documents=["source:b"], properties={"y": 2})
    )
    rows = await execute_query(
        f"SELECT properties FROM {row_id};", config=live_surrealdb
    )
    assert "grounding" not in rows[0]["properties"]
    # And the ordinary overlay still works for everything else.
    assert rows[0]["properties"]["x"] == 1 and rows[0]["properties"]["y"] == 2
