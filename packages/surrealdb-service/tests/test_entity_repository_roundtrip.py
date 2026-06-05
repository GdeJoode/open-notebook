"""Round-trip tests for EntityRepository.upsert_entity (Phase B.1a).

Exercises the canonical write-path against a real SurrealDB container via the
B.0 ``live_surrealdb`` fixture. The migration-44 fields (``type_tags``,
``primary_type``) MUST round-trip; the upsert merge semantics on second-write
must hold (``array::union``, ``math::max``).
"""

from __future__ import annotations

import uuid

import pytest

from shared.models import Entity
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_upsert_creates_entity_with_type_tags(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Create path: type_tags + primary_type land verbatim, schema field names align."""
    repo = EntityRepository(config=live_surrealdb)
    name = _unique("type-tags-create")

    ent = Entity(
        canonical_name=name,
        entity_type="Person",
        type_tags=["Person", "Researcher"],
        primary_type="Researcher",
        confidence=0.87,
        description="Round-trip canary for B.1a",
        source_documents=["source:doc-a"],
        embedding=[],
        properties={"affiliation": "ACME Labs"},
    )
    record_id = await repo.upsert_entity(ent)
    assert record_id, "upsert_entity returned an empty record id"

    rows = await execute_query(
        "SELECT * FROM entity WHERE canonical_name = $name;",
        {"name": name},
        config=live_surrealdb,
    )
    assert len(rows) == 1, f"Expected exactly 1 row, got {len(rows)}"
    row = rows[0]
    assert row["canonical_name"] == name
    assert row["entity_type"] == "Person"
    assert row["description"] == "Round-trip canary for B.1a"
    # Migration-44 additive fields
    assert sorted(row["type_tags"]) == ["Person", "Researcher"]
    assert row["primary_type"] == "Researcher"
    # Confidence + properties survive
    assert row["confidence"] == pytest.approx(0.87)
    assert row["properties"]["affiliation"] == "ACME Labs"
    # Default-typed fields populated by SurrealDB
    assert row["status"] == "active"
    assert row["extraction_method"] == "llm"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_upsert_merges_on_second_call(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Update path: second upsert merges type_tags / source_documents / confidence."""
    repo = EntityRepository(config=live_surrealdb)
    name = _unique("merge-test")

    # First write
    first = Entity(
        canonical_name=name,
        entity_type="Org",
        type_tags=["Org"],
        primary_type="Org",
        confidence=0.6,
        source_documents=["source:doc-1"],
        embedding=[],
        properties={"key_a": "value_a"},
    )
    await repo.upsert_entity(first)

    # Second write — higher confidence, new type_tag, new source, new property
    second = Entity(
        canonical_name=name,
        entity_type="Org",
        type_tags=["Org", "GovAgency"],
        primary_type="GovAgency",
        confidence=0.92,
        source_documents=["source:doc-2"],
        embedding=[],
        properties={"key_b": "value_b"},
    )
    await repo.upsert_entity(second)

    rows = await execute_query(
        "SELECT * FROM entity WHERE canonical_name = $name;",
        {"name": name},
        config=live_surrealdb,
    )
    assert len(rows) == 1, "Upsert produced a duplicate row instead of merging"
    row = rows[0]

    # array::union dedups + accumulates
    assert sorted(row["type_tags"]) == ["GovAgency", "Org"]
    assert sorted(row["source_documents"]) == ["source:doc-1", "source:doc-2"]
    # math::max(confidence, $confidence): keep the higher value
    assert row["confidence"] == pytest.approx(0.92)
    # primary_type replaced with the new value
    assert row["primary_type"] == "GovAgency"
    # object::extend keeps existing keys and adds new ones
    assert row["properties"]["key_a"] == "value_a"
    assert row["properties"]["key_b"] == "value_b"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_upsert_handles_empty_embedding(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The SCHEMAFULL ``entity.embedding`` field has no default — empty list must work."""
    repo = EntityRepository(config=live_surrealdb)
    name = _unique("empty-embedding")

    ent = Entity(
        canonical_name=name,
        entity_type="Concept",
        embedding=[],
    )
    record_id = await repo.upsert_entity(ent)
    assert record_id

    rows = await execute_query(
        "SELECT canonical_name, embedding FROM entity WHERE canonical_name = $name;",
        {"name": name},
        config=live_surrealdb,
    )
    assert len(rows) == 1
    assert rows[0]["embedding"] == []
