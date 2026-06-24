"""Round-trip tests for EntityRepository.upsert_entity (Phase B.1a).

Exercises the canonical write-path against a real SurrealDB container via the
B.0 ``live_surrealdb`` fixture. The migration-44 fields (``type_tags``,
``primary_type``) MUST round-trip; the upsert merge semantics on second-write
must hold (``array::union``, ``math::max``).
"""

from __future__ import annotations

import asyncio
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


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_upsert_roundtrips_created_at_and_updated_at(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """created_at + updated_at must survive the round-trip through the Pydantic model.

    Mirrors the major from the attempt-1 review: ``Entity(**row)`` previously
    dropped the timestamps silently (the inherited ``ObjectModel.created`` /
    ``updated`` field names didn't match the schema's ``created_at`` /
    ``updated_at``). This guards against the regression.

    Flow:
      1. Upsert → assert ``get_entity`` reads back ``created_at`` populated.
      2. Sleep a tick, upsert again with higher confidence → assert
         ``updated_at`` ≥ ``created_at`` (the UPDATE branch refreshes it
         via ``time::now()``).
    """
    repo = EntityRepository(config=live_surrealdb)
    name = _unique("timestamp-roundtrip")

    first = Entity(
        canonical_name=name,
        entity_type="Person",
        confidence=0.5,
        embedding=[],
    )
    record_id = await repo.upsert_entity(first)
    assert record_id

    after_create = await repo.get_entity(record_id)
    assert after_create is not None, "get_entity returned None after CREATE"
    assert after_create.created_at is not None, "created_at should be populated by DB default"
    assert after_create.updated_at is not None, "updated_at should be populated by DB default"
    created_at_initial = after_create.created_at

    # Bump confidence so the merge UPDATE branch fires.
    await asyncio.sleep(0.05)
    second = Entity(
        canonical_name=name,
        entity_type="Person",
        confidence=0.9,
        embedding=[],
    )
    await repo.upsert_entity(second)

    after_update = await repo.get_entity(record_id)
    assert after_update is not None
    assert after_update.created_at == created_at_initial, "created_at must not change on UPDATE"
    assert after_update.updated_at is not None
    assert after_update.updated_at >= created_at_initial, (
        "updated_at should be refreshed on the UPDATE branch"
    )
    assert after_update.confidence == pytest.approx(0.9)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_list_entities_survives_order_by_name_with_fulltext_index(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Regression: ``list_entities`` must return rows despite ``ORDER BY name``.

    Production databases carry a full-text SEARCH index ``idx_entity_fulltext``
    on ``entity`` FIELDS (name, description). SurrealDB v2's query planner cannot
    satisfy ``ORDER BY name`` from a SEARCH index and aborts the transaction with
    "No iterator has been found." ``list_entities`` swallows the error and
    returns ``[]`` — so before the ``WITH NOINDEX`` fix this silently dropped
    every entity, leaving the KG / entity browser empty although the rows exist.

    NOTE: that index is NOT created by the current migration files (it exists
    only as out-of-band schema drift on the live DB), so this test reconstructs
    the exact failing condition: a ``name`` field plus the SEARCH index. The
    ``my_analyzer`` analyzer is defined by migration 1.
    """
    cfg = live_surrealdb

    # Recreate the production drift: a `name` field + full-text SEARCH index.
    # ``OVERWRITE`` because migration 50 now also defines ``name`` on ``entity``.
    await execute_query(
        "DEFINE FIELD OVERWRITE name ON entity TYPE option<string>;",
        {},
        config=cfg,
    )
    await execute_query(
        "DEFINE INDEX idx_entity_fulltext ON entity FIELDS name, description "
        "SEARCH ANALYZER my_analyzer BM25 HIGHLIGHTS;",
        {},
        config=cfg,
    )

    prefix = _unique("listcanary")
    # Insert out of alphabetical order. ``hash_id`` is a required UNIQUE string.
    names = [f"{prefix}-charlie", f"{prefix}-alpha", f"{prefix}-bravo"]
    for n in names:
        await execute_query(
            "CREATE entity SET canonical_name = $n, name = $n, hash_id = $n, "
            "entity_type = 'concept', confidence = 0.9, embedding = [];",
            {"n": n},
            config=cfg,
        )

    # Sanity: the unfixed query pattern MUST fail under this index, otherwise the
    # test is not exercising the bug it claims to guard.
    with pytest.raises(Exception):
        await execute_query(
            "SELECT id, name FROM entity ORDER BY name LIMIT 10", {}, config=cfg
        )

    repo = EntityRepository(config=cfg)

    rows = await repo.list_entities(limit=500, offset=0)
    got = [r["name"] for r in rows if str(r.get("name", "")).startswith(prefix)]
    assert set(got) == set(names), (
        f"list_entities dropped rows (ORDER BY name vs FTS index). "
        f"Missing: {set(names) - set(got)}"
    )
    assert got == sorted(names), f"rows not ascending by name: {got}"

    # The type-filtered branch travels the same planner path and must also hold.
    filtered = await repo.list_entities(limit=500, offset=0, entity_type="concept")
    fgot = [r["name"] for r in filtered if str(r.get("name", "")).startswith(prefix)]
    assert set(fgot) == set(names), (
        f"type-filtered list_entities dropped rows. Missing: {set(names) - set(fgot)}"
    )


@pytest.mark.requires_docker
async def test_list_entities_status_filter(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Q.5: the optional ``status`` filter restricts the page to that status.

    Seeds an ``active`` and a ``reference`` entity and asserts each status filter
    returns only its own rows, that ``count_entities`` agrees with the filtered
    page, and that the projection surfaces ``status`` for the KG table.
    """
    cfg = live_surrealdb
    repo = EntityRepository(config=cfg)
    prefix = _unique("statusfilter")

    active_name = f"{prefix}-active"
    reference_name = f"{prefix}-reference"
    await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, hash_id=$n, "
        "entity_type='concept', confidence=0.9, embedding=[], status='active';",
        {"n": active_name},
        config=cfg,
    )
    await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, hash_id=$n, "
        "entity_type='concept', confidence=0.9, embedding=[], status='reference';",
        {"n": reference_name},
        config=cfg,
    )

    ref_rows = await repo.list_entities(limit=500, offset=0, status="reference")
    ref_names = {
        r["name"] for r in ref_rows if str(r.get("name", "")).startswith(prefix)
    }
    assert ref_names == {reference_name}
    assert all(r["status"] == "reference" for r in ref_rows)

    active_rows = await repo.list_entities(limit=500, offset=0, status="active")
    active_names = {
        r["name"] for r in active_rows if str(r.get("name", "")).startswith(prefix)
    }
    assert active_names == {active_name}

    # Count agrees with the filtered page (scoped to this test's prefix is not
    # possible via count, so assert the reference count is >= 1 and that an
    # unmatched status returns 0 for a freshly-unique status value).
    assert await repo.count_entities(status="reference") >= 1
    assert await repo.count_entities(status=f"nonexistent-{prefix}") == 0


# --- P.1: entity-embedding backfill primitives --------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_backfill_lists_missing_and_updates_embedding(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """P.1 backfill roundtrip: list entities with an empty embedding, update one,
    confirm it drops out of the missing set and the vector round-trips.

    Covers the idempotency contract: a row that already has a vector is excluded
    from ``list_active_entities_missing_embedding`` so a re-run only ever touches
    the not-yet-done remainder.
    """
    repo = EntityRepository(config=live_surrealdb)
    missing_name = _unique("p1-missing")
    has_vec_name = _unique("p1-hasvec")
    vector = [0.01 * i for i in range(8)]

    # One active entity with embedding=[] (the backfill target) and one that
    # already carries a vector (must be skipped — idempotency).
    await execute_query(
        "CREATE entity SET canonical_name = $n, name = $n, hash_id = $n, "
        "entity_type = 'concept', confidence = 0.9, status = 'active', "
        "embedding = [];",
        {"n": missing_name},
        config=live_surrealdb,
    )
    await execute_query(
        "CREATE entity SET canonical_name = $n, name = $n, hash_id = $n, "
        "entity_type = 'concept', confidence = 0.9, status = 'active', "
        "embedding = $emb;",
        {"n": has_vec_name, "emb": [0.5, 0.5]},
        config=live_surrealdb,
    )

    rows = await repo.list_active_entities_missing_embedding()
    by_name = {r["canonical_name"]: r for r in rows}
    assert missing_name in by_name, "empty-embedding entity not listed for backfill"
    assert has_vec_name not in by_name, "entity WITH a vector must be skipped"

    target_id = str(by_name[missing_name]["id"])
    ok = await repo.update_entity_embedding(target_id, vector)
    assert ok is True

    # Vector round-trips verbatim.
    fetched = await repo.get_entity_with_embedding(target_id)
    assert fetched is not None
    assert [pytest.approx(v) for v in fetched["embedding"]] == [
        pytest.approx(v) for v in vector
    ]

    # And the entity no longer appears in the missing set (re-run is a no-op
    # for it — idempotency).
    rows_after = await repo.list_active_entities_missing_embedding()
    assert missing_name not in {r["canonical_name"] for r in rows_after}


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_update_entity_embedding_rejects_empty_vector(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """An empty vector is rejected (it would be a no-op leaving the row 'missing')."""
    repo = EntityRepository(config=live_surrealdb)
    assert await repo.update_entity_embedding("entity:whatever", []) is False
    assert await repo.update_entity_embedding("", [0.1]) is False
