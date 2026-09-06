"""`name_key` is identity; `canonical_name` is display (PC.3, migration 79).

The defect: `upsert_entity` keys on `(canonical_name, entity_type)` with the raw
surface form, so `Brede Welvaart` and `brede welvaart` are two rows regardless of
whether cross-document resolution runs.

Every fact these tests rest on was measured against a container rather than
assumed, because two of them are counter-intuitive:

* a UNIQUE index treats NONE as a **value**, so a nullable column plus a later
  backfill would reject every second row per type;
* `THROW` inside `IF { }` surfaces only because the runner routes bodies through
  `execute_transaction`; the plain `execute_query` seam returns the first
  statement's result and swallows the rest.
"""

from __future__ import annotations

import uuid

import pytest
from shared.utils.name_normalizer import normalize_entity_name
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query, execute_transaction
from surrealdb_service.testing import fixtures as fx


def _migration(version: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(
        fx._MIGRATIONS_DIR / f"{version}{suffix}.surrealql"
    ).sql


async def _entity(config, name: str, etype: str = "organization") -> str:
    rows = await execute_query(
        "CREATE entity SET canonical_name = $n, name = $n, entity_type = $t, "
        "confidence = 0.5, embedding = [], name_key = $k RETURN id;",
        {"n": name, "t": etype, "k": normalize_entity_name(name)},
        config,
    )
    return str(rows[0]["id"])


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_case_variants_cannot_both_exist(live_surrealdb: SurrealDBConfig) -> None:
    """The point of the phase, at the storage layer.

    Before migration 79 both of these are accepted, because the UNIQUE index is on
    the raw `canonical_name`.
    """
    stem = f"Brede Welvaart {uuid.uuid4().hex[:6]}"
    await _entity(live_surrealdb, stem)

    with pytest.raises(Exception) as excinfo:
        await _entity(live_surrealdb, stem.lower())
    assert "idx_entity_identity" in str(excinfo.value)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_the_display_name_survives(live_surrealdb: SurrealDBConfig) -> None:
    """`canonical_name` is not normalised — the curator and the exports read it.

    The whole reason for a second column rather than normalising in place.
    """
    name = f"Gemeente Súdwest-Fryslân {uuid.uuid4().hex[:6]}"
    row_id = await _entity(live_surrealdb, name)
    rows = await execute_query(
        f"SELECT canonical_name, name_key FROM {row_id};", config=live_surrealdb
    )
    assert rows[0]["canonical_name"] == name
    assert rows[0]["name_key"] == normalize_entity_name(name)
    assert rows[0]["name_key"] != name


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_the_same_key_under_two_types_is_not_a_collision(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The index is on `(name_key, entity_type)`, and that is deliberate.

    PC.2 measured names the graph holds twice under two types and routed them to
    a curator as `fold_equal_cross_type`, review-only, precisely because a
    machine must not decide that one. A UNIQUE index on `name_key` alone would
    decide it at write time.
    """
    stem = f"Regio Deal {uuid.uuid4().hex[:6]}"
    await _entity(live_surrealdb, stem, etype="programme")
    await _entity(live_surrealdb, stem, etype="topic")  # must not raise


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_the_migration_refuses_on_an_unkeyed_row(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The refusal, in the state it claims to prevent — not one of that shape.

    Forged the way migrations 61/64/65's tests forge drift: make the column
    nullable, create a row without a key, then re-run the real migration body
    through the real runner path. Asserting on a hand-written THROW would prove
    nothing about migration 79.
    """
    await execute_query(
        "DEFINE FIELD OVERWRITE name_key ON entity TYPE option<string>;",
        config=live_surrealdb,
    )
    await execute_query(
        "CREATE entity SET canonical_name = $n, name = $n, entity_type = 'org', "
        "confidence = 0.5, embedding = [], name_key = NONE;",
        {"n": f"unkeyed-{uuid.uuid4().hex[:8]}"},
        live_surrealdb,
    )
    try:
        with pytest.raises(RuntimeError) as excinfo:
            await execute_transaction(_migration(79), config=live_surrealdb)
        message = str(excinfo.value)
        assert "migration 79 refuses" in message
        assert "backfill_name_key.py" in message, "the refusal must name the fix"
        assert "curator queue" in message
    finally:
        await execute_query(
            "DELETE entity WHERE name_key = NONE;", config=live_surrealdb
        )
        await execute_query(
            "DEFINE FIELD OVERWRITE name_key ON entity TYPE string;",
            config=live_surrealdb,
        )


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_the_migration_applies_when_every_row_is_keyed(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The counterweight: a refusal that fires on a clean database is a blocker.

    Re-applying the real body over an already-migrated table must be a no-op,
    which is also what makes the migration safe to re-run after a backfill.
    """
    await _entity(live_surrealdb, f"Keyed {uuid.uuid4().hex[:6]}")
    await execute_transaction(_migration(79), config=live_surrealdb)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_down_leaves_the_table_usable(live_surrealdb: SurrealDBConfig) -> None:
    """And the migration-39 index still guards, weakly, after a rollback."""
    name = f"After down {uuid.uuid4().hex[:6]}"
    try:
        await execute_transaction(_migration(79, down=True), config=live_surrealdb)
        await execute_query(
            "CREATE entity SET canonical_name = $n, name = $n, "
            "entity_type = 'org', confidence = 0.5, embedding = [];",
            {"n": name},
            live_surrealdb,
        )
        with pytest.raises(Exception) as excinfo:
            await execute_query(
                "CREATE entity SET canonical_name = $n, name = $n, "
                "entity_type = 'org', confidence = 0.5, embedding = [];",
                {"n": name},
                live_surrealdb,
            )
        assert "idx_entity_name_type" in str(excinfo.value)
    finally:
        # Rows created while the column was gone have no key, so re-applying 79
        # would refuse — which is the migration working, not a fault. The test
        # clears its own rows before restoring the state it borrowed.
        await execute_query(
            "DELETE entity WHERE canonical_name = $n;", {"n": name}, live_surrealdb
        )
        await execute_transaction(_migration(79), config=live_surrealdb)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_upsert_merges_case_variants_into_one_row(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Two spellings, one row — through the production writer.

    Found by mutation: reverting `upsert_entity`'s lookup to `canonical_name`
    broke nothing. It would not have merged; it would have missed, taken the
    create path, and been rejected by the UNIQUE index — turning a merge into an
    exception with nothing to say about why. The index alone cannot prove this,
    because the index only proves the second row is refused.
    """
    from shared.models import Entity
    from surrealdb_service.repositories.entity import EntityRepository

    repo = EntityRepository(config=live_surrealdb)
    stem = f"Brede Welvaart {uuid.uuid4().hex[:6]}"

    first = await repo.upsert_entity(
        Entity(canonical_name=stem, entity_type="topic", confidence=0.6,
               source_documents=["source:a"])
    )
    second = await repo.upsert_entity(
        Entity(canonical_name=stem.lower(), entity_type="topic", confidence=0.9,
               source_documents=["source:b"])
    )

    assert first == second, "the case variant created a second row"

    rows = await execute_query(
        f"SELECT canonical_name, name_key, confidence, source_documents "
        f"FROM {first};", config=live_surrealdb
    )
    row = rows[0]
    # The merge semantics still hold across the two spellings.
    assert row["confidence"] == pytest.approx(0.9)
    assert set(row["source_documents"]) == {"source:a", "source:b"}
    # And the display name is the one written first, not the lowercased variant.
    assert row["canonical_name"] == stem
    assert row["name_key"] == normalize_entity_name(stem)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_upsert_still_separates_genuinely_different_names(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The counterweight: keying on the identity rule must not over-merge.

    `normalize_entity_name` expands curated org aliases, which is a stronger
    transform than a fold — so this is the pair PC.2 warned about, checked at the
    write path rather than assumed from the function's docstring.
    """
    from shared.models import Entity
    from surrealdb_service.repositories.entity import EntityRepository

    repo = EntityRepository(config=live_surrealdb)
    suffix = uuid.uuid4().hex[:6]
    a = await repo.upsert_entity(
        Entity(canonical_name=f"Ministerie van Onderwijs {suffix}",
               entity_type="organization", confidence=0.8)
    )
    b = await repo.upsert_entity(
        Entity(canonical_name=f"Onderwijs {suffix}",
               entity_type="organization", confidence=0.8)
    )
    assert a != b, "two distinct entities were collapsed by the identity rule"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_history_may_share_a_key_with_the_live_row(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A merged tombstone keeps its key; a second LIVE row still cannot exist.

    Found by running the backfill against the working database rather than by
    reasoning: of 854 rows in colliding groups, 571 were `archived` and 75
    `merged` against 103 `active`. A UNIQUE index over every status would have
    rejected the migration for history it should keep — and worse, it would break
    the merge PC.2 ships, which produces exactly this pair: two case variants, one
    surviving active and the other becoming `merged` with the SAME key.

    The index sits on a computed column — the key while active, the row's own id
    otherwise — so non-active rows occupy a namespace of one each.
    """
    stem = f"Brede Welvaart {uuid.uuid4().hex[:6]}"
    live = await _entity(live_surrealdb, stem)

    # The tombstone the merge flow leaves behind: same key, retired status.
    await execute_query(
        "CREATE entity SET canonical_name = $n, name = $n, entity_type = "
        "'organization', confidence = 0.5, embedding = [], name_key = $k, "
        f"status = 'merged', merged_into = '{live}';",
        {"n": stem.lower(), "k": normalize_entity_name(stem)},
        live_surrealdb,
    )
    await execute_query(
        "CREATE entity SET canonical_name = $n, name = $n, entity_type = "
        "'organization', confidence = 0.5, embedding = [], name_key = $k, "
        "status = 'archived';",
        {"n": stem.upper(), "k": normalize_entity_name(stem)},
        live_surrealdb,
    )

    # But a second ACTIVE row with that key is still impossible.
    with pytest.raises(Exception) as excinfo:
        await _entity(live_surrealdb, stem.title())
    assert "idx_entity_identity" in str(excinfo.value)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_retiring_a_row_frees_its_key(live_surrealdb: SurrealDBConfig) -> None:
    """`identity_key` is a VALUE clause, so it is recomputed on every write.

    The transition that matters: a live row is merged away, and the name becomes
    available again. Without recomputation the retired row would hold the key
    forever and the entity could never be re-created — which is what a plain
    stored column would have done.
    """
    stem = f"Regio Deal {uuid.uuid4().hex[:6]}"
    first = await _entity(live_surrealdb, stem)
    await execute_query(
        f"UPDATE {first} SET status = 'merged';", config=live_surrealdb
    )
    second = await _entity(live_surrealdb, stem)
    assert second != first
