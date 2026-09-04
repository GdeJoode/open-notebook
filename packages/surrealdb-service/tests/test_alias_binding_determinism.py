"""`find_by_alias` must resolve one surface form to one canonical (PC.2).

Nothing constrains ``entity_alias.alias_text`` to be unique, and `register_alias`
writes a row per resolution, so two rows can bind the same text to two different
entities. With a bare ``LIMIT 1`` the caller gets whichever row the storage engine
returns — the same input can then resolve to a different entity between two runs,
and neither answer looks any less confident than the other.

These tests run against a real container because the ordering they assert is
performed by SurrealDB. A unit test could only assert that the SQL string contains
``ORDER BY``, which passes just as well when the clause names the wrong columns.
"""

from __future__ import annotations

import uuid

import pytest
from shared.models import Entity
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository


async def _entity(config: SurrealDBConfig, name: str) -> str:
    """Create a canonical entity through the PRODUCTION writer.

    Not a hand-rolled ``CREATE``: the entity table is SCHEMAFULL and a fixture
    that assembles its own field set drifts from what `upsert_entity` actually
    writes, so the test would pass against rows no run can produce.
    """
    entity = Entity(
        canonical_name=name,
        entity_type="organization",
        confidence=0.9,
    )
    return str(await EntityRepository(config=config).upsert_entity(entity))


async def _alias(
    config: SurrealDBConfig,
    text: str,
    canonical: str,
    score: float,
    verified: bool,
) -> None:
    # `canonical_entity` is `record<entity>`; a bound string parameter is rejected,
    # so the validated id is interpolated. `canonical` here is always an id this
    # test just created.
    await execute_query(
        f"CREATE entity_alias SET alias_text = $t, canonical_entity = {canonical}, "
        "match_type = 'exact', similarity_score = $s, verified = $v;",
        {"t": text, "s": score, "v": verified},
        config,
    )


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_verified_alias_outranks_a_higher_scoring_unverified_one(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A human decision beats a machine one, even at a much lower score.

    The scores are deliberately adverse: ordering by ``similarity_score`` alone
    would pick the unverified row. Measured on the live database before the fix,
    the unordered query returned exactly that row.
    """
    text = f"probe-{uuid.uuid4().hex[:10]}"
    machine = await _entity(live_surrealdb, f"{text}-machine")
    human = await _entity(live_surrealdb, f"{text}-human")
    await _alias(live_surrealdb, text, machine, 0.99, verified=False)
    await _alias(live_surrealdb, text, human, 0.50, verified=True)

    repo = EntityRepository(config=live_surrealdb)
    got = await repo.find_by_alias(text)
    assert got is not None
    assert str(got["id"]) == human
    assert got["verified"] is True


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_repeated_lookups_return_the_same_canonical(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Same input, same answer — every time, not usually.

    Three rows at the SAME trust level and the SAME score, so ``verified`` and
    ``similarity_score`` both tie and only the ``id`` tie-break can decide. Without
    it this is where the non-determinism lives.
    """
    text = f"probe-{uuid.uuid4().hex[:10]}"
    ids = [await _entity(live_surrealdb, f"{text}-{i}") for i in range(3)]
    for eid in ids:
        await _alias(live_surrealdb, text, eid, 0.80, verified=False)

    repo = EntityRepository(config=live_surrealdb)
    answers = {str((await repo.find_by_alias(text))["id"]) for _ in range(12)}
    assert len(answers) == 1, f"non-deterministic alias binding: {answers}"
    assert answers == {min(ids)}, "the tie-break must be the lowest id, not chance"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_an_unverified_alias_still_resolves_when_it_is_all_there_is(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Ranking, not filtering.

    Every writer today writes ``verified = false``, so filtering unverified rows
    out would make tier-1 alias resolution inert — the failure mode PC.1 shipped a
    whole phase to undo. This test is what stops the ordering being tightened into
    a filter later.
    """
    text = f"probe-{uuid.uuid4().hex[:10]}"
    only = await _entity(live_surrealdb, f"{text}-only")
    await _alias(live_surrealdb, text, only, 0.42, verified=False)

    got = await EntityRepository(config=live_surrealdb).find_by_alias(text)
    assert got is not None and str(got["id"]) == only


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_alias_provenance_survives_a_fresh_migrated_database(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Everything `register_alias` writes must still be there when read back.

    The drift this pins is the reason migration 78 exists. `entity_alias` is
    SCHEMAFULL; migration 39 declared five fields and `register_alias` inserts
    nine, and SurrealDB drops the undeclared ones **silently** — no error, no log,
    a row that looks fine and has lost how it was matched, how strongly, by what
    method, and whether a human confirmed it.

    It is invisible to any probe against `staging`, where the table predates being
    schema-locked and all nine columns exist. So this test asserts the round trip
    through the PRODUCTION writer against a FRESH container, which is the only
    place the loss is observable.
    """
    text = f"probe-{uuid.uuid4().hex[:10]}"
    canonical = await _entity(live_surrealdb, f"{text}-canonical")
    repo = EntityRepository(config=live_surrealdb)
    assert await repo.register_alias(
        alias_text=text,
        canonical_entity_id=canonical,
        match_type="fuzzy",
        similarity_score=0.87,
        method="levenshtein",
    )

    rows = await execute_query(
        "SELECT * FROM entity_alias WHERE alias_text = $t;", {"t": text}, live_surrealdb
    )
    assert len(rows) == 1
    row = rows[0]
    # Field by field, because a single missing one is the whole failure.
    assert row["match_type"] == "fuzzy", "match_type dropped by the schema"
    assert row["similarity_score"] == pytest.approx(0.87), "similarity_score dropped"
    assert row["method"] == "levenshtein", "method dropped by the schema"
    assert row["verified"] is False, "verified dropped by the schema"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_a_row_that_predates_the_field_is_repaired_not_bricked(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """An `entity_alias` row written before migration 78 must stay writable.

    The drift class migrations 61, 64 and 65 exist for: a SurrealDB column
    DEFAULT applies only to NEWLY created rows, so a row predating the DEFINE
    keeps NONE, and a strict type then rejects the WHOLE record on the next
    UPDATE because a SCHEMAFULL update re-validates every field. `entity_alias`
    has four live UPDATE paths — the K.2 duplicate-merge alias transfer (twice),
    the K.3 apply, and the vault round trip — every one of which would have
    started failing on pre-78 rows had 78 shipped as a bare DEFINE.

    The previous test covers a NEWLY created row, which is exactly the case a
    DEFAULT does handle, so it cannot see this. Here the legacy timeline is
    replayed with the migration-64/65 forging technique — OVERWRITE the field as
    `option<>`, create the row at NONE, re-DEFINE it strict without backfilling —
    and then the migration's own coalescing UPDATE is run against it.
    """
    text = f"probe-{uuid.uuid4().hex[:10]}"
    canonical = await _entity(live_surrealdb, f"{text}-canonical")
    rid = f"entity_alias:{uuid.uuid4().hex[:12]}"

    await execute_query(
        "DEFINE FIELD OVERWRITE verified ON entity_alias TYPE option<bool>;",
        config=live_surrealdb,
    )
    await execute_query(
        f"CREATE {rid} SET alias_text = $t, canonical_entity = {canonical}, "
        "verified = NONE;",
        {"t": text},
        live_surrealdb,
    )
    await execute_query(
        "DEFINE FIELD OVERWRITE verified ON entity_alias TYPE bool DEFAULT false;",
        config=live_surrealdb,
    )

    none_rows = await execute_query(
        f"SELECT type::is::none(verified) AS isnone FROM {rid};", config=live_surrealdb
    )
    assert none_rows and none_rows[0]["isnone"] is True, "setup: expected NONE"

    # Pre-repair: the K.2 alias transfer — a real production write — is blocked.
    with pytest.raises(Exception) as excinfo:
        await execute_query(
            f"UPDATE {rid} SET canonical_entity = {canonical};", config=live_surrealdb
        )
    assert "verified" in str(excinfo.value), (
        f"expected the strict `verified` field to be what blocks the write, got: "
        f"{excinfo.value}"
    )

    # The migration's own repair line, run verbatim.
    await execute_query(
        "UPDATE entity_alias SET verified = verified ?? false;", config=live_surrealdb
    )

    await execute_query(
        f"UPDATE {rid} SET canonical_entity = {canonical};", config=live_surrealdb
    )
    rows = await execute_query(f"SELECT verified FROM {rid};", config=live_surrealdb)
    assert rows[0]["verified"] is False, "the repaired row must read as unverified"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_the_repair_line_is_in_the_migration_and_is_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The coalesce must be in migration 78 itself, not only in this test.

    The test above would pass just as well if the repair existed nowhere but the
    test body. Migration 65's guard sweep runs BEFORE 78 and so cannot cover a
    field defined by it, which is why the line has to live here.
    """
    from surrealdb_service.testing import fixtures as fx

    body = (fx._MIGRATIONS_DIR / "78.surrealql").read_text()
    sql = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("--")
    )
    assert "verified = verified ?? false" in sql, (
        "migration 78 declares a strict field without repairing the rows that "
        "predate it — the drift class migrations 61/64/65 already fixed twice"
    )

    # Idempotent: running it twice on a clean database changes nothing.
    for _ in range(2):
        await execute_query(
            "UPDATE entity_alias SET verified = verified ?? false;",
            config=live_surrealdb,
        )
