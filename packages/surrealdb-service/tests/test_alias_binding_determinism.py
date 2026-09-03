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
