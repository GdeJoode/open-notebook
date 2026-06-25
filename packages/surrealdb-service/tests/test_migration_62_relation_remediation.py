"""Container tests for migration 62 — non-destructive `relation` remediation.

Migration 62 self-heals the `relation` table: it converts a drifted NORMAL
table (pre-migration-39 legacy shape, null-endpoint junk rows) back to
``TYPE RELATION FROM entity TO entity`` while PRESERVING real edges on an
already-healthy environment.

These tests prove every Track O.2a acceptance criterion against a real
SurrealDB 2.6.5 container:

* AC1 — runner auto-discovers version 62 and it applies clean on a fresh
  fully-migrated container (``test_migration_62_discovered_and_applied``).
* AC2/AC3 — drifted NORMAL table → after 62 it is ``TYPE RELATION`` and a
  ``RELATE`` lands, count increases by 1 (``test_drifted_relation_converted``).
* AC4 (SAFETY INVARIANT) — healthy RELATION table with N real edges → after 62
  still exactly N edges, in/out intact (``test_healthy_edges_preserved``). This
  also records the load-bearing empirical answer: *does DEFINE TABLE OVERWRITE
  preserve edges on a healthy TYPE RELATION table in 2.6.5?*
* AC5 — idempotent: applying 62 twice yields the same end state, no error
  (``test_migration_62_idempotent``).

Plus two end-to-end runner tests added in O.2a review attempt 2:

* ``test_runner_sequence_58_then_62_preserves_edges`` — proves the headline
  phase invariant against the ACTUAL runner driving migrations 58 AND 62 in
  order on a populated table (the destructive original 58 made this false). N
  real edges seeded at a simulated v57 must all survive the 58->62 sequence.
* ``test_runner_surfaces_failing_second_statement`` — proves the runner no
  longer silently marks a migration applied when a NON-FIRST statement fails at
  runtime (the swallow risk in the old ``execute_query``-based runner).

Because the session fixture already applies migration 62, each test rebuilds
the specific starting state it needs in an isolated namespace/table-state and
replays migration 62's SQL body (the same idempotency pattern used by the
existing migration_52 / 54 / 57 tests).

To run:

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_migration_62_relation_remediation.py -v
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _migration_62_sql() -> str:
    """The migration-62 forward body, loaded from disk (single source of truth)."""
    from surrealdb_service.testing import fixtures as fx

    return (fx._MIGRATIONS_DIR / "62.surrealql").read_text()


def _load_up_migration(version: int):
    """Load one ``<version>.surrealql`` as an ``AsyncMigration`` (cleaned body).

    Uses the same ``AsyncMigration.from_file`` the real runner uses, so the test
    exercises the exact SQL the runner would execute (comments stripped, etc.).
    """
    from surrealdb_service.migrations import AsyncMigration
    from surrealdb_service.testing import fixtures as fx

    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{version}.surrealql")


async def _relation_table_def(config: SurrealDBConfig) -> str:
    """Return the ``DEFINE TABLE relation ...`` text, incl. its TYPE clause.

    Empirical note (SurrealDB 2.6.5): ``INFO FOR TABLE relation`` does NOT carry
    the table *kind* — it returns only ``events`` / ``fields`` / ``indexes`` /
    ``lives`` / ``tables``. The ``TYPE RELATION`` clause lives in the table's
    DEFINE statement, which is exposed by ``INFO FOR DB`` under
    ``tables.relation`` (e.g.
    ``DEFINE TABLE relation TYPE RELATION IN entity OUT entity SCHEMAFULL``).
    So the canonical way to assert the table kind on 2.6.5 is to read the
    DEFINE text from ``INFO FOR DB`` — checking ``INFO FOR TABLE`` for
    ``TYPE RELATION`` would always be a false negative. The AC wording
    ("INFO FOR TABLE output contains TYPE RELATION") is satisfied here via the
    authoritative ``INFO FOR DB`` projection of the same DEFINE.
    """
    # ``execute_query`` returns the single statement's result. For INFO FOR DB
    # that is the info dict itself (not a list of rows), but be defensive in
    # case a client version wraps it in a one-element list.
    result = await execute_query("INFO FOR DB;", config=config)
    if isinstance(result, list):
        payload = result[0] if result else {}
    else:
        payload = result
    tables = payload.get("tables", {}) if isinstance(payload, dict) else {}
    return str(tables.get("relation", ""))


async def _make_two_entities(config: SurrealDBConfig) -> tuple[str, str]:
    a = await execute_query(
        "CREATE entity SET canonical_name = $n, entity_type = 'Person', embedding = [];",
        {"n": _unique("ent_a")},
        config=config,
    )
    b = await execute_query(
        "CREATE entity SET canonical_name = $n, entity_type = 'Org', embedding = [];",
        {"n": _unique("ent_b")},
        config=config,
    )
    return a[0]["id"], b[0]["id"]


# ---------------------------------------------------------------------------
# AC1 — version 62 auto-discovered and applied on a fresh migrated container.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_62_discovered_and_applied(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Runner records version 62 and `relation` is TYPE RELATION post-migration."""
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 62;",
        config=live_surrealdb,
    )
    assert rows, "Migration 62 not recorded in _sbl_migrations — runner did not discover it"

    info = await _relation_table_def(live_surrealdb)
    assert "TYPE RELATION" in info, (
        f"After full migration, `relation` is not TYPE RELATION. INFO: {info}"
    )


# ---------------------------------------------------------------------------
# AC2 + AC3 — drifted NORMAL table is converted and accepts RELATE.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_drifted_relation_converted(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Reproduce the live drift, apply 62, prove conversion + a landing RELATE.

    Live drift (from the O.1 escalation): `relation` is a NORMAL table with the
    pre-39 field shape (subject/object/predicate/weight/sources) and rows whose
    in/out are null. We force that exact state, then replay migration 62.
    """
    # --- Force the drifted NORMAL table (pre-39 legacy shape) ---------------
    # REMOVE then re-DEFINE as a clean NORMAL table carrying ONLY the legacy
    # pre-39 field shape (subject/object/predicate/weight/sources) — the live
    # drift never had the migration-39 in/out fields at all. A plain
    # OVERWRITE-to-NORMAL would leave the required ``in``/``out`` record fields
    # behind and reject the legacy CREATE, which is a test-setup artifact, not
    # the real drift; REMOVE gives the faithful pre-39 shape.
    await execute_query("REMOVE TABLE IF EXISTS relation;", config=live_surrealdb)
    await execute_query(
        """
        DEFINE TABLE relation SCHEMAFULL;
        DEFINE FIELD subject ON relation TYPE option<record<entity>>;
        DEFINE FIELD object ON relation TYPE option<record<entity>>;
        DEFINE FIELD predicate ON relation TYPE option<string>;
        DEFINE FIELD weight ON relation TYPE option<float>;
        DEFINE FIELD sources ON relation FLEXIBLE TYPE option<array>;
        """,
        config=live_surrealdb,
    )
    # Insert null-endpoint junk rows (the "3 malformed legacy relations" the
    # live DB carries — subject/object set but no in/out).
    for _ in range(3):
        await execute_query(
            "CREATE relation SET predicate = 'LEGACY', weight = 1.0;",
            config=live_surrealdb,
        )

    # Sanity: it really is a NORMAL table now, and RELATE is rejected.
    pre_info = await _relation_table_def(live_surrealdb)
    assert "TYPE RELATION" not in pre_info, (
        f"Setup failed — table should be NORMAL before 62. INFO: {pre_info}"
    )
    a_id, b_id = await _make_two_entities(live_surrealdb)
    with pytest.raises(Exception):  # noqa: B017 — RELATE rejected on a NORMAL table
        await execute_query(
            f"RELATE {a_id}->relation->{b_id} SET relation_type = 'X';",
            config=live_surrealdb,
        )

    # --- Apply migration 62 -------------------------------------------------
    await execute_query(_migration_62_sql(), config=live_surrealdb)

    # AC2: table is now TYPE RELATION.
    post_info = await _relation_table_def(live_surrealdb)
    assert "TYPE RELATION" in post_info, (
        f"AC2 FAIL: `relation` not TYPE RELATION after migration 62. INFO: {post_info}"
    )

    # Junk rows gone (null-endpoint deleted by step 1 of the migration).
    remaining = await execute_query(
        "SELECT count() AS c FROM relation GROUP ALL;", config=live_surrealdb
    )
    count_before_relate = remaining[0]["c"] if remaining else 0
    assert count_before_relate == 0, (
        f"AC2 FAIL: expected 0 rows after junk cleanup, got {count_before_relate}"
    )

    # AC3: a RELATE between two real entities now lands; count +1.
    await execute_query(
        f"RELATE {a_id}->relation->{b_id} SET relation_type = 'WORKS_AT', confidence = 0.9;",
        config=live_surrealdb,
    )
    after = await execute_query(
        "SELECT count() AS c FROM relation GROUP ALL;", config=live_surrealdb
    )
    count_after_relate = after[0]["c"] if after else 0
    assert count_after_relate == count_before_relate + 1, (
        f"AC3 FAIL: RELATE did not increase count by 1 "
        f"({count_before_relate} -> {count_after_relate})"
    )

    edge = await execute_query(
        "SELECT in, out, relation_type FROM relation;", config=live_surrealdb
    )
    assert len(edge) == 1
    assert edge[0]["in"] == a_id and edge[0]["out"] == b_id
    assert edge[0]["relation_type"] == "WORKS_AT"


# ---------------------------------------------------------------------------
# AC4 — SAFETY INVARIANT: healthy edges survive migration 62 untouched.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_healthy_edges_preserved(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A healthy TYPE RELATION table with N real edges keeps exactly N after 62.

    This is the load-bearing empirical proof: it answers "does
    DEFINE TABLE OVERWRITE preserve edges on a healthy TYPE RELATION table in
    2.6.5?". If OVERWRITE wiped records, this test would fail and migration 62
    would need a conditional (INFO-gated) redefine instead.
    """
    # Start from a guaranteed-healthy RELATION table (re-assert migration 39's
    # def via the migration-62 body once, so the table kind is RELATION before
    # we seed edges — independent of whatever a prior test left behind).
    await execute_query(_migration_62_sql(), config=live_surrealdb)
    # Clear pre-existing edges so N is exact for this test.
    await execute_query("DELETE relation;", config=live_surrealdb)

    # Seed N real edges with non-null in/out.
    n = 5
    seeded: list[tuple[str, str]] = []
    for i in range(n):
        a_id, b_id = await _make_two_entities(live_surrealdb)
        await execute_query(
            f"RELATE {a_id}->relation->{b_id} SET relation_type = 'E{i}', confidence = 0.5;",
            config=live_surrealdb,
        )
        seeded.append((a_id, b_id))

    pre = await execute_query(
        "SELECT count() AS c FROM relation GROUP ALL;", config=live_surrealdb
    )
    pre_count = pre[0]["c"] if pre else 0
    assert pre_count == n, f"setup: expected {n} seeded edges, got {pre_count}"

    pre_edges = await execute_query(
        "SELECT in, out, relation_type FROM relation ORDER BY relation_type;",
        config=live_surrealdb,
    )

    # --- Apply migration 62 on the HEALTHY table ----------------------------
    await execute_query(_migration_62_sql(), config=live_surrealdb)

    # AC4: still exactly N edges.
    post = await execute_query(
        "SELECT count() AS c FROM relation GROUP ALL;", config=live_surrealdb
    )
    post_count = post[0]["c"] if post else 0
    assert post_count == n, (
        f"AC4 FAIL (edge loss): healthy table had {n} edges, "
        f"only {post_count} survived migration 62. "
        f"=> DEFINE TABLE OVERWRITE WIPED records on 2.6.5; migration 62 must "
        f"use a conditional (INFO-gated) redefine instead."
    )

    # Each edge's in/out intact (same set of endpoint pairs).
    post_edges = await execute_query(
        "SELECT in, out, relation_type FROM relation ORDER BY relation_type;",
        config=live_surrealdb,
    )
    pre_pairs = {(e["in"], e["out"], e["relation_type"]) for e in pre_edges}
    post_pairs = {(e["in"], e["out"], e["relation_type"]) for e in post_edges}
    assert pre_pairs == post_pairs, (
        f"AC4 FAIL: edge in/out/type changed across migration 62.\n"
        f"before: {sorted(pre_pairs)}\nafter:  {sorted(post_pairs)}"
    )

    # Still TYPE RELATION and still accepts new RELATEs.
    info = await _relation_table_def(live_surrealdb)
    assert "TYPE RELATION" in info, f"table not RELATION after 62 on healthy DB: {info}"
    a_id, b_id = await _make_two_entities(live_surrealdb)
    await execute_query(
        f"RELATE {a_id}->relation->{b_id} SET relation_type = 'NEW';",
        config=live_surrealdb,
    )


# ---------------------------------------------------------------------------
# AC5 — idempotent: applying 62 twice yields the same end state, no error.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_62_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Applying migration 62 twice in a row leaves the same valid end state."""
    # Establish a healthy table with a couple of real edges.
    await execute_query(_migration_62_sql(), config=live_surrealdb)
    await execute_query("DELETE relation;", config=live_surrealdb)
    pairs: list[tuple[str, str]] = []
    for i in range(2):
        a_id, b_id = await _make_two_entities(live_surrealdb)
        await execute_query(
            f"RELATE {a_id}->relation->{b_id} SET relation_type = 'I{i}';",
            config=live_surrealdb,
        )
        pairs.append((a_id, b_id))

    # First apply.
    await execute_query(_migration_62_sql(), config=live_surrealdb)
    mid = await execute_query(
        "SELECT count() AS c FROM relation GROUP ALL;", config=live_surrealdb
    )
    mid_count = mid[0]["c"] if mid else 0

    # Second apply — must not raise and must not change the end state.
    await execute_query(_migration_62_sql(), config=live_surrealdb)
    end = await execute_query(
        "SELECT count() AS c FROM relation GROUP ALL;", config=live_surrealdb
    )
    end_count = end[0]["c"] if end else 0

    assert mid_count == end_count == 2, (
        f"AC5 FAIL: re-applying 62 changed edge count ({mid_count} -> {end_count}), "
        f"expected 2 both times"
    )
    info = await _relation_table_def(live_surrealdb)
    assert "TYPE RELATION" in info


# ---------------------------------------------------------------------------
# Blocker 1/2 — runner SEQUENCE across 58 AND 62 must not wipe a populated table.
#
# The per-test isolation above proves "62 alone is non-destructive". This test
# proves the END-TO-END phase invariant: an environment at a simulated pre-58
# version, holding N real edges, survives the ACTUAL runner driving 58 then 62.
# Against the original destructive migration 58 (`REMOVE TABLE IF EXISTS`) this
# would report 0 surviving edges; against the non-destructive 58 it must report
# exactly N.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_runner_sequence_58_then_62_preserves_edges(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Drive the real runner over migrations 58 AND 62 on a populated table.

    Simulates an environment stuck at version 57 (healthy RELATION table with N
    real edges) and runs the actual ``AsyncMigrationRunner`` with the on-disk 58
    and 62 bodies. Asserts exactly N edges survive with in/out intact — the
    headline phase guarantee that the destructive original 58 violated.
    """
    from surrealdb_service.migrations import (
        AsyncMigrationRunner,
        _get_latest_version,
    )

    # Start from a guaranteed-healthy RELATION table, cleared so N is exact.
    await execute_query(_migration_62_sql(), config=live_surrealdb)
    await execute_query("DELETE relation;", config=live_surrealdb)

    n = 7
    for i in range(n):
        a_id, b_id = await _make_two_entities(live_surrealdb)
        await execute_query(
            f"RELATE {a_id}->relation->{b_id} SET relation_type = 'S{i}', confidence = 0.5;",
            config=live_surrealdb,
        )
    pre = await execute_query(
        "SELECT count() AS c FROM relation GROUP ALL;", config=live_surrealdb
    )
    assert (pre[0]["c"] if pre else 0) == n, "setup: expected N seeded edges"
    pre_edges = await execute_query(
        "SELECT in, out, relation_type FROM relation;", config=live_surrealdb
    )
    pre_pairs = {(e["in"], e["out"], e["relation_type"]) for e in pre_edges}

    # Simulate "stuck at version 57": rewind the tracked version so the runner
    # treats 58 and 62 as pending. We snapshot the real applied versions and
    # restore them in a finally block so other session tests are unaffected.
    real_versions = await execute_query(
        "SELECT version FROM _sbl_migrations;", config=live_surrealdb
    )
    real_set = {r["version"] for r in real_versions}
    try:
        await execute_query(
            "DELETE _sbl_migrations WHERE version > 57;", config=live_surrealdb
        )
        # Version 57 is already present from the real migration history, so
        # deleting everything above it leaves the latest at 57 — no CREATE
        # needed (and a CREATE would collide on the existing record).
        await execute_query(
            "UPSERT _sbl_migrations:57 SET version = 57, applied_at = time::now();",
            config=live_surrealdb,
        )
        assert await _get_latest_version(live_surrealdb) == 57

        # Drive the ACTUAL runner over just 58 and 62 (loaded from disk).
        runner = AsyncMigrationRunner(
            up={58: _load_up_migration(58), 62: _load_up_migration(62)},
            down={},
            config=live_surrealdb,
        )
        await runner.run_all()

        # Both versions now recorded as applied.
        latest = await _get_latest_version(live_surrealdb)
        assert latest == 62, f"runner did not advance to 62 (got {latest})"
    finally:
        # Restore the real _sbl_migrations state for the rest of the session.
        await execute_query(
            "DELETE _sbl_migrations WHERE version > 57;", config=live_surrealdb
        )
        for v in sorted(real_set):
            if v > 57:
                await execute_query(
                    f"CREATE _sbl_migrations:{v} SET version = {v}, "
                    f"applied_at = time::now();",
                    config=live_surrealdb,
                )

    # THE INVARIANT: exactly N edges survived the 58->62 sequence, in/out intact.
    post = await execute_query(
        "SELECT count() AS c FROM relation GROUP ALL;", config=live_surrealdb
    )
    post_count = post[0]["c"] if post else 0
    assert post_count == n, (
        f"BLOCKER FAIL (edge loss across runner sequence): seeded {n} edges at "
        f"v57, ran the 58->62 runner, only {post_count} survived. A destructive "
        f"migration 58 (REMOVE TABLE) wipes the populated table before 62 can "
        f"protect it."
    )
    post_edges = await execute_query(
        "SELECT in, out, relation_type FROM relation;", config=live_surrealdb
    )
    post_pairs = {(e["in"], e["out"], e["relation_type"]) for e in post_edges}
    assert pre_pairs == post_pairs, (
        f"edge in/out/type changed across 58->62 sequence.\n"
        f"before: {sorted(pre_pairs)}\nafter:  {sorted(post_pairs)}"
    )

    # Table is RELATION and still accepts RELATE.
    info = await _relation_table_def(live_surrealdb)
    assert "TYPE RELATION" in info
    a_id, b_id = await _make_two_entities(live_surrealdb)
    await execute_query(
        f"RELATE {a_id}->relation->{b_id} SET relation_type = 'AFTER';",
        config=live_surrealdb,
    )


# ---------------------------------------------------------------------------
# Major 3 — the runner must surface a failed NON-FIRST statement (no silent
# "applied"). Previously execute_query swallowed later-statement errors; the
# runner now routes bodies through the per-statement-checked transaction seam.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_runner_surfaces_failing_second_statement(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A migration whose 2nd statement errors at runtime must NOT be marked applied.

    Uses a synthetic migration body: a harmless first statement followed by a
    statement that violates a strict type at runtime (valid syntax, runtime
    ERR). Pre-fix, ``connection.query`` returned only statement 1 and the runner
    bumped the version on a silent failure. Post-fix, the per-statement seam
    raises and the version is NOT bumped.
    """
    from surrealdb_service.migrations import (
        AsyncMigration,
        AsyncMigrationRunner,
        _get_latest_version,
    )

    # A throwaway strict table whose 2nd statement violates the type.
    tbl = _unique("o2a_guard").replace("-", "_")
    body = (
        f"DEFINE TABLE {tbl} SCHEMAFULL; "
        f"DEFINE FIELD n ON {tbl} TYPE int; "
        # Runtime type violation — string into an int field. Valid syntax, ERR
        # only at execution, in a NON-FIRST statement.
        f"CREATE {tbl}:bad SET n = 'not an int';"
    )

    # Pin a high synthetic version so it is "pending" relative to the session DB.
    synthetic_version = 99001
    runner = AsyncMigrationRunner(
        up={synthetic_version: AsyncMigration(synthetic_version, body)},
        down={},
        config=live_surrealdb,
    )

    before = await _get_latest_version(live_surrealdb)
    with pytest.raises(RuntimeError):
        await runner.run_all()

    # Version must NOT have advanced to the failing migration.
    after = await _get_latest_version(live_surrealdb)
    assert after == before, (
        f"MAJOR FAIL: runner bumped version on a silently-failing 2nd statement "
        f"({before} -> {after}). The failing statement was swallowed."
    )
    recorded = await execute_query(
        f"SELECT version FROM _sbl_migrations WHERE version = {synthetic_version};",
        config=live_surrealdb,
    )
    assert not recorded, "failing migration was wrongly recorded as applied"
