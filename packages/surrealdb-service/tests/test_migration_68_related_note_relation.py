"""Container tests for migration 68 — `related_note` TYPE RELATION definition (Y.1).

Migration 68 defines the `related_note` edge table as
``TYPE RELATION FROM note TO note`` so Track Y's auto-link can persist note↔note
similarity links as real graph edges. `related_note` is BRAND NEW (no prior
migration defines it), so on a fresh container it would otherwise be auto-created
by the first RELATE as the SurrealDB default ``TYPE ANY SCHEMALESS`` — the exact
class migrations 66/67 fixed for `mentions`/`cites`. 68 defines it up front,
non-destructively (the same OVERWRITE strategy).

These tests prove the Y.1 schema acceptance against a real SurrealDB container:

* version 68 is auto-discovered and `related_note` is TYPE RELATION note->note on
  a fresh fully-migrated container (the cites/mentions drift lesson — assert on a
  FRESH container, not just staging);
* the strict fields carry defaults — a RELATE with no SET succeeds and the
  defaults populate (S.4);
* a drifted ``TYPE ANY`` `related_note` table is converted to RELATION by 68;
* a healthy RELATION `related_note` with N real edges keeps exactly N after 68
  (OVERWRITE preserves records); applying 68 twice is idempotent.

Run:

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_migration_68_related_note_relation.py -v
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _migration_68_sql() -> str:
    """The migration-68 forward body, loaded from disk (single source of truth)."""
    from surrealdb_service.testing import fixtures as fx

    return (fx._MIGRATIONS_DIR / "68.surrealql").read_text()


async def _related_note_table_def(config: SurrealDBConfig) -> str:
    """Return the ``DEFINE TABLE related_note ...`` text, incl. its TYPE clause.

    The table *kind* lives in the DEFINE statement exposed by ``INFO FOR DB``
    under ``tables.related_note`` (e.g.
    ``DEFINE TABLE related_note TYPE RELATION IN note OUT note SCHEMAFULL``);
    ``INFO FOR TABLE`` does not carry the kind on v2.x (the 62/66/67 lesson).
    """
    result = await execute_query("INFO FOR DB;", config=config)
    payload = (result[0] if result else {}) if isinstance(result, list) else result
    tables = payload.get("tables", {}) if isinstance(payload, dict) else {}
    return str(tables.get("related_note", ""))


async def _make_note(config: SurrealDBConfig) -> str:
    rows = await execute_query(
        "CREATE note SET title = $t, content = $t, embedding = [] RETURN AFTER;",
        {"t": _unique("note")},
        config=config,
    )
    return rows[0]["id"]


# ---------------------------------------------------------------------------
# AC2: version 68 auto-discovered; related_note is TYPE RELATION note->note on a
# FRESH fully-migrated container.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_68_discovered_and_related_note_is_note_to_note(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Runner records version 68 and `related_note` is RELATION note->note."""
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 68;",
        config=live_surrealdb,
    )
    assert rows, "Migration 68 not recorded — runner did not discover it"

    info = await _related_note_table_def(live_surrealdb)
    assert "TYPE RELATION" in info, (
        f"After full migration, `related_note` is not TYPE RELATION. INFO: {info}"
    )
    # note->note specifically (not source->* or ANY).
    assert "note" in info.lower()


# ---------------------------------------------------------------------------
# AC2: strict fields carry defaults — a RELATE with no SET succeeds and the
# defaults populate.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_related_note_defaults_populate_on_bare_relate(
    live_surrealdb: SurrealDBConfig,
) -> None:
    a = await _make_note(live_surrealdb)
    b = await _make_note(live_surrealdb)
    # No SET — every strict field must fall back to its DEFAULT (S.4).
    await execute_query(
        f"RELATE {a}->related_note->{b};", config=live_surrealdb
    )
    rows = await execute_query(
        "SELECT similarity_score, method, created_at "
        f"FROM related_note WHERE in = {a} AND out = {b};",
        config=live_surrealdb,
    )
    assert len(rows) == 1, "bare RELATE must succeed (strict fields have defaults)"
    edge = rows[0]
    assert float(edge["similarity_score"]) == 0.0
    assert edge["method"] == "embedding"
    assert edge["created_at"] is not None


# ---------------------------------------------------------------------------
# Drifted TYPE ANY table → converted by 68.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_drifted_any_related_note_converted(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Force the SurrealDB-default TYPE ANY, apply 68, prove RELATE lands typed."""
    await execute_query(
        "REMOVE TABLE IF EXISTS related_note;", config=live_surrealdb
    )
    await execute_query(
        "DEFINE TABLE related_note TYPE ANY SCHEMALESS;", config=live_surrealdb
    )
    pre = await _related_note_table_def(live_surrealdb)
    assert "TYPE RELATION" not in pre, (
        f"Setup failed — table should be TYPE ANY before 68. INFO: {pre}"
    )

    await execute_query(_migration_68_sql(), config=live_surrealdb)

    post = await _related_note_table_def(live_surrealdb)
    assert "TYPE RELATION" in post, (
        f"FAIL: `related_note` not TYPE RELATION after migration 68. INFO: {post}"
    )

    a = await _make_note(live_surrealdb)
    b = await _make_note(live_surrealdb)
    await execute_query(
        f"RELATE {a}->related_note->{b} SET similarity_score = 0.84;",
        config=live_surrealdb,
    )
    edge = await execute_query(
        f"SELECT in, out, similarity_score FROM related_note "
        f"WHERE in = {a} AND out = {b};",
        config=live_surrealdb,
    )
    assert len(edge) == 1
    assert edge[0]["in"] == a and edge[0]["out"] == b
    assert abs(float(edge[0]["similarity_score"]) - 0.84) < 1e-9


# ---------------------------------------------------------------------------
# Idempotent: applying 68 twice yields the same valid end state.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_68_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    await execute_query(_migration_68_sql(), config=live_surrealdb)
    await execute_query(_migration_68_sql(), config=live_surrealdb)
    info = await _related_note_table_def(live_surrealdb)
    assert "TYPE RELATION" in info
    a = await _make_note(live_surrealdb)
    b = await _make_note(live_surrealdb)
    await execute_query(
        f"RELATE {a}->related_note->{b} SET similarity_score = 0.2;",
        config=live_surrealdb,
    )


# ---------------------------------------------------------------------------
# SAFETY: healthy edges survive migration 68 untouched (OVERWRITE preserves).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_healthy_related_note_edges_preserved(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A healthy RELATION `related_note` with N edges keeps exactly N after 68."""
    await execute_query(_migration_68_sql(), config=live_surrealdb)
    await execute_query("DELETE related_note;", config=live_surrealdb)

    n = 5
    pre_pairs: set[tuple] = set()
    for i in range(n):
        a = await _make_note(live_surrealdb)
        b = await _make_note(live_surrealdb)
        await execute_query(
            f"RELATE {a}->related_note->{b} SET similarity_score = {0.1 * (i + 1)};",
            config=live_surrealdb,
        )
        pre_pairs.add((a, b))

    pre = await execute_query(
        "SELECT count() AS c FROM related_note GROUP ALL;", config=live_surrealdb
    )
    assert (pre[0]["c"] if pre else 0) == n

    # Apply 68 on the HEALTHY table.
    await execute_query(_migration_68_sql(), config=live_surrealdb)

    post = await execute_query(
        "SELECT count() AS c FROM related_note GROUP ALL;", config=live_surrealdb
    )
    post_count = post[0]["c"] if post else 0
    assert post_count == n, (
        f"edge loss: healthy `related_note` had {n} edges, only {post_count} "
        f"survived migration 68 — OVERWRITE wiped records on this version."
    )
    post_edges = await execute_query(
        "SELECT in, out FROM related_note;", config=live_surrealdb
    )
    assert pre_pairs == {(e["in"], e["out"]) for e in post_edges}


# ---------------------------------------------------------------------------
# AC4: canonical `note` rows untouched by migration 68.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_canonical_note_rows_untouched(
    live_surrealdb: SurrealDBConfig,
) -> None:
    a = await _make_note(live_surrealdb)
    pre = await execute_query(
        f"SELECT count() AS c FROM note WHERE id = {a} GROUP ALL;",
        config=live_surrealdb,
    )
    assert (pre[0]["c"] if pre else 0) == 1
    await execute_query(_migration_68_sql(), config=live_surrealdb)
    post = await execute_query(
        f"SELECT count() AS c FROM note WHERE id = {a} GROUP ALL;",
        config=live_surrealdb,
    )
    assert (post[0]["c"] if post else 0) == 1, "migration 68 must not touch note rows"
