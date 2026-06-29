"""Container tests for migration 70 — `note_about` TYPE RELATION definition (NS.1).

Migration 70 defines the `note_about` edge table as
``TYPE RELATION FROM note TO source`` so Track NS (note→source auto-link) can
persist a note↔source link ("this note is about this source", found by embedding
cosine) as a real graph edge — alongside the note↔note `related_note` (68) and
source↔source `source_verdict` (69) edges. `note_about` is BRAND NEW (no prior
migration defines it), so on a fresh container it would otherwise be auto-created
by the first RELATE as the SurrealDB default ``TYPE ANY SCHEMALESS`` — the exact
drift class migrations 66/67/68/69 fixed. 70 defines it up front, non-destructively
(the same OVERWRITE strategy).

These tests prove the NS.1 schema acceptance against a real SurrealDB container:

* version 70 is auto-discovered and `note_about` is TYPE RELATION note->source on
  a FRESH fully-migrated container (the cites/mentions drift lesson — assert on a
  FRESH container, not just staging);
* the strict fields carry defaults — a bare RELATE with no SET succeeds and the
  defaults populate (S.4);
* a drifted ``TYPE ANY`` `note_about` table is converted to RELATION by 70;
* a healthy RELATION `note_about` with N real edges keeps exactly N after 70
  (OVERWRITE preserves records); applying 70 twice is idempotent;
* canonical `note` and `source` rows are untouched by migration 70.

Run:

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_migration_70_note_about_relation.py -v
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _migration_70_sql() -> str:
    """The migration-70 forward body, loaded from disk (single source of truth)."""
    from surrealdb_service.testing import fixtures as fx

    return (fx._MIGRATIONS_DIR / "70.surrealql").read_text()


async def _table_def(config: SurrealDBConfig, table: str) -> str:
    """Return the ``DEFINE TABLE <table> ...`` text, incl. its TYPE clause.

    The table *kind* lives in the DEFINE statement exposed by ``INFO FOR DB``
    under ``tables.<table>`` (e.g.
    ``DEFINE TABLE note_about TYPE RELATION IN note OUT source SCHEMAFULL``);
    ``INFO FOR TABLE`` does not carry the kind on v2.x (the 62/66/67/68/69 lesson).
    """
    result = await execute_query("INFO FOR DB;", config=config)
    payload = (result[0] if result else {}) if isinstance(result, list) else result
    tables = payload.get("tables", {}) if isinstance(payload, dict) else {}
    return str(tables.get(table, ""))


async def _make_note(config: SurrealDBConfig) -> str:
    """Create a note (strict `embedding` requires a value — use `[]`)."""
    rows = await execute_query(
        "CREATE note SET title = $t, content = $t, embedding = [] RETURN AFTER;",
        {"t": _unique("note")},
        config=config,
    )
    return rows[0]["id"]


async def _make_source(config: SurrealDBConfig) -> str:
    rows = await execute_query(
        "CREATE source SET title = $t RETURN AFTER;",
        {"t": _unique("source")},
        config=config,
    )
    return rows[0]["id"]


# ---------------------------------------------------------------------------
# AC2: version 70 auto-discovered; note_about is TYPE RELATION note->source
# on a FRESH fully-migrated container.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_70_discovered_and_note_about_is_note_to_source(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Runner records version 70 and `note_about` is RELATION note->source."""
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 70;",
        config=live_surrealdb,
    )
    assert rows, "Migration 70 not recorded — runner did not discover it"

    info = await _table_def(live_surrealdb, "note_about")
    assert "TYPE RELATION" in info, (
        f"After full migration, `note_about` is not TYPE RELATION. INFO: {info}"
    )
    # note->source specifically (IN note OUT source).
    lower = info.lower()
    assert "note" in lower and "source" in lower, (
        f"`note_about` must be note->source. INFO: {info}"
    )


# ---------------------------------------------------------------------------
# AC2: strict fields carry defaults — a bare RELATE with no SET succeeds and the
# defaults populate.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_note_about_defaults_populate_on_bare_relate(
    live_surrealdb: SurrealDBConfig,
) -> None:
    note = await _make_note(live_surrealdb)
    src = await _make_source(live_surrealdb)
    # No SET — every strict field must fall back to its DEFAULT (S.4).
    await execute_query(
        f"RELATE {note}->note_about->{src};", config=live_surrealdb
    )
    rows = await execute_query(
        "SELECT similarity_score, method, created_at "
        f"FROM note_about WHERE in = {note} AND out = {src};",
        config=live_surrealdb,
    )
    assert len(rows) == 1, "bare RELATE must succeed (strict fields have defaults)"
    edge = rows[0]
    assert float(edge["similarity_score"]) == 0.0
    assert edge["method"] == "embedding"
    assert edge["created_at"] is not None


# ---------------------------------------------------------------------------
# Drifted TYPE ANY table → converted by 70.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_drifted_any_note_about_converted(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Force the SurrealDB-default TYPE ANY, apply 70, prove RELATE lands typed."""
    await execute_query(
        "REMOVE TABLE IF EXISTS note_about;", config=live_surrealdb
    )
    await execute_query(
        "DEFINE TABLE note_about TYPE ANY SCHEMALESS;", config=live_surrealdb
    )
    pre = await _table_def(live_surrealdb, "note_about")
    assert "TYPE RELATION" not in pre, (
        f"Setup failed — table should be TYPE ANY before 70. INFO: {pre}"
    )

    await execute_query(_migration_70_sql(), config=live_surrealdb)

    post = await _table_def(live_surrealdb, "note_about")
    assert "TYPE RELATION" in post, (
        f"FAIL: `note_about` not TYPE RELATION after migration 70. INFO: {post}"
    )

    note = await _make_note(live_surrealdb)
    src = await _make_source(live_surrealdb)
    await execute_query(
        f"RELATE {note}->note_about->{src} SET similarity_score = 0.8;",
        config=live_surrealdb,
    )
    edge = await execute_query(
        f"SELECT in, out, similarity_score FROM note_about "
        f"WHERE in = {note} AND out = {src};",
        config=live_surrealdb,
    )
    assert len(edge) == 1
    assert edge[0]["in"] == note and edge[0]["out"] == src
    assert abs(float(edge[0]["similarity_score"]) - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# Idempotent: applying 70 twice yields the same valid end state.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_70_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    await execute_query(_migration_70_sql(), config=live_surrealdb)
    await execute_query(_migration_70_sql(), config=live_surrealdb)
    info = await _table_def(live_surrealdb, "note_about")
    assert "TYPE RELATION" in info
    note = await _make_note(live_surrealdb)
    src = await _make_source(live_surrealdb)
    await execute_query(
        f"RELATE {note}->note_about->{src} SET similarity_score = 0.9;",
        config=live_surrealdb,
    )


# ---------------------------------------------------------------------------
# SAFETY: healthy edges survive migration 70 untouched (OVERWRITE preserves).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_healthy_note_about_edges_preserved(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A healthy RELATION `note_about` with N edges keeps exactly N after 70."""
    await execute_query(_migration_70_sql(), config=live_surrealdb)
    await execute_query("DELETE note_about;", config=live_surrealdb)

    n = 5
    pre_pairs: set[tuple] = set()
    for i in range(n):
        note = await _make_note(live_surrealdb)
        src = await _make_source(live_surrealdb)
        await execute_query(
            f"RELATE {note}->note_about->{src} "
            f"SET similarity_score = {0.1 * (i + 1)};",
            config=live_surrealdb,
        )
        pre_pairs.add((note, src))

    pre = await execute_query(
        "SELECT count() AS c FROM note_about GROUP ALL;", config=live_surrealdb
    )
    assert (pre[0]["c"] if pre else 0) == n

    # Apply 70 on the HEALTHY table.
    await execute_query(_migration_70_sql(), config=live_surrealdb)

    post = await execute_query(
        "SELECT count() AS c FROM note_about GROUP ALL;", config=live_surrealdb
    )
    post_count = post[0]["c"] if post else 0
    assert post_count == n, (
        f"edge loss: healthy `note_about` had {n} edges, only {post_count} "
        f"survived migration 70 — OVERWRITE wiped records on this version."
    )
    post_edges = await execute_query(
        "SELECT in, out FROM note_about;", config=live_surrealdb
    )
    assert pre_pairs == {(e["in"], e["out"]) for e in post_edges}


# ---------------------------------------------------------------------------
# AC4: canonical `note` and `source` rows untouched by migration 70.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_canonical_note_and_source_rows_untouched(
    live_surrealdb: SurrealDBConfig,
) -> None:
    note = await _make_note(live_surrealdb)
    src = await _make_source(live_surrealdb)
    for rid in (note, src):
        pre = await execute_query(
            f"SELECT count() AS c FROM (SELECT id FROM {rid}) GROUP ALL;",
            config=live_surrealdb,
        )
        assert (pre[0]["c"] if pre else 0) == 1

    await execute_query(_migration_70_sql(), config=live_surrealdb)

    for rid, table in ((note, "note"), (src, "source")):
        post = await execute_query(
            f"SELECT count() AS c FROM (SELECT id FROM {rid}) GROUP ALL;",
            config=live_surrealdb,
        )
        assert (post[0]["c"] if post else 0) == 1, (
            f"migration 70 must not touch {table} rows"
        )
