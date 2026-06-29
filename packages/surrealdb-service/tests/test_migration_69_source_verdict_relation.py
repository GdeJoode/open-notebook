"""Container tests for migration 69 — `source_verdict` TYPE RELATION definition (Z.1).

Migration 69 defines the `source_verdict` edge table as
``TYPE RELATION FROM source TO source`` so Track Z (contradiction detection) can
persist a source↔source verdict (an LLM judge's reinforces/contradicts/neutral
judgment of two RELATED sources) as a real graph edge. `source_verdict` is BRAND
NEW (no prior migration defines it), so on a fresh container it would otherwise
be auto-created by the first RELATE as the SurrealDB default ``TYPE ANY
SCHEMALESS`` — the exact drift class migrations 66/67/68 fixed for
`mentions`/`cites`/`related_note`. 69 defines it up front, non-destructively (the
same OVERWRITE strategy).

These tests prove the Z.1 schema acceptance against a real SurrealDB container:

* version 69 is auto-discovered and `source_verdict` is TYPE RELATION
  source->source on a FRESH fully-migrated container (the cites/mentions drift
  lesson — assert on a FRESH container, not just staging);
* the strict fields carry defaults — a bare RELATE with no SET succeeds and the
  defaults populate (S.4);
* a drifted ``TYPE ANY`` `source_verdict` table is converted to RELATION by 69;
* a healthy RELATION `source_verdict` with N real edges keeps exactly N after 69
  (OVERWRITE preserves records); applying 69 twice is idempotent;
* the app-side `claim`/`contradicts` scaffolding is NOT broken by 69 (the
  reconciliation: source_verdict is a separate, cleaner source↔source edge).

Run:

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_migration_69_source_verdict_relation.py -v
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _migration_69_sql() -> str:
    """The migration-69 forward body, loaded from disk (single source of truth)."""
    from surrealdb_service.testing import fixtures as fx

    return (fx._MIGRATIONS_DIR / "69.surrealql").read_text()


async def _table_def(config: SurrealDBConfig, table: str) -> str:
    """Return the ``DEFINE TABLE <table> ...`` text, incl. its TYPE clause.

    The table *kind* lives in the DEFINE statement exposed by ``INFO FOR DB``
    under ``tables.<table>`` (e.g.
    ``DEFINE TABLE source_verdict TYPE RELATION IN source OUT source SCHEMAFULL``);
    ``INFO FOR TABLE`` does not carry the kind on v2.x (the 62/66/67/68 lesson).
    """
    result = await execute_query("INFO FOR DB;", config=config)
    payload = (result[0] if result else {}) if isinstance(result, list) else result
    tables = payload.get("tables", {}) if isinstance(payload, dict) else {}
    return str(tables.get(table, ""))


async def _make_source(config: SurrealDBConfig) -> str:
    rows = await execute_query(
        "CREATE source SET title = $t RETURN AFTER;",
        {"t": _unique("source")},
        config=config,
    )
    return rows[0]["id"]


# ---------------------------------------------------------------------------
# AC1: version 69 auto-discovered; source_verdict is TYPE RELATION source->source
# on a FRESH fully-migrated container.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_69_discovered_and_source_verdict_is_source_to_source(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Runner records version 69 and `source_verdict` is RELATION source->source."""
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 69;",
        config=live_surrealdb,
    )
    assert rows, "Migration 69 not recorded — runner did not discover it"

    info = await _table_def(live_surrealdb, "source_verdict")
    assert "TYPE RELATION" in info, (
        f"After full migration, `source_verdict` is not TYPE RELATION. INFO: {info}"
    )
    # source->source specifically (not source->claim or ANY).
    assert "source" in info.lower()
    assert "claim" not in info.lower(), (
        f"`source_verdict` must be source->source, not source->claim. INFO: {info}"
    )


# ---------------------------------------------------------------------------
# AC1: strict fields carry defaults — a bare RELATE with no SET succeeds and the
# defaults populate.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_source_verdict_defaults_populate_on_bare_relate(
    live_surrealdb: SurrealDBConfig,
) -> None:
    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)
    # No SET — every strict field must fall back to its DEFAULT (S.4).
    await execute_query(
        f"RELATE {a}->source_verdict->{b};", config=live_surrealdb
    )
    rows = await execute_query(
        "SELECT verdict, confidence, reasoning, judge_model, created_at "
        f"FROM source_verdict WHERE in = {a} AND out = {b};",
        config=live_surrealdb,
    )
    assert len(rows) == 1, "bare RELATE must succeed (strict fields have defaults)"
    edge = rows[0]
    assert edge["verdict"] == "neutral"
    assert float(edge["confidence"]) == 0.0
    assert edge["reasoning"] == ""
    assert edge["judge_model"] == ""
    assert edge["created_at"] is not None


# ---------------------------------------------------------------------------
# Drifted TYPE ANY table → converted by 69.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_drifted_any_source_verdict_converted(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Force the SurrealDB-default TYPE ANY, apply 69, prove RELATE lands typed."""
    await execute_query(
        "REMOVE TABLE IF EXISTS source_verdict;", config=live_surrealdb
    )
    await execute_query(
        "DEFINE TABLE source_verdict TYPE ANY SCHEMALESS;", config=live_surrealdb
    )
    pre = await _table_def(live_surrealdb, "source_verdict")
    assert "TYPE RELATION" not in pre, (
        f"Setup failed — table should be TYPE ANY before 69. INFO: {pre}"
    )

    await execute_query(_migration_69_sql(), config=live_surrealdb)

    post = await _table_def(live_surrealdb, "source_verdict")
    assert "TYPE RELATION" in post, (
        f"FAIL: `source_verdict` not TYPE RELATION after migration 69. INFO: {post}"
    )

    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)
    await execute_query(
        f"RELATE {a}->source_verdict->{b} SET verdict = 'contradicts';",
        config=live_surrealdb,
    )
    edge = await execute_query(
        f"SELECT in, out, verdict FROM source_verdict "
        f"WHERE in = {a} AND out = {b};",
        config=live_surrealdb,
    )
    assert len(edge) == 1
    assert edge[0]["in"] == a and edge[0]["out"] == b
    assert edge[0]["verdict"] == "contradicts"


# ---------------------------------------------------------------------------
# Idempotent: applying 69 twice yields the same valid end state.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_69_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    await execute_query(_migration_69_sql(), config=live_surrealdb)
    await execute_query(_migration_69_sql(), config=live_surrealdb)
    info = await _table_def(live_surrealdb, "source_verdict")
    assert "TYPE RELATION" in info
    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)
    await execute_query(
        f"RELATE {a}->source_verdict->{b} SET confidence = 0.9;",
        config=live_surrealdb,
    )


# ---------------------------------------------------------------------------
# SAFETY: healthy edges survive migration 69 untouched (OVERWRITE preserves).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_healthy_source_verdict_edges_preserved(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A healthy RELATION `source_verdict` with N edges keeps exactly N after 69."""
    await execute_query(_migration_69_sql(), config=live_surrealdb)
    await execute_query("DELETE source_verdict;", config=live_surrealdb)

    n = 5
    pre_pairs: set[tuple] = set()
    for i in range(n):
        a = await _make_source(live_surrealdb)
        b = await _make_source(live_surrealdb)
        await execute_query(
            f"RELATE {a}->source_verdict->{b} SET confidence = {0.1 * (i + 1)};",
            config=live_surrealdb,
        )
        pre_pairs.add((a, b))

    pre = await execute_query(
        "SELECT count() AS c FROM source_verdict GROUP ALL;", config=live_surrealdb
    )
    assert (pre[0]["c"] if pre else 0) == n

    # Apply 69 on the HEALTHY table.
    await execute_query(_migration_69_sql(), config=live_surrealdb)

    post = await execute_query(
        "SELECT count() AS c FROM source_verdict GROUP ALL;", config=live_surrealdb
    )
    post_count = post[0]["c"] if post else 0
    assert post_count == n, (
        f"edge loss: healthy `source_verdict` had {n} edges, only {post_count} "
        f"survived migration 69 — OVERWRITE wiped records on this version."
    )
    post_edges = await execute_query(
        "SELECT in, out FROM source_verdict;", config=live_surrealdb
    )
    assert pre_pairs == {(e["in"], e["out"]) for e in post_edges}


# ---------------------------------------------------------------------------
# AC4: the app-side `claim`/`contradicts` scaffolding is NOT broken by 69
# (the reconciliation: source_verdict is a SEPARATE, source↔source edge; the
# existing source→claim `contradicts` edge keeps working).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_claim_contradicts_scaffolding_not_broken(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """69 must not touch the app-side `claim`/`contradicts` shape or rows.

    The app-side pair (`claim` TYPE ANY, `contradicts` source->claim) is NOT
    migration-managed and is a different unit from Z's source↔source verdict.
    69 introduces a separate `source_verdict` and leaves `claim`/`contradicts`
    alone. This recreates their app-side shape, applies 69, and asserts a
    source->contradicts->claim edge still RELATEs.
    """
    # Recreate the app-side scaffolding shape (as observed on staging).
    await execute_query(
        "DEFINE TABLE claim TYPE ANY SCHEMALESS;", config=live_surrealdb
    )
    await execute_query(
        "DEFINE TABLE contradicts TYPE RELATION IN source OUT claim SCHEMAFULL;",
        config=live_surrealdb,
    )
    src = await _make_source(live_surrealdb)
    claim_rows = await execute_query(
        "CREATE claim SET statement = $s RETURN AFTER;",
        {"s": _unique("claim")},
        config=live_surrealdb,
    )
    claim_id = claim_rows[0]["id"]
    await execute_query(
        f"RELATE {src}->contradicts->{claim_id};", config=live_surrealdb
    )

    # Apply 69 (only asserts source_verdict — must not touch claim/contradicts).
    await execute_query(_migration_69_sql(), config=live_surrealdb)

    # The existing source->claim edge survives and is still typed.
    cdef = await _table_def(live_surrealdb, "contradicts")
    assert "TYPE RELATION" in cdef and "claim" in cdef.lower(), (
        f"69 must not alter `contradicts` (source->claim). INFO: {cdef}"
    )
    edge = await execute_query(
        f"SELECT in, out FROM contradicts WHERE in = {src} AND out = {claim_id};",
        config=live_surrealdb,
    )
    assert len(edge) == 1, "the app-side contradicts edge must survive migration 69"
    # And a fresh source->claim RELATE still works post-69.
    src2 = await _make_source(live_surrealdb)
    await execute_query(
        f"RELATE {src2}->contradicts->{claim_id};", config=live_surrealdb
    )


# ---------------------------------------------------------------------------
# AC4: canonical `source` rows untouched by migration 69.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_canonical_source_rows_untouched(
    live_surrealdb: SurrealDBConfig,
) -> None:
    a = await _make_source(live_surrealdb)
    pre = await execute_query(
        f"SELECT count() AS c FROM source WHERE id = {a} GROUP ALL;",
        config=live_surrealdb,
    )
    assert (pre[0]["c"] if pre else 0) == 1
    await execute_query(_migration_69_sql(), config=live_surrealdb)
    post = await execute_query(
        f"SELECT count() AS c FROM source WHERE id = {a} GROUP ALL;",
        config=live_surrealdb,
    )
    assert (post[0]["c"] if post else 0) == 1, (
        "migration 69 must not touch source rows"
    )
