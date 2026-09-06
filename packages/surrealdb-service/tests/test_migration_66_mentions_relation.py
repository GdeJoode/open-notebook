"""Container tests for migration 66 — `mentions` TYPE RELATION definition (U.2).

Migration 66 defines the `mentions` edge table as
``TYPE RELATION FROM source TO entity`` so the document↔entity bipartite graph
can be RELATEd / traversed / drawn / exported. On staging the table had drifted
to the SurrealDB default ``TYPE ANY SCHEMALESS`` (0 rows); 66 self-heals it
non-destructively (the same OVERWRITE strategy migration 62 used for
`relation`).

These tests prove the U.2 schema acceptance against a real SurrealDB container:

* version 66 is auto-discovered and `mentions` is TYPE RELATION on a fresh
  fully-migrated container;
* a drifted ``TYPE ANY`` `mentions` table → after 66 it is ``TYPE RELATION`` and
  a ``RELATE source -> mentions -> entity`` lands carrying its weight;
* applying 66 twice is idempotent;
* a healthy RELATION `mentions` with N real edges keeps exactly N after 66
  (OVERWRITE preserves records — the migration-62 empirical fact, re-checked for
  `mentions`).

Run:

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_migration_66_mentions_relation.py -v
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _migration_66_sql() -> str:
    """The migration-66 forward body, loaded from disk (single source of truth)."""
    from surrealdb_service.testing import fixtures as fx

    return (fx._MIGRATIONS_DIR / "66.surrealql").read_text()


async def _mentions_table_def(config: SurrealDBConfig) -> str:
    """Return the ``DEFINE TABLE mentions ...`` text, incl. its TYPE clause.

    Same approach as the migration-62 test: the table *kind* lives in the DEFINE
    statement exposed by ``INFO FOR DB`` under ``tables.mentions`` (e.g.
    ``DEFINE TABLE mentions TYPE RELATION IN source OUT entity SCHEMAFULL``);
    ``INFO FOR TABLE`` does not carry the kind on v2.x.
    """
    result = await execute_query("INFO FOR DB;", config=config)
    payload = (result[0] if result else {}) if isinstance(result, list) else result
    tables = payload.get("tables", {}) if isinstance(payload, dict) else {}
    return str(tables.get("mentions", ""))


async def _make_source(config: SurrealDBConfig) -> str:
    rows = await execute_query(
        "CREATE source SET title = $t;",
        {"t": _unique("src")},
        config=config,
    )
    return rows[0]["id"]


async def _make_entity(config: SurrealDBConfig) -> str:
    rows = await execute_query(
        "CREATE entity SET canonical_name = $n, name_key = string::lowercase(string::trim($n)), entity_type = 'programme', "
        "embedding = [];",
        {"n": _unique("ent")},
        config=config,
    )
    return rows[0]["id"]


# ---------------------------------------------------------------------------
# version 66 auto-discovered and applied on a fresh migrated container.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_66_discovered_and_applied(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Runner records version 66 and `mentions` is TYPE RELATION post-migration."""
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 66;",
        config=live_surrealdb,
    )
    assert rows, "Migration 66 not recorded — runner did not discover it"

    info = await _mentions_table_def(live_surrealdb)
    assert "TYPE RELATION" in info, (
        f"After full migration, `mentions` is not TYPE RELATION. INFO: {info}"
    )


# ---------------------------------------------------------------------------
# Drifted TYPE ANY table → converted, accepts a weighted RELATE.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_drifted_any_mentions_converted(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Reproduce the staging drift (TYPE ANY), apply 66, prove RELATE lands.

    Staging drift (from the U.2 probe): `mentions` is ``TYPE ANY SCHEMALESS``
    with 0 rows. A ``RELATE`` against it would store a non-edge record. We force
    that exact state, then replay migration 66.
    """
    # Force the drifted TYPE ANY table.
    await execute_query("REMOVE TABLE IF EXISTS mentions;", config=live_surrealdb)
    await execute_query(
        "DEFINE TABLE mentions TYPE ANY SCHEMALESS;", config=live_surrealdb
    )
    pre_info = await _mentions_table_def(live_surrealdb)
    assert "TYPE RELATION" not in pre_info, (
        f"Setup failed — table should be TYPE ANY before 66. INFO: {pre_info}"
    )

    # Apply migration 66.
    await execute_query(_migration_66_sql(), config=live_surrealdb)

    post_info = await _mentions_table_def(live_surrealdb)
    assert "TYPE RELATION" in post_info, (
        f"FAIL: `mentions` not TYPE RELATION after migration 66. INFO: {post_info}"
    )

    # A weighted RELATE source -> mentions -> entity now lands.
    src = await _make_source(live_surrealdb)
    ent = await _make_entity(live_surrealdb)
    await execute_query(
        f"RELATE {src}->mentions->{ent} SET weight = 1.336, "
        "concept_name = 'Regio Deal', concept_type = 'programme', "
        "document_frequency = 4;",
        config=live_surrealdb,
    )
    edge = await execute_query(
        "SELECT in, out, weight, concept_name, document_frequency "
        "FROM mentions;",
        config=live_surrealdb,
    )
    assert len(edge) == 1
    assert edge[0]["in"] == src and edge[0]["out"] == ent
    assert abs(float(edge[0]["weight"]) - 1.336) < 1e-9
    assert edge[0]["concept_name"] == "Regio Deal"
    assert int(edge[0]["document_frequency"]) == 4


# ---------------------------------------------------------------------------
# Idempotent: applying 66 twice yields the same valid end state.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_66_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Applying migration 66 twice leaves the same valid end state."""
    await execute_query(_migration_66_sql(), config=live_surrealdb)
    await execute_query(_migration_66_sql(), config=live_surrealdb)
    info = await _mentions_table_def(live_surrealdb)
    assert "TYPE RELATION" in info
    # Still accepts a RELATE after the double apply.
    src = await _make_source(live_surrealdb)
    ent = await _make_entity(live_surrealdb)
    await execute_query(
        f"RELATE {src}->mentions->{ent} SET weight = 0.2;",
        config=live_surrealdb,
    )


# ---------------------------------------------------------------------------
# SAFETY: healthy edges survive migration 66 untouched (OVERWRITE preserves).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_healthy_mentions_edges_preserved(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A healthy RELATION `mentions` with N edges keeps exactly N after 66."""
    # Guarantee a RELATION table, then clear so N is exact.
    await execute_query(_migration_66_sql(), config=live_surrealdb)
    await execute_query("DELETE mentions;", config=live_surrealdb)

    n = 5
    pairs: list[tuple[str, str]] = []
    for i in range(n):
        src = await _make_source(live_surrealdb)
        ent = await _make_entity(live_surrealdb)
        await execute_query(
            f"RELATE {src}->mentions->{ent} SET weight = {0.1 * (i + 1)};",
            config=live_surrealdb,
        )
        pairs.append((src, ent))

    pre = await execute_query(
        "SELECT count() AS c FROM mentions GROUP ALL;", config=live_surrealdb
    )
    assert (pre[0]["c"] if pre else 0) == n
    pre_edges = await execute_query(
        "SELECT in, out FROM mentions;", config=live_surrealdb
    )
    pre_pairs = {(e["in"], e["out"]) for e in pre_edges}

    # Apply 66 on the HEALTHY table.
    await execute_query(_migration_66_sql(), config=live_surrealdb)

    post = await execute_query(
        "SELECT count() AS c FROM mentions GROUP ALL;", config=live_surrealdb
    )
    post_count = post[0]["c"] if post else 0
    assert post_count == n, (
        f"edge loss: healthy `mentions` had {n} edges, only {post_count} "
        f"survived migration 66 — OVERWRITE wiped records on this version."
    )
    post_edges = await execute_query(
        "SELECT in, out FROM mentions;", config=live_surrealdb
    )
    assert pre_pairs == {(e["in"], e["out"]) for e in post_edges}
