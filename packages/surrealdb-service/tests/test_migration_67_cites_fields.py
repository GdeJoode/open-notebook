"""Container tests for migration 67 — `cites` U.3 materialization fields.

`cites` is in the same class as the `mentions` drift migration 66 fixed: on
STAGING it was already ``TYPE RELATION IN source OUT source SCHEMAFULL`` (defined
at runtime by the relation write path, verified 2026-06-27), but on a FRESH
migration-only container it comes up as the SurrealDB default
``TYPE ANY SCHEMALESS`` (the migration suite never defined it). So migration 67
ASSERTS the relation definition (non-destructively, like 66) AND adds the U.3
fields (`confidence`, `reference_text`, `match_method`, `created_at`) — without
which a SCHEMAFULL table silently drops those values on RELATE-with-SET.

These tests prove the U.3 schema acceptance against a real SurrealDB container:

* version 67 is auto-discovered and `cites` is TYPE RELATION source->source on a
  fresh fully-migrated container (AC5);
* after 67 a ``RELATE source -> cites -> source SET confidence=..., ...`` lands
  carrying ALL the U.3 fields (proving the SCHEMAFULL table no longer drops them);
* a drifted ``TYPE ANY`` `cites` table is converted to RELATION source->source by
  67 (the fresh-container case);
* a healthy RELATION `cites` with N real edges keeps exactly N after 67
  (OVERWRITE preserves records); applying 67 twice is idempotent.

Run:

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_migration_67_cites_fields.py -v
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _migration_67_sql() -> str:
    """The migration-67 forward body, loaded from disk (single source of truth)."""
    from surrealdb_service.testing import fixtures as fx

    return (fx._MIGRATIONS_DIR / "67.surrealql").read_text()


async def _cites_table_def(config: SurrealDBConfig) -> str:
    """Return the ``DEFINE TABLE cites ...`` text, incl. its TYPE clause."""
    result = await execute_query("INFO FOR DB;", config=config)
    payload = (result[0] if result else {}) if isinstance(result, list) else result
    tables = payload.get("tables", {}) if isinstance(payload, dict) else {}
    return str(tables.get("cites", ""))


async def _cites_fields(config: SurrealDBConfig) -> dict:
    """Return the `cites` field definitions (name -> DEFINE FIELD text)."""
    result = await execute_query("INFO FOR TABLE cites;", config=config)
    payload = (result[0] if result else {}) if isinstance(result, list) else result
    return payload.get("fields", {}) if isinstance(payload, dict) else {}


async def _make_source(config: SurrealDBConfig) -> str:
    rows = await execute_query(
        "CREATE source SET title = $t;",
        {"t": _unique("src")},
        config=config,
    )
    return rows[0]["id"]


# ---------------------------------------------------------------------------
# version 67 auto-discovered; cites is TYPE RELATION source->source (AC5).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_67_discovered_and_cites_is_source_to_source(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Runner records 67; `cites` is TYPE RELATION source->source (AC5)."""
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 67;",
        config=live_surrealdb,
    )
    assert rows, "Migration 67 not recorded — runner did not discover it"

    info = await _cites_table_def(live_surrealdb)
    assert "TYPE RELATION" in info, f"`cites` is not TYPE RELATION. INFO: {info}"
    # AC5: source -> source specifically (not the mentions source -> entity).
    assert "IN source OUT source" in info, (
        f"`cites` is not RELATION source->source. INFO: {info}"
    )


# ---------------------------------------------------------------------------
# After 67 the U.3 fields exist and a full RELATE-with-SET lands them.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_cites_u3_fields_defined_and_persisted(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The U.3 fields are DEFINEd and a RELATE-with-SET carries them through.

    The whole point of 67: on a SCHEMAFULL table an undefined field is dropped
    silently on RELATE. After 67, confidence / reference_text / match_method must
    DEFINE'd and a written edge must round-trip them.
    """
    await execute_query(_migration_67_sql(), config=live_surrealdb)

    fields = await _cites_fields(live_surrealdb)
    for f in ("confidence", "reference_text", "match_method", "created_at"):
        assert f in fields, f"migration 67 did not DEFINE `cites.{f}`"

    src_a = await _make_source(live_surrealdb)
    src_b = await _make_source(live_surrealdb)
    await execute_query(
        f"RELATE {src_a}->cites->{src_b} SET "
        "confidence = 0.91, reference_text = $rt, match_method = 'title_author';",
        {"rt": "Ali (2025) Cohesion Policy and Institutional Quality."},
        config=live_surrealdb,
    )
    # Scope to the edge THIS test created. The container is session-scoped, so
    # other suites (e.g. the W.3 MCP `cite` roundtrip) legitimately add their
    # own `cites` edges into the same DB — a global ``len == 1`` would be a
    # false cross-test coupling, not a real assertion about this RELATE.
    edge = await execute_query(
        "SELECT in, out, confidence, reference_text, match_method FROM cites "
        "WHERE in = type::thing($a) AND out = type::thing($b);",
        {"a": src_a, "b": src_b},
        config=live_surrealdb,
    )
    assert len(edge) == 1
    e = edge[0]
    assert e["in"] == src_a and e["out"] == src_b
    assert abs(float(e["confidence"]) - 0.91) < 1e-9
    assert e["match_method"] == "title_author"
    assert "Ali (2025)" in e["reference_text"]


# ---------------------------------------------------------------------------
# Drifted TYPE ANY `cites` (the fresh-container case) → converted to RELATION.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_drifted_any_cites_converted(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Force a TYPE ANY `cites`, apply 67, prove it is RELATION source->source.

    This is the fresh-migration-container reality: `cites` comes up as the
    SurrealDB default ``TYPE ANY SCHEMALESS``. Migration 67 must convert it (the
    same self-healing OVERWRITE migration 66 does for `mentions`) so a
    ``RELATE source -> cites -> source`` stores a real graph edge.
    """
    await execute_query("REMOVE TABLE IF EXISTS cites;", config=live_surrealdb)
    await execute_query(
        "DEFINE TABLE cites TYPE ANY SCHEMALESS;", config=live_surrealdb
    )
    pre = await _cites_table_def(live_surrealdb)
    assert "TYPE RELATION" not in pre, f"setup should be TYPE ANY. INFO: {pre}"

    await execute_query(_migration_67_sql(), config=live_surrealdb)

    post = await _cites_table_def(live_surrealdb)
    assert "IN source OUT source" in post, (
        f"FAIL: `cites` not RELATION source->source after 67. INFO: {post}"
    )
    # A full cites RELATE now lands with its U.3 fields.
    src_a = await _make_source(live_surrealdb)
    src_b = await _make_source(live_surrealdb)
    await execute_query(
        f"RELATE {src_a}->cites->{src_b} SET confidence = 1.0, "
        "reference_text = 'doi:10.1/x', match_method = 'doi';",
        config=live_surrealdb,
    )
    edge = await execute_query(
        "SELECT in, out, confidence, match_method FROM cites;",
        config=live_surrealdb,
    )
    assert len(edge) == 1
    assert edge[0]["in"] == src_a and edge[0]["out"] == src_b
    assert edge[0]["match_method"] == "doi"


# ---------------------------------------------------------------------------
# SAFETY: healthy edges survive migration 67 untouched (OVERWRITE preserves).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_healthy_cites_edges_preserved(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A healthy RELATION `cites` with N edges keeps exactly N after 67."""
    # Guarantee a RELATION table, then clear so N is exact.
    await execute_query(_migration_67_sql(), config=live_surrealdb)
    await execute_query("DELETE cites;", config=live_surrealdb)

    n = 4
    for i in range(n):
        src_a = await _make_source(live_surrealdb)
        src_b = await _make_source(live_surrealdb)
        await execute_query(
            f"RELATE {src_a}->cites->{src_b} SET confidence = {0.2 * (i + 1)}, "
            "reference_text = 'r', match_method = 'doi';",
            config=live_surrealdb,
        )

    pre = await execute_query(
        "SELECT count() AS c FROM cites GROUP ALL;", config=live_surrealdb
    )
    assert (pre[0]["c"] if pre else 0) == n
    pre_edges = await execute_query(
        "SELECT in, out FROM cites;", config=live_surrealdb
    )
    pre_pairs = {(e["in"], e["out"]) for e in pre_edges}

    await execute_query(_migration_67_sql(), config=live_surrealdb)

    post = await execute_query(
        "SELECT count() AS c FROM cites GROUP ALL;", config=live_surrealdb
    )
    post_count = post[0]["c"] if post else 0
    assert post_count == n, (
        f"edge loss: healthy `cites` had {n} edges, only {post_count} survived "
        f"migration 67 — OVERWRITE wiped records on this version."
    )
    post_edges = await execute_query(
        "SELECT in, out FROM cites;", config=live_surrealdb
    )
    assert pre_pairs == {(e["in"], e["out"]) for e in post_edges}


# ---------------------------------------------------------------------------
# Idempotent: applying 67 twice yields the same valid end state.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_67_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Applying migration 67 twice leaves the same valid end state."""
    await execute_query(_migration_67_sql(), config=live_surrealdb)
    await execute_query(_migration_67_sql(), config=live_surrealdb)
    fields = await _cites_fields(live_surrealdb)
    assert "confidence" in fields
    # Still accepts a full cites RELATE after the double apply.
    src_a = await _make_source(live_surrealdb)
    src_b = await _make_source(live_surrealdb)
    await execute_query(
        f"RELATE {src_a}->cites->{src_b} SET confidence = 1.0, "
        "reference_text = 'x', match_method = 'doi';",
        config=live_surrealdb,
    )
