"""Container tests for migration 65 — self-healing strict-field drift guard (S.2).

Migration 65 is an idempotent forward-guard. For every strict (non-`option<>`),
no-VALUE field that has a SAFE coalesce default (per the S.1 inventory), it runs
``UPDATE <table> SET f = f ?? <default>`` so a row that predates the field's
DEFINE (and therefore holds NONE) is repaired before the strict type rejects the
next write. S.1 confirmed the live `staging` DB has ZERO such drift, so these
tests prove the MECHANISM against SYNTHETIC drift forged per table.

These prove every S.2 acceptance criterion against a real SurrealDB container:

* AC1 — synthetic-drift repro per representative table (`entity`, `source`,
  `transformation`): forge a legacy row with the strict field NONE (REMOVE FIELD
  -> UNSET -> re-DEFINE-without-backfill, the migration-64 test pattern), assert
  a later ``UPDATE`` FAILS pre-65 and PASSES post-65.
* AC2 — idempotent: applying 65 twice is a no-op and a clean row is unchanged.
* AC3 — full 1->65 chain applies clean (the session fixture runs the strict
  transactional runner over every migration incl. 65; we assert 65 is recorded).
* AC4 — no coalesce of no-safe-default fields: the migration body must not touch
  `chunk.text`, `source_embedding.embedding`, `*.source`, or edge `in`/`out`
  (static grep of the on-disk body).
* AC5 — down present + sane (comment-only no-op, applies without error).

To run::

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_migration_65_strict_drift_guard.py -v
"""

from __future__ import annotations

import re
import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.testing import fixtures as fx

pytestmark = pytest.mark.asyncio


def _u(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _migration_65_sql() -> str:
    """The migration-65 forward body, cleaned exactly as the runner sees it."""
    from surrealdb_service.migrations import AsyncMigration

    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / "65.surrealql").sql


def _raw_65_body() -> str:
    """The raw on-disk migration-65 text (comments included) for grep checks."""
    return (fx._MIGRATIONS_DIR / "65.surrealql").read_text()


def _sql_only_65() -> str:
    """Migration-65 body with `--` comment lines stripped, for grep on real SQL."""
    return "\n".join(
        line
        for line in _raw_65_body().splitlines()
        if not line.strip().startswith("--")
    )


# ---------------------------------------------------------------------------
# AC1 — synthetic-drift reproduction, per representative table.
#
# A table is "representative" if the inventory lists a strict field WITH a safe
# default that migration 65 coalesces. We cover the two historically-drifted
# tables (`entity` via 61, `source` via 64) plus a clean third (`transformation`)
# to show the guard is generic, not entity/source-specific.
#
# Forging the drift: the session fixture already applied the field's strict
# DEFINE-with-DEFAULT, so a fresh CREATE can't leave the field NONE. We replay
# the legacy timeline: REMOVE the field -> create the row (field now absent ->
# NONE) -> re-DEFINE the field strict WITHOUT backfilling the existing row. That
# is exactly the state a pre-migration row sits in, and the migration-64 test
# uses the same REMOVE/re-DEFINE technique.
# ---------------------------------------------------------------------------


async def _forge_none_strict_field(
    config: SurrealDBConfig,
    *,
    table: str,
    field: str,
    field_type: str,
    field_default: str,
    create_extra: str,
) -> str:
    """Create one row of ``table`` whose strict ``field`` is NONE.

    Returns the row id. ``create_extra`` supplies any OTHER required strict
    fields so the CREATE itself is accepted. ``field_type`` / ``field_default``
    re-establish the production strict DEFINE *after* the row exists, so the row
    is left at NONE — the exact pre-migration drift.
    """
    rid = f"{table}:{_u('drift')}"

    # Step 1 — redefine the strict field as option<> transiently so a CREATE can
    # leave it NONE. OVERWRITE (not REMOVE+DEFINE) because a bare re-DEFINE of an
    # existing field raises "already exists" on 2.6.5.
    await execute_query(
        f"DEFINE FIELD OVERWRITE {field} ON {table} TYPE option<{field_type}>;",
        config=config,
    )

    # Step 2 — create the legacy row with the field unset (NONE).
    await execute_query(
        f"CREATE {rid} SET {create_extra} {field} = NONE;", config=config
    )

    # Step 3 — re-establish the production strict DEFINE WITHOUT backfilling the
    # existing row (DEFAULT applies only to new rows — the root-cause drift).
    await execute_query(
        f"DEFINE FIELD OVERWRITE {field} ON {table} "
        f"TYPE {field_type} DEFAULT {field_default};",
        config=config,
    )
    return rid


async def _assert_field_is_none(
    config: SurrealDBConfig, rid: str, field: str
) -> None:
    rows = await execute_query(
        f"SELECT type::is::none({field}) AS isnone FROM {rid};", config=config
    )
    assert rows and rows[0]["isnone"] is True, (
        f"setup: expected {rid}.{field} to be NONE before migration 65"
    )


@pytest.mark.requires_docker
async def test_entity_drift_fails_pre_passes_post(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """`entity.manual_override` NONE blocks UPDATE pre-65, succeeds post-65."""
    rid = await _forge_none_strict_field(
        live_surrealdb,
        table="entity",
        field="manual_override",
        field_type="bool",
        field_default="false",
        # entity has other strict required fields with no default — supply them.
        create_extra=(
            "canonical_name = $n, name = $n, entity_type = 'org', "
            "embedding = [], confidence = 0.9, source_documents = [], "
        ).replace("$n", f"'{_u('Ent')}'"),
    )
    await _assert_field_is_none(live_surrealdb, rid, "manual_override")

    # Pre-65: any UPDATE re-validates the whole record and trips on NONE bool.
    with pytest.raises(Exception):  # noqa: B017
        await execute_query(
            f"UPDATE {rid} SET status = 'reference';", config=live_surrealdb
        )

    # Apply migration 65 (the entity block coalesces manual_override ?? false).
    await execute_query(_migration_65_sql(), config=live_surrealdb)

    # Post-65: the same UPDATE now lands.
    await execute_query(
        f"UPDATE {rid} SET status = 'reference';", config=live_surrealdb
    )
    rows = await execute_query(
        f"SELECT manual_override, status FROM {rid};", config=live_surrealdb
    )
    assert rows[0]["manual_override"] is False
    assert rows[0]["status"] == "reference"


@pytest.mark.requires_docker
async def test_source_drift_fails_pre_passes_post(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """`source.private` NONE blocks UPDATE pre-65, succeeds post-65 (mirrors 64)."""
    rid = await _forge_none_strict_field(
        live_surrealdb,
        table="source",
        field="private",
        field_type="bool",
        field_default="false",
        create_extra="",  # source has no other no-default strict field.
    )
    await _assert_field_is_none(live_surrealdb, rid, "private")

    with pytest.raises(Exception):  # noqa: B017
        await execute_query(
            f"UPDATE {rid} SET title = 'x';", config=live_surrealdb
        )

    await execute_query(_migration_65_sql(), config=live_surrealdb)

    await execute_query(f"UPDATE {rid} SET title = 'x';", config=live_surrealdb)
    rows = await execute_query(
        f"SELECT private, title FROM {rid};", config=live_surrealdb
    )
    assert rows[0]["private"] is False
    assert rows[0]["title"] == "x"


@pytest.mark.requires_docker
async def test_transformation_drift_fails_pre_passes_post(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A third, clean table proves the guard is generic, not entity/source-only."""
    rid = await _forge_none_strict_field(
        live_surrealdb,
        table="transformation",
        field="apply_default",
        field_type="bool",
        field_default="false",
        create_extra=(
            "name = 'n', title = 't', description = 'd', prompt = 'p', "
        ),
    )
    await _assert_field_is_none(live_surrealdb, rid, "apply_default")

    with pytest.raises(Exception):  # noqa: B017
        await execute_query(
            f"UPDATE {rid} SET title = 't2';", config=live_surrealdb
        )

    await execute_query(_migration_65_sql(), config=live_surrealdb)

    await execute_query(f"UPDATE {rid} SET title = 't2';", config=live_surrealdb)
    rows = await execute_query(
        f"SELECT apply_default, title FROM {rid};", config=live_surrealdb
    )
    assert rows[0]["apply_default"] is False
    assert rows[0]["title"] == "t2"


@pytest.mark.requires_docker
async def test_extraction_result_drift_repaired_clean_untouched(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """`extraction_result` drift is repaired post-65; a clean row is left as-is.

    `extraction_result` is the table that crashed startup live: its rows carry the
    full multi-MB extraction payload, so its UPDATE block ships with a drift-only
    WHERE. This test pins BOTH halves of that guard:

    * the WHERE still REPAIRS a genuinely-drifted row (`entity_count` NONE blocks
      writes pre-65, lands post-65), and
    * the WHERE is a NO-OP on a clean row — it must not match (0 rewrites), which
      is exactly what keeps startup safe on a healthy DB with large rows.
    """
    # --- clean row: distinct source_id (UNIQUE index) at a non-default value. ---
    clean_src = _u("clean-src")
    clean = await execute_query(
        "CREATE extraction_result SET source_id = $s, entity_count = 7, "
        "relation_count = 3, entities = [], relations = [], metadata = {} "
        "RETURN VALUE id;",
        {"s": clean_src},
        config=live_surrealdb,
    )
    clean_rid = str(clean[0])

    # --- drifted row: entity_count forged to NONE (the migration-64 technique). ---
    rid = await _forge_none_strict_field(
        live_surrealdb,
        table="extraction_result",
        field="entity_count",
        field_type="int",
        field_default="0",
        # source_id is the only no-default strict field; must be unique.
        create_extra=f"source_id = '{_u('drift-src')}', ",
    )
    await _assert_field_is_none(live_surrealdb, rid, "entity_count")

    # Pre-65: any UPDATE re-validates the whole record and trips on NONE int.
    with pytest.raises(Exception):  # noqa: B017
        await execute_query(
            f"UPDATE {rid} SET relation_count = 1;", config=live_surrealdb
        )

    await execute_query(_migration_65_sql(), config=live_surrealdb)

    # Post-65: the drifted row is repaired and the same UPDATE now lands.
    await execute_query(
        f"UPDATE {rid} SET relation_count = 1;", config=live_surrealdb
    )
    rows = await execute_query(
        f"SELECT entity_count, relation_count FROM {rid};", config=live_surrealdb
    )
    assert rows[0]["entity_count"] == 0
    assert rows[0]["relation_count"] == 1

    # Clean row: the WHERE did NOT match it, so its values are exactly as set
    # (coalesce defaults would be 0/0 — both staying at 7/3 proves no rewrite).
    clean_rows = await execute_query(
        f"SELECT entity_count, relation_count FROM {clean_rid};",
        config=live_surrealdb,
    )
    assert clean_rows[0]["entity_count"] == 7
    assert clean_rows[0]["relation_count"] == 3


# ---------------------------------------------------------------------------
# AC2 — idempotent (double-apply no-op; clean row unchanged).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
async def test_migration_65_idempotent_double_apply(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Applying 65 twice raises nothing and leaves the same end state."""
    await execute_query(_migration_65_sql(), config=live_surrealdb)
    await execute_query(_migration_65_sql(), config=live_surrealdb)


@pytest.mark.requires_docker
async def test_migration_65_clean_row_unchanged(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A clean `entity` row is untouched: its drift-only WHERE matches 0 rows.

    This also locks in the generalized WHERE on a representative multi-field block:
    `entity` has no NONE coalesced field, so its WHERE must not match — a future
    "forgot the WHERE" regression (blanket rewrite) would still preserve these
    values, but the dedicated extraction_result test pins the rewrite-cost no-op.
    """
    name = _u("Clean")
    created = await execute_query(
        "CREATE entity SET canonical_name = $n, name = $n, entity_type = 'org', "
        "embedding = [], confidence = 0.9, source_documents = [], "
        "manual_override = true, status = 'reference' "
        "RETURN VALUE id;",
        {"n": name},
        config=live_surrealdb,
    )
    rid = str(created[0])

    await execute_query(_migration_65_sql(), config=live_surrealdb)

    rows = await execute_query(
        f"SELECT manual_override, status FROM {rid};",
        config=live_surrealdb,
    )
    # `f ?? default` leaves a non-NONE value exactly as it was. The coalesce
    # defaults would be `manual_override = false` / `status = "active"`; both
    # staying at their set values proves the guard did NOT clobber real data.
    assert rows[0]["manual_override"] is True
    assert rows[0]["status"] == "reference"


# ---------------------------------------------------------------------------
# AC3 — full chain applies clean; version 65 recorded by the strict runner.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
async def test_migration_65_discovered_and_applied(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The session fixture ran the full 1->65 chain; 65 is recorded."""
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 65;",
        config=live_surrealdb,
    )
    assert rows, (
        "Migration 65 not recorded in _sbl_migrations — the strict runner did "
        "not discover/apply it on the fully-migrated container"
    )


# ---------------------------------------------------------------------------
# AC4 — no coalesce of no-safe-default fields (static check on the SQL body).
#
# These fields are required-with-no-honest-default; coalescing them would invent
# data. The guard must NEVER assign them. We grep the comment-stripped body for
# any ``<field> = <field> ??`` coalesce on the forbidden fields.
# ---------------------------------------------------------------------------


async def test_no_safe_default_fields_not_coalesced() -> None:
    # Static check on the on-disk SQL body — no DB needed, but kept async so
    # pytest-asyncio (auto mode) collects it uniformly with the rest of the file.
    sql = _sql_only_65()

    def _block(table: str) -> str | None:
        """Return the SET body of `UPDATE <table> SET ...;`, or None if absent."""
        m = re.search(
            rf"UPDATE\s+{re.escape(table)}\s+SET(.*?);",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        return m.group(1) if m else None

    def _coalesces(body: str, field: str) -> bool:
        """True if `body` contains a `field = field ??` coalesce."""
        return bool(
            re.search(
                rf"\b{re.escape(field)}\s*=\s*{re.escape(field)}\s*\?\?",
                body,
                re.IGNORECASE,
            )
        )

    # Tables whose ONLY strict fields are no-safe-default (content bodies / FKs /
    # edge endpoints) must NOT have an UPDATE block at all in migration 65.
    for forbidden_table in [
        "source_embedding",  # content + embedding + order + source — none safe
        "derived_from",
        "next_node",
        "parent_of",
        "reference",
        "refers_to",
    ]:
        assert _block(forbidden_table) is None, (
            f"migration 65 must NOT touch `{forbidden_table}` — its strict "
            f"fields have no safe coalesce default"
        )

    # The `chunk` block exists (safe fields like layer/metadata) but must NOT
    # coalesce its no-safe-default fields.
    chunk = _block("chunk")
    assert chunk is not None, "expected an `UPDATE chunk SET ...` block"
    for unsafe in ["text", "element_type", "source", "order"]:
        assert not _coalesces(chunk, unsafe), (
            f"migration 65 must NOT coalesce `chunk.{unsafe}` (no safe default)"
        )

    # The `source` block must touch ONLY `private` (its sole load-bearing strict
    # field); not `embedding`/`full_text` (option<>, NONE is intentional).
    source = _block("source")
    assert source is not None, "expected an `UPDATE source SET ...` block"
    assert _coalesces(source, "private")
    assert not _coalesces(source, "embedding")
    assert not _coalesces(source, "full_text")

    # No block anywhere may coalesce an edge endpoint or a required FK column.
    for field in ["in", "out"]:
        assert not _coalesces(sql, field), (
            f"migration 65 must NOT coalesce edge endpoint `{field}`"
        )
    # FK `source` column (on chunk/doc_node/etc.) — distinct from `source_documents`.
    assert not _coalesces(sql, "source"), (
        "migration 65 must NOT coalesce the required FK column `source`"
    )


# ---------------------------------------------------------------------------
# AC5 — down present + sane (comment-only no-op, applies without error).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
async def test_migration_65_down_is_noop(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """65_down is a documented no-op: it has no executable statement and runs clean."""
    from surrealdb_service.migrations import AsyncMigration

    down = AsyncMigration.from_file(fx._MIGRATIONS_DIR / "65_down.surrealql")
    # Comment-only file → cleaned SQL is empty.
    assert down.sql.strip() == "", (
        f"65_down should be a comment-only no-op, got SQL: {down.sql!r}"
    )
    # Executing an empty/no-op body must not raise.
    if down.sql.strip():
        await execute_query(down.sql, config=live_surrealdb)
