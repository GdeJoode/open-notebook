"""Static lockstep test for migration 65 — WHERE mirrors SET on every block (S.4).

Migration 65 ships a drift-only ``WHERE`` on each ``UPDATE`` so a healthy DB
matches 0 rows (no rewrites, startup-safe) and only genuinely-drifted rows are
touched. That guard is only correct if, for EVERY block, the set of fields
coalesced in the ``SET`` (``f = f ?? default``) is exactly the set of fields
tested in the ``WHERE`` (``f = NONE`` disjuncts). If they drift apart — e.g. a
future change adds a ``SET`` field but forgets the matching ``WHERE`` disjunct —
the ``WHERE`` no longer covers that field, so a row drifted ONLY on the new field
is silently left unrepaired (and the strict type keeps blocking its writes).

This test enforces that lockstep invariant by PARSING the on-disk
``migrations/65.surrealql`` (no DB, no Docker) and asserting per-block
``set(WHERE fields) == set(SET fields)``. It must pass now (all 19 blocks
mirror) and FAIL if any ``WHERE`` disjunct is removed — so the regression the
S.2b reviewer flagged ("added a SET field, forgot the WHERE disjunct") fails CI.

To run::

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_migration_65_where_mirrors_set.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

# migrations/ lives at the repo root: tests/ -> surrealdb-service -> packages -> root.
_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"
_MIGRATION_65 = _MIGRATIONS_DIR / "65.surrealql"

# Number of UPDATE blocks migration 65 is expected to carry. Pinned so an
# accidental drop of a whole block is caught, not silently passed over.
_EXPECTED_BLOCK_COUNT = 19


def _sql_body() -> str:
    """Migration-65 text with `--` comment lines stripped.

    Comments contain prose like "= NONE" and "??" that would confuse the
    field-extraction regexes, so strip them before parsing.
    """
    raw = _MIGRATION_65.read_text()
    return "\n".join(
        line for line in raw.splitlines() if not line.strip().startswith("--")
    )


def _update_blocks(sql: str) -> list[tuple[str, str, str]]:
    """Return ``(table, set_clause, where_clause)`` for each UPDATE statement.

    Matches ``UPDATE <table> SET <...> WHERE <...> RETURN NONE;``. Every block in
    migration 65 carries a ``WHERE`` and ``RETURN NONE`` (that IS the invariant),
    so a block missing either should not parse here — which itself is a failure
    signal we assert on via the expected block count.
    """
    pattern = re.compile(
        r"UPDATE\s+(?P<table>\w+)\s+SET\s+"
        r"(?P<set>.*?)\s+WHERE\s+"
        r"(?P<where>.*?)\s+RETURN\s+NONE\s*;",
        re.IGNORECASE | re.DOTALL,
    )
    return [
        (m.group("table"), m.group("set"), m.group("where"))
        for m in pattern.finditer(sql)
    ]


def _set_fields(set_clause: str) -> set[str]:
    """Fields coalesced in a SET clause: every ``<field> = <field> ?? <default>``.

    Only counts assignments that are a self-coalesce (``f = f ?? ...``); a plain
    ``f = g`` would not be a drift backfill and is intentionally excluded.
    """
    fields: set[str] = set()
    for m in re.finditer(
        r"(?P<lhs>\w+)\s*=\s*(?P<rhs>\w+)\s*\?\?",
        set_clause,
    ):
        if m.group("lhs") == m.group("rhs"):
            fields.add(m.group("lhs"))
    return fields


def _where_fields(where_clause: str) -> set[str]:
    """Fields tested in a WHERE clause: every ``<field> = NONE`` disjunct."""
    return {
        m.group(1)
        for m in re.finditer(r"(\w+)\s*=\s*NONE\b", where_clause, re.IGNORECASE)
    }


def test_migration_65_exists_and_parses() -> None:
    assert _MIGRATION_65.is_file(), f"missing {_MIGRATION_65}"
    blocks = _update_blocks(_sql_body())
    assert len(blocks) == _EXPECTED_BLOCK_COUNT, (
        f"expected {_EXPECTED_BLOCK_COUNT} `UPDATE ... WHERE ... RETURN NONE;` "
        f"blocks in migration 65, parsed {len(blocks)}: "
        f"{[b[0] for b in blocks]}. A block missing its WHERE/RETURN NONE would "
        f"not parse here — that is itself the regression this test guards."
    )


def test_each_where_mirrors_its_set() -> None:
    """For every UPDATE block: set(WHERE `= NONE` fields) == set(SET coalesced fields).

    This is the lockstep invariant that keeps the drift-only WHERE correct. A
    mismatch in either direction is a bug:
      * field in SET but not WHERE -> a row drifted only on that field is never
        matched, so it stays NONE and keeps blocking writes (the regression).
      * field in WHERE but not SET -> the WHERE matches rows the SET won't repair.
    """
    blocks = _update_blocks(_sql_body())
    assert blocks, "no UPDATE blocks parsed from migration 65"

    mismatches: list[str] = []
    for table, set_clause, where_clause in blocks:
        set_fields = _set_fields(set_clause)
        where_fields = _where_fields(where_clause)
        assert set_fields, f"`{table}` block has no `f = f ?? default` coalesce"
        if set_fields != where_fields:
            only_set = sorted(set_fields - where_fields)
            only_where = sorted(where_fields - set_fields)
            mismatches.append(
                f"  {table}: SET-only={only_set or '-'} "
                f"WHERE-only={only_where or '-'}"
            )

    assert not mismatches, (
        "migration 65 WHERE/SET field lists are out of lockstep "
        "(drift-only WHERE no longer mirrors SET):\n" + "\n".join(mismatches)
    )
