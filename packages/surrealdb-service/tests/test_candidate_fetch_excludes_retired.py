"""`find_by_type` offers the resolver live entities, not retired ones (PC.3).

WHAT THIS EXISTS FOR. The fetch used to select every row of a type regardless of
status, and it is the ONLY source of candidates for stage 10's fuzzy and semantic
tiers and for stage 15's concept alignment. Measured on the working database
before the fix:

    type       live   ALL rows   the capped 100 contained
    topic       785      1408    31 active
    concept     574      1892     0 active

For `concept` the resolver could not reach one of the three active entities — a
correct match was structurally impossible, and nothing said so. `("active",
"reference")` is the live set, taken from `audit_service` / `deep_audit_service`,
which already use exactly that pair to decide what counts as graph content.

The second half is the ordering. `LIMIT` with no `ORDER BY` is not guaranteed to
return any particular rows. MEASURED: on SurrealDB v2 this fetch returned id order
with and without the clause, so the behavioural test below cannot catch the
clause's removal and does not claim to — `test_the_query_states_both_rules` reads
the query text for that. The clause is worth having because the ordering becomes
a real choice once `weight` carries centrality, and because an ordering nobody
wrote down is one nobody can rely on.
"""

from __future__ import annotations

import uuid
from typing import List

import pytest
from shared.models.entity import Entity

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository

pytestmark = pytest.mark.asyncio


async def _make(config: SurrealDBConfig, name: str, etype: str, status: str) -> str:
    """Create through the production writer, then set status as triage would."""
    repo = EntityRepository(config=config)
    rid = str(await repo.upsert_entity(
        Entity(canonical_name=name, entity_type=etype, confidence=0.9)
    ))
    if status != "active":
        await execute_query(
            "UPDATE type::thing($id) SET status = $status;",
            {"id": rid, "status": status},
            config,
        )
    return rid


async def test_retired_rows_are_not_offered_as_candidates(
    live_surrealdb: SurrealDBConfig,
) -> None:
    etype = f"probe-{uuid.uuid4().hex[:8]}"
    live = {
        await _make(live_surrealdb, f"{etype}-active", etype, "active"),
        await _make(live_surrealdb, f"{etype}-reference", etype, "reference"),
    }
    retired = {
        await _make(live_surrealdb, f"{etype}-archived", etype, "archived"),
        await _make(live_surrealdb, f"{etype}-merged", etype, "merged"),
    }

    got = {
        str(r["id"])
        for r in await EntityRepository(config=live_surrealdb).find_by_type(etype)
    }

    assert live <= got, (
        f"the live rows must be offered as candidates; missing {live - got}"
    )
    assert not (retired & got), (
        f"triage retired these and the resolver was still offered them: "
        f"{retired & got}"
    )


async def test_the_capped_slice_follows_the_declared_order(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The cap returns the slice the ORDER BY names, not whatever comes back.

    An earlier version of this test fetched twice and compared, which passed with
    the ORDER BY deleted: twelve rows in a fresh table came back the same way
    twice by luck. Comparing against the order the query DECLARES is the claim
    itself. Record ids are random, so id order and insertion order differ, and a
    fetch that ignores the clause lands on a different five.
    """
    etype = f"probe-{uuid.uuid4().hex[:8]}"
    made: List[str] = []
    for i in range(20):
        made.append(await _make(live_surrealdb, f"{etype}-{i:02d}", etype, "active"))

    repo = EntityRepository(config=live_surrealdb)
    got = [str(r["id"]) for r in await repo.find_by_type(etype, limit=5)]

    assert len(got) == 5, f"the cap must bite: got {len(got)} of a limit of 5"
    # weight is 0.0 for every row here, so `weight DESC, id ASC` is id order.
    assert got == sorted(made)[:5], (
        f"the capped slice does not follow `ORDER BY weight DESC, id ASC`.\n"
        f"  returned: {got}\n"
        f"  expected: {sorted(made)[:5]}"
    )


def _sql_literals(func) -> str:
    """The string literals in a function's BODY, excluding its docstring.

    `inspect.getsource` includes the docstring, and the docstring of
    `find_by_type` explains the two rules in prose — so asserting
    `"ORDER BY" in getsource(...)` is true whether or not the clause is in the
    query. That guard passed with the clause deleted. Verified by deleting it.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    fn = tree.body[0]
    body = fn.body[1:] if ast.get_docstring(fn) else fn.body
    parts = []
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                parts.append(sub.value)
    return " ".join(parts)


async def test_the_query_states_both_rules(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Read the SQL, not the prose around it.

    The behaviour tests above can hold by accident — four rows of one type fit
    inside any cap. This reads the shipped query so neither rule can quietly
    leave while the fixtures keep passing.
    """
    sql = _sql_literals(EntityRepository.find_by_type)
    assert "status IN ['active', 'reference']" in sql, (
        f"the status filter is gone from the QUERY — retired rows compete for "
        f"identity again. Query literals: {sql!r}"
    )
    assert "ORDER BY" in sql, (
        f"the ORDER BY is gone from the QUERY — the capped slice is arbitrary "
        f"again. Query literals: {sql!r}"
    )
