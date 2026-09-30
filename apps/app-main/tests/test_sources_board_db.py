"""WB.3: the Sources board against a real SurrealDB.

* ``GET /api/sources/counts`` equals the lists its chips open: every stage
  (taken from the enum, not a sample), stopped, and the whole list, also
  within each notebook.
* ``?sort_by=attention`` lists the sources that need you first, and that set
  is The Tray's failed, gated and stopped groups.
* Each row says whether it stopped, by the ``?stalled=true`` predicate.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from shared.types.enums import ProcessingStage

from tests.test_attention_db import (  # noqa: F401  (``env`` is a fixture)
    _app,
    _client,
    _paged_sources,
    _seed,
    env,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


async def _counts(c, **params: Any) -> Dict[str, Any]:
    r = await c.get("/api/sources/counts", params=params)
    assert r.status_code == 200, r.text
    return r.json()


async def test_counts_equal_the_lists_they_open(env) -> None:
    ids = await _seed()
    async with _client(_app()) as c:
        for scope in ([], [("notebook_id", ids["nb1"])], [("notebook_id", ids["nb2"])]):
            counts = await _counts(c, **dict(scope))
            assert set(counts["stages"]) == {s.value for s in ProcessingStage}
            for stage in ProcessingStage:
                listed = await _paged_sources(c, scope + [("stage", stage.value)])
                assert counts["stages"][stage.value] == len(listed), (scope, stage)
            stalled = await _paged_sources(c, scope + [("stalled", "true")])
            assert counts["stalled"] == len(stalled), scope
            assert counts["total"] == len(await _paged_sources(c, scope)), scope
    # Not all zero: the seed has a source at most stages.
    assert counts["stages"]["embedded"] == 3 and counts["stalled"] == 2


async def test_needs_you_is_the_trays_source_groups(env) -> None:
    await _seed()
    async with _client(_app()) as c:
        counts = await _counts(c)
        groups = {g["key"]: g for g in (await c.get("/api/attention")).json()["groups"]}
    tray = sum(groups[k]["count"] for k in
               ("sources_failed", "sources_awaiting_review", "sources_stalled"))
    assert counts["needs_you"] == tray == 5


def _needs_you(row: Dict[str, Any]) -> bool:
    return row["stalled"] or row["processing_stage"] in (
        "failed", "awaiting_schema_review")


async def _attention_pages(c, scope: List[tuple], page: int) -> List[Dict[str, Any]]:
    """The attention order read in small pages, so the order holds across them."""
    out: List[Dict[str, Any]] = []
    while True:
        r = await c.get("/api/sources/", params=scope + [
            ("sort_by", "attention"), ("limit", page), ("offset", len(out))])
        assert r.status_code == 200, r.text
        out += r.json()
        if len(r.json()) < page:
            return out


async def test_attention_sort_puts_needs_you_first(env) -> None:
    ids = await _seed()
    needs = {ids[k] for k in ("failed", "gated", "gated2", "stalled", "empty")}
    async with _client(_app()) as c:
        rows = await _attention_pages(c, [], page=2)
        in_nb2 = await _attention_pages(c, [("notebook_id", ids["nb2"])], page=2)
        counts = await _counts(c)
        by_updated = await _paged_sources(c, [])
    head = rows[: counts["needs_you"]]
    assert {r["id"] for r in head} == needs
    assert not any(_needs_you(r) for r in rows[len(head):])
    assert len(rows) == len(by_updated)
    # Each part keeps the default order (updated, newest first).
    order = [r["id"] for r in by_updated]
    for part in (head, rows[len(head):]):
        ids_in_part = [r["id"] for r in part]
        assert ids_in_part == [i for i in order if i in set(ids_in_part)]

    head_nb2 = [r["id"] for r in in_nb2[:3]]
    assert set(head_nb2) == {ids["gated2"], ids["stalled"], ids["empty"]}
    assert not any(_needs_you(r) for r in in_nb2[3:])


async def test_rows_say_whether_they_stopped(env) -> None:
    ids = await _seed()
    async with _client(_app()) as c:
        rows = await _paged_sources(c, [])
        stalled = await _paged_sources(c, [("stalled", "true")])
    flagged = {r["id"] for r in rows if r["stalled"]}
    assert flagged == {r["id"] for r in stalled} == {ids["stalled"], ids["empty"]}


async def test_counts_for_an_unknown_notebook(env) -> None:
    async with _client(_app()) as c:
        r = await c.get("/api/sources/counts", params={"notebook_id": "notebook:nope"})
        bad = await c.get("/api/sources/", params={"sort_by": "title"})
    assert r.status_code == 404
    assert bad.status_code == 400
