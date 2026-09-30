"""WB.3: the source-title backfill script against a real SurrealDB.

The script would write the live ``source`` table, so what it plans and what it
writes and undoes are pinned here (it is not run on the live DB; the owner
decides): the dry run writes nothing; only prefixed titles change; an empty
result is left alone; the old-title guard skips a row changed meanwhile; the
rollback file undoes exactly what was written.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_attention_db import _q, _source, env  # noqa: F401  (``env`` is a fixture)

_PATH = Path(__file__).resolve().parents[3] / "scripts" / "wb3_backfill_source_titles.py"
_spec = importlib.util.spec_from_file_location("wb3_backfill_source_titles", _PATH)
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


async def _titles() -> dict:
    return {str(r["id"]): r.get("title") for r in await _q("SELECT id, title FROM source;")}


async def test_plan_apply_and_rollback(env, tmp_path, capsys) -> None:
    ids = {
        "prefixed": await _source("410a65bba3f8__Convenant Regio Deal Waterwegregio", "complete"),
        "copy": await _source("347adc9d1dcd__Convenant Regio Deal Waterwegregio", "complete"),
        "plain": await _source("Convenant Regio Deal Het Hogeland", "complete"),
        "only_prefix": await _source("49b9320d0e1e__", "complete"),
        "moved": await _source("5753cb08f382__Convenant Het Hogeland", "embedded"),
    }
    before = await _titles()

    # Dry run: the plan, and nothing written.
    result = await backfill.plan()
    assert {c["id"] for c in result["changes"]} == {ids["prefixed"], ids["copy"], ids["moved"]}
    assert result["shared"] == ["Convenant Regio Deal Waterwegregio"]
    assert await _titles() == before

    # Someone renames one source between the dry run and the apply.
    await _q("UPDATE type::thing($id) SET title = 'Renamed by the owner';", {"id": ids["moved"]})

    outcome = await backfill.apply(result, rollback_dir=tmp_path)
    assert outcome["written"] == 2 and outcome["skipped"] == [ids["moved"]]
    after = await _titles()
    assert after[ids["prefixed"]] == after[ids["copy"]] == "Convenant Regio Deal Waterwegregio"
    assert after[ids["plain"]] == "Convenant Regio Deal Het Hogeland"
    assert after[ids["only_prefix"]] == "49b9320d0e1e__"
    assert after[ids["moved"]] == "Renamed by the owner"
    log = Path(outcome["rollback_file"])
    assert {json.loads(line)["id"] for line in log.read_text().splitlines()} == {
        ids["prefixed"], ids["copy"]}

    undone = await backfill.rollback(log)
    assert undone["written"] == 2
    restored = await _titles()
    assert restored[ids["prefixed"]] == before[ids["prefixed"]]
    assert restored[ids["copy"]] == before[ids["copy"]]
    assert restored[ids["moved"]] == "Renamed by the owner"
