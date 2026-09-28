"""PC.4b: the retype script — what it plans, and what it writes and undoes.

The script writes to the live graph, so both halves are pinned:

* the plan (rows shaped as the script's own SELECT returns them): which label
  it types, which changes it refuses, what it flags;
* the writes, against a REAL SurrealDB (the container): the old-type guard,
  the flag, a failing row not aborting the run, and the rollback file undoing
  exactly what was written.
"""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path

import pytest
import surrealdb_service.connection as connection_module
from surrealdb_service.config import SurrealDBConfig

_PATH = Path(__file__).resolve().parents[3] / "scripts" / "pc4b_retype_entities.py"
_spec = importlib.util.spec_from_file_location("pc4b_retype_entities", _PATH)
retype = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retype)

pytestmark = pytest.mark.asyncio


def _row(i, name, key, etype, label, status="active", flagged=None, raw=None):
    return {"id": f"entity:{i}", "canonical_name": name, "name_key": key,
            "entity_type": etype, "status": status, "primary_type": label,
            "raw": raw, "flagged": flagged}


async def _plan(monkeypatch, rows):
    """The script's plan, reading `rows` where it reads the entity table (the
    planner module, shared with the ontology editor's preview since OE.1)."""
    from app_main.services import retype_planner

    async def _query(sql, params=None, config=None):
        assert "FROM entity" in sql
        return rows

    monkeypatch.setattr(retype_planner, "execute_query", _query)
    return await retype.plan()


# -- the plan ------------------------------------------------------------------


async def test_a_mapping_miss_is_retyped(monkeypatch):
    result = await _plan(monkeypatch, [_row(1, "Waterschap Aa", "waterschap aa", "concept", "Waterschap")])
    [c] = result["changes"]
    assert (c["old"], c["new"]) == ("concept", "government_organization")


async def test_the_raw_label_is_typed_as_persistence_types_it(monkeypatch):
    """Review M1: 99 live rows carry `primary_type = "other"` (a canonical name)
    beside the raw LLM label persistence actually typed."""
    rows = [_row(1, "Ministerie van BZK", "ministerie van bzk", "other", "other",
                 raw="Ministerie")]
    [c] = (await _plan(monkeypatch, rows))["changes"]
    assert (c["label"], c["new"]) == ("Ministerie", "government_organization")


async def test_a_retype_onto_a_held_identity_is_refused(monkeypatch):
    rows = [
        _row(1, "Plaats X", "plaats x", "concept", "Plaats"),
        _row(2, "Plaats X", "plaats x", "location", "Location"),
    ]
    result = await _plan(monkeypatch, rows)
    assert result["changes"] == []
    [c] = result["collisions"]
    assert (c["id"], c["held_by"]) == ("entity:1", ["entity:2"])


async def test_a_reference_row_holding_the_identity_also_refuses(monkeypatch):
    """Review M2: no index clash, but `upsert_entity` looks up on (name_key,
    type) without a status filter, so the pair would split new mentions."""
    rows = [
        _row(1, "Justitie en Veiligheid", "justitie en veiligheid", "concept", "Overheid"),
        _row(2, "Justitie en Veiligheid", "justitie en veiligheid",
             "government_organization", "Ministerie", status="reference"),
    ]
    result = await _plan(monkeypatch, rows)
    assert result["changes"] == [] and [c["id"] for c in result["collisions"]] == ["entity:1"]


async def test_two_rows_retyped_onto_one_identity_only_the_first_goes(monkeypatch):
    rows = [
        _row(1, "Overheid Y", "overheid y", "concept", "Overheid"),
        _row(2, "Overheid Y", "overheid y", "other", "GovernmentBody", status="reference"),
    ]
    result = await _plan(monkeypatch, rows)
    assert [c["id"] for c in result["changes"]] == ["entity:1"]
    assert [c["id"] for c in result["collisions"]] == ["entity:2"]


async def test_a_literal_goes_to_other_and_is_flagged(monkeypatch):
    rows = [_row(1, "2024", "2024", "concept", "Date"),
            _row(2, "EZK", "ezk", "other", "ABBREVIATION")]
    result = await _plan(monkeypatch, rows)
    [c] = result["changes"]
    assert (c["new"], c["new_flag"], c["old_flag"]) == ("other", True, None)
    [f] = result["flag_only"]
    assert f["id"] == "entity:2"


async def test_an_unchanged_or_unlabelled_row_is_left_alone(monkeypatch):
    rows = [_row(1, "Jan", "jan", "person", "Person"),
            _row(2, "X", "x", "concept", None),
            _row(3, "EZK", "ezk", "other", "ABBREVIATION", flagged=True)]
    result = await _plan(monkeypatch, rows)
    assert result["changes"] == result["collisions"] == result["flag_only"] == []


# -- the writes, against a real SurrealDB ---------------------------------------


@pytest.fixture
def bound(live_surrealdb: SurrealDBConfig, monkeypatch):
    real = connection_module.execute_query

    async def _bound(q, params=None, config=None):
        return await real(q, params, config or live_surrealdb)

    monkeypatch.setattr(retype, "execute_query", _bound)
    return _bound


async def _entity(bound, etype, flagged=None, name_key=None):
    props = {} if flagged is None else {"non_type_label": flagged}
    [row] = await bound(
        "CREATE entity CONTENT $c RETURN id;",
        {"c": {"canonical_name": f"R {uuid.uuid4().hex[:6]}",
               "name_key": name_key or uuid.uuid4().hex,
               "entity_type": etype, "status": "active", "properties": props, "embedding": []}},
    )
    return str(row["id"])


async def _state(bound, rid):
    [row] = await bound("SELECT entity_type, properties.non_type_label AS f FROM $id;",
                        {"id": connection_module.ensure_record_id(rid)})
    return row["entity_type"], row.get("f")


@pytest.mark.requires_docker
async def test_apply_writes_guards_and_rolls_back(bound, tmp_path):
    a = await _entity(bound, "concept")               # a type change
    b = await _entity(bound, "concept")               # a literal: type + flag
    c = await _entity(bound, "other", flagged=None)   # flag only
    d = await _entity(bound, "location")              # changed since the plan
    # A write the database refuses: `idx_entity_identity` (active rows).
    shared = uuid.uuid4().hex
    await _entity(bound, "organization", name_key=shared)
    e = await _entity(bound, "concept", name_key=shared)
    result = {
        "changes": [
            {"id": a, "old": "concept", "new": "government_organization", "old_flag": None, "new_flag": None},
            {"id": b, "old": "concept", "new": "other", "old_flag": None, "new_flag": True},
            {"id": d, "old": "concept", "new": "organization", "old_flag": None, "new_flag": None},
            {"id": e, "old": "concept", "new": "organization", "old_flag": None, "new_flag": None},
        ],
        "flag_only": [{"id": c, "old": "other", "new": "other", "old_flag": None, "new_flag": True}],
    }
    outcome = await retype.apply(result, rollback_dir=tmp_path)

    assert await _state(bound, a) == ("government_organization", None)
    assert await _state(bound, b) == ("other", True)
    assert await _state(bound, c) == ("other", True)
    assert await _state(bound, d) == ("location", None)  # guard: left alone
    assert outcome["written"] == 3 and outcome["skipped"] == [d]
    # The refused row is reported and did not abort the run (c came after it).
    assert [f["id"] for f in outcome["failed"]] == [e]
    assert await _state(bound, e) == ("concept", None)

    logged = [json.loads(line) for line in Path(outcome["rollback_file"]).read_text().splitlines()]
    assert sorted(r["id"] for r in logged) == sorted([a, b, c])  # what was WRITTEN

    back = await retype.rollback(Path(outcome["rollback_file"]))
    assert back["written"] == 3
    assert await _state(bound, a) == ("concept", None)
    assert await _state(bound, b) == ("concept", None)
    assert await _state(bound, c) == ("other", None)
