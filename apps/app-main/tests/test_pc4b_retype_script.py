"""PC.4b: the retype script's plan — what it changes, flags and refuses.

The script is a one-off, but it writes to the live graph, so the decision logic
is pinned: a retype that would give an ACTIVE row the (name_key, type) another
active row already holds is refused (that is a merge, and the unique identity
index would reject it); a literal is flagged; an unchanged row is left alone.
The rows are shaped like the script's own SELECT returns them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "scripts" / "pc4b_retype_entities.py"
_spec = importlib.util.spec_from_file_location("pc4b_retype_entities", _PATH)
retype = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retype)

pytestmark = pytest.mark.asyncio


def _row(i, name, key, etype, label, status="active", flagged=None):
    return {"id": f"entity:{i}", "canonical_name": name, "name_key": key,
            "entity_type": etype, "status": status, "primary_type": label,
            "raw": None, "flagged": flagged}


async def _plan(monkeypatch, rows):
    async def _query(sql, params=None, config=None):
        assert "FROM entity" in sql
        return rows

    monkeypatch.setattr(retype, "execute_query", _query)
    return await retype.plan()


async def test_a_mapping_miss_is_retyped(monkeypatch):
    result = await _plan(monkeypatch, [_row(1, "Waterschap Aa", "waterschap aa", "concept", "Waterschap")])
    [c] = result["changes"]
    assert (c["old"], c["new"]) == ("concept", "government_organization")


async def test_a_retype_onto_an_occupied_identity_is_refused(monkeypatch):
    rows = [
        _row(1, "Plaats X", "plaats x", "concept", "Plaats"),
        _row(2, "Plaats X", "plaats x", "location", "Location"),
    ]
    result = await _plan(monkeypatch, rows)
    assert result["changes"] == []
    assert [c["id"] for c in result["collisions"]] == ["entity:1"]


async def test_two_rows_retyped_onto_one_identity_only_the_first_goes(monkeypatch):
    rows = [
        _row(1, "Overheid Y", "overheid y", "concept", "Overheid"),
        _row(2, "Overheid Y", "overheid y", "other", "GovernmentBody"),
    ]
    result = await _plan(monkeypatch, rows)
    assert [c["id"] for c in result["changes"]] == ["entity:1"]
    assert [c["id"] for c in result["collisions"]] == ["entity:2"]


async def test_a_reference_row_never_collides(monkeypatch):
    """Its identity key is its own id (migration 79), so the index cannot clash."""
    rows = [
        _row(1, "Plaats X", "plaats x", "concept", "Plaats", status="reference"),
        _row(2, "Plaats X", "plaats x", "location", "Location"),
    ]
    result = await _plan(monkeypatch, rows)
    assert [c["id"] for c in result["changes"]] == ["entity:1"]


async def test_a_literal_goes_to_other_and_is_flagged(monkeypatch):
    rows = [_row(1, "2024", "2024", "concept", "Date"),
            _row(2, "EZK", "ezk", "other", "ABBREVIATION")]
    result = await _plan(monkeypatch, rows)
    [c] = result["changes"]
    assert (c["new"], c["non_type_label"]) == ("other", True)
    [f] = result["flag_only"]
    assert f["id"] == "entity:2"


async def test_an_unchanged_or_unlabelled_row_is_left_alone(monkeypatch):
    rows = [_row(1, "Jan", "jan", "person", "Person"),
            _row(2, "X", "x", "concept", None),
            _row(3, "EZK", "ezk", "other", "ABBREVIATION", flagged=True)]
    result = await _plan(monkeypatch, rows)
    assert result["changes"] == result["collisions"] == result["flag_only"] == []
