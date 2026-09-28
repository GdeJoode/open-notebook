"""OE.1: the preview — what an edit would do, measured before it takes effect.

The active set is the shipped YAML (the conftest keeps the registry off any
database); the candidate is that set with one edit. Entity rows are shaped as
``retype_planner.ENTITY_ROWS`` returns them.
"""

from __future__ import annotations

import pytest
from app_main.services.entity_persistence_service import _resolve_entity_type
from app_main.services.ontology_preview_service import (
    load_own_ontologies,
    preview,
    vocabulary_of,
)
from app_main.services.retype_planner import plan_rows

pytestmark = pytest.mark.asyncio


def _row(i, name, etype, label):
    return {"id": f"entity:{i}", "canonical_name": name, "name_key": name.lower(),
            "entity_type": etype, "status": "active", "primary_type": label,
            "raw": None, "flagged": None}


ROWS = [
    _row(1, "Gezonde Huiskamer", "programme", "Project"),
    _row(2, "Dutch TechZone", "programme", "Project"),
    _row(3, "Waterschap Aa", "government_organization", "Waterschap"),
    _row(4, "Jan", "person", "Person"),
]


async def _project_back_under_creative_work():
    active = await load_own_ontologies()
    candidate = {k: v.model_copy(deep=True) for k, v in active.items()}
    candidate["deals"].entity_types["Project"].parent_type = "CreativeWork"
    return active, candidate


async def test_an_unchanged_candidate_changes_nothing():
    active = await load_own_ontologies()
    result = await preview(active, active, rows=ROWS, include_detection=False, count_notebook_edits=False)
    assert result["blocking"] is False
    assert result["label_changes"] == []
    assert result["retype"]["changes"] == result["retype"]["caused_by_this_edit"] == 0


async def test_an_edit_shows_the_labels_and_the_entities_it_moves():
    active, candidate = await _project_back_under_creative_work()
    result = await preview(candidate, active, rows=ROWS, include_detection=False, count_notebook_edits=False)
    project = [c for c in result["label_changes"] if c["label"] == "Project"]
    assert project == [{"label": "Project", "before": "programme",
                        "after": "creative_work", "entities": 2}]
    assert result["retype"]["caused_by_this_edit"] == 2
    assert {c["id"] for c in result["retype"]["plan"]["changes"]} == {"entity:1", "entity:2"}


async def test_the_preview_counts_what_the_retype_script_would_do(monkeypatch):
    """The OE.1 AC, through the SCRIPT's own path (review M6): the script builds
    its vocabulary with `typing_vocabulary(None, load_typing_ontologies())` from
    the registry; the preview with `vocabulary_of(own)` from own content. After
    a publish the registry serves the candidate — stand that in, run the
    script's `plan()`, and the counts must match the preview's."""
    import importlib.util
    from pathlib import Path

    from app_main.services import retype_planner
    from ontology_manager.validation import resolve_set

    active, candidate = await _project_back_under_creative_work()

    async def _published():  # what the registry serves once `candidate` is active
        return list(resolve_set(candidate)[0].values())

    async def _rows(sql, params=None, config=None):
        return ROWS

    path = Path(__file__).resolve().parents[3] / "scripts" / "pc4b_retype_entities.py"
    spec = importlib.util.spec_from_file_location("pc4b_retype_entities", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "load_typing_ontologies", _published)
    monkeypatch.setattr(retype_planner, "execute_query", _rows)
    dry_run = await script.plan()

    result = await preview(candidate, active, rows=ROWS, include_detection=False,
                           count_notebook_edits=False)
    assert result["retype"]["changes"] == len(dry_run["changes"]) == 2
    assert result["retype"]["collisions"] == len(dry_run["collisions"])


async def test_already_due_changes_are_not_blamed_on_the_edit():
    """A row whose stored type is stale under the ACTIVE set too is in the plan
    but not caused by this edit."""
    active = await load_own_ontologies()
    stale = [_row(9, "Waterschap Bb", "concept", "Waterschap")]  # active types it gov_org
    result = await preview(active, active, rows=stale, include_detection=False, count_notebook_edits=False)
    assert result["retype"]["changes"] == 1
    assert result["retype"]["caused_by_this_edit"] == 0


async def test_a_blocking_finding_is_reported():
    active = await load_own_ontologies()
    candidate = {k: v.model_copy(deep=True) for k, v in active.items()}
    candidate["policy"].entity_types["Legislation"].schema_org_type = None
    result = await preview(candidate, active, rows=ROWS, include_detection=False, count_notebook_edits=False)
    assert result["blocking"] is True
    assert any(f["rule"] == "base_named_type_misses_its_base" for f in result["findings"])


async def test_the_active_set_is_a_copy():
    """Building a candidate must not edit the registry's shared cached objects."""
    first = await load_own_ontologies()
    first["deals"].entity_types["Project"].parent_type = "CreativeWork"
    again = await load_own_ontologies()
    assert again["deals"].entity_types["Project"].parent_type == "GovernmentService"
    assert _resolve_entity_type("Project", vocabulary_of(again)).entity_type == "programme"


async def test_the_detection_replay_names_documents_whose_selection_changes(monkeypatch):
    """Hints and descriptions drive detection. Give `government` a hint the
    document is full of, and the document's selection changes."""
    from app_main.services import ontology_preview_service as svc
    from surrealdb_service.repositories import NotebookSchemaRepository

    text = "Waterschapsbestuur " * 40 + "brede welvaart leefbaarheid regio deal"

    async def _query(sql, params=None, config=None):
        if "FROM source" in sql:
            return [{"id": "source:a", "title": "Convenant", "dt": None}]
        if "FROM chunk" in sql:
            return [{"source": "source:a", "text": text, "order": 0}]
        if "FROM reference" in sql:
            return [{"s": "source:a", "nb": "notebook:n"}]
        raise AssertionError(sql)

    async def _no_declared_base(self, notebook_id):
        return None

    monkeypatch.setattr(svc, "execute_query", _query)
    monkeypatch.setattr(NotebookSchemaRepository, "get_by_notebook", _no_declared_base)
    active = await load_own_ontologies()
    candidate = {k: v.model_copy(deep=True) for k, v in active.items()}
    for definition in candidate["government"].entity_types.values():
        definition.extraction_hints = ["Waterschapsbestuur"]

    [change] = await svc.detection_changes(active, candidate)
    assert (change["source"], change["notebook"]) == ("source:a", "notebook:n")
    assert "government" in change["after"] and "government" not in change["before"]
    assert await svc.detection_changes(active, active) == []


def _r(i, etype, label):
    return {"id": f"entity:{i}", "canonical_name": "X", "name_key": "x", "entity_type": etype,
            "status": "active", "primary_type": label, "raw": None, "flagged": None}


async def test_no_two_rows_can_end_up_with_one_identity():
    """Review OE.1 round 2 B1: releasing identities within one plan let B take
    the identity a REFUSED row A kept. A wants organization (held by C) and is
    refused, staying person; B must then be refused for person too."""
    vocabulary = vocabulary_of(await load_own_ontologies())
    rows = [_r(1, "person", "Organization"), _r(2, "organization", "Organization"),
            _r(3, "event", "Person")]
    plan = plan_rows(rows, vocabulary)
    assert plan["changes"] == []
    assert {c["id"] for c in plan["collisions"]} == {"entity:1", "entity:3"}
    final = {r["id"]: r["entity_type"] for r in rows}
    final.update({c["id"]: c["new"] for c in plan["changes"]})
    assert len(set(final.values())) == len(final)


async def test_a_chain_resolves_over_two_runs_in_any_write_order():
    """Row 1 wants what row 2 holds, and row 2 moves away. Plan 1 moves row 2
    only (row 1 would trip the unique index if written first); plan 2 moves
    row 1 into the freed identity."""
    vocabulary = vocabulary_of(await load_own_ontologies())
    rows = [_r(1, "concept", "Waterschap"), _r(2, "government_organization", "Date")]
    first = plan_rows(rows, vocabulary)
    assert [(c["id"], c["new"]) for c in first["changes"]] == [("entity:2", "other")]
    assert [c["id"] for c in first["collisions"]] == ["entity:1"]
    after = [dict(rows[0]), {**rows[1], "entity_type": "other"}]
    second = plan_rows(after, vocabulary)
    assert [(c["id"], c["new"]) for c in second["changes"]] == [("entity:1", "government_organization")]
