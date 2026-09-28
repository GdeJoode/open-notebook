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
    assert change["source"] == "source:a"
    assert "government" in change["after"] and "government" not in change["before"]
    assert await svc.detection_changes(active, active) == []


async def test_a_row_that_moves_releases_the_identity_it_held():
    """Review m11: two rows swapping types were reported as a collision."""
    vocabulary = vocabulary_of(await load_own_ontologies())
    rows = [
        {"id": "entity:1", "canonical_name": "X", "name_key": "x", "entity_type": "concept",
         "status": "active", "primary_type": "Waterschap", "raw": None, "flagged": None},
        {"id": "entity:2", "canonical_name": "X", "name_key": "x",
         "entity_type": "government_organization", "status": "active",
         "primary_type": "Date", "raw": None, "flagged": None},
    ]
    plan = plan_rows(rows, vocabulary)
    assert {(c["id"], c["new"]) for c in plan["changes"]} == {
        ("entity:1", "government_organization"), ("entity:2", "other")}
    assert plan["collisions"] == []
