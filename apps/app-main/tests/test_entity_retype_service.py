"""OE.4: the retype after a publish — dry run, an apply of exactly the plan the
dry run showed, undo — against a real SurrealDB, through the registry's
database-first read (as ``test_ontology_edit_service.py``).

The graph here is small and built per test: a label the shipped YAML types
``concept`` (``Streekfonds``) that a publish moves under ``Grant``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter

import httpx
import pytest
import surrealdb_service.connection as conn
from app_main.api.routers import ontologies as ontologies_router
from app_main.dependencies import get_entity_retype_service
from app_main.services.entity_persistence_service import (
    _resolve_entity_type,
    load_typing_ontologies,
    typing_vocabulary,
)
from app_main.services.entity_retype_service import EntityRetypeService, fingerprint
from app_main.services.ontology_edit_service import (
    OntologyEditError,
    OntologyEditService,
)
from fastapi import FastAPI
from ontology_manager.registry import OntologyRegistry, get_ontology_registry

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

_LOAD_FROM_DB = OntologyRegistry._load_from_db
_LIST_ONTOLOGIES = OntologyRegistry.list_ontologies
_CLEAN = ("DELETE ontology_draft; DELETE ontology_version; DELETE entity; "
          "DELETE notebook_schema;")


@pytest.fixture
async def env(live_surrealdb, monkeypatch, tmp_path):
    monkeypatch.setattr(OntologyRegistry, "_load_from_db", _LOAD_FROM_DB)
    monkeypatch.setattr(OntologyRegistry, "list_ontologies", _LIST_ONTOLOGIES)
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    registry = get_ontology_registry()
    registry.clear_cache()
    await conn.execute_query(_CLEAN, {})
    yield OntologyEditService(), EntityRetypeService(rollback_dir=tmp_path / "retype_rollback")
    await conn.execute_query(_CLEAN, {})
    registry.clear_cache()
    await conn._pool.close_all()
    conn._pool = original


async def _entity(name, etype, label, status="active"):
    [row] = await conn.execute_query(
        "CREATE entity CONTENT $c RETURN id;",
        {"c": {"canonical_name": name, "name_key": name.lower(), "entity_type": etype,
               "status": status, "primary_type": label, "properties": {}, "embedding": []}},
    )
    return str(row["id"])


async def _types():
    rows = await conn.execute_query("SELECT id, entity_type FROM entity;", {})
    return {str(r["id"]): r["entity_type"] for r in rows}


async def _publish_streekfonds_under_grant(editor):
    await editor.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    return await editor.publish("deals")


async def _graph():
    """A row the publish moves; a row it would move onto an identity another row
    holds (a collision); a row nothing moves."""
    moved = await _entity("Fonds Noord", "concept", "Streekfonds")
    refused = await _entity("Fonds Zuid", "concept", "Streekfonds")
    holder = await _entity("Fonds Zuid", "grant", "Grant")
    stable = await _entity("Jan Jansen", "person", "Person")
    return moved, refused, holder, stable


async def test_after_publish_and_apply_the_dry_run_is_empty_and_no_identity_splits(env):
    """The OE.4 AC. "No new identity split": every row a new mention of it would
    be typed onto is the row itself — the label types, under the active
    ontologies, exactly as the row is stored — except the collision, which is
    left for the curator's merge and not touched."""
    editor, retype = env
    moved, refused, holder, stable = await _graph()
    assert (await retype.dry_run())["counts"] == {"changes": 0, "collisions": 0, "flag_only": 0}

    published = await _publish_streekfonds_under_grant(editor)
    shown = await retype.dry_run()
    assert [(c["id"], c["old"], c["new"]) for c in shown["changes"]] == [(moved, "concept", "grant")]
    assert [(c["id"], c["held_by"]) for c in shown["collisions"]] == [(refused, [holder])]
    # The producer (publish) and its consumer (this dry run) count the same plan.
    assert {k: published["retype"][k] for k in shown["counts"]} == shown["counts"]
    assert published["retype"]["caused_by_this_edit"] == 1

    outcome = await retype.apply(shown["fingerprint"])
    assert outcome["written"] == 1 and outcome["skipped"] == outcome["failed"] == []
    assert outcome["rollback_id"]

    after = await retype.dry_run()
    assert after["changes"] == after["flag_only"] == []
    assert [c["id"] for c in after["collisions"]] == [refused]  # left, not merged
    assert await _types() == {moved: "grant", refused: "concept", holder: "grant",
                              stable: "person"}
    vocabulary = typing_vocabulary(None, await load_typing_ontologies())
    rows = await conn.execute_query(
        "SELECT id, name_key, entity_type, primary_type FROM entity;", {})
    assert not [k for k, n in Counter((r["name_key"], r["entity_type"]) for r in rows).items()
                if n > 1]
    for r in rows:
        if str(r["id"]) != refused:
            assert _resolve_entity_type(r["primary_type"], vocabulary).entity_type == r["entity_type"]


async def test_undo_restores_what_the_apply_wrote(env):
    editor, retype = env
    moved, *_ = await _graph()
    await _publish_streekfonds_under_grant(editor)
    shown = await retype.dry_run()
    outcome = await retype.apply(shown["fingerprint"])
    back = await retype.undo(outcome["rollback_id"])
    assert back["written"] == 1
    assert (await _types())[moved] == "concept"
    assert (await retype.dry_run())["fingerprint"] == shown["fingerprint"]


async def test_an_apply_of_a_plan_that_changed_since_the_dry_run_writes_nothing(env):
    """OE-D3 (a): the owner confirms the plan they saw. A row added since, or
    the typing changed since (a rollback), makes it another plan: refused."""
    editor, retype = env
    moved, *_ = await _graph()
    await _publish_streekfonds_under_grant(editor)
    shown = await retype.dry_run()

    late = await _entity("Fonds West", "concept", "Streekfonds")
    with pytest.raises(OntologyEditError) as refused:
        await retype.apply(shown["fingerprint"])
    assert refused.value.status == 409
    assert (await _types())[moved] == (await _types())[late] == "concept"

    shown = await retype.dry_run()
    seed = next(v["version"] for v in await get_ontology_registry().list_versions("deals")
                if v["note"] == "YAML seed")
    await editor.rollback("deals", seed)
    with pytest.raises(OntologyEditError) as refused:
        await retype.apply(shown["fingerprint"])
    assert refused.value.status == 409
    assert (await _types())[moved] == "concept"


async def test_an_apply_waits_for_a_publish_under_way(env, monkeypatch):
    """The apply shares the publish lock. Here the rows are typed ``grant`` while
    the active typing says ``concept``, so the dry run offers grant → concept;
    a publish then makes ``grant`` right. An apply that ran during that publish
    would write the pre-publish plan onto rows the new typing wants left alone.
    Under the lock it waits, sees the plan changed, and writes nothing."""
    from app_main.services import ontology_edit_service as mod

    editor, retype = env
    ahead = await _entity("Fonds Oost", "grant", "Streekfonds")
    shown = await retype.dry_run()
    assert [(c["id"], c["new"]) for c in shown["changes"]] == [(ahead, "concept")]

    real_preview = mod.preview

    async def _slow_preview(*args, **kwargs):
        await asyncio.sleep(0.3)
        return await real_preview(*args, **kwargs)

    monkeypatch.setattr(mod, "preview", _slow_preview)
    await editor.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    publishing = asyncio.create_task(editor.publish("deals"))
    await asyncio.sleep(0.1)  # the publish holds the lock, in its preview
    with pytest.raises(OntologyEditError) as refused:
        await retype.apply(shown["fingerprint"])
    await publishing
    assert refused.value.status == 409
    assert (await _types())[ahead] == "grant"
    assert (await retype.dry_run())["changes"] == []


async def test_stored_notebook_edits_do_not_hold_up_an_apply(env):
    """OE.5 retired the per-notebook layer: typing projects nothing per notebook,
    so a notebook row that still stores old edits no longer refuses the apply
    (OE.4 refused it with a 409)."""
    editor, retype = env
    moved, *_ = await _graph()
    await _publish_streekfonds_under_grant(editor)
    await conn.execute_query(
        "CREATE notebook_schema CONTENT $c;",
        {"c": {"notebook": conn.ensure_record_id("notebook:oe4"), "base_ontology": "deals",
               "accepted_extensions": [
                   {"type_name": "Streekfonds"},
                   {"op": "reparent", "type_name": "Streekfonds", "new_parent": "Deal",
                    "parent_type": "Deal"}]}},
    )
    shown = await retype.dry_run()
    assert "notebook_edits" not in shown
    outcome = await retype.apply(shown["fingerprint"])
    assert outcome["written"] >= 1
    assert (await _types())[moved] == "grant"


async def test_undo_names_only_files_in_the_rollback_directory(env, tmp_path):
    """The rollback id is a bare file name the apply wrote: a path reaching
    outside the rollback directory is refused, and its rows are not written."""
    _editor, retype = env
    victim = await _entity("Fonds Noord", "grant", "Streekfonds")
    retype.rollback_dir.mkdir(parents=True)
    outside = tmp_path / "pc4b_retype_20260928T120000Z.jsonl"
    outside.write_text(json.dumps({"id": victim, "old": "concept", "new": "grant",
                                   "old_flag": None, "new_flag": None}) + "\n")
    with pytest.raises(OntologyEditError) as refused:
        await retype.undo(f"../{outside.name}")
    assert refused.value.status == 422
    assert (await _types())[victim] == "grant"
    with pytest.raises(OntologyEditError) as missing:
        await retype.undo("pc4b_retype_20260928T120001Z.jsonl")
    assert missing.value.status == 404


async def test_the_fingerprint_is_the_plan_not_its_order_or_total():
    item = {"id": "entity:1", "old": "concept", "new": "grant", "old_flag": None, "new_flag": None}
    other = {**item, "id": "entity:2"}
    plan = {"changes": [item, other], "flag_only": [], "collisions": [], "total": 5}
    assert fingerprint(plan) == fingerprint({**plan, "changes": [other, item], "total": 9})
    assert fingerprint(plan) != fingerprint({**plan, "changes": [item, {**other, "new": "other"}]})
    assert fingerprint(plan) != fingerprint({**plan, "changes": [item], "flag_only": [other]})


async def test_the_http_surface(env):
    editor, retype = env
    moved, *_ = await _graph()
    await _publish_streekfonds_under_grant(editor)
    app = FastAPI()
    app.include_router(ontologies_router.router, prefix="/api")
    app.dependency_overrides[get_entity_retype_service] = lambda: retype
    base = "/api/ontologies/retype"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        shown = (await client.get(f"{base}/dry-run")).json()
        assert shown["counts"]["changes"] == 1
        assert (await client.post(f"{base}/apply", json={})).status_code == 422
        assert (await client.post(f"{base}/apply",
                                  json={"fingerprint": uuid.uuid4().hex})).status_code == 409
        assert (await _types())[moved] == "concept"
        applied = await client.post(f"{base}/apply", json={"fingerprint": shown["fingerprint"]})
        assert applied.status_code == 200 and applied.json()["written"] == 1
        assert (await client.get(f"{base}/dry-run")).json()["counts"]["changes"] == 0
        undone = await client.post(f"{base}/undo/{applied.json()['rollback_id']}")
        assert undone.status_code == 200 and undone.json()["written"] == 1
        assert (await client.post(f"{base}/undo/nope")).status_code == 422
