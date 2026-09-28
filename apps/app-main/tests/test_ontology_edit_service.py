"""OE.2: editing the shared ontologies — draft, validate, publish, versions,
rollback — against a real SurrealDB, through the registry's database-first read
(the unit-test conftest keeps the registry on the YAML files; undone here).
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import surrealdb_service.connection as conn
from app_main.api.routers import ontologies as ontologies_router
from app_main.dependencies import get_ontology_edit_service
from app_main.services.entity_persistence_service import (
    _resolve_entity_type,
    load_typing_ontologies,
    typing_vocabulary,
)
from app_main.services.ontology_edit_service import (
    OntologyEditError,
    OntologyEditService,
)
from fastapi import FastAPI
from ontology_manager.registry import OntologyRegistry, get_ontology_registry
from ontology_manager.validation import labels_of

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

_LOAD_FROM_DB = OntologyRegistry._load_from_db
_LIST_ONTOLOGIES = OntologyRegistry.list_ontologies


@pytest.fixture
async def svc(live_surrealdb, monkeypatch):
    monkeypatch.setattr(OntologyRegistry, "_load_from_db", _LOAD_FROM_DB)
    monkeypatch.setattr(OntologyRegistry, "list_ontologies", _LIST_ONTOLOGIES)
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    registry = get_ontology_registry()
    registry.clear_cache()
    await conn.execute_query("DELETE ontology_draft; DELETE ontology_version;", {})
    yield OntologyEditService()
    await conn.execute_query("DELETE ontology_draft; DELETE ontology_version;", {})
    registry.clear_cache()
    await conn._pool.close_all()
    conn._pool = original


async def _typed(label):
    return _resolve_entity_type(label, typing_vocabulary(None, await load_typing_ontologies())).entity_type


async def _typing_map():
    """The type of EVERY declared label, as extraction types it now (through the
    registry's cache — warm it first to prove a publish reaches it)."""
    vocabulary = typing_vocabulary(None, await load_typing_ontologies())
    labels = {label for o in vocabulary for label, _ in labels_of(o)}
    return {label: _resolve_entity_type(label, vocabulary).entity_type for label in labels}


async def _active_versions(name):
    return [v["version"] for v in await get_ontology_registry().list_versions(name) if v["is_active"]]


async def test_a_draft_starts_from_the_active_content_and_changes_nothing_active(svc):
    draft = await svc.get_draft("deals")
    assert draft["exists"] is False and "Project" in draft["ontology"].entity_types
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    assert (await svc.get_draft("deals"))["exists"] is True
    assert await _typed("Streekfonds") == "concept"  # not published: nothing changed


async def test_publish_makes_the_draft_the_active_typing(svc):
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    result = await svc.publish("deals", note="a grant type")
    assert await _typed("Streekfonds") == "grant"
    versions = await get_ontology_registry().list_versions("deals")
    assert {v["note"] for v in versions} == {"YAML seed", "a grant type"}
    assert await _active_versions("deals") == [result["version"]]
    assert (await svc.get_draft("deals"))["exists"] is False  # the draft is consumed


async def test_a_blocking_finding_refuses_the_publish_and_keeps_the_draft(svc):
    draft = await svc.get_draft("policy")
    fields = draft["ontology"].entity_types["Legislation"].model_dump(exclude_unset=True)
    fields["schema_org_type"] = None
    await svc.put_entity_type("policy", "Legislation", fields)
    assert any(f["rule"] == "base_named_type_misses_its_base" and f["severity"] == "block"
               for f in await svc.validate_draft("policy"))
    with pytest.raises(OntologyEditError) as refused:
        await svc.publish("policy")
    assert refused.value.status == 422
    assert (await svc.get_draft("policy"))["exists"] is True
    assert await get_ontology_registry().list_versions("policy") == []
    assert await _typed("Legislation") == "legislation"


async def test_a_publish_on_a_stale_draft_is_refused(svc):
    """Optimistic concurrency: a version published after the draft was started
    makes the draft stale; publishing it would overwrite that version's work."""
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    underneath = (await svc.get_draft("deals"))["ontology"].model_copy(deep=True)
    underneath.metadata.version = "1.0.0+published-meanwhile"
    assert await get_ontology_registry().register(underneath, activate=True)
    with pytest.raises(OntologyEditError) as refused:
        await svc.publish("deals")
    assert refused.value.status == 409
    assert await _active_versions("deals") == ["1.0.0+published-meanwhile"]


async def test_a_draft_is_published_once_even_when_two_publishes_race(svc, monkeypatch):
    """OE.0 review minor 2 / OE.2: two publishes of the same draft that both get
    past the checks before either stores would store it twice. Forced here: the
    preview yields control mid-publish, and each publish gets its own timestamp.
    The lock lets exactly one through; the other finds no draft."""
    from datetime import datetime, timedelta, timezone

    from app_main.services import ontology_edit_service as mod

    real_preview = mod.preview

    async def _slow_preview(*args, **kwargs):
        await asyncio.sleep(0.3)
        return await real_preview(*args, **kwargs)

    ticks = iter(range(100))

    class _Clock:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 9, 28, tzinfo=timezone.utc) + timedelta(seconds=next(ticks))

    monkeypatch.setattr(mod, "preview", _slow_preview)
    monkeypatch.setattr(mod, "datetime", _Clock)
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    results = await asyncio.gather(svc.publish("deals"), svc.publish("deals"),
                                   return_exceptions=True)
    assert sum(1 for r in results if isinstance(r, dict)) == 1
    assert [r.status for r in results if isinstance(r, OntologyEditError)] == [404]
    versions = await get_ontology_registry().list_versions("deals")
    assert len(versions) == 2  # the seed and ONE publish
    assert len(await _active_versions("deals")) == 1


async def test_publishes_of_different_ontologies_do_not_interfere(svc):
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    await svc.put_entity_type("general", "Werkplaats", {"parent_type": "Place"})
    await asyncio.gather(svc.publish("deals"), svc.publish("general"))
    for name in ("deals", "general"):
        assert len(await _active_versions(name)) == 1
    assert await _typed("Streekfonds") == "grant" and await _typed("Werkplaats") == "location"


async def test_rename_keeps_the_old_name_resolving(svc):
    await svc.rename_entity_type("deals", "Project", "Programmaproject")
    draft = (await svc.get_draft("deals"))["ontology"]
    assert "Project" in draft.entity_types["Programmaproject"].aliases
    await svc.publish("deals")
    assert await _typed("Project") == await _typed("Programmaproject") == "programme"


async def test_rollback_restores_the_previous_typing(svc):
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    await svc.publish("deals")
    seed = next(v["version"] for v in await get_ontology_registry().list_versions("deals")
                if v["note"] == "YAML seed")
    await svc.rollback("deals", seed)
    assert await _typed("Streekfonds") == "concept"
    assert await _active_versions("deals") == [seed]


async def test_a_rollback_that_would_block_is_refused(svc):
    registry = get_ontology_registry()
    broken = (await svc.get_draft("policy"))["ontology"]
    broken.entity_types["Legislation"].schema_org_type = None
    broken.metadata.version = "0.0.1-broken"
    assert await registry.register(broken)  # stored, not active
    with pytest.raises(OntologyEditError) as refused:
        await svc.rollback("policy", "0.0.1-broken")
    assert refused.value.status == 422
    assert await _typed("Legislation") == "legislation"


async def test_export_is_the_active_content(svc):
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    await svc.publish("deals")
    assert "Streekfonds" in await svc.export_yaml("deals")


async def test_the_http_round_trip(svc):
    app = FastAPI()
    app.include_router(ontologies_router.router, prefix="/api")
    app.dependency_overrides[get_ontology_edit_service] = lambda: svc
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        r = await client.put("/api/ontologies/deals/draft/entity-types/Streekfonds",
                             json={"parent_type": "Grant"})
        assert r.status_code == 200 and "Streekfonds" in r.json()["ontology"]["entity_types"]
        r = await client.post("/api/ontologies/deals/draft/preview",
                              params={"include_detection": "false"})
        assert r.status_code == 200 and r.json()["blocking"] is False
        r = await client.post("/api/ontologies/deals/draft/publish", json={"note": "via http"})
        assert r.status_code == 200
        r = await client.post("/api/ontologies/deals/draft/publish", json={})
        assert r.status_code == 404  # no draft left
        r = await client.get("/api/ontologies/deals/versions")
        assert {v["note"] for v in r.json()} == {"YAML seed", "via http"}
        r = await client.put("/api/ontologies/deals/draft/entity-types/X",
                             json={"parent_type": ["not", "a", "string"]})
        assert r.status_code == 422


async def test_a_draft_takes_many_edits(svc):
    """Review OE.2 B1: `UPSERT ... CONTENT` wiped `created_at` on the second
    edit, and every later edit of a draft failed."""
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    await svc.put_entity_type("deals", "Tweedefonds", {"parent_type": "Grant"})
    await svc.rename_entity_type("deals", "Tweedefonds", "Derdefonds")
    await svc.delete_entity_type("deals", "Streekfonds")
    types = (await svc.get_draft("deals"))["ontology"].entity_types
    assert "Derdefonds" in types and "Streekfonds" not in types and "Tweedefonds" not in types


async def test_a_publish_reaches_every_label_through_a_warm_cache(svc):
    """Review M4: the cache the publish must drop was never filled, so removing
    `clear_cache` passed. Warm it, publish, and every label types as the
    candidate says — then roll back and every label types as before."""
    from app_main.services.ontology_preview_service import (
        load_own_ontologies,
        vocabulary_of,
    )

    before = await _typing_map()  # warms the merged cache for every ontology
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    project = (await svc.get_draft("deals"))["ontology"].entity_types["Project"]
    await svc.put_entity_type("deals", "Project", {**project.model_dump(exclude_unset=True),
                                                   "parent_type": "CreativeWork"})
    candidate = {**await load_own_ontologies(), "deals": (await svc.get_draft("deals"))["ontology"]}
    expected_vocabulary = vocabulary_of(candidate)
    await svc.publish("deals")

    after = await _typing_map()
    assert after["Streekfonds"] == "grant" and after["Project"] == "creative_work"
    assert after == {label: _resolve_entity_type(label, expected_vocabulary).entity_type
                     for label in after}

    seed = next(v["version"] for v in await get_ontology_registry().list_versions("deals")
                if v["note"] == "YAML seed")
    await svc.rollback("deals", seed)
    assert await _typing_map() == before  # every label, exactly


async def test_a_read_in_flight_during_a_publish_does_not_re_cache_the_old_ontology(svc, monkeypatch):
    """Review M2: `get` awaits the DB; a publish clearing the cache in between
    let the reader write the PRE-publish ontology back, for good."""
    registry = get_ontology_registry()
    gate = asyncio.Event()
    real = OntologyRegistry._load_from_db
    delayed = {"n": 0}

    async def _slow(self, name, version=None):
        # The reader has already read `deals` (the old content) and now resolves
        # its parent chain: the window a publish lands in.
        if name == "instruments" and delayed["n"] == 0:
            delayed["n"] += 1
            await gate.wait()
        return await real(self, name, version)

    monkeypatch.setattr(OntologyRegistry, "_load_from_db", _slow)
    reader = asyncio.create_task(registry.get("deals"))
    await asyncio.sleep(0.1)  # the reader holds the old `deals`, waiting on its parent
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    await svc.publish("deals")
    gate.set()
    await reader
    assert "Streekfonds" in (await registry.get("deals")).entity_types


async def test_an_edit_during_a_publish_is_not_lost(svc, monkeypatch):
    """Review M3: an edit landing between publish reading the draft and
    deleting it was accepted, then silently dropped with the draft."""
    from app_main.services import ontology_edit_service as mod

    real_preview = mod.preview

    async def _slow_preview(*args, **kwargs):
        await asyncio.sleep(0.3)
        return await real_preview(*args, **kwargs)

    monkeypatch.setattr(mod, "preview", _slow_preview)
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    publishing = asyncio.create_task(svc.publish("deals"))
    await asyncio.sleep(0.1)
    await svc.put_entity_type("deals", "Laatfonds", {"parent_type": "Grant"})
    published = await publishing
    draft = await svc.get_draft("deals")
    assert draft["exists"] and "Laatfonds" in draft["ontology"].entity_types
    assert draft["base_version"] == published["version"]  # a new draft on the new version


async def test_the_seed_is_stored_when_only_inactive_versions_exist(svc):
    """Review minor 6: any stored row used to skip the seed, so the YAML could
    not be rolled back to."""
    imported = (await svc.get_draft("deals"))["ontology"].model_copy(deep=True)
    imported.metadata.version = "0.9-imported"
    assert await get_ontology_registry().register(imported)  # inactive
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    await svc.publish("deals")
    notes = {v["note"] for v in await get_ontology_registry().list_versions("deals")}
    assert "YAML seed" in notes


async def test_activating_a_version_stamps_it(svc):
    await svc.put_entity_type("deals", "Streekfonds", {"parent_type": "Grant"})
    await svc.publish("deals")
    seed = next(v for v in await get_ontology_registry().list_versions("deals")
                if v["note"] == "YAML seed")
    await asyncio.sleep(1.1)
    await svc.rollback("deals", seed["version"])
    [row] = await conn.execute_query(
        "SELECT created_at, updated_at FROM ontology_version WHERE version = $v;",
        {"v": seed["version"]})
    assert row["updated_at"] > row["created_at"]


async def test_rename_moves_this_ontologys_own_references(svc):
    await svc.put_entity_type("deals", "Kindfonds", {"parent_type": "Project"})
    await svc.rename_entity_type("deals", "Project", "Programmaproject")
    types = (await svc.get_draft("deals"))["ontology"].entity_types
    assert types["Kindfonds"].parent_type == "Programmaproject"


async def test_the_http_surface(svc):
    """Review minor 14: every route, the refusals and a second edit."""
    app = FastAPI()
    app.include_router(ontologies_router.router, prefix="/api")
    app.dependency_overrides[get_ontology_edit_service] = lambda: svc
    base = "/api/ontologies/deals"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        assert (await client.put(f"{base}/draft/entity-types/A", json={"parent_type": "Grant"})).status_code == 200
        assert (await client.put(f"{base}/draft/entity-types/B", json={"parent_type": "Grant"})).status_code == 200
        r = await client.post(f"{base}/draft/entity-types/B/rename", json={"new_name": "C"})
        assert r.status_code == 200 and "C" in r.json()["ontology"]["entity_types"]
        assert (await client.post(f"{base}/draft/entity-types/C/rename",
                                  json={"new_name": "  "})).status_code == 422
        assert (await client.delete(f"{base}/draft/entity-types/A")).status_code == 200
        assert (await client.delete(f"{base}/draft/entity-types/Nope")).status_code == 404
        r = await client.put(f"{base}/draft/relationship-types/FUNDS",
                             json={"domain": ["Grant"], "range": ["Nergens"]})
        assert r.status_code == 200
        findings = (await client.post(f"{base}/draft/validate")).json()
        assert any(f["rule"] == "relationship_endpoint_unknown" and f["severity"] == "block"
                   for f in findings)
        assert (await client.post(f"{base}/draft/publish", json={})).status_code == 422
        assert (await client.delete(f"{base}/draft/relationship-types/FUNDS")).status_code == 200
        published = await client.post(f"{base}/draft/publish", json={"note": "via http"})
        assert published.status_code == 200
        r = await client.get(f"{base}/export.yaml")
        assert r.status_code == 200 and "C:" in r.text
        versions = (await client.get(f"{base}/versions")).json()
        seed = next(v["version"] for v in versions if v["note"] == "YAML seed")
        assert (await client.post(f"{base}/rollback/{seed}")).status_code == 200
        assert (await client.post(f"{base}/rollback/nope")).status_code == 404
        # A draft started from the seed, published over by someone else: 409.
        assert (await client.put(f"{base}/draft/entity-types/D", json={"parent_type": "Grant"})).status_code == 200
        underneath = (await svc.get_draft("deals"))["ontology"].model_copy(deep=True)
        underneath.metadata.version = "9.9.9"
        assert await get_ontology_registry().register(underneath, activate=True)
        assert (await client.post(f"{base}/draft/publish", json={})).status_code == 409
        assert (await client.get("/api/ontologies/nope/draft")).status_code == 404
