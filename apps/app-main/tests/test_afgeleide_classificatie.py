"""Afgeleide classificatie (F2-03, ADR 0001) tegen een echte SurrealDB.

Een vermelding (PC.3-herkomst: een bron in ``source_documents``, met het
fragment in ``properties.grounding``) neemt de waarde van haar fragment over.
Een entiteit, alias, relatie en triage-item zijn ``intern`` zodra één
onderliggende vermelding intern is, en ``intern`` zonder vermelding. De
waarde wordt berekend bij elk schrijven van een vermelding en bij elke
herclassificatie.

Alle testdata is synthetisch.
"""

from __future__ import annotations

import uuid

import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main.services.classification_service import ClassificationService
from shared.models.agents import Right
from shared.models.entity import Entity
from surrealdb_service.repositories.entity import EntityRepository
from surrealdb_service.testing.herkomst import herkomst

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


@pytest.fixture
async def env(live_surrealdb, monkeypatch):
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    yield
    await conn._pool.close_all()
    # Zonder eerdere pool een nieuwe op de testdatabase achterlaten: een latere
    # aanroep zonder config zou anders een pool op de standaardinstelling maken.
    conn._pool = original or conn.ConnectionPool(config=live_surrealdb)


async def _q(sql: str, params: dict | None = None):
    return await conn.execute_query(sql, params or {})


async def _bron(classification: str) -> str:
    [row] = await _q(
        "CREATE source SET title = 'Openbare testbron', classification = $c RETURN id",
        {"c": classification},
    )
    return str(row["id"])


async def _fragment(bron: str) -> str:
    [row] = await _q(
        "CREATE chunk CONTENT $c RETURN id",
        {"c": {"source": conn.ensure_record_id(bron), "text": "t", "order": 0,
               "physical_page": 0, "element_type": "paragraph", "positions": [],
               "metadata": {}, **await herkomst(bron)}},
    )
    return str(row["id"])


async def _entiteit(bronnen: list[str], grounding: dict | None = None) -> str:
    """Via ``EntityRepository.upsert_entity``, het schrijfpad van de extractie."""
    naam = f"Testentiteit {uuid.uuid4().hex[:8]}"
    entity = Entity(
        canonical_name=naam,
        entity_type="organization",
        source_documents=bronnen,
        properties={"grounding": grounding} if grounding else {},
    )
    return str(await EntityRepository().upsert_entity(entity))


async def _alias(entiteit: str) -> str:
    [row] = await _q(
        "CREATE entity_alias SET canonical_entity = type::thing($e), alias_text = 'alias',"
        " match_type = 'exact' RETURN id",
        {"e": entiteit},
    )
    return str(row["id"])


async def _relatie(van: str, naar: str, bronnen: list[str]) -> str:
    [row] = await _q(
        "RELATE (type::thing($a))->relation->(type::thing($b))"
        " SET relation_type = 'related_to', source_documents = $s RETURN id",
        {"a": van, "b": naar, "s": bronnen},
    )
    return str(row["id"])


async def _triage(entiteit: str) -> str:
    [row] = await _q(
        "CREATE triage_queue SET entity = type::thing($e), batch_id = 'b', name = 'x',"
        " reason = 'r' RETURN id",
        {"e": entiteit},
    )
    return str(row["id"])


async def _c(rid: str) -> str:
    [row] = await _q("SELECT classification FROM type::thing($id)", {"id": rid})
    return row["classification"]


async def _herclassificeer(bron: str, naar: str) -> None:
    await ClassificationService().reclassify(
        bron, naar, "Synthetische test", permission=Right.BEHEERDER,
        person="Testpersoon", key_id="agent_keys:test",
    )


async def test_een_interne_vermelding_maakt_alles_intern(env) -> None:
    extern, intern = await _bron("extern"), await _bron("intern")
    open_ = await _entiteit([extern])
    gemengd = await _entiteit([extern, intern])
    assert await _c(open_) == "extern"
    assert await _c(gemengd) == "intern"

    assert await _c(await _alias(open_)) == "extern"
    assert await _c(await _alias(gemengd)) == "intern"
    assert await _c(await _triage(open_)) == "extern"
    assert await _c(await _triage(gemengd)) == "intern"
    assert await _c(await _relatie(open_, open_, [extern])) == "extern"
    assert await _c(await _relatie(open_, open_, [extern, intern])) == "intern"
    # Een relatie met een interne entiteit is intern.
    assert await _c(await _relatie(open_, gemengd, [extern])) == "intern"


async def test_een_vermelding_neemt_de_waarde_van_haar_fragment_over(env) -> None:
    extern = await _bron("extern")
    fragment = await _fragment(extern)
    assert await _c(await _entiteit([extern], {extern: {"chunk_id": fragment}})) == "extern"
    # Een fragment strenger dan zijn bron: de vermelding is intern.
    await _q("UPDATE type::thing($id) SET classification = 'intern'", {"id": fragment})
    assert await _c(await _entiteit([extern], {extern: {"chunk_id": fragment}})) == "intern"
    # Een onbekend fragment: sinds F2-04 (migratie 90) geweigerd.
    with pytest.raises(RuntimeError, match="grounding"):
        await _entiteit([extern], {extern: {"chunk_id": "chunk:onbekend"}})


async def test_een_fragment_van_een_andere_bron_maakt_een_interne_vermelding_niet_extern(
    env,
) -> None:
    # Reviewbevinding F2-03: het fragment in `grounding` wijst naar een fragment
    # van een externe bron B, de vermelding staat in interne bron A. Sinds F2-04
    # (migratie 90) weigert de database zo'n vermelding, in beide richtingen.
    intern, extern = await _bron("intern"), await _bron("extern")
    vreemd = await _fragment(extern)
    with pytest.raises(RuntimeError, match="grounding"):
        await _entiteit([intern], {intern: {"chunk_id": vreemd}})
    eigen = await _fragment(intern)
    with pytest.raises(RuntimeError, match="grounding"):
        await _entiteit([extern], {extern: {"chunk_id": eigen}})
    # Met het eigen fragment: de vermelding volgt haar bron.
    entiteit = await _entiteit([intern], {intern: {"chunk_id": eigen}})
    assert await _c(entiteit) == "intern"
    assert await _c(await _alias(entiteit)) == "intern"
    assert await _c(await _triage(entiteit)) == "intern"


async def test_zonder_vermelding_intern(env) -> None:
    entiteit = await _entiteit([])
    assert await _c(entiteit) == "intern"
    assert await _c(await _alias(entiteit)) == "intern"
    assert await _c(await _triage(entiteit)) == "intern"
    extern = await _entiteit([await _bron("extern")])
    assert await _c(await _relatie(extern, extern, [])) == "intern"
    # Een onbekende of onleesbare bron telt als intern.
    assert await _c(await _entiteit(["source:onbekend", "geen-id"])) == "intern"


async def test_een_nieuwe_interne_vermelding_werkt_alles_bij(env) -> None:
    extern, intern = await _bron("extern"), await _bron("intern")
    entiteit = await _entiteit([extern])
    alias, triage = await _alias(entiteit), await _triage(entiteit)
    relatie = await _relatie(entiteit, entiteit, [extern])
    assert {await _c(r) for r in (entiteit, alias, triage, relatie)} == {"extern"}

    # Dezelfde entiteit opnieuw geschreven met een vermelding in een interne bron.
    await EntityRepository().upsert_entity(Entity(
        canonical_name=(await _q("SELECT VALUE canonical_name FROM type::thing($id)",
                                 {"id": entiteit}))[0],
        entity_type="organization",
        source_documents=[intern],
    ))
    assert {await _c(r) for r in (entiteit, alias, triage, relatie)} == {"intern"}


async def test_herclassificatie_van_de_enige_interne_bron_rekent_opnieuw(env) -> None:
    extern, intern = await _bron("extern"), await _bron("intern")
    entiteit = await _entiteit([extern, intern])
    alias, triage = await _alias(entiteit), await _triage(entiteit)
    relatie = await _relatie(entiteit, entiteit, [extern, intern])
    afgeleid = (entiteit, alias, triage, relatie)
    assert {await _c(r) for r in afgeleid} == {"intern"}

    await _herclassificeer(intern, "extern")
    assert {await _c(r) for r in afgeleid} == {"extern"}

    await _herclassificeer(intern, "intern")
    assert {await _c(r) for r in afgeleid} == {"intern"}
