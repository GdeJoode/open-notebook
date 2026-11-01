"""De strengste classificatie wint over alle versies (F2-04, ADR 0001, 0003).

Tegen een echte SurrealDB:

* een interne bronversie maakt de bron, al haar versies, fragmenten,
  ``doc_node``-records en embeddings intern, in één transactie: wordt die
  transactie afgebroken, dan verandert niets;
* uit de herreview van F2-03: de pijplijn weigert een vermelding waarvan
  ``grounding[bron].chunk_id`` niet bij die bron hoort. Zo blijft een entiteit
  niet extern als een andere bron intern wordt: getoetst met twee externe
  bronnen, waarvan er één intern wordt.

Alle testdata is synthetisch.
"""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import AsyncMock

import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.entity_persistence_service import EntityPersistenceService
from surrealdb_service.repositories.processing_run import ProcessingRunRepository
from surrealdb_service.repositories.source_version import SourceVersionRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


@pytest.fixture
async def env(live_surrealdb, monkeypatch):
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    yield
    await conn._pool.close_all()
    conn._pool = original or conn.ConnectionPool(config=live_surrealdb)


async def _q(sql: str, params: dict | None = None):
    return await conn.execute_query(sql, params or {})


def _rid(waarde: str):
    return conn.ensure_record_id(waarde)


def _hash() -> str:
    return hashlib.sha256(uuid.uuid4().bytes).hexdigest()


async def _bron() -> str:
    [rij] = await _q("CREATE source SET title = 'Openbare testbron', classification = 'extern' RETURN id")
    return str(rij["id"])


async def _run() -> str:
    return str((await ProcessingRunRepository().create(
        trigger="schema", pipeline_id="test", flow_vastgelegd=True
    )).id)


async def _versie_met_kinderen(bron: str) -> list[str]:
    """Een externe versie met een fragment, ``doc_node`` en embedding."""
    run = await _run()
    versie, _ = await SourceVersionRepository().register(bron, _hash(), run)
    herkomst = {"source": _rid(bron), "source_version": _rid(str(versie.id)), "run": _rid(run)}
    [chunk] = await _q("CREATE chunk CONTENT $d RETURN id", {"d": {
        **herkomst, "text": "Openbare tekst", "order": 0, "physical_page": 0,
        "element_type": "paragraph", "positions": [], "metadata": {}}})
    [node] = await _q("CREATE doc_node CONTENT $d RETURN id", {"d": {
        **herkomst, "element_type": "paragraph", "self_ref": "#/texts/0", "sequence": 0}})
    [emb] = await _q("CREATE source_embedding CONTENT $d RETURN id", {"d": {
        "source": _rid(bron), "chunk": _rid(str(chunk["id"])), "run": _rid(run), "content": "x",
        "order": 0, "embedding": [], "embedding_model": "test", "afgekapt": False}})
    return [str(versie.id), str(chunk["id"]), str(node["id"]), str(emb["id"])]


async def _classificaties(ids: list[str]) -> set[str]:
    rows = await _q("SELECT VALUE classification FROM $ids", {"ids": [_rid(i) for i in ids]})
    return set(rows)


# --------------------------------------------------------------------------
# Een interne versie
# --------------------------------------------------------------------------


async def test_interne_versie_maakt_alles_intern(env) -> None:
    bron = await _bron()
    records = [bron, *await _versie_met_kinderen(bron), *await _versie_met_kinderen(bron)]
    assert await _classificaties(records) == {"extern"}

    run = await _run()
    nieuw, _ = await SourceVersionRepository().register(bron, _hash(), run, classification="intern")
    assert await _classificaties([*records, str(nieuw.id)]) == {"intern"}
    # De bron wijst naar de nieuwe versie.
    [huidig] = await _q("SELECT VALUE current_version FROM $b", {"b": _rid(bron)})
    assert str(huidig) == str(nieuw.id)


async def test_afgebroken_transactie_verandert_niets(env) -> None:
    bron = await _bron()
    records = [bron, *await _versie_met_kinderen(bron)]
    run = await _run()
    with pytest.raises(RuntimeError):
        await conn.execute_transaction(
            "BEGIN TRANSACTION;"
            " CREATE source_version CONTENT $d;"
            " THROW 'afgebroken';"
            " COMMIT TRANSACTION;",
            {"d": {"source": _rid(bron), "version_no": 9, "content_hash": _hash(),
                   "created_by_run": _rid(run), "classification": "intern"}},
        )
    assert await _classificaties(records) == {"extern"}
    assert len(await SourceVersionRepository().list_by_source(bron)) == 1


# --------------------------------------------------------------------------
# Vermeldingen: het fragment hoort bij de bron
# --------------------------------------------------------------------------


def _entiteit(naam: str, chunk_id: str) -> dict:
    return {"text": naam, "label": "topic", "confidence": 0.9, "properties": {},
            "source_chunk_id": chunk_id}


async def _entiteit_rij(naam: str) -> dict | None:
    rows = await _q("SELECT * FROM entity WHERE canonical_name = $n", {"n": naam})
    return rows[0] if rows else None


async def _run_van(chunk: str) -> str:
    [run] = await _q("SELECT VALUE run FROM $c", {"c": _rid(chunk)})
    return str(run)


async def test_vermelding_na_herverwerking_krijgt_de_nieuwe_run(env) -> None:
    """Twee runs op dezelfde versie: de vermelding volgt haar fragment."""
    bron = await _bron()
    [versie, oud, *_] = await _versie_met_kinderen(bron)
    tweede = await _run()
    [nieuw] = await _q("CREATE chunk CONTENT $d RETURN id", {"d": {
        "source": _rid(bron), "source_version": _rid(versie), "run": _rid(tweede),
        "text": "Openbare tekst", "order": 0, "physical_page": 0,
        "element_type": "paragraph", "positions": [], "metadata": {}}})
    nieuw = str(nieuw["id"])
    await SourceVersionRepository().set_current_run(versie, tweede)
    naam = f"Brede Welvaart {uuid.uuid4().hex[:6]}"
    # De extractie geeft geen of een oude run mee; de database neemt die van het fragment.
    await EntityPersistenceService().persist_filtered_result(
        bron, [_entiteit(naam, nieuw)], [], run_id=await _run_van(oud)
    )
    rij = await _entiteit_rij(naam)
    assert rij["properties"]["grounding"][bron]["run"] == tweede


async def test_pijplijn_weigert_een_fragment_van_een_andere_bron(env) -> None:
    a, b = await _bron(), await _bron()
    [_, chunk_a, *_] = await _versie_met_kinderen(a)
    [_, chunk_b, *_] = await _versie_met_kinderen(b)
    dienst = EntityPersistenceService()
    run = await _run()
    naam = f"Regio Deal {uuid.uuid4().hex[:6]}"

    # Een vermelding in bron A met een fragment uit bron B: geweigerd.
    with pytest.raises(RuntimeError, match="wrote 0"):
        await dienst.persist_filtered_result(a, [_entiteit(naam, chunk_b)], [], run_id=run)
    assert await _entiteit_rij(naam) is None

    # Met de eigen fragmenten: twee vermeldingen, beide extern, met de run van
    # hun fragment.
    await dienst.persist_filtered_result(a, [_entiteit(naam, chunk_a)], [], run_id=run)
    await dienst.persist_filtered_result(b, [_entiteit(naam, chunk_b)], [], run_id=run)
    rij = await _entiteit_rij(naam)
    assert rij["classification"] == "extern"
    grounding = rij["properties"]["grounding"]
    assert {k: (v["chunk_id"], v["run"], v["classification"]) for k, v in grounding.items()} == {
        a: (chunk_a, await _run_van(chunk_a), "extern"),
        b: (chunk_b, await _run_van(chunk_b), "extern")}

    # Bron B wordt intern (een interne versie): de entiteit wordt intern.
    await SourceVersionRepository().register(b, _hash(), await _run(), classification="intern")
    rij = await _entiteit_rij(naam)
    assert rij["classification"] == "intern"
    assert rij["properties"]["grounding"][b]["classification"] == "intern"
    assert rij["properties"]["grounding"][a]["classification"] == "extern"

    # Een latere poging om in A naar het fragment van B te wijzen: geweigerd.
    with pytest.raises(RuntimeError, match="wrote 0"):
        await dienst.persist_filtered_result(a, [_entiteit(naam, chunk_b)], [], run_id=run)
    rij = await _entiteit_rij(naam)
    assert rij["properties"]["grounding"][a]["chunk_id"] == chunk_a
