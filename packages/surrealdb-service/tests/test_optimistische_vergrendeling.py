"""Optimistische vergrendeling met ``wijzig_met_rev`` (F2-06, ADR 0003).

Tegen een echte SurrealDB (migratie 92):

* de database telt ``rev`` bij elke schrijfactie op, ook buiten de hulpfunctie;
* een wijziging met een verouderde ``rev`` geeft :class:`RevConflictError`, en
  bij meerdere records in één aanroep verandert dan niets;
* ``modified_by`` en ``modified_at`` worden gezet en teruggelezen;
* een run wijzigt een record met ``manual_override = true`` niet, en legt het
  conflict vast in zijn logboek.

Alle testdata is synthetisch.
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.base import (
    RecordNotFoundError,
    RevConflictError,
    RevWijziging,
    wijzig_met_rev,
)
from surrealdb_service.repositories.processing_run import ProcessingRunRepository

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _entiteit(config: SurrealDBConfig, **velden) -> str:
    naam = f"Openbare organisatie {uuid.uuid4().hex[:8]}"
    [rij] = await _q(config, "CREATE entity CONTENT $d RETURN id", {"d": {
        "canonical_name": naam, "name": naam, "name_key": naam.lower(),
        "entity_type": "organization", "hash_id": uuid.uuid4().hex,
        "embedding": [], "status": "active", **velden,
    }})
    return str(rij["id"])


async def _rev(config: SurrealDBConfig, record_id: str) -> int:
    [rij] = await _q(config, "SELECT rev FROM $id", {"id": ensure_record_id(record_id)})
    return rij["rev"]


async def test_rev_begint_op_nul_en_telt_bij_elke_schrijfactie(live_surrealdb) -> None:
    eid = await _entiteit(live_surrealdb)
    assert await _rev(live_surrealdb, eid) == 0
    await _q(live_surrealdb, "UPDATE $id SET status = 'reference'", {"id": ensure_record_id(eid)})
    assert await _rev(live_surrealdb, eid) == 1
    # Een meegestuurde waarde telt niet: de database telt zelf.
    await _q(live_surrealdb, "UPDATE $id SET rev = 40", {"id": ensure_record_id(eid)})
    assert await _rev(live_surrealdb, eid) == 2


async def test_wijziging_met_actuele_rev_zet_auteur_en_tijdstip(live_surrealdb) -> None:
    eid = await _entiteit(live_surrealdb)
    [na] = await wijzig_met_rev(
        [RevWijziging(eid, 0, {"status": "reference"})], door="Beheerder Test", config=live_surrealdb
    )
    assert na["status"] == "reference"
    assert na["rev"] == 1
    assert na["modified_by"] == "Beheerder Test"
    assert na["modified_at"] is not None


async def test_verouderde_rev_geeft_conflict_en_wijzigt_niets(live_surrealdb) -> None:
    eid = await _entiteit(live_surrealdb)
    await wijzig_met_rev([RevWijziging(eid, 0, {"status": "reference"})], door="A", config=live_surrealdb)
    with pytest.raises(RevConflictError) as fout:
        await wijzig_met_rev([RevWijziging(eid, 0, {"status": "active"})], door="B", config=live_surrealdb)
    assert fout.value.record_ids == [eid]
    [rij] = await _q(live_surrealdb, "SELECT status, rev, modified_by FROM $id", {"id": ensure_record_id(eid)})
    assert (rij["status"], rij["rev"], rij["modified_by"]) == ("reference", 1, "A")


async def test_meerdere_records_alles_of_niets(live_surrealdb) -> None:
    a = await _entiteit(live_surrealdb)
    b = await _entiteit(live_surrealdb)
    await _q(live_surrealdb, "UPDATE $id SET status = 'reference'", {"id": ensure_record_id(b)})
    with pytest.raises(RevConflictError) as fout:
        await wijzig_met_rev(
            [RevWijziging(a, 0, {"status": "merged"}), RevWijziging(b, 0, {"status": "merged"})],
            door="Beheerder Test", config=live_surrealdb,
        )
    assert fout.value.record_ids == [b]
    [rij] = await _q(live_surrealdb, "SELECT status, rev FROM $id", {"id": ensure_record_id(a)})
    assert (rij["status"], rij["rev"]) == ("active", 0)


async def test_onbekend_record(live_surrealdb) -> None:
    with pytest.raises(RecordNotFoundError):
        await wijzig_met_rev(
            [RevWijziging("entity:bestaat_niet_f206", 0, {"status": "active"})],
            door="Beheerder Test", config=live_surrealdb,
        )


@pytest.mark.parametrize("tabel", ["entity_alias", "relation", "triage_queue"])
async def test_velden_op_de_andere_tabellen(live_surrealdb, tabel) -> None:
    eid = await _entiteit(live_surrealdb)
    if tabel == "entity_alias":
        [rij] = await _q(live_surrealdb, "CREATE entity_alias CONTENT $d RETURN id", {"d": {
            "alias_text": "Openbare alias", "canonical_entity": ensure_record_id(eid),
            "match_type": "exact"}})
    elif tabel == "relation":
        ander = await _entiteit(live_surrealdb)
        [rij] = await _q(live_surrealdb, "RELATE $a->relation->$b CONTENT $d RETURN id", {
            "a": ensure_record_id(eid), "b": ensure_record_id(ander),
            "d": {"relation_type": "related_to", "source_documents": []}})
    else:
        [rij] = await _q(live_surrealdb, "CREATE triage_queue CONTENT $d RETURN id", {"d": {
            "entity": ensure_record_id(eid), "name": "x", "reason": "test", "batch_id": "b"}})
    rid = str(rij["id"])
    assert await _rev(live_surrealdb, rid) == 0
    [na] = await wijzig_met_rev([RevWijziging(rid, 0, {})], door="Beheerder Test", config=live_surrealdb)
    assert (na["rev"], na["modified_by"]) == (1, "Beheerder Test")
    with pytest.raises(RevConflictError):
        await wijzig_met_rev([RevWijziging(rid, 0, {})], door="Beheerder Test", config=live_surrealdb)


async def test_manual_override_blokkeert_een_run_die_het_conflict_logt(live_surrealdb) -> None:
    vast = await _entiteit(live_surrealdb, manual_override=True)
    vrij = await _entiteit(live_surrealdb)
    run = await ProcessingRunRepository(live_surrealdb).create(
        trigger="schema", pipeline_id="test", flow_vastgelegd=True
    )
    uitkomst = await wijzig_met_rev(
        [RevWijziging(vast, None, {"status": "reference"}), RevWijziging(vrij, None, {"status": "reference"})],
        door="pijplijn", run_id=str(run.id), config=live_surrealdb,
    )
    assert uitkomst[0] is None
    assert uitkomst[1]["status"] == "reference" and uitkomst[1]["modified_by"] == str(run.id)
    [rij] = await _q(live_surrealdb, "SELECT status, rev FROM $id", {"id": ensure_record_id(vast)})
    assert (rij["status"], rij["rev"]) == ("active", 0)
    log = (await ProcessingRunRepository(live_surrealdb).get(str(run.id))).log
    assert [(r["step"], r["status"]) for r in log] == [("vergrendeling", "conflict")]
    assert vast in log[0]["message"]


async def test_hulpfunctie_vraagt_wie_wijzigt(live_surrealdb) -> None:
    eid = await _entiteit(live_surrealdb)
    with pytest.raises(ValueError):
        await wijzig_met_rev([RevWijziging(eid, 0, {})], door=" ", config=live_surrealdb)


async def test_set_status_door_een_run_respecteert_de_vastgezette_status(live_surrealdb) -> None:
    from surrealdb_service.repositories.entity import EntityRepository

    vast = await _entiteit(live_surrealdb, manual_override=True)
    vrij = await _entiteit(live_surrealdb)
    run = await ProcessingRunRepository(live_surrealdb).create(
        trigger="schema", pipeline_id="test", flow_vastgelegd=True
    )
    repo = EntityRepository(config=live_surrealdb)
    assert await repo.set_status(vast, "reference", run_id=str(run.id)) is False
    assert await repo.set_status(vrij, "reference", run_id=str(run.id)) is True
    [rij] = await _q(live_surrealdb, "SELECT status FROM $id", {"id": ensure_record_id(vast)})
    assert rij["status"] == "active"
    log = (await ProcessingRunRepository(live_surrealdb).get(str(run.id))).log
    assert [(r["step"], r["status"]) for r in log] == [("vergrendeling", "conflict")]


# --------------------------------------------------------------------------
# Grens voor automatische schrijvers op een vastgezette entiteit (voorstel B28)
# --------------------------------------------------------------------------


async def _melding(config: SurrealDBConfig, eid: str) -> list:
    return await _q(config, "SELECT soort, melding FROM beheer_melding WHERE entity = $e",
                    {"e": ensure_record_id(eid)})


async def test_naamwijziging_door_een_run_wordt_overgeslagen_en_gelogd(live_surrealdb) -> None:
    vast = await _entiteit(live_surrealdb, manual_override=True)
    [voor] = await _q(live_surrealdb, "SELECT canonical_name FROM $id", {"id": ensure_record_id(vast)})
    run = await ProcessingRunRepository(live_surrealdb).create(
        trigger="schema", pipeline_id="test", flow_vastgelegd=True
    )
    [uit] = await wijzig_met_rev(
        [RevWijziging(vast, None, {"canonical_name": "Andere naam"})],
        door="pijplijn", run_id=str(run.id), config=live_surrealdb,
    )
    assert uit is None
    [na] = await _q(live_surrealdb, "SELECT canonical_name FROM $id", {"id": ensure_record_id(vast)})
    assert na["canonical_name"] == voor["canonical_name"]
    log = (await ProcessingRunRepository(live_surrealdb).get(str(run.id))).log
    assert [(r["step"], r["status"]) for r in log] == [("vergrendeling", "conflict")]


async def test_upsert_voegt_op_een_vastgezette_entiteit_alleen_vermeldingen_toe(live_surrealdb) -> None:
    from shared.models.entity import Entity
    from surrealdb_service.repositories.entity import EntityRepository

    repo = EntityRepository(config=live_surrealdb)
    [b1] = await _q(live_surrealdb, "CREATE source SET title = 'Openbare bron 1' RETURN id")
    [b2] = await _q(live_surrealdb, "CREATE source SET title = 'Openbare bron 2' RETURN id")
    naam = f"Openbaar programma {uuid.uuid4().hex[:8]}"
    eid = await repo.upsert_entity(Entity(
        canonical_name=naam, entity_type="programme", confidence=0.5, embedding=[],
        source_documents=[str(b1["id"])], description="Openbare omschrijving",
    ))
    await _q(live_surrealdb, "UPDATE $id SET manual_override = true", {"id": ensure_record_id(eid)})
    run = await ProcessingRunRepository(live_surrealdb).create(
        trigger="schema", pipeline_id="test", flow_vastgelegd=True
    )
    await repo.upsert_entity(Entity(
        canonical_name=naam, entity_type="programme", confidence=0.9, embedding=[],
        source_documents=[str(b2["id"])], type_tags=["Nieuw"],
        properties={"grounding": {str(b2["id"]): {"grounding": "openbare zin"}}},
    ), run_id=str(run.id))
    [rij] = await _q(live_surrealdb, "SELECT * FROM $id", {"id": ensure_record_id(eid)})
    # De vermelding is toegevoegd ...
    assert set(rij["source_documents"]) == {str(b1["id"]), str(b2["id"])}
    assert str(b2["id"]) in rij["properties"]["grounding"]
    # ... de rest niet.
    assert rij["confidence"] == 0.5 and "Nieuw" not in (rij.get("type_tags") or [])
    log = (await ProcessingRunRepository(live_surrealdb).get(str(run.id))).log
    assert [(r["step"], r["status"]) for r in log] == [("vergrendeling", "conflict")]


async def test_automatische_external_ids_alias_en_status_zonder_run_geven_een_melding(live_surrealdb) -> None:
    from surrealdb_service.repositories.entity import EntityRepository

    repo = EntityRepository(config=live_surrealdb)
    vast = await _entiteit(live_surrealdb, manual_override=True)
    assert await repo.update_external_ids(vast, ["https://example.org/id/1"], ["Alias"]) is False
    assert await repo.register_alias(vast, "Andere alias", "exact", 1.0) is False
    assert await repo.set_status(vast, "reference") is False
    [rij] = await _q(live_surrealdb, "SELECT external_ids, status FROM $id", {"id": ensure_record_id(vast)})
    assert not rij.get("external_ids") and rij["status"] == "active"
    assert not await _q(live_surrealdb, "SELECT id FROM entity_alias WHERE canonical_entity = $e",
                        {"e": ensure_record_id(vast)})
    meldingen = await _melding(live_surrealdb, vast)
    assert sorted(m["melding"] for m in meldingen) == [
        "alias overgeslagen: manual_override",
        "external_ids overgeslagen: manual_override",
        "status overgeslagen: manual_override",
    ]
    assert {m["soort"] for m in meldingen} == {"vastgezet_overgeslagen"}
    # De beheerder mag wel.
    assert await repo.register_alias(vast, "Andere alias", "exact", 1.0, door="Beheerder Test") is True


async def test_schrijfconflict_van_de_opslag_wordt_een_revconflict(live_surrealdb, monkeypatch) -> None:
    import surrealdb_service.repositories.base as base

    async def _conflict(*_a, **_k):
        raise RuntimeError(
            "SurrealQL transaction failed: ['Failed to commit transaction due to a read or"
            " write conflict. This transaction can be retried']"
        )

    monkeypatch.setattr(base, "execute_transaction", _conflict)
    eid = await _entiteit(live_surrealdb)
    with pytest.raises(RevConflictError):
        await wijzig_met_rev([RevWijziging(eid, 0, {"status": "reference"})],
                             door="Beheerder Test", config=live_surrealdb)
