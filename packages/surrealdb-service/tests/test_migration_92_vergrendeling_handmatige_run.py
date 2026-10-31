"""Migratie 92: vergrendeling, auteur, actor en bescherming van handmatige runs (F2-06).

Onder SurrealDB v2.7.0 laat een SCHEMAFULL-tabel een onbekend veld stil weg
(P4, P10 K6). Daarom schrijft elke test een waarde én leest die terug.
Alle testdata is synthetisch.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query, execute_transaction
from surrealdb_service.repositories.processing_run import ProcessingRunRepository
from surrealdb_service.repositories.source_version import SourceVersionRepository
from surrealdb_service.testing import fixtures as fx

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]

TABELLEN = ("entity", "entity_alias", "relation", "triage_queue")


def _migratie(versie: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}{suffix}.surrealql").sql


@pytest.fixture(autouse=True, scope="module")
def _migratie_92_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    """De database is per sessie gedeeld; de tests van migratie 90 en 91 voeren
    die opnieuw uit en zetten `current_run` en `trigger` terug. 92 is idempotent."""
    import asyncio

    import surrealdb_service.connection as conn

    asyncio.run(execute_transaction(_migratie(92), config=live_surrealdb))
    conn._pool = None


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _info(config: SurrealDBConfig, sql: str) -> dict:
    rij = await _q(config, sql)
    return rij[0] if isinstance(rij, list) else rij


async def _run(config: SurrealDBConfig, trigger: str, **extra):
    return await ProcessingRunRepository(config).create(
        trigger=trigger, pipeline_id="test", flow_vastgelegd=True, **extra
    )


@pytest.mark.parametrize("tabel", TABELLEN)
async def test_vergrendelvelden(live_surrealdb, tabel) -> None:
    velden = (await _info(live_surrealdb, f"INFO FOR TABLE {tabel}"))["fields"]
    assert {"rev", "modified_by", "modified_at"} <= set(velden)


async def test_auteur_verplicht_en_teruggelezen(live_surrealdb) -> None:
    rij = {"entity_a_text": "a", "entity_b_text": "b", "match_method": "test"}
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE resolution_log CONTENT $d", {"d": rij})
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE resolution_log CONTENT $d", {"d": {**rij, "author": " "}})
    [terug] = await _q(live_surrealdb, "CREATE resolution_log CONTENT $d RETURN author",
                       {"d": {**rij, "author": "Beheerder Test"}})
    assert terug["author"] == "Beheerder Test"


async def test_handmatige_run_vraagt_een_actor(live_surrealdb) -> None:
    with pytest.raises(RuntimeError):
        await _run(live_surrealdb, "handmatig")
    run = await _run(live_surrealdb, "handmatig", actor="Beheerder Test")
    [terug] = await _q(live_surrealdb, "SELECT actor FROM $id", {"id": ensure_record_id(str(run.id))})
    assert terug["actor"] == "Beheerder Test"
    # Een automatische run mag zonder actor.
    assert (await _run(live_surrealdb, "schema")).status == "bezig"


async def _versie(config: SurrealDBConfig, run_id: str) -> str:
    [bron] = await _q(config, "CREATE source SET title = 'Openbare testbron', classification = 'extern' RETURN id")
    versie, _ = await SourceVersionRepository(config).register(
        str(bron["id"]), hashlib.sha256(uuid.uuid4().bytes).hexdigest(), run_id
    )
    return str(versie.id)


async def test_handmatige_run_blijft_de_huidige(live_surrealdb) -> None:
    auto = await _run(live_surrealdb, "inname")
    versie = await _versie(live_surrealdb, str(auto.id))
    handmatig = await _run(live_surrealdb, "handmatig", actor="Beheerder Test")
    vid = ensure_record_id(versie)
    await _q(live_surrealdb, "UPDATE $v SET current_run = $r", {"v": vid, "r": ensure_record_id(str(handmatig.id))})

    # Een automatische run verzet de huidige run niet, ook niet rechtstreeks.
    later = await _run(live_surrealdb, "herverwerking")
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "UPDATE $v SET current_run = $r", {"v": vid, "r": ensure_record_id(str(later.id))})
    [rij] = await _q(live_surrealdb, "SELECT current_run FROM $v", {"v": vid})
    assert str(rij["current_run"]) == str(handmatig.id)

    # Een volgende handmatige run wel.
    tweede = await _run(live_surrealdb, "handmatig", actor="Beheerder Test")
    await _q(live_surrealdb, "UPDATE $v SET current_run = $r", {"v": vid, "r": ensure_record_id(str(tweede.id))})
    [rij] = await _q(live_surrealdb, "SELECT current_run FROM $v", {"v": vid})
    assert str(rij["current_run"]) == str(tweede.id)


async def test_beheer_melding(live_surrealdb) -> None:
    run = await _run(live_surrealdb, "inname")
    [rij] = await _q(live_surrealdb, "CREATE beheer_melding CONTENT $d RETURN AFTER", {"d": {
        "soort": "handmatige_run_beschermd", "melding": "test", "run": ensure_record_id(str(run.id))}})
    assert (rij["soort"], rij["status"], str(rij["run"])) == ("handmatige_run_beschermd", "open", str(run.id))
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE beheer_melding CONTENT {soort: 'iets anders', melding: 'x'}")


async def test_twee_keer_uitvoeren_en_down(live_surrealdb) -> None:
    await execute_transaction(_migratie(92), config=live_surrealdb)
    await execute_transaction(_migratie(92), config=live_surrealdb)
    run = await _run(live_surrealdb, "handmatig", actor="Beheerder Test")
    try:
        await execute_transaction(_migratie(92, down=True), config=live_surrealdb)
        await execute_transaction(_migratie(92, down=True), config=live_surrealdb)
        for tabel in TABELLEN:
            velden = (await _info(live_surrealdb, f"INFO FOR TABLE {tabel}"))["fields"]
            assert not {"rev", "modified_by", "modified_at"} & set(velden)
        assert "beheer_melding" not in (await _info(live_surrealdb, "INFO FOR DB"))["tables"]
        # De herkomst blijft: de actor van een run en de auteur in het log.
        [rij] = await _q(live_surrealdb, "SELECT actor FROM $id", {"id": ensure_record_id(str(run.id))})
        assert rij["actor"] == "Beheerder Test"
        assert "author" in (await _info(live_surrealdb, "INFO FOR TABLE resolution_log"))["fields"]
        assert "handmatig" not in (await _info(live_surrealdb, "INFO FOR TABLE source_version"))["fields"]["current_run"]
    finally:
        await execute_transaction(_migratie(92), config=live_surrealdb)
    assert "rev" in (await _info(live_surrealdb, "INFO FOR TABLE entity"))["fields"]
    assert "handmatig" in (await _info(live_surrealdb, "INFO FOR TABLE source_version"))["fields"]["current_run"]
