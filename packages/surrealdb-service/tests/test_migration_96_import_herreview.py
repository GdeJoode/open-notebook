"""Migratie 96: versieteller, index op batchhashes, strengste wint per inhoud
(F2-14, herreview). Elke test schrijft een waarde én leest die terug. Alle
testdata is synthetisch.
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    ensure_record_id,
    execute_query,
    execute_transaction,
)
from surrealdb_service.testing import fixtures as fx

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]


def _migratie(versie: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}{suffix}.surrealql").sql


@pytest.fixture(autouse=True, scope="module")
def _migraties_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    import asyncio

    import surrealdb_service.connection as conn

    for versie in (93, 94, 95, 96, 97):
        asyncio.run(execute_transaction(_migratie(versie), config=live_surrealdb))
    conn._pool = None


async def _q(config, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _externe_bron_met_versie(config, hexhash: str) -> str:
    [run] = await _q(config, "CREATE processing_run CONTENT {trigger: 'schema', pipeline_id: 'p',"
                     " flow_vastgelegd: true} RETURN id")
    [bron] = await _q(config, "CREATE source CONTENT {title: 'Openbare testbron',"
                      " classification: 'extern'} RETURN id")
    run_id, bron_id = ensure_record_id(str(run["id"])), ensure_record_id(str(bron["id"]))
    [versie] = await _q(config, "CREATE source_version CONTENT $d RETURN id", {"d": {
        "source": bron_id, "version_no": 1, "content_hash": hexhash,
        "created_by_run": run_id, "current_run": run_id}})
    await _q(config, "UPDATE $b SET current_version = $v",
             {"b": bron_id, "v": ensure_record_id(str(versie["id"]))})
    return str(bron["id"])


async def test_versieteller(live_surrealdb) -> None:
    [bron] = await _q(live_surrealdb, "CREATE source CONTENT {title: 't', version_seq: 3} RETURN AFTER")
    assert bron["version_seq"] == 3
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE source CONTENT {title: 't', version_seq: -1}")


async def test_intern_besluit_over_een_item_maakt_de_bron_intern(live_surrealdb) -> None:
    hexhash = uuid.uuid4().hex * 2
    bron = await _externe_bron_met_versie(live_surrealdb, hexhash)
    [item] = await _q(live_surrealdb, "CREATE intake_item CONTENT $d RETURN id", {"d": {
        "origin": "watchfolder", "path": f"/inname/{uuid.uuid4().hex}.pdf",
        "content_hash": f"sha256:{hexhash}", "registered_by": "langflow"}})
    await _q(live_surrealdb, "UPDATE $id SET status = 'besloten', decision = {route: 'r',"
             " classification: 'intern', decided_by: 'Testpersoon', decided_at: time::now()}",
             {"id": ensure_record_id(str(item["id"]))})
    [klasse] = await _q(live_surrealdb, "SELECT VALUE classification FROM $b",
                        {"b": ensure_record_id(bron)})
    assert klasse == "intern"
    meldingen = await _q(live_surrealdb, "SELECT VALUE soort FROM beheer_melding WHERE source = $b",
                         {"b": ensure_record_id(bron)})
    assert meldingen == ["inhoud_intern"]


async def test_down_en_weer_op(live_surrealdb) -> None:
    for _ in range(2):
        await execute_transaction(_migratie(97, down=True), config=live_surrealdb)
        await execute_transaction(_migratie(96, down=True), config=live_surrealdb)
    hexhash = uuid.uuid4().hex * 2
    bron = await _externe_bron_met_versie(live_surrealdb, hexhash)
    [run] = await _q(live_surrealdb, "CREATE processing_run CONTENT {trigger: 'schema',"
                     " pipeline_id: 'p', flow_vastgelegd: true} RETURN id")
    [ander] = await _q(live_surrealdb, "CREATE source CONTENT {title: 'intern'} RETURN id")
    await _q(live_surrealdb, "CREATE source_version CONTENT $d", {"d": {
        "source": ensure_record_id(str(ander["id"])), "version_no": 1, "content_hash": hexhash,
        "created_by_run": ensure_record_id(str(run["id"])),
        "current_run": ensure_record_id(str(run["id"]))}})
    # Zonder migratie 96 geen strengste-wint per inhoud.
    [klasse] = await _q(live_surrealdb, "SELECT VALUE classification FROM $b",
                        {"b": ensure_record_id(bron)})
    assert klasse == "extern"
    for _ in range(2):
        await execute_transaction(_migratie(96), config=live_surrealdb)
        await execute_transaction(_migratie(97), config=live_surrealdb)
    info = await _q(live_surrealdb, "INFO FOR TABLE intake_batch")
    info = info[0] if isinstance(info, list) else info
    assert "idx_intake_batch_file_hash" in info["indexes"]
