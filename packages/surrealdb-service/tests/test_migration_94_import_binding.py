"""Migratie 94: binding van run en bron voor het importeindpunt (F2-14).

Onder SurrealDB v2.7.0 laat een SCHEMAFULL-tabel een onbekend veld stil weg;
daarom schrijft elke test een waarde én leest die terug. Alle testdata is
synthetisch.
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
def _migratie_94_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    import asyncio

    import surrealdb_service.connection as conn

    asyncio.run(execute_transaction(_migratie(93), config=live_surrealdb))
    asyncio.run(execute_transaction(_migratie(94), config=live_surrealdb))
    conn._pool = None


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _regel(config) -> str:
    [rij] = await _q(config, "CREATE source_map_rule CONTENT $d RETURN id", {"d": {
        "name": "Openbare testregel", "route_type": "api", "source_type": "api",
        "location": "https://example.org/api", "pipeline_start": "automatisch",
        "route": "inname_api", "fetch_content": "volledig_document"}})
    return str(rij["id"])


async def test_run_legt_aanmaker_batch_en_item_vast(live_surrealdb) -> None:
    [batch] = await _q(live_surrealdb, "CREATE intake_batch CONTENT {route: 'inname_pdf',"
                       " created_by: 'Testpersoon'} RETURN id")
    [item] = await _q(live_surrealdb, "CREATE intake_item CONTENT $d RETURN id", {"d": {
        "origin": "watchfolder", "path": f"/inname/{uuid.uuid4().hex}.pdf",
        "content_hash": "sha256:" + "a" * 64, "registered_by": "langflow"}})
    [run] = await _q(live_surrealdb, "CREATE processing_run CONTENT $d RETURN AFTER", {"d": {
        "trigger": "schema", "pipeline_id": "inname_pdf", "created_by_key": "agent_keys:langflow",
        "intake_batch": ensure_record_id(str(batch["id"])),
        "intake_item": ensure_record_id(str(item["id"]))}})
    assert run["created_by_key"] == "agent_keys:langflow"
    assert str(run["intake_batch"]) == str(batch["id"])
    assert str(run["intake_item"]) == str(item["id"])
    # Na het aanmaken liggen ze vast.
    for veld, waarde in (("created_by_key", "agent_keys:ander"), ("intake_item", None)):
        with pytest.raises(RuntimeError):
            await _q(live_surrealdb, f"UPDATE $id SET {veld} = $w",
                     {"id": ensure_record_id(str(run["id"])), "w": waarde})
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE processing_run CONTENT {trigger: 'schema',"
                 " pipeline_id: 'x', created_by_key: ' '}")


async def test_bron_met_extern_id_en_regel(live_surrealdb) -> None:
    regel = await _regel(live_surrealdb)
    extern_id = f"test-{uuid.uuid4().hex}"
    [bron] = await _q(live_surrealdb, "CREATE source CONTENT $d RETURN AFTER", {"d": {
        "title": "Openbare testbron", "external_id": extern_id,
        "source_map_rule": ensure_record_id(regel)}})
    assert bron["external_id"] == extern_id
    assert str(bron["source_map_rule"]) == regel
    for veld in ("external_id", "source_map_rule"):
        with pytest.raises(RuntimeError):
            await _q(live_surrealdb, f"UPDATE $id SET {veld} = NONE",
                     {"id": ensure_record_id(str(bron["id"]))})
    # Uniek; bronnen zonder extern id mogen naast elkaar bestaan.
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE source CONTENT {title: 'tweede', external_id: $e}",
                 {"e": extern_id})
    await _q(live_surrealdb, "CREATE source CONTENT {title: 'zonder id 1'}")
    await _q(live_surrealdb, "CREATE source CONTENT {title: 'zonder id 2'}")


async def test_down_en_weer_op(live_surrealdb) -> None:
    await execute_transaction(_migratie(94, down=True), config=live_surrealdb)
    await execute_transaction(_migratie(94, down=True), config=live_surrealdb)
    info = await _q(live_surrealdb, "INFO FOR TABLE source")
    info = info[0] if isinstance(info, list) else info
    assert "idx_source_external_id" not in info["indexes"]
    # De velden blijven bestaan (herkomst blijft), maar zijn weer wijzigbaar.
    [bron] = await _q(live_surrealdb, "CREATE source CONTENT {title: 'terug', external_id: 'x-terug'}"
                      " RETURN AFTER")
    assert bron["external_id"] == "x-terug"
    await _q(live_surrealdb, "UPDATE $id SET external_id = 'x-anders'", {"id": ensure_record_id(str(bron["id"]))})

    await execute_transaction(_migratie(94), config=live_surrealdb)
    await execute_transaction(_migratie(94), config=live_surrealdb)
    info = await _q(live_surrealdb, "INFO FOR TABLE source")
    info = info[0] if isinstance(info, list) else info
    assert "idx_source_external_id" in info["indexes"]
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "UPDATE $id SET external_id = 'x-derde'", {"id": ensure_record_id(str(bron["id"]))})
