"""Migratie 97: record ``inhoud`` per hash, strengste wint over alle versies,
batch die intern wordt, oplopende versieteller (F2-14, derde review).

Elke test schrijft een waarde én leest die terug. Alle testdata is synthetisch.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

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


async def _een(config, sql: str, params: dict | None = None):
    rijen = await _q(config, sql, params)
    return rijen[0] if isinstance(rijen, list) else rijen


async def _run(config) -> object:
    rij = await _een(config, "CREATE processing_run CONTENT {trigger: 'schema', pipeline_id: 'p',"
                     " flow_vastgelegd: true} RETURN id")
    return ensure_record_id(str(rij["id"]))


async def _versie(config, bron, hexhash: str, nr: int) -> str:
    run = await _run(config)
    rij = await _een(config, "CREATE source_version CONTENT $d RETURN id", {"d": {
        "source": bron, "version_no": nr, "content_hash": hexhash,
        "created_by_run": run, "current_run": run}})
    await _q(config, "UPDATE $b SET current_version = $v",
             {"b": bron, "v": ensure_record_id(str(rij["id"]))})
    return str(rij["id"])


async def _externe_bron(config) -> object:
    rij = await _een(config, "CREATE source CONTENT {title: 'Openbare testbron',"
                     " classification: 'extern'} RETURN id")
    return ensure_record_id(str(rij["id"]))


async def test_inhoud_alleen_strenger(live_surrealdb) -> None:
    h = uuid.uuid4().hex * 2
    rij = await _een(live_surrealdb, "UPSERT type::thing('inhoud', $h) SET classification ="
                     " classification ?? 'extern' RETURN AFTER", {"h": h})
    assert rij["classification"] == "extern" and rij["gewijzigd_op"]
    await _q(live_surrealdb, "UPSERT type::thing('inhoud', $h) SET classification = 'intern'",
             {"h": h})
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "UPDATE type::thing('inhoud', $h) SET classification = 'extern'",
                 {"h": h})
    assert await _een(live_surrealdb, "RETURN fn::inhoud_intern_bekend($h)", {"h": h}) is True
    assert await _een(live_surrealdb, "RETURN fn::inhoud_intern_bekend($h)",
                      {"h": uuid.uuid4().hex * 2}) is False


async def test_oudere_versie_met_de_hash_maakt_de_bron_intern(live_surrealdb) -> None:
    oud, nieuw = uuid.uuid4().hex * 2, uuid.uuid4().hex * 2
    bron = await _externe_bron(live_surrealdb)
    await _versie(live_surrealdb, bron, oud, 1)
    await _versie(live_surrealdb, bron, nieuw, 2)
    await _q(live_surrealdb, "RETURN fn::inhoud_intern($h)", {"h": oud})
    await _q(live_surrealdb, "RETURN fn::inhoud_intern($h)", {"h": oud})
    assert await _een(live_surrealdb, "SELECT VALUE classification FROM $b", {"b": bron}) == "intern"
    klassen = await _q(live_surrealdb, "SELECT VALUE classification FROM source_version"
                       " WHERE source = $b", {"b": bron})
    assert set(klassen) == {"intern"}
    meldingen = await _q(live_surrealdb, "SELECT VALUE soort FROM beheer_melding WHERE source = $b",
                         {"b": bron})
    assert meldingen == ["inhoud_intern"]


async def test_batch_die_intern_wordt(live_surrealdb) -> None:
    h = uuid.uuid4().hex * 2
    bron = await _externe_bron(live_surrealdb)
    await _versie(live_surrealdb, bron, h, 1)
    batch = await _een(live_surrealdb, "CREATE intake_batch CONTENT $d RETURN id", {"d": {
        "route": "r", "created_by": "Testpersoon", "classification": "extern",
        "classification_confirmed_by": "Testpersoon",
        "classification_confirmed_at": datetime.now(timezone.utc),
        "file_hashes": [{"path": "/inname/a.pdf", "content_hash": f"sha256:{h}"}]}})
    assert await _een(live_surrealdb, "SELECT VALUE classification FROM $b", {"b": bron}) == "extern"
    await _q(live_surrealdb, "UPDATE $id SET classification_confirmed_by = NONE,"
             " classification_confirmed_at = NONE", {"id": ensure_record_id(str(batch["id"]))})
    assert await _een(live_surrealdb, "SELECT VALUE classification FROM $b", {"b": bron}) == "intern"


async def test_versieteller_daalt_niet(live_surrealdb) -> None:
    rij = await _een(live_surrealdb, "CREATE source CONTENT {title: 't', version_seq: 2} RETURN id")
    bron = ensure_record_id(str(rij["id"]))
    await _q(live_surrealdb, "UPDATE $b SET version_seq = 3", {"b": bron})
    await _q(live_surrealdb, "UPDATE $b SET title = 'ander'", {"b": bron})
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "UPDATE $b SET version_seq = 1", {"b": bron})


async def test_down_en_weer_op(live_surrealdb) -> None:
    for _ in range(2):
        await execute_transaction(_migratie(97, down=True), config=live_surrealdb)
    db = await _een(live_surrealdb, "INFO FOR DB")
    assert "inhoud" not in db["tables"] and "inhoud_intern_bekend" not in db["functions"]
    for _ in range(2):
        await execute_transaction(_migratie(97), config=live_surrealdb)
    db = await _een(live_surrealdb, "INFO FOR DB")
    assert "inhoud" in db["tables"] and "inhoud_intern_bekend" in db["functions"]
