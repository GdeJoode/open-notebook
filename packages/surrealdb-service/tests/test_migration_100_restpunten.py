"""Migratie 100: grens van de groep laat diepte N toe; melding bij een interne
batch met inhoud van een externe bron (F2-14, zesde review). Alle testdata is
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
def _migraties_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    import asyncio

    import surrealdb_service.connection as conn

    for versie in (93, 94, 95, 96, 97, 98, 99, 100):
        asyncio.run(execute_transaction(_migratie(versie), config=live_surrealdb,
                                        _herclassificatie=True))
    conn._pool = None


async def _q(config, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _een(config, sql: str, params: dict | None = None):
    rijen = await _q(config, sql, params)
    return rijen[0] if isinstance(rijen, list) else rijen


async def _bron(config, hashes: list[str]):
    rij = await _een(config, "CREATE source CONTENT {title: 'Openbare testbron',"
                     " classification: 'extern'} RETURN id")
    bron = ensure_record_id(str(rij["id"]))
    for nr, h in enumerate(hashes, start=1):
        run = await _een(config, "CREATE processing_run CONTENT {trigger: 'schema',"
                         " pipeline_id: 'p', flow_vastgelegd: true} RETURN id")
        run = ensure_record_id(str(run["id"]))
        await _q(config, "CREATE source_version CONTENT $d", {"d": {
            "source": bron, "version_no": nr, "content_hash": h,
            "created_by_run": run, "current_run": run}})
    return bron


async def test_grens_n_laat_diepte_n_toe(live_surrealdb) -> None:
    # Een keten van 5 bronnen: h0-S0-h1-S1-...-S4-h5.
    hashes = [uuid.uuid4().hex * 2 for _ in range(6)]
    for i in range(5):
        await _bron(live_surrealdb, hashes[i:i + 2])
    groep = await _een(live_surrealdb, "RETURN fn::inhoud_groep($h, 5)", {"h": hashes[0]})
    assert len(groep["bronnen"]) == 5 and len(groep["hashes"]) == 6
    with pytest.raises(RuntimeError, match="inhoud:groep_te_groot"):
        await _q(live_surrealdb, "RETURN fn::inhoud_groep($h, 4)", {"h": hashes[0]})


async def test_interne_batch_met_inhoud_van_een_externe_bron(live_surrealdb) -> None:
    h = uuid.uuid4().hex * 2
    bron = await _bron(live_surrealdb, [h])

    async def batch(klasse: str):
        rij = await _een(live_surrealdb, "CREATE intake_batch CONTENT $d RETURN id", {"d": {
            "route": "r", "created_by": "Testpersoon", "classification": klasse,
            "file_hashes": [{"path": "/inname/a.pdf", "content_hash": f"sha256:{h}"}]}})
        return ensure_record_id(str(rij["id"]))

    extern = await batch("extern")
    assert await _q(live_surrealdb, "SELECT id FROM beheer_melding WHERE intake_batch = $b",
                    {"b": extern}) == []
    intern = await batch("intern")
    [melding] = await _q(live_surrealdb, "SELECT * FROM beheer_melding WHERE intake_batch = $b",
                         {"b": intern})
    assert melding["soort"] == "batch_inhoud_extern" and str(melding["source"]) == str(bron)
    # De bron blijft extern.
    assert await _een(live_surrealdb, "SELECT VALUE classification FROM $b", {"b": bron}) == "extern"
    # Ook bij het wijzigen naar intern.
    await _q(live_surrealdb, "UPDATE $b SET classification = 'intern'", {"b": extern})
    meldingen = await _q(live_surrealdb, "SELECT VALUE soort FROM beheer_melding"
                         " WHERE intake_batch = $b", {"b": extern})
    assert meldingen == ["batch_inhoud_extern"]


async def test_down_en_weer_op(live_surrealdb) -> None:
    hashes = [uuid.uuid4().hex * 2 for _ in range(3)]
    for i in range(2):
        await _bron(live_surrealdb, hashes[i:i + 2])
    for _ in range(2):
        await execute_transaction(_migratie(100, down=True), config=live_surrealdb)
    # De vorm van 99: grens 2 is voor een keten van 2 te krap.
    with pytest.raises(RuntimeError, match="inhoud:groep_te_groot"):
        await _q(live_surrealdb, "RETURN fn::inhoud_groep($h, 2)", {"h": hashes[0]})
    for _ in range(2):
        await execute_transaction(_migratie(100), config=live_surrealdb)
    groep = await _een(live_surrealdb, "RETURN fn::inhoud_groep($h, 2)", {"h": hashes[0]})
    assert len(groep["bronnen"]) == 2
