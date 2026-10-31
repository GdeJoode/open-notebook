"""Migratie 102: restpunten van de review van F2-07 (ADR 0007).

- ``model_call.model_version`` is een korte code zonder vrije tekst;
- ``beheer_melding`` kent de soort ``cloudaanroep_intern`` met een verwijzing
  naar de regel in ``model_call``.

Onder SurrealDB v2.7.0 laat een SCHEMAFULL-tabel een onbekend veld stil weg;
daarom schrijft elke test een waarde én leest die terug. Alle testdata is
synthetisch.
"""

from __future__ import annotations

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    ensure_record_id,
    execute_query,
    execute_transaction,
)
from surrealdb_service.repositories import ModelCallRepository
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

    for versie in (101, 102):
        asyncio.run(execute_transaction(_migratie(versie), config=live_surrealdb))
    conn._pool = None


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


def _regel(**velden) -> dict:
    regel = {"model": "werkmodel", "model_version": "repo/model:q4_k_m:ctx-8192",
             "invocation": "llama-swap", "location": "lokaal", "task": "chat",
             "classification": "intern", "status": "geslaagd", "origin": "fork"}
    regel.update(velden)
    return regel


@pytest.mark.parametrize("versie", ["v1 met vrije tekst", "x" * 129, "v1\nregel"])
async def test_versie_zonder_vrije_tekst(live_surrealdb, versie) -> None:
    with pytest.raises(RuntimeError):
        await ModelCallRepository(live_surrealdb).create(_regel(model_version=versie))


async def test_versie_teruggelezen(live_surrealdb) -> None:
    rij = await ModelCallRepository(live_surrealdb).create(_regel())
    [terug] = await _q(live_surrealdb, "SELECT model_version FROM $id",
                       {"id": ensure_record_id(str(rij["id"]))})
    assert terug["model_version"] == "repo/model:q4_k_m:ctx-8192"


async def test_melding_cloudaanroep_intern(live_surrealdb) -> None:
    [run] = await _q(live_surrealdb, "CREATE processing_run CONTENT {trigger: 'schema',"
                     " pipeline_id: 'p', flow_vastgelegd: true} RETURN id")
    repo = ModelCallRepository(live_surrealdb)
    rij = await repo.create(_regel(model="cloudmodel", invocation="api", location="cloud",
                                   origin="langflow", run=str(run["id"])))
    await repo.meld_cloudaanroep_intern(str(rij["id"]), str(run["id"]))
    [melding] = await _q(live_surrealdb, "SELECT * FROM beheer_melding WHERE model_call = $m",
                         {"m": ensure_record_id(str(rij["id"]))})
    assert melding["soort"] == "cloudaanroep_intern" and melding["status"] == "open"
    assert str(melding["run"]) == str(run["id"])


async def test_twee_keer_uitvoeren_en_down(live_surrealdb) -> None:
    for _ in range(2):
        await execute_transaction(_migratie(102, down=True), config=live_surrealdb)
    # Terug: de versie mag weer langer, de soort blijft toegestaan.
    await ModelCallRepository(live_surrealdb).create(_regel(model_version="v1 oud formaat"))
    await _q(live_surrealdb, "CREATE beheer_melding CONTENT {soort: 'cloudaanroep_intern',"
             " melding: 'test'} RETURN NONE")
    for _ in range(2):
        await execute_transaction(_migratie(102), config=live_surrealdb)
    with pytest.raises(RuntimeError):
        await ModelCallRepository(live_surrealdb).create(_regel(model_version="v1 oud formaat"))
