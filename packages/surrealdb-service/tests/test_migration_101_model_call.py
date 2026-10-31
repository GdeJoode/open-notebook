"""Migratie 101: auditlog van modelaanroepen, ``model_call`` (F2-07, ADR 0007).

Stopconditie: de tabel heeft geen tekstveld voor prompt of antwoord. Onder
SurrealDB v2.7.0 laat een SCHEMAFULL-tabel een onbekend veld stil weg; daarom
schrijft elke test een waarde én leest die terug. Alle testdata is synthetisch.
"""

from __future__ import annotations

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    ensure_record_id,
    execute_query,
    execute_transaction,
)
from surrealdb_service.repositories import ModelCallRepository, ProcessingRunRepository
from surrealdb_service.testing import fixtures as fx

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]

GEHEIM = "Interne tekst die nooit in het auditlog mag"

#: De velden van ``model_call``; geen enkel veld is bedoeld voor tekst.
VELDEN = {
    "at", "recorded_at", "run", "model", "model_version", "invocation", "location",
    "task", "classification", "input_tokens", "output_tokens", "duration_ms",
    "status", "attempts", "error_code", "origin",
}


def _migratie(versie: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}{suffix}.surrealql").sql


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _info(config) -> dict:
    info = await _q(config, "INFO FOR TABLE model_call")
    return info[0] if isinstance(info, list) else info


async def _run(config) -> str:
    [rij] = await _q(config, "CREATE processing_run CONTENT {trigger: 'schema',"
                     " pipeline_id: 'p', flow_vastgelegd: true} RETURN id")
    return str(rij["id"])


def _regel(**velden) -> dict:
    regel = {"model": "werkmodel", "model_version": "v1", "invocation": "llama-swap",
             "location": "lokaal", "task": "chat", "classification": "intern",
             "status": "geslaagd", "attempts": 1, "origin": "fork",
             "input_tokens": 10, "output_tokens": 2, "duration_ms": 30}
    regel.update(velden)
    return regel


async def test_geen_tekstveld_voor_prompt_of_antwoord(live_surrealdb) -> None:
    info = await _info(live_surrealdb)
    assert set(info["fields"]) == VELDEN

    # Een meegestuurd tekstveld komt niet in de tabel.
    regel = await ModelCallRepository(live_surrealdb).create(
        _regel(prompt=GEHEIM, response=GEHEIM, text=GEHEIM))
    [terug] = await _q(live_surrealdb, "SELECT * FROM $id", {"id": ensure_record_id(str(regel["id"]))})
    assert GEHEIM not in str(terug)
    assert set(terug) - {"id"} <= VELDEN


async def test_schema_is_vast(live_surrealdb) -> None:
    info = await _q(live_surrealdb, "INFO FOR DB")
    info = info[0] if isinstance(info, list) else info
    assert "SCHEMAFULL" in info["tables"]["model_call"]


async def test_regel_teruggelezen(live_surrealdb) -> None:
    run = await _run(live_surrealdb)
    repo = ModelCallRepository(live_surrealdb)
    await repo.create(_regel(run=run))
    [terug] = await repo.for_run(run)
    for veld, waarde in _regel().items():
        assert terug[veld] == waarde, veld
    assert str(terug["run"]) == run
    assert terug["at"] and terug["recorded_at"]


@pytest.mark.parametrize("fout", [
    {"model": "vrije tekst " + GEHEIM},
    {"task": "Vat deze tekst samen: " + GEHEIM},
    {"error_code": GEHEIM},
    {"model_version": "x" * 300},
    {"invocation": "ollama"},
    {"location": "elders"},
    {"classification": "geheim"},
    {"status": "onbekend"},
    {"origin": "gateway"},
    {"input_tokens": -1},
    {"attempts": -1},
    {"run": "processing_run:bestaat_niet"},
])
async def test_ongeldige_regel_geweigerd(live_surrealdb, fout) -> None:
    with pytest.raises(RuntimeError):
        await ModelCallRepository(live_surrealdb).create(_regel(**fout))


async def test_alleen_aanvulbaar(live_surrealdb) -> None:
    regel = await ModelCallRepository(live_surrealdb).create(_regel())
    rid = ensure_record_id(str(regel["id"]))
    with pytest.raises(RuntimeError, match="readonly"):
        await _q(live_surrealdb, "UPDATE $id SET attempts = 2", {"id": rid})
    with pytest.raises(RuntimeError, match="alleen aanvulbaar"):
        await _q(live_surrealdb, "DELETE $id", {"id": rid})
    [terug] = await _q(live_surrealdb, "SELECT attempts FROM $id", {"id": rid})
    assert terug["attempts"] == 1


async def test_models_van_de_run_uit_model_call(live_surrealdb) -> None:
    run = await _run(live_surrealdb)
    repo = ModelCallRepository(live_surrealdb)
    runs = ProcessingRunRepository(live_surrealdb)
    await repo.create(_regel(run=run))
    await repo.create(_regel(run=run, status="mislukt", error_code="http_500"))
    await repo.create(_regel(run=run, model="embed", task="embedding"))
    # Een geweigerd model heeft de invoer niet gezien en telt niet mee.
    await repo.create(_regel(run=run, model="cloudmodel", invocation="api",
                             location="cloud", status="geweigerd", attempts=0,
                             error_code="niet_toegestaan_intern"))
    await runs.ververs_models(run)
    opgeslagen = (await runs.get(run)).models
    assert sorted((m["model"], m["model_version"], m["invocation"]) for m in opgeslagen) == [
        ("embed", "v1", "llama-swap"), ("werkmodel", "v1", "llama-swap")]
    assert sorted(m["model"] for m in await runs.models_from_model_calls(run)) == [
        "embed", "werkmodel"]


async def test_twee_keer_uitvoeren_en_down(live_surrealdb) -> None:
    regel = await ModelCallRepository(live_surrealdb).create(_regel())
    for _ in range(2):
        await execute_transaction(_migratie(101, down=True), config=live_surrealdb)
    info = await _info(live_surrealdb)
    assert "alleen_aanvullen" not in info["events"]
    assert "idx_model_call_run" not in info["indexes"]
    # De regels blijven staan (herkomst), en de tabel blijft zonder tekstveld.
    [terug] = await _q(live_surrealdb, "SELECT id FROM $id", {"id": ensure_record_id(str(regel["id"]))})
    assert terug
    assert set(info["fields"]) == VELDEN

    for _ in range(2):
        await execute_transaction(_migratie(101), config=live_surrealdb)
    info = await _info(live_surrealdb)
    assert "alleen_aanvullen" in info["events"]
    assert "idx_model_call_run" in info["indexes"]
