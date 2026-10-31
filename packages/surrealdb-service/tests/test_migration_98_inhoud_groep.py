"""Migratie 98: inhoud zonder recursie, alleen intern bij een expliciet besluit,
losser alleen in de herclassificatie, versieteller per stap (F2-14, vierde
review). Elke test schrijft een waarde én leest die terug. Alle testdata is
synthetisch.
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


async def _op(config, versie: int) -> None:
    # Migratie 98 noemt de vlag van de herclassificatie; zoals de migratieloader.
    await execute_transaction(_migratie(versie), config=config, _herclassificatie=True)


@pytest.fixture(autouse=True, scope="module")
def _migraties_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    import asyncio

    import surrealdb_service.connection as conn

    for versie in (93, 94, 95, 96, 97, 98):
        asyncio.run(_op(live_surrealdb, versie))
    conn._pool = None


async def _q(config, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _een(config, sql: str, params: dict | None = None):
    rijen = await _q(config, sql, params)
    return rijen[0] if isinstance(rijen, list) else rijen


async def _run(config):
    rij = await _een(config, "CREATE processing_run CONTENT {trigger: 'schema', pipeline_id: 'p',"
                     " flow_vastgelegd: true} RETURN id")
    return ensure_record_id(str(rij["id"]))


async def _bron_met_versies(config, hashes: list[str]):
    rij = await _een(config, "CREATE source CONTENT {title: 'Openbare testbron',"
                     " classification: 'extern'} RETURN id")
    bron = ensure_record_id(str(rij["id"]))
    for nr, h in enumerate(hashes, start=1):
        run = await _run(config)
        await _q(config, "CREATE source_version CONTENT $d", {"d": {
            "source": bron, "version_no": nr, "content_hash": h,
            "created_by_run": run, "current_run": run}})
    return bron


async def test_keten_zonder_recursie(live_surrealdb) -> None:
    hashes = [uuid.uuid4().hex * 2 for _ in range(31)]
    bronnen = [await _bron_met_versies(live_surrealdb, hashes[i:i + 2]) for i in range(30)]
    await execute_transaction("BEGIN TRANSACTION; RETURN fn::inhoud_intern($h, true);"
                              " COMMIT TRANSACTION;", {"h": hashes[0]}, live_surrealdb)
    klassen = await _q(live_surrealdb, "SELECT VALUE classification FROM $b", {"b": bronnen})
    assert set(klassen) == {"intern"} and len(klassen) == 30
    inhoud = await _q(live_surrealdb, "SELECT VALUE classification FROM $ids",
                      {"ids": [ensure_record_id(f"inhoud:{h}") for h in hashes]})
    assert set(inhoud) == {"intern"} and len(inhoud) == 31
    meldingen = await _q(live_surrealdb, "SELECT VALUE source FROM beheer_melding"
                         " WHERE soort = 'inhoud_intern'")
    assert {str(b) for b in bronnen} <= {str(m) for m in meldingen}


async def test_onbevestigde_externe_batch_zet_het_record_niet(live_surrealdb) -> None:
    h = uuid.uuid4().hex * 2
    bron = await _bron_met_versies(live_surrealdb, [h])
    rij = await _een(live_surrealdb, "CREATE intake_batch CONTENT $d RETURN id", {"d": {
        "route": "r", "created_by": "Testpersoon", "classification": "extern",
        "file_hashes": [{"path": "/inname/a.pdf", "content_hash": f"sha256:{h}"}]}})
    assert await _een(live_surrealdb, "SELECT VALUE classification FROM $b", {"b": bron}) == "extern"
    assert await _een(live_surrealdb, "SELECT VALUE classification FROM type::thing('inhoud', $h)",
                      {"h": h}) == "extern"
    # Telt wel mee in de controle.
    assert await _een(live_surrealdb, "RETURN fn::inhoud_intern_bekend($h)", {"h": h}) is True
    await _q(live_surrealdb, "UPDATE $id SET classification_confirmed_by = 'Testpersoon',"
             " classification_confirmed_at = $t",
             {"id": ensure_record_id(str(rij["id"])), "t": datetime.now(timezone.utc)})
    assert await _een(live_surrealdb, "RETURN fn::inhoud_intern_bekend($h)", {"h": h}) is False


async def test_inhoud_alleen_losser_met_de_vlag(live_surrealdb) -> None:
    h = uuid.uuid4().hex * 2
    await _q(live_surrealdb, "UPSERT type::thing('inhoud', $h) SET classification = 'intern'",
             {"h": h})
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "UPDATE type::thing('inhoud', $h) SET classification = 'extern'",
                 {"h": h})
    await execute_transaction(
        "BEGIN TRANSACTION; LET $beheer_inhoud_los = true;"
        " UPDATE type::thing('inhoud', $h) SET classification = 'extern'; COMMIT TRANSACTION;",
        {"h": h}, live_surrealdb, _herclassificatie=True)
    assert await _een(live_surrealdb, "SELECT VALUE classification FROM type::thing('inhoud', $h)",
                      {"h": h}) == "extern"


async def test_versieteller(live_surrealdb) -> None:
    rij = await _een(live_surrealdb, "CREATE source CONTENT {title: 't'} RETURN AFTER")
    bron = ensure_record_id(str(rij["id"]))
    assert rij["version_seq"] == 0
    await _q(live_surrealdb, "UPDATE $b SET version_seq = 1", {"b": bron})
    await _q(live_surrealdb, "UPDATE $b SET title = 'ander'", {"b": bron})
    for fout in ("version_seq = 3", "version_seq = NONE", "version_seq = 0"):
        with pytest.raises(RuntimeError):
            await _q(live_surrealdb, f"UPDATE $b SET {fout}", {"b": bron})
    assert await _een(live_surrealdb, "SELECT VALUE version_seq FROM $b", {"b": bron}) == 1


async def test_down_en_weer_op(live_surrealdb) -> None:
    for _ in range(2):
        await execute_transaction(_migratie(98, down=True), config=live_surrealdb)
    db = await _een(live_surrealdb, "INFO FOR DB")
    assert "inhoud_groep" not in db["tables"] and "inhoud_elders_intern" not in db["functions"]
    for _ in range(2):
        await _op(live_surrealdb, 98)
    db = await _een(live_surrealdb, "INFO FOR DB")
    assert "inhoud_groep" in db["tables"]
