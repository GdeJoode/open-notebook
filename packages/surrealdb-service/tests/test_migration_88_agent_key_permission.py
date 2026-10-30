"""Migratie 88: de vijf rechten op `agent_keys` en het veld `person` (F2-01).

Onder SurrealDB v2.7.0 laat een SCHEMAFULL-tabel een onbekend veld stil weg
(P10, K6). Daarom schrijft elke test een waarde én leest die terug.

De fixture is per sessie: een test die `_down` draait, zet 88 daarna terug.
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query, execute_transaction
from surrealdb_service.testing import fixtures as fx

NIEUWE_RECHTEN = ["lezen_extern", "lezen_intern", "pijplijn", "ophalen", "beheerder"]


def _migratie(versie: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}{suffix}.surrealql").sql


async def _maak(config: SurrealDBConfig, permission: str, person: str | None = None):
    data = {
        "agent_id": f"test-{uuid.uuid4().hex[:8]}",
        "key_hash": uuid.uuid4().hex,
        "key_prefix": "ak_test",
        "permission": permission,
    }
    if person is not None:
        data["person"] = person
    rows = await execute_query("CREATE agent_keys CONTENT $data", {"data": data}, config)
    return rows[0]


async def _lees(config: SurrealDBConfig, rid) -> dict:
    rows = await execute_query("SELECT * FROM type::thing($id)", {"id": rid}, config)
    return rows[0]


@pytest.mark.requires_docker
@pytest.mark.asyncio
@pytest.mark.parametrize("recht", NIEUWE_RECHTEN)
async def test_nieuwe_rechten_worden_aanvaard_en_teruggelezen(
    live_surrealdb: SurrealDBConfig, recht: str
) -> None:
    person = "Testpersoon" if recht == "beheerder" else None
    row = await _maak(live_surrealdb, recht, person)
    teruggelezen = await _lees(live_surrealdb, row["id"])
    assert teruggelezen["permission"] == recht
    if person:
        assert teruggelezen["person"] == person


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_person_wordt_bewaard_bij_een_dienstsleutel(
    live_surrealdb: SurrealDBConfig,
) -> None:
    row = await _maak(live_surrealdb, "pijplijn", "Dienst Langflow")
    assert (await _lees(live_surrealdb, row["id"]))["person"] == "Dienst Langflow"


@pytest.mark.requires_docker
@pytest.mark.asyncio
@pytest.mark.parametrize("oud", ["read", "write", "admin"])
async def test_oude_rechten_worden_geweigerd(
    live_surrealdb: SurrealDBConfig, oud: str
) -> None:
    with pytest.raises(Exception):
        await _maak(live_surrealdb, oud)


@pytest.mark.requires_docker
@pytest.mark.asyncio
@pytest.mark.parametrize("person", [None, "", "   "])
async def test_beheerder_zonder_person_wordt_geweigerd(
    live_surrealdb: SurrealDBConfig, person: str | None
) -> None:
    with pytest.raises(Exception):
        await _maak(live_surrealdb, "beheerder", person)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_person_weghalen_bij_beheerder_wordt_geweigerd(
    live_surrealdb: SurrealDBConfig,
) -> None:
    row = await _maak(live_surrealdb, "beheerder", "Testpersoon")
    with pytest.raises(Exception):
        await execute_query("UPDATE type::thing($id) UNSET person", {"id": row["id"]}, live_surrealdb)
    assert (await _lees(live_surrealdb, row["id"]))["person"] == "Testpersoon"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migratie_twee_keer_na_elkaar(live_surrealdb: SurrealDBConfig) -> None:
    await execute_transaction(_migratie(88), config=live_surrealdb)
    await execute_transaction(_migratie(88), config=live_surrealdb)
    row = await _maak(live_surrealdb, "lezen_extern")
    assert (await _lees(live_surrealdb, row["id"]))["permission"] == "lezen_extern"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_down_draait_terug_en_trekt_nieuwe_sleutels_in(
    live_surrealdb: SurrealDBConfig,
) -> None:
    beheerder = await _maak(live_surrealdb, "beheerder", "Testpersoon")
    try:
        await execute_transaction(_migratie(88, down=True), config=live_surrealdb)
        # Twee keer na elkaar: idempotent.
        await execute_transaction(_migratie(88, down=True), config=live_surrealdb)

        assert (await _lees(live_surrealdb, beheerder["id"]))["revoked"] is True
        oud = await _maak(live_surrealdb, "read")
        assert (await _lees(live_surrealdb, oud["id"]))["permission"] == "read"
        with pytest.raises(Exception):
            await _maak(live_surrealdb, "lezen_extern")
        # Na `_down` bestaat `person` niet meer: het veld valt stil weg.
        zonder = await _maak(live_surrealdb, "write", "Testpersoon")
        assert "person" not in await _lees(live_surrealdb, zonder["id"])
    finally:
        # Terug naar de stand na alle migraties: de fixture is per sessie.
        await execute_query(
            "DELETE agent_keys WHERE permission IN ['read', 'write', 'admin']",
            None,
            live_surrealdb,
        )
        await execute_transaction(_migratie(88), config=live_surrealdb)

    row = await _maak(live_surrealdb, "beheerder", "Testpersoon")
    assert (await _lees(live_surrealdb, row["id"]))["person"] == "Testpersoon"
