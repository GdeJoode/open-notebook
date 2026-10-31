"""Migratie 103: Track J-routering en oude privacyvelden weg (F2-09).

- de velddefinities ``source.private``, ``notebook.privacy_mode`` en
  ``open_notebook.default_privacy_mode`` vervallen;
- de tabel ``model_route`` vervalt;
- de ``_down`` zet ze terug en zet ``private = true`` op alle bronnen;
- heen en terug zijn idempotent, en een bron blijft bestaan.

Onder SurrealDB v2.7.0 laat een SCHEMAFULL-tabel een onbekend veld stil weg;
daarom schrijft de test een waarde én leest die terug. Alle testdata is
synthetisch.
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query, execute_transaction
from surrealdb_service.testing import fixtures as fx

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]

VELDEN = (("source", "private"), ("notebook", "privacy_mode"),
          ("open_notebook", "default_privacy_mode"))


def _migratie(versie: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}{suffix}.surrealql").sql


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _info(config: SurrealDBConfig, tabel: str | None = None) -> dict:
    info = await _q(config, f"INFO FOR TABLE {tabel}" if tabel else "INFO FOR DB")
    return info[0] if isinstance(info, list) else info


async def _velden_weg(config: SurrealDBConfig) -> None:
    for tabel, veld in VELDEN:
        assert veld not in (await _info(config, tabel))["fields"], (tabel, veld)
    assert "model_route" not in (await _info(config))["tables"]


async def _bron(config: SurrealDBConfig) -> str:
    [rij] = await _q(config, "CREATE source SET title = $t RETURN id",
                     {"t": f"bron-{uuid.uuid4().hex[:8]}"})
    return str(rij["id"])


async def test_velden_en_tabel_weg_na_de_migraties(live_surrealdb) -> None:
    # Een andere test kan de oude definitie tijdelijk terugzetten (migratie 65).
    await execute_transaction(_migratie(103), config=live_surrealdb)
    await _velden_weg(live_surrealdb)


async def test_private_wordt_niet_meer_bewaard(live_surrealdb) -> None:
    await execute_transaction(_migratie(103), config=live_surrealdb)
    [rij] = await _q(live_surrealdb, "CREATE source SET title = 'x', private = true RETURN id")
    [terug] = await _q(live_surrealdb, "SELECT * FROM type::thing($id)", {"id": str(rij["id"])})
    assert terug["title"] == "x"
    assert "private" not in terug


async def test_heen_en_terug_idempotent(live_surrealdb) -> None:
    bron = await _bron(live_surrealdb)
    try:
        for _ in range(2):
            await execute_transaction(_migratie(103, down=True), config=live_surrealdb)
        for tabel, veld in VELDEN:
            assert veld in (await _info(live_surrealdb, tabel))["fields"], (tabel, veld)
        assert "model_route" in (await _info(live_surrealdb))["tables"]
        # Terugdraaien maakt niets extern: elke bron is private.
        rows = await _q(live_surrealdb, "SELECT VALUE private FROM source")
        assert rows and all(p is True for p in rows)
        [nb] = await _q(live_surrealdb,
                        "CREATE notebook SET name = 'nb', privacy_mode = 'private' RETURN id")
        [terug] = await _q(live_surrealdb, "SELECT privacy_mode FROM type::thing($id)",
                           {"id": str(nb["id"])})
        assert terug["privacy_mode"] == "private"
    finally:
        for _ in range(2):
            await execute_transaction(_migratie(103), config=live_surrealdb)
    await _velden_weg(live_surrealdb)
    # Alleen de definities gaan weg: de bron zelf blijft.
    assert await _q(live_surrealdb, "SELECT id FROM type::thing($id)", {"id": bron})
