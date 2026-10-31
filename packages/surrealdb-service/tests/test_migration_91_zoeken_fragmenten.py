"""Migratie 91: zoeken leest alleen fragmenten, `is_derived`, `full_text` weg (F2-05).

Onder SurrealDB v2.7.0 laat een SCHEMAFULL-tabel een onbekend veld stil weg
(P4, P10 K6). Daarom schrijft elke test een waarde én leest die terug.
Alle testdata is synthetisch.
"""

from __future__ import annotations

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query, execute_transaction
from surrealdb_service.testing import fixtures as fx

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]


def _migratie(versie: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}{suffix}.surrealql").sql


async def _opnieuw_89_tot_91(config: SurrealDBConfig) -> None:
    for versie in (89, 90, 91):
        await execute_transaction(_migratie(versie), config=config, _herclassificatie=True)


@pytest.fixture(autouse=True, scope="module")
def _migratie_91_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    """De database is per sessie gedeeld; eerdere tests voeren oude migraties
    opnieuw uit. 89 tot en met 91 zijn idempotent."""
    import asyncio

    import surrealdb_service.connection as conn

    asyncio.run(_opnieuw_89_tot_91(live_surrealdb))
    conn._pool = None


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _info(config: SurrealDBConfig, sql: str) -> dict:
    rij = await _q(config, sql)
    return rij[0] if isinstance(rij, list) else rij


async def test_full_text_bestaat_niet_meer(live_surrealdb) -> None:
    info = await _info(live_surrealdb, "INFO FOR TABLE source")
    assert "full_text" not in info["fields"]
    assert "idx_source_full_text" not in info["indexes"]
    # Een schrijfactie met `full_text` komt niet terug (SCHEMAFULL laat weg).
    [rij] = await _q(
        live_surrealdb,
        "CREATE source CONTENT {title: 'Openbare testbron', full_text: 'x'} RETURN AFTER",
    )
    assert "full_text" not in rij


async def test_index_op_fragmenttekst(live_surrealdb) -> None:
    info = await _info(live_surrealdb, "INFO FOR TABLE chunk")
    assert "BM25" in info["indexes"]["idx_chunk_text"]


@pytest.mark.parametrize("tabel", ["note", "source_insight"])
async def test_is_derived_altijd_waar(live_surrealdb, tabel) -> None:
    info = await _info(live_surrealdb, f"INFO FOR TABLE {tabel}")
    assert "is_derived" in info["fields"]
    if tabel == "note":
        [rij] = await _q(live_surrealdb, "CREATE note CONTENT {content: 'Openbare notitie', embedding: []} RETURN AFTER")
    else:
        [bron] = await _q(live_surrealdb, "CREATE source CONTENT {title: 'b'} RETURN id")
        [rij] = await _q(
            live_surrealdb,
            "CREATE source_insight CONTENT {source: type::thing($b), insight_type: 't', content: 'c',"
            " embedding: []} RETURN AFTER",
            {"b": bron["id"]},
        )
    assert rij["is_derived"] is True
    with pytest.raises(Exception):
        await _q(live_surrealdb, "UPDATE type::thing($id) SET is_derived = false", {"id": rij["id"]})
    [terug] = await _q(live_surrealdb, "SELECT VALUE is_derived FROM type::thing($id)", {"id": rij["id"]})
    assert terug is True


async def test_zoekfuncties_hebben_de_nieuwe_parameters(live_surrealdb) -> None:
    db = await _info(live_surrealdb, "INFO FOR DB")
    tekst = db["functions"]["text_search"]
    vector = db["functions"]["vector_search"]
    for definitie in (tekst, vector):
        assert "$classifications" in definitie and "$history" in definitie
        assert "show_notes" not in definitie
        assert "source_insight" not in definitie and "FROM note" not in definitie


async def test_down_draait_terug_en_weer_op(live_surrealdb) -> None:
    try:
        await execute_transaction(_migratie(91, down=True), config=live_surrealdb)
        await execute_transaction(_migratie(91, down=True), config=live_surrealdb)
        source = await _info(live_surrealdb, "INFO FOR TABLE source")
        assert "full_text" in source["fields"]
        assert "idx_source_full_text" in source["indexes"]
        for tabel in ("note", "source_insight"):
            assert "is_derived" not in (await _info(live_surrealdb, f"INFO FOR TABLE {tabel}"))["fields"]
        assert "idx_chunk_text" not in (await _info(live_surrealdb, "INFO FOR TABLE chunk"))["indexes"]
        db = await _info(live_surrealdb, "INFO FOR DB")
        assert "show_notes" in db["functions"]["text_search"]
        assert "show_notes" in db["functions"]["vector_search"]
    finally:
        await _opnieuw_89_tot_91(live_surrealdb)
    await _opnieuw_89_tot_91(live_surrealdb)
    assert "full_text" not in (await _info(live_surrealdb, "INFO FOR TABLE source"))["fields"]
