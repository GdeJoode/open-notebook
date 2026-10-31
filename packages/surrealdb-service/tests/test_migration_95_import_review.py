"""Migratie 95: indexen op inhoudshash, hashes van batchbestanden, structuur per
versie (F2-14, review). Elke test schrijft een waarde én leest die terug.
Alle testdata is synthetisch.
"""

from __future__ import annotations

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query, execute_transaction
from surrealdb_service.testing import fixtures as fx

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]

HASH = "sha256:" + "c" * 64


def _migratie(versie: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}{suffix}.surrealql").sql


@pytest.fixture(autouse=True, scope="module")
def _migratie_95_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    import asyncio

    import surrealdb_service.connection as conn

    for versie in (93, 94, 95):
        asyncio.run(execute_transaction(_migratie(versie), config=live_surrealdb))
    conn._pool = None


async def _q(config, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _indexen(config, tabel: str) -> dict:
    info = await _q(config, f"INFO FOR TABLE {tabel}")
    return (info[0] if isinstance(info, list) else info)["indexes"]


async def test_batch_bewaart_bestandshashes(live_surrealdb) -> None:
    [rij] = await _q(live_surrealdb, "CREATE intake_batch CONTENT $d RETURN AFTER", {"d": {
        "route": "inname_pdf", "created_by": "Testpersoon", "files": ["/inname/a.pdf"],
        "file_hashes": [{"path": "/inname/a.pdf", "content_hash": HASH}]}})
    assert rij["file_hashes"] == [{"path": "/inname/a.pdf", "content_hash": HASH}]
    [leeg] = await _q(live_surrealdb, "CREATE intake_batch CONTENT {route: 'x',"
                      " created_by: 'Testpersoon'} RETURN AFTER")
    assert leeg["file_hashes"] == []
    for fout in ({"path": "/a", "content_hash": "abc"}, {"path": " ", "content_hash": HASH}):
        with pytest.raises(RuntimeError):
            await _q(live_surrealdb, "CREATE intake_batch CONTENT $d", {"d": {
                "route": "x", "created_by": "Testpersoon", "file_hashes": [fout]}})


async def test_indexen_en_heen_en_terug(live_surrealdb) -> None:
    assert "idx_source_version_content_hash" in await _indexen(live_surrealdb, "source_version")
    assert "idx_intake_item_content_hash" in await _indexen(live_surrealdb, "intake_item")
    assert "source_version" in (await _indexen(live_surrealdb, "doc_node"))["dn_run_ref"]

    await execute_transaction(_migratie(95, down=True), config=live_surrealdb)
    await execute_transaction(_migratie(95, down=True), config=live_surrealdb)
    assert "idx_source_version_content_hash" not in await _indexen(live_surrealdb, "source_version")
    ref = (await _indexen(live_surrealdb, "doc_node"))["dn_run_ref"]
    assert "source_version" not in ref and "UNIQUE" not in ref

    await execute_transaction(_migratie(95), config=live_surrealdb)
    await execute_transaction(_migratie(95), config=live_surrealdb)
    assert "UNIQUE" in (await _indexen(live_surrealdb, "doc_node"))["dn_run_ref"]
    [rij] = await _q(live_surrealdb, "CREATE intake_batch CONTENT $d RETURN AFTER", {"d": {
        "route": "inname_pdf", "created_by": "Testpersoon",
        "file_hashes": [{"path": "/inname/b.pdf", "content_hash": HASH}]}})
    assert rij["file_hashes"][0]["content_hash"] == HASH
