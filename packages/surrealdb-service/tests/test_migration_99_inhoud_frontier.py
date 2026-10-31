"""Migratie 99: groep met frontier en grens, interne versie schrijft altijd het
record, batch laat bronnen ongemoeid, eigen item of batch blokkeert de
herclassificatie niet (F2-14, vijfde review). Alle testdata is synthetisch.
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
    await execute_transaction(_migratie(versie), config=config, _herclassificatie=True)


@pytest.fixture(autouse=True, scope="module")
def _migraties_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    import asyncio

    import surrealdb_service.connection as conn

    for versie in (93, 94, 95, 96, 97, 98, 99):
        asyncio.run(_op(live_surrealdb, versie))
    conn._pool = None


async def _q(config, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _een(config, sql: str, params: dict | None = None):
    rijen = await _q(config, sql, params)
    return rijen[0] if isinstance(rijen, list) else rijen


async def _run(config, **extra):
    rij = await _een(config, "CREATE processing_run CONTENT $d RETURN id", {"d": {
        "trigger": "schema", "pipeline_id": "p", "flow_vastgelegd": True, **extra}})
    return ensure_record_id(str(rij["id"]))


async def _bron(config, hashes: list[str], klasse: str = "extern", **run_extra):
    rij = await _een(config, "CREATE source CONTENT {title: 'Openbare testbron',"
                     " classification: $k} RETURN id", {"k": klasse})
    bron = ensure_record_id(str(rij["id"]))
    for nr, h in enumerate(hashes, start=1):
        run = await _run(config, **run_extra)
        await _q(config, "CREATE source_version CONTENT $d", {"d": {
            "source": bron, "version_no": nr, "content_hash": h,
            "created_by_run": run, "current_run": run}})
    return bron


async def test_groep_met_frontier_en_grens(live_surrealdb) -> None:
    hashes = [uuid.uuid4().hex * 2 for _ in range(6)]
    bronnen = [await _bron(live_surrealdb, hashes[i:i + 2]) for i in range(5)]
    groep = await _een(live_surrealdb, "RETURN fn::inhoud_groep($h, 100)", {"h": hashes[0]})
    assert sorted(groep["hashes"]) == sorted(hashes)
    assert sorted(str(b) for b in groep["bronnen"]) == sorted(str(b) for b in bronnen)
    # Te lage grens: geen stille halve groep.
    with pytest.raises(RuntimeError, match="inhoud:groep_te_groot"):
        await _q(live_surrealdb, "RETURN fn::inhoud_groep($h, 2)", {"h": hashes[0]})


async def test_interne_versie_schrijft_altijd_het_record(live_surrealdb) -> None:
    h = uuid.uuid4().hex * 2
    await _q(live_surrealdb, "UPSERT type::thing('inhoud', $h) SET classification = 'intern'",
             {"h": h})
    voor = await _een(live_surrealdb, "SELECT VALUE gewijzigd_op FROM type::thing('inhoud', $h)",
                      {"h": h})
    await _bron(live_surrealdb, [h], klasse="intern")
    na = await _een(live_surrealdb, "SELECT VALUE gewijzigd_op FROM type::thing('inhoud', $h)",
                    {"h": h})
    assert na > voor


async def test_batch_laat_bestaande_bronnen_ongemoeid(live_surrealdb) -> None:
    h = uuid.uuid4().hex * 2
    bron = await _bron(live_surrealdb, [h])
    for klasse in ("intern", "extern"):
        await _q(live_surrealdb, "CREATE intake_batch CONTENT $d", {"d": {
            "route": "r", "created_by": "Testpersoon", "classification": klasse,
            "file_hashes": [{"path": "/inname/a.pdf", "content_hash": f"sha256:{h}"}]}})
    assert await _een(live_surrealdb, "SELECT VALUE classification FROM $b", {"b": bron}) == "extern"
    assert await _een(live_surrealdb, "RETURN fn::inhoud_intern_bekend($h)", {"h": h}) is True


async def test_eigen_item_en_batch_blokkeren_niet(live_surrealdb) -> None:
    h = uuid.uuid4().hex * 2
    item = await _een(live_surrealdb, "CREATE intake_item CONTENT $d RETURN id", {"d": {
        "origin": "watchfolder", "path": f"/inname/{uuid.uuid4().hex}.pdf",
        "content_hash": f"sha256:{h}", "registered_by": "langflow"}})
    item_id = ensure_record_id(str(item["id"]))
    await _q(live_surrealdb, "UPDATE $id SET status = 'besloten', decision = {route: 'r',"
             " classification: 'intern', decided_by: 'Testpersoon', decided_at: $t}",
             {"id": item_id, "t": datetime.now(timezone.utc)})
    batch = await _een(live_surrealdb, "CREATE intake_batch CONTENT $d RETURN id", {"d": {
        "route": "r", "created_by": "Testpersoon", "classification": "intern",
        "file_hashes": [{"path": "/inname/a.pdf", "content_hash": f"sha256:{h}"}]}})
    batch_id = ensure_record_id(str(batch["id"]))
    eigen = await _bron(live_surrealdb, [h], klasse="intern", intake_item=item_id)
    elders = await _q(live_surrealdb, "RETURN fn::inhoud_elders_intern([$h], $b)",
                      {"h": h, "b": eigen})
    # Het item hoort bij de eigen run; de batch nog niet.
    assert [str(r) for r in elders] == [str(batch_id)]
    andere = await _bron(live_surrealdb, [h], klasse="intern", intake_batch=batch_id)
    elders = await _q(live_surrealdb, "RETURN fn::inhoud_elders_intern([$h], $b)",
                      {"h": h, "b": andere})
    assert {str(r) for r in elders} == {str(item_id)} | {
        str(v) for v in await _q(live_surrealdb, "SELECT VALUE id FROM source_version"
                                 " WHERE source = $b", {"b": eigen})}


async def test_down_en_weer_op(live_surrealdb) -> None:
    for _ in range(2):
        await execute_transaction(_migratie(99, down=True), config=live_surrealdb)
    db = await _een(live_surrealdb, "INFO FOR DB")
    assert "inhoud_groep" not in db["functions"]
    for _ in range(2):
        await _op(live_surrealdb, 99)
    db = await _een(live_surrealdb, "INFO FOR DB")
    assert "inhoud_groep" in db["functions"]
