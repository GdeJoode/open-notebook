"""Migratie 105: reviewpunten F2-15 (ADR 0011, 0003).

- Een embedding van een fragment hoort bij de bron en de run van dat fragment;
  zonder run kan niet meer bij een fragment met een run.
- Hoogstens één embedding per fragment en run.
- ``run_step.review_item`` mag een notebook zijn.
- Gaat een run dicht, dan gaan zijn open stappen op ``mislukt``.

Elke test schrijft en leest terug (SurrealDB v2.7.0). Testdata is synthetisch.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    ensure_record_id,
    execute_query,
    execute_transaction,
)
from surrealdb_service.repositories import ProcessingRunRepository, SourceVersionRepository
from surrealdb_service.testing import fixtures as fx

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]


def _migratie(versie: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}{suffix}.surrealql").sql


@pytest.fixture(autouse=True, scope="module")
def _migratie_nog_eens(live_surrealdb: SurrealDBConfig) -> None:
    import asyncio

    import surrealdb_service.connection as conn

    asyncio.run(execute_transaction(_migratie(105), config=live_surrealdb))
    conn._pool = None


async def _q(config, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _een(config, sql, params=None) -> dict:
    rows = await _q(config, sql, params)
    return rows[0] if isinstance(rows, list) else rows


async def _opzet(config) -> dict:
    run = await ProcessingRunRepository(config).create(
        trigger="schema", pipeline_id="test", flow_vastgelegd=True)
    run_id = str(run.id)
    bron = str((await _een(config, "CREATE source SET title = 'Openbare testbron',"
                                   " classification = 'intern' RETURN id"))["id"])
    versie, _ = await SourceVersionRepository(config).register(
        bron, hashlib.sha256(uuid.uuid4().bytes).hexdigest(), run_id)
    fragment = await _een(config, "CREATE chunk CONTENT $c RETURN id", {"c": {
        "source": ensure_record_id(bron), "text": "Openbare tekst.", "order": 0,
        "physical_page": 1, "element_type": "paragraph", "positions": [], "metadata": {},
        "source_version": ensure_record_id(str(versie.id)), "run": ensure_record_id(run_id)}})
    return {"run": run_id, "bron": bron, "versie": str(versie.id),
            "fragment": str(fragment["id"])}


def _embedding(o: dict, **velden) -> dict:
    d = {"source": ensure_record_id(o["bron"]), "order": 0, "content": "Openbare tekst.",
         "embedding": [0.1] * 4, "chunk": ensure_record_id(o["fragment"]),
         "run": ensure_record_id(o["run"]), "embedding_model": "embed:v1", "afgekapt": False}
    d.update(velden)
    return {k: v for k, v in d.items() if v is not None}


async def test_embedding_volgt_bron_en_run_van_het_fragment(live_surrealdb) -> None:
    o = await _opzet(live_surrealdb)
    ander = await _opzet(live_surrealdb)
    # Zonder run bij een fragment met een run (de oude route).
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE source_embedding CONTENT $d",
                 {"d": _embedding(o, run=None, embedding_model=None, afgekapt=None)})
    # Fragment van een andere bron.
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE source_embedding CONTENT $d",
                 {"d": _embedding(o, source=ensure_record_id(ander["bron"]))})
    rij = await _een(live_surrealdb, "CREATE ONLY source_embedding CONTENT $d",
                     {"d": _embedding(o)})
    gelezen = await _een(live_surrealdb, "SELECT * FROM ONLY $id",
                         {"id": ensure_record_id(rij["id"])})
    assert (str(gelezen["run"]), str(gelezen["chunk"])) == (o["run"], o["fragment"])
    # Eén embedding per fragment en run.
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE source_embedding CONTENT $d", {"d": _embedding(o)})


async def test_embeddings_zonder_fragment_blijven_mogelijk(live_surrealdb) -> None:
    o = await _opzet(live_surrealdb)
    for _ in range(2):
        await _q(live_surrealdb, "CREATE source_embedding CONTENT $d", {"d": {
            "source": ensure_record_id(o["bron"]), "order": 0, "content": "t",
            "embedding": [0.1] * 4}})
    rijen = await _q(live_surrealdb, "SELECT id FROM source_embedding WHERE source = $b",
                     {"b": ensure_record_id(o["bron"])})
    assert len(rijen) == 2


async def test_reviewitem_mag_een_notebook_zijn(live_surrealdb) -> None:
    o = await _opzet(live_surrealdb)
    nb = await _een(live_surrealdb, "CREATE ONLY notebook SET name = 'Openbaar notebook',"
                                    " description = '' RETURN id")
    stap = await _een(live_surrealdb, "CREATE ONLY run_step CONTENT $d", {"d": {
        "run": ensure_record_id(o["run"]), "stap": "entity_extraction",
        "source_version": ensure_record_id(o["versie"]), "source": ensure_record_id(o["bron"]),
        "status": "wacht_op_review", "review_item": ensure_record_id(nb["id"])}})
    gelezen = await _een(live_surrealdb, "SELECT * FROM ONLY $id",
                         {"id": ensure_record_id(stap["id"])})
    assert str(gelezen["review_item"]) == str(nb["id"])


async def test_sluiten_van_een_run_zet_open_stappen_op_mislukt(live_surrealdb) -> None:
    o = await _opzet(live_surrealdb)
    stappen = {}
    for stap, status in (("embed", "geslaagd"), ("entity_extraction", "bezig"),
                         ("graph", "wachtrij"), ("entity_dedup", "wacht_op_review")):
        rij = await _een(live_surrealdb, "CREATE ONLY run_step CONTENT $d", {"d": {
            "run": ensure_record_id(o["run"]), "stap": stap, "status": status,
            "source_version": ensure_record_id(o["versie"]),
            "source": ensure_record_id(o["bron"])}})
        stappen[stap] = rij["id"]
    await ProcessingRunRepository(live_surrealdb).close(o["run"], status="mislukt")
    gelezen = {r["stap"]: (r["status"], r.get("error")) for r in await _q(
        live_surrealdb, "SELECT stap, status, error FROM run_step WHERE run = $r",
        {"r": ensure_record_id(o["run"])})}
    assert gelezen == {
        "embed": ("geslaagd", None),
        "entity_extraction": ("mislukt", "Stap entity_extraction mislukt na 0 pogingen"
                                         " (RunAfgesloten)"),
        "graph": ("mislukt", "Stap graph mislukt na 0 pogingen (RunAfgesloten)"),
        "entity_dedup": ("mislukt", "Stap entity_dedup mislukt na 0 pogingen (RunAfgesloten)"),
    }


async def test_down_en_weer_op(live_surrealdb) -> None:
    await execute_transaction(_migratie(105, down=True), config=live_surrealdb)
    try:
        o = await _opzet(live_surrealdb)
        await _q(live_surrealdb, "CREATE run_step CONTENT $d", {"d": {
            "run": ensure_record_id(o["run"]), "stap": "graph",
            "source_version": ensure_record_id(o["versie"]),
            "source": ensure_record_id(o["bron"])}})
        await ProcessingRunRepository(live_surrealdb).close(o["run"], status="mislukt")
        [status] = await _q(live_surrealdb, "SELECT VALUE status FROM run_step WHERE run = $r",
                            {"r": ensure_record_id(o["run"])})
        assert status == "wachtrij"
        info = await _een(live_surrealdb, "INFO FOR TABLE source_embedding")
        assert "uniek_embedding_run" not in info["indexes"]
    finally:
        await execute_transaction(_migratie(105), config=live_surrealdb)
    info = await _een(live_surrealdb, "INFO FOR TABLE source_embedding")
    assert "uniek_embedding_run" in info["indexes"]
