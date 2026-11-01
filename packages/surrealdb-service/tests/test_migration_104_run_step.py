"""Migratie 104: stappen van een run en embeddings met run (F2-15, ADR 0011, 0003, 0008).

- ``run_step``: uniek op run, stap en versie; alleen in een lopende run; de bron
  is die van de versie; ``geslaagd`` en ``mislukt`` zijn eindtoestanden; ``error``
  is een vaste melding zonder vrije tekst.
- ``source_embedding`` met een run: bij een fragment van die run, in een lopende
  run, met ``embedding_model`` en ``afgekapt``. Zonder run blijft het als voorheen.
- ``beheer_melding`` kent ``stap_mislukt`` met ``run_step``.
- ``_down`` draait terug en de migratie is idempotent.

Onder SurrealDB v2.7.0 laat een SCHEMAFULL-tabel een onbekend veld stil weg;
daarom schrijft elke test een waarde én leest die terug. Alle testdata is
synthetisch.
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
    """Idempotent: nog een keer bovenop de al toegepaste migraties."""
    import asyncio

    import surrealdb_service.connection as conn

    asyncio.run(execute_transaction(_migratie(104), config=live_surrealdb))
    conn._pool = None


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _een(config, sql, params=None) -> dict:
    rows = await _q(config, sql, params)
    return rows[0] if isinstance(rows, list) else rows


async def _opzet(config) -> dict:
    """Een lopende run met een interne bron, een versie en een fragment."""
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
    return {"run": run_id, "bron": bron, "versie": str(versie.id), "fragment": str(fragment["id"])}


def _stap(o: dict, stap: str = "embed", **velden) -> dict:
    d = {"run": ensure_record_id(o["run"]), "stap": stap,
         "source_version": ensure_record_id(o["versie"]), "source": ensure_record_id(o["bron"])}
    d.update(velden)
    return d


def _embedding(o: dict, **velden) -> dict:
    d = {"source": ensure_record_id(o["bron"]), "order": 0, "content": "Openbare tekst.",
         "embedding": [0.1] * 4, "chunk": ensure_record_id(o["fragment"]),
         "run": ensure_record_id(o["run"]), "embedding_model": "embed:v1", "afgekapt": False}
    d.update(velden)
    return {k: v for k, v in d.items() if v is not None}


async def test_stap_schrijven_en_teruglezen(live_surrealdb) -> None:
    o = await _opzet(live_surrealdb)
    rij = await _een(live_surrealdb, "CREATE ONLY run_step CONTENT $d", {"d": _stap(o)})
    gelezen = await _een(live_surrealdb, "SELECT * FROM ONLY $id",
                         {"id": ensure_record_id(rij["id"])})
    assert (gelezen["stap"], gelezen["status"], gelezen["attempts"]) == ("embed", "wachtrij", 0)
    assert str(gelezen["source_version"]) == o["versie"] and str(gelezen["source"]) == o["bron"]
    assert gelezen["created_at"] and gelezen["updated_at"]

    melding = "Stap embed mislukt na 3 pogingen (RuntimeError)"
    await _q(live_surrealdb, "UPDATE $id SET status = 'mislukt', error = $e",
             {"id": ensure_record_id(rij["id"]), "e": melding})
    gelezen = await _een(live_surrealdb, "SELECT * FROM ONLY $id",
                         {"id": ensure_record_id(rij["id"])})
    assert (gelezen["status"], gelezen["error"]) == ("mislukt", melding)
    # Een eindtoestand blijft.
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "UPDATE $id SET status = 'bezig'",
                 {"id": ensure_record_id(rij["id"])})


@pytest.mark.parametrize("fout", [
    "Interne zin uit een bron",
    "Stap embed mislukt na 3 pogingen (Runtime Error met tekst)",
    "Poging 1 mislukt (RuntimeError)\nregel",
])
async def test_geen_vrije_tekst_in_error(live_surrealdb, fout) -> None:
    o = await _opzet(live_surrealdb)
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE run_step CONTENT $d", {"d": _stap(o, error=fout)})


async def test_stap_uniek_in_een_lopende_run_bij_de_eigen_bron(live_surrealdb) -> None:
    o = await _opzet(live_surrealdb)
    await _q(live_surrealdb, "CREATE run_step CONTENT $d", {"d": _stap(o)})
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE run_step CONTENT $d", {"d": _stap(o)})
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE run_step CONTENT $d", {"d": _stap(o, stap="onbekend")})
    ander = await _opzet(live_surrealdb)
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE run_step CONTENT $d",
                 {"d": _stap(o, stap="graph", source=ensure_record_id(ander["bron"]))})
    await ProcessingRunRepository(live_surrealdb).close(o["run"], status="mislukt")
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE run_step CONTENT $d", {"d": _stap(o, stap="graph")})


async def test_embedding_met_run_schrijven_en_teruglezen(live_surrealdb) -> None:
    o = await _opzet(live_surrealdb)
    rij = await _een(live_surrealdb, "CREATE ONLY source_embedding CONTENT $d",
                     {"d": _embedding(o, afgekapt=True)})
    gelezen = await _een(live_surrealdb, "SELECT * FROM ONLY $id",
                         {"id": ensure_record_id(rij["id"])})
    assert (str(gelezen["run"]), gelezen["embedding_model"], gelezen["afgekapt"]) == (
        o["run"], "embed:v1", True)
    # Zonder run en zonder fragment blijft het als voorheen; zonder run bij een
    # fragment met een run kan sinds migratie 105 niet meer.
    await _q(live_surrealdb, "CREATE source_embedding CONTENT $d",
             {"d": _embedding(o, chunk=None, run=None, embedding_model=None, afgekapt=None)})


@pytest.mark.parametrize("weg", ["embedding_model", "afgekapt", "chunk"])
async def test_embedding_met_run_vraagt_model_afgekapt_en_fragment(live_surrealdb, weg) -> None:
    o = await _opzet(live_surrealdb)
    d = _embedding(o)
    d.pop(weg)
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE source_embedding CONTENT $d", {"d": d})


async def test_embedding_alleen_bij_een_fragment_van_een_lopende_run(live_surrealdb) -> None:
    o = await _opzet(live_surrealdb)
    ander = await ProcessingRunRepository(live_surrealdb).create(
        trigger="schema", pipeline_id="test", flow_vastgelegd=True)
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE source_embedding CONTENT $d",
                 {"d": _embedding(o, run=ensure_record_id(str(ander.id)))})
    await ProcessingRunRepository(live_surrealdb).close(o["run"], status="geslaagd")
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE source_embedding CONTENT $d", {"d": _embedding(o)})


async def test_melding_stap_mislukt(live_surrealdb) -> None:
    o = await _opzet(live_surrealdb)
    stap = await _een(live_surrealdb, "CREATE ONLY run_step CONTENT $d", {"d": _stap(o)})
    rij = await _een(live_surrealdb, "CREATE ONLY beheer_melding CONTENT $d", {"d": {
        "soort": "stap_mislukt", "melding": "Stap embed mislukt na 3 pogingen (RuntimeError)",
        "run": ensure_record_id(o["run"]), "run_step": ensure_record_id(stap["id"])}})
    gelezen = await _een(live_surrealdb, "SELECT * FROM ONLY $id",
                         {"id": ensure_record_id(rij["id"])})
    assert (gelezen["soort"], str(gelezen["run_step"])) == ("stap_mislukt", str(stap["id"]))


async def test_down_en_weer_op(live_surrealdb) -> None:
    await execute_transaction(_migratie(104, down=True), config=live_surrealdb)
    try:
        o = await _opzet(live_surrealdb)
        # Na `_down`: geen ASSERT meer op run, model en afgekapt; de tabel blijft.
        d = _embedding(o)
        d.pop("afgekapt")
        d.pop("embedding_model")
        await _q(live_surrealdb, "CREATE source_embedding CONTENT $d", {"d": d})
        await _q(live_surrealdb, "CREATE run_step CONTENT $d", {"d": _stap(o)})
        info = await _een(live_surrealdb, "INFO FOR TABLE source_embedding")
        assert "idx_source_embedding_run" not in info["indexes"]
    finally:
        await execute_transaction(_migratie(104), config=live_surrealdb)
    info = await _een(live_surrealdb, "INFO FOR TABLE source_embedding")
    assert "idx_source_embedding_run" in info["indexes"]
