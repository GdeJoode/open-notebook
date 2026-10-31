"""Migratie 90: bronversie, verwerkingsrun en herkomst (F2-04, ADR 0003).

Onder SurrealDB v2.7.0 laat een SCHEMAFULL-tabel een onbekend veld stil weg
(P4, P10 K6). Daarom schrijft elke test een waarde én leest die terug.
Alle testdata is synthetisch.
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
from surrealdb_service.repositories.processing_run import ProcessingRunRepository
from surrealdb_service.repositories.source_version import SourceVersionRepository
from surrealdb_service.testing import fixtures as fx

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]


def _migratie(versie: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}{suffix}.surrealql").sql


async def _opnieuw_89_en_90(config: SurrealDBConfig) -> None:
    for versie in (89, 90):
        await execute_transaction(_migratie(versie), config=config, _herclassificatie=True)


@pytest.fixture(autouse=True, scope="module")
def _migratie_90_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    """De database is per sessie gedeeld; eerdere tests voeren oude migraties
    opnieuw uit. 89 en 90 zijn idempotent."""
    import asyncio

    import surrealdb_service.connection as conn

    asyncio.run(_opnieuw_89_en_90(live_surrealdb))
    conn._pool = None


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _lees(config: SurrealDBConfig, rid: str) -> dict:
    rows = await _q(config, "SELECT * FROM type::thing($id)", {"id": rid})
    return rows[0]


def _hash(tekst: str | None = None) -> str:
    return hashlib.sha256((tekst or uuid.uuid4().hex).encode()).hexdigest()


async def _bron(config: SurrealDBConfig, classification: str = "extern") -> str:
    [rij] = await _q(
        config,
        "CREATE source CONTENT {title: 'Openbare testbron', classification: $c} RETURN id",
        {"c": classification},
    )
    return str(rij["id"])


async def _run(config: SurrealDBConfig) -> str:
    run = await ProcessingRunRepository(config).create(
        trigger="inname", pipeline_id="test", flow_vastgelegd=True
    )
    return str(run.id)


async def _versie(config: SurrealDBConfig, bron: str, run: str, **extra) -> str:
    versie, nieuw = await SourceVersionRepository(config).register(bron, _hash(), run, **extra)
    assert nieuw
    return str(versie.id)


async def _fragment(config: SurrealDBConfig, bron: str, **herkomst) -> str:
    data = {"source": ensure_record_id(bron), "text": "x", "order": 0, "physical_page": 0,
            "element_type": "paragraph", "metadata": {}, "positions": []}
    data.update({k: ensure_record_id(v) if k in ("source_version", "run") else v
                 for k, v in herkomst.items()})
    [rij] = await _q(config, "CREATE chunk CONTENT $d RETURN id", {"d": data})
    return str(rij["id"])


async def _doc_node(config: SurrealDBConfig, bron: str, **herkomst) -> str:
    data = {"source": ensure_record_id(bron), "element_type": "paragraph",
            "self_ref": f"#/texts/{uuid.uuid4().hex[:6]}", "sequence": 0}
    data.update({k: ensure_record_id(v) for k, v in herkomst.items()})
    [rij] = await _q(config, "CREATE doc_node CONTENT $d RETURN id", {"d": data})
    return str(rij["id"])


# --------------------------------------------------------------------------
# Verplichte herkomst op `chunk` en `doc_node`
# --------------------------------------------------------------------------


@pytest.mark.parametrize("maak", [_fragment, _doc_node], ids=["chunk", "doc_node"])
@pytest.mark.parametrize("weg", ["source_version", "run"])
async def test_zonder_versie_of_run_geweigerd(live_surrealdb, maak, weg) -> None:
    bron = await _bron(live_surrealdb)
    run = await _run(live_surrealdb)
    herkomst = {"source_version": await _versie(live_surrealdb, bron, run), "run": run}
    herkomst.pop(weg)
    with pytest.raises(RuntimeError, match=weg):
        await maak(live_surrealdb, bron, **herkomst)


@pytest.mark.parametrize("maak", [_fragment, _doc_node], ids=["chunk", "doc_node"])
async def test_met_herkomst_schrijven_en_teruglezen(live_surrealdb, maak) -> None:
    bron = await _bron(live_surrealdb)
    run = await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, bron, run)
    extra = {"extractor": "docling-serve 1.0"} if maak is _fragment else {}
    rid = await maak(live_surrealdb, bron, source_version=versie, run=run, **extra)
    rij = await _lees(live_surrealdb, rid)
    assert (str(rij["source_version"]), str(rij["run"])) == (versie, run)
    if extra:
        assert rij["extractor"] == "docling-serve 1.0"


@pytest.mark.parametrize("maak", [_fragment, _doc_node], ids=["chunk", "doc_node"])
async def test_versie_van_een_andere_bron_geweigerd(live_surrealdb, maak) -> None:
    bron, ander = await _bron(live_surrealdb), await _bron(live_surrealdb)
    run = await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, ander, run)
    with pytest.raises(RuntimeError, match="source_version"):
        await maak(live_surrealdb, bron, source_version=versie, run=run)


async def test_onbekende_run_geweigerd(live_surrealdb) -> None:
    bron = await _bron(live_surrealdb)
    run = await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, bron, run)
    with pytest.raises(RuntimeError, match="run"):
        await _fragment(live_surrealdb, bron, source_version=versie, run="processing_run:bestaatniet")


async def test_herkomst_van_een_fragment_verandert_niet(live_surrealdb) -> None:
    bron = await _bron(live_surrealdb)
    run, andere_run = await _run(live_surrealdb), await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, bron, run)
    rid = await _fragment(live_surrealdb, bron, source_version=versie, run=run)
    with pytest.raises(RuntimeError, match="readonly"):
        await _q(live_surrealdb, "UPDATE type::thing($id) SET run = $r",
                 {"id": rid, "r": ensure_record_id(andere_run)})


async def test_nieuwe_run_maakt_structuur_naast_de_oude(live_surrealdb) -> None:
    bron = await _bron(live_surrealdb)
    run, tweede = await _run(live_surrealdb), await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, bron, run)
    for r in (run, tweede):
        await _q(live_surrealdb, "CREATE doc_node CONTENT $d", {"d": {
            "source": ensure_record_id(bron), "element_type": "paragraph",
            "self_ref": "#/texts/0", "sequence": 0,
            "source_version": ensure_record_id(versie), "run": ensure_record_id(r)}})
    rows = await _q(live_surrealdb, "SELECT VALUE id FROM doc_node WHERE source = $b",
                    {"b": ensure_record_id(bron)})
    assert len(rows) == 2


@pytest.mark.parametrize("tabel", ["chunk", "doc_node", "source_embedding"])
async def test_verplaatsen_naar_een_andere_bron_geweigerd(live_surrealdb, tabel) -> None:
    bron, ander = await _bron(live_surrealdb, "intern"), await _bron(live_surrealdb, "extern")
    run = await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, bron, run)
    if tabel == "chunk":
        rid = await _fragment(live_surrealdb, bron, source_version=versie, run=run)
    elif tabel == "doc_node":
        rid = await _doc_node(live_surrealdb, bron, source_version=versie, run=run)
    else:
        [rij] = await _q(live_surrealdb, "CREATE source_embedding CONTENT $d RETURN id", {"d": {
            "source": ensure_record_id(bron), "content": "x", "order": 0, "embedding": []}})
        rid = str(rij["id"])
    with pytest.raises(RuntimeError, match="readonly"):
        await _q(live_surrealdb, "UPDATE type::thing($id) SET source = $b",
                 {"id": rid, "b": ensure_record_id(ander)})
    assert str((await _lees(live_surrealdb, rid))["source"]) == bron


async def test_intern_fragment_blijft_intern_bij_herclassificatie_van_een_andere_bron(
    live_surrealdb,
) -> None:
    from surrealdb_service.repositories.source import SourceRepository

    bron, ander = await _bron(live_surrealdb, "intern"), await _bron(live_surrealdb, "intern")
    run = await _run(live_surrealdb)
    frag = await _fragment(live_surrealdb, bron,
                           source_version=await _versie(live_surrealdb, bron, run), run=run)
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "UPDATE type::thing($id) SET source = $b",
                 {"id": frag, "b": ensure_record_id(ander)})
    await SourceRepository(live_surrealdb).reclassify(
        ander, from_classification="intern", to_classification="extern",
        reason="Openbaar gepubliceerd", agent_key="agent_keys:test", person="Testpersoon",
    )
    assert (await _lees(live_surrealdb, ander))["classification"] == "extern"
    assert (await _lees(live_surrealdb, frag))["classification"] == "intern"


async def test_afgesloten_run_mag_geen_fragment_of_structuur_maken(live_surrealdb) -> None:
    bron = await _bron(live_surrealdb, "extern")
    run = await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, bron, run)
    await ProcessingRunRepository(live_surrealdb).close(run, status="geslaagd")
    with pytest.raises(RuntimeError, match="run"):
        await _fragment(live_surrealdb, bron, source_version=versie, run=run)
    with pytest.raises(RuntimeError, match="run"):
        await _doc_node(live_surrealdb, bron, source_version=versie, run=run)


@pytest.mark.parametrize("classification, toegestaan", [("intern", False), ("extern", True)])
async def test_run_met_flow_buiten_git(live_surrealdb, classification, toegestaan) -> None:
    bron = await _bron(live_surrealdb, classification)
    run = str((await ProcessingRunRepository(live_surrealdb).create(
        trigger="schema", pipeline_id="test")).id)
    assert (await _lees(live_surrealdb, run))["flow_vastgelegd"] is False
    if not toegestaan:
        # Bij een interne bron mag zo'n run geen versie maken ...
        with pytest.raises(RuntimeError, match="created_by_run"):
            await _versie(live_surrealdb, bron, run)
        # ... en geen fragment of structuur in een bestaande versie.
        versie = await _versie(live_surrealdb, bron, await _run(live_surrealdb))
    else:
        versie = await _versie(live_surrealdb, bron, run)
    for maak in (_fragment, _doc_node):
        if toegestaan:
            await maak(live_surrealdb, bron, source_version=versie, run=run)
        else:
            with pytest.raises(RuntimeError, match="run"):
                await maak(live_surrealdb, bron, source_version=versie, run=run)


async def test_versie_alleen_verleggen_naar_een_toegestane_run(live_surrealdb) -> None:
    repo = SourceVersionRepository(live_surrealdb)
    runs = ProcessingRunRepository(live_surrealdb)
    bron = await _bron(live_surrealdb, "intern")
    versie = await _versie(live_surrealdb, bron, await _run(live_surrealdb))

    gesloten = await _run(live_surrealdb)
    await runs.close(gesloten, status="geslaagd")
    buiten_git = str((await runs.create(trigger="schema", pipeline_id="test")).id)
    for run in (gesloten, buiten_git):
        with pytest.raises(RuntimeError, match="current_run"):
            await repo.set_current_run(versie, run)
    goed = await _run(live_surrealdb)
    await repo.set_current_run(versie, goed)
    rij = await _lees(live_surrealdb, versie)
    assert str(rij["current_run"]) == goed
    # Een afgesloten huidige run blijft geldig bij een andere wijziging.
    await runs.close(goed, status="geslaagd")
    await _q(live_surrealdb, "UPDATE type::thing($id) SET classification = 'intern'", {"id": versie})


async def test_alleen_fragmenten_van_de_huidige_versie_en_run(live_surrealdb) -> None:
    from surrealdb_service.repositories.source import SourceRepository

    bron = await _bron(live_surrealdb, "extern")
    eerste = await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, bron, eerste)
    await _fragment(live_surrealdb, bron, source_version=versie, run=eerste)
    tweede = await _run(live_surrealdb)
    nieuw = await _fragment(live_surrealdb, bron, source_version=versie, run=tweede)
    await SourceVersionRepository(live_surrealdb).set_current_run(versie, tweede)
    repo = SourceRepository(live_surrealdb)
    assert len(await repo.get_chunks(bron)) == 2
    assert [c.id for c in await repo.get_chunks(bron, huidig=True)] == [nieuw]
    assert await repo.get_chunks(await _bron(live_surrealdb), huidig=True) == []


async def test_herverwerking_laat_de_fragmenten_van_de_oude_run_staan(live_surrealdb) -> None:
    from surrealdb_service.repositories.source import ChunkRepository

    bron = await _bron(live_surrealdb, "extern")
    eerste = await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, bron, eerste)

    def _set(run: str, n: int) -> list[dict]:
        return [{"source": bron, "source_version": versie, "run": run, "text": f"t{i}",
                 "order": i, "physical_page": 0, "element_type": "paragraph",
                 "positions": [], "metadata": {}} for i in range(n)]

    await ChunkRepository(live_surrealdb).bulk_create(_set(eerste, 3))
    await ProcessingRunRepository(live_surrealdb).close(eerste, status="geslaagd")
    tweede = await _run(live_surrealdb)
    await ChunkRepository(live_surrealdb).bulk_create(_set(tweede, 5))
    await SourceVersionRepository(live_surrealdb).set_current_run(versie, tweede)

    async def _aantal(run: str) -> int:
        return len(await _q(live_surrealdb, "SELECT VALUE id FROM chunk WHERE run = $r",
                            {"r": ensure_record_id(run)}))

    assert (await _aantal(eerste), await _aantal(tweede)) == (3, 5)
    assert str((await _lees(live_surrealdb, versie))["current_run"]) == tweede


# --------------------------------------------------------------------------
# Bronversie
# --------------------------------------------------------------------------


async def test_unieke_index_weigert_een_dubbele_hash(live_surrealdb) -> None:
    bron, ander = await _bron(live_surrealdb), await _bron(live_surrealdb)
    run = await _run(live_surrealdb)
    h = _hash()
    data = {"source": ensure_record_id(bron), "version_no": 1, "content_hash": h,
            "created_by_run": ensure_record_id(run)}
    await _q(live_surrealdb, "CREATE source_version CONTENT $d", {"d": data})
    with pytest.raises(RuntimeError, match="idx_source_version_hash"):
        await _q(live_surrealdb, "CREATE source_version CONTENT $d", {"d": {**data, "version_no": 2}})
    # Dezelfde hash bij een andere bron mag wel.
    await _q(live_surrealdb, "CREATE source_version CONTENT $d",
             {"d": {**data, "source": ensure_record_id(ander)}})


async def test_versie_schrijven_en_teruglezen(live_surrealdb) -> None:
    bron = await _bron(live_surrealdb, "extern")
    run = await _run(live_surrealdb)
    repo = SourceVersionRepository(live_surrealdb)
    h1, h2 = _hash(), _hash()
    eerste, nieuw = await repo.register(bron, h1, run, original="bronnen/rapport.pdf")
    assert nieuw
    rij = await _lees(live_surrealdb, str(eerste.id))
    assert rij["version_no"] == 1 and rij["content_hash"] == h1
    assert rij["original"] == "bronnen/rapport.pdf"
    assert str(rij["created_by_run"]) == run and str(rij["current_run"]) == run
    assert rij["classification"] == "extern"
    assert "predecessor" not in rij or rij["predecessor"] is None

    tweede, nieuw = await repo.register(bron, h2, run)
    assert nieuw and tweede.version_no == 2 and tweede.predecessor == eerste.id
    assert (await _lees(live_surrealdb, bron))["current_version"] == tweede.id

    # Een bekende hash geeft de bestaande versie en verandert niets.
    weer, nieuw = await repo.register(bron, h1, run)
    assert not nieuw and weer.id == eerste.id
    assert (await _lees(live_surrealdb, bron))["current_version"] == tweede.id


async def test_ongeldige_hash_geweigerd(live_surrealdb) -> None:
    bron = await _bron(live_surrealdb)
    run = await _run(live_surrealdb)
    with pytest.raises(RuntimeError, match="content_hash"):
        await SourceVersionRepository(live_surrealdb).register(bron, "geen-hash", run)


async def test_versie_niet_losser_dan_de_bron(live_surrealdb) -> None:
    from surrealdb_service.connection import ClassificationConflictError

    bron = await _bron(live_surrealdb, "intern")
    run = await _run(live_surrealdb)
    with pytest.raises(ClassificationConflictError):
        await SourceVersionRepository(live_surrealdb).register(
            bron, _hash(), run, classification="extern"
        )


async def test_interne_versie_maakt_de_bron_intern(live_surrealdb) -> None:
    bron = await _bron(live_surrealdb, "extern")
    run = await _run(live_surrealdb)
    oude = await _versie(live_surrealdb, bron, run)
    await _versie(live_surrealdb, bron, run, classification="intern")
    assert (await _lees(live_surrealdb, bron))["classification"] == "intern"
    assert (await _lees(live_surrealdb, oude))["classification"] == "intern"


# --------------------------------------------------------------------------
# Verwerkingsrun
# --------------------------------------------------------------------------


async def test_run_schrijven_en_teruglezen(live_surrealdb) -> None:
    repo = ProcessingRunRepository(live_surrealdb)
    run = await repo.create(
        trigger="schema", pipeline_id="inname_pdf",
        pipeline_version={"git_commit": "a" * 40, "flow_hash": "b" * 64},
        input_classification="extern", source_map_rule_id="source_map_rule:x",
        source_map_rule_rev=3, label_schema={"schema": "iv3", "version": "2025"},
        log=[{"step": "start", "status": "ok", "message": ""}],
    )
    rij = await _lees(live_surrealdb, str(run.id))
    assert rij["status"] == "bezig" and rij["trigger"] == "schema"
    assert rij["pipeline_version"] == {"git_commit": "a" * 40, "flow_hash": "b" * 64}
    assert rij["input_classification"] == "extern"
    assert (rij["source_map_rule_id"], rij["source_map_rule_rev"]) == ("source_map_rule:x", 3)
    assert rij["label_schema"] == {"schema": "iv3", "version": "2025"}
    assert rij["log"][0]["step"] == "start" and rij["attempts"] == 0
    assert rij["models"] == [] and rij["started_at"]


async def test_run_standaard_invoer_onbekend(live_surrealdb) -> None:
    run = await ProcessingRunRepository(live_surrealdb).create(trigger="feed", pipeline_id="x")
    assert run.input_classification == "onbekend"


@pytest.mark.parametrize("veld,waarde", [("status", "klaar"), ("trigger", "cron")])
async def test_run_ongeldige_waarde_geweigerd(live_surrealdb, veld, waarde) -> None:
    data = {"trigger": "feed", "pipeline_id": "x", veld: waarde}
    with pytest.raises(RuntimeError, match=veld):
        await _q(live_surrealdb, "CREATE processing_run CONTENT $d", {"d": data})


async def test_afgesloten_run_blijft_afgesloten(live_surrealdb) -> None:
    repo = ProcessingRunRepository(live_surrealdb)
    run = await repo.create(trigger="feed", pipeline_id="x")
    gesloten = await repo.close(str(run.id), status="geslaagd")
    assert gesloten.status == "geslaagd" and gesloten.ended_at
    assert await repo.close(str(run.id), status="mislukt") is None
    with pytest.raises(RuntimeError, match="status"):
        await _q(live_surrealdb, "UPDATE type::thing($id) SET status = 'bezig'", {"id": str(run.id)})
    assert (await repo.get(str(run.id))).status == "geslaagd"


async def test_run_zonder_inhoud_heeft_niets_nieuws(live_surrealdb) -> None:
    repo = ProcessingRunRepository(live_surrealdb)
    leeg = await repo.create(trigger="feed", pipeline_id="x")
    assert not await repo.has_new_content(str(leeg.id))
    bron = await _bron(live_surrealdb)
    await _versie(live_surrealdb, bron, str(leeg.id))
    assert await repo.has_new_content(str(leeg.id))


# --------------------------------------------------------------------------
# Embedding, job en bron
# --------------------------------------------------------------------------


async def test_embedding_job_en_bron_velden(live_surrealdb) -> None:
    bron = await _bron(live_surrealdb)
    run = await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, bron, run)
    [emb] = await _q(live_surrealdb, "CREATE source_embedding CONTENT $d RETURN id", {"d": {
        "source": ensure_record_id(bron), "content": "x", "order": 0, "embedding": [],
        "run": ensure_record_id(run), "embedding_model": "mxbai-embed-large Q8_0"}})
    rij = await _lees(live_surrealdb, str(emb["id"]))
    assert str(rij["run"]) == run and rij["embedding_model"] == "mxbai-embed-large Q8_0"

    [job] = await _q(live_surrealdb, "CREATE job CONTENT $d RETURN id",
                     {"d": {"job_type": "embed", "run": ensure_record_id(run)}})
    assert str((await _lees(live_surrealdb, str(job["id"])))["run"]) == run

    assert str((await _lees(live_surrealdb, bron))["current_version"]) == versie
    ander = await _bron(live_surrealdb)
    with pytest.raises(RuntimeError, match="current_version"):
        await _q(live_surrealdb, "UPDATE type::thing($id) SET current_version = $v",
                 {"id": ander, "v": ensure_record_id(versie)})


# --------------------------------------------------------------------------
# Vermeldingen en `mentions`
# --------------------------------------------------------------------------


async def _entiteit(config, bronnen: list[str], grounding: dict) -> str:
    naam = f"testentiteit {uuid.uuid4().hex[:8]}"
    [rij] = await _q(config, "CREATE entity CONTENT $d RETURN id", {"d": {
        "canonical_name": naam, "name_key": naam, "hash_id": naam,
        "entity_type": "topic", "embedding": [],
        "source_documents": bronnen, "properties": {"grounding": grounding}}})
    return str(rij["id"])


async def _bron_met_fragment(config, classification: str) -> tuple[str, str]:
    bron = await _bron(config, classification)
    run = await _run(config)
    versie = await _versie(config, bron, run)
    return bron, await _fragment(config, bron, source_version=versie, run=run)


async def test_vermelding_slaat_run_en_classificatie_op(live_surrealdb) -> None:
    extern, frag_e = await _bron_met_fragment(live_surrealdb, "extern")
    intern, frag_i = await _bron_met_fragment(live_surrealdb, "intern")
    run = await _run(live_surrealdb)
    eid = await _entiteit(live_surrealdb, [extern, intern], {
        extern: {"chunk_id": frag_e, "run": run, "classification": "intern"},
        intern: {"chunk_id": frag_i, "run": run, "classification": "extern"},
    })
    grounding = (await _lees(live_surrealdb, eid))["properties"]["grounding"]
    # De opgeslagen waarden zijn berekend; een meegegeven waarde telt niet. De
    # run komt van het fragment.
    assert grounding[extern]["classification"] == "extern"
    assert grounding[intern]["classification"] == "intern"
    run_e = (await _lees(live_surrealdb, frag_e))["run"]
    assert grounding[extern]["run"] == str(run_e) != run


async def test_vermelding_na_herverwerking_krijgt_de_run_van_haar_fragment(live_surrealdb) -> None:
    bron = await _bron(live_surrealdb, "extern")
    eerste = await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, bron, eerste)
    oud = await _fragment(live_surrealdb, bron, source_version=versie, run=eerste)
    tweede = await _run(live_surrealdb)
    nieuw = await _fragment(live_surrealdb, bron, source_version=versie, run=tweede)
    eid = await _entiteit(live_surrealdb, [bron], {bron: {"chunk_id": nieuw, "run": eerste}})
    assert (await _lees(live_surrealdb, eid))["properties"]["grounding"][bron]["run"] == tweede
    # Een vermelding zonder fragment houdt de meegegeven run.
    eid = await _entiteit(live_surrealdb, [bron], {bron: {"context": "x", "run": eerste}})
    assert (await _lees(live_surrealdb, eid))["properties"]["grounding"][bron]["run"] == eerste
    assert oud != nieuw


async def test_vermelding_met_fragment_van_een_andere_bron_geweigerd(live_surrealdb) -> None:
    bron, _ = await _bron_met_fragment(live_surrealdb, "extern")
    _, vreemd = await _bron_met_fragment(live_surrealdb, "extern")
    with pytest.raises(RuntimeError, match="grounding"):
        await _entiteit(live_surrealdb, [bron], {bron: {"chunk_id": vreemd}})
    with pytest.raises(RuntimeError, match="grounding"):
        await _entiteit(live_surrealdb, [bron], {bron: {"chunk_id": "chunk:bestaatniet"}})


async def test_vermelding_die_geen_object_is_geweigerd(live_surrealdb) -> None:
    bron, _ = await _bron_met_fragment(live_surrealdb, "extern")
    for waarde in ("tekst", 5, ["chunk:x"]):
        with pytest.raises(RuntimeError, match="grounding"):
            await _entiteit(live_surrealdb, [bron], {bron: waarde})


async def test_vermelding_van_een_bron_buiten_source_documents_geweigerd(live_surrealdb) -> None:
    bron, _ = await _bron_met_fragment(live_surrealdb, "extern")
    ander, frag = await _bron_met_fragment(live_surrealdb, "intern")
    with pytest.raises(RuntimeError, match="grounding"):
        await _entiteit(live_surrealdb, [bron], {ander: {"chunk_id": frag}})


async def test_vermelding_wordt_intern_als_haar_bron_intern_wordt(live_surrealdb) -> None:
    bron, frag = await _bron_met_fragment(live_surrealdb, "extern")
    eid = await _entiteit(live_surrealdb, [bron], {bron: {"chunk_id": frag}})
    assert (await _lees(live_surrealdb, eid))["properties"]["grounding"][bron]["classification"] == "extern"
    await _q(live_surrealdb, "UPDATE type::thing($id) SET classification = 'intern'", {"id": bron})
    rij = await _lees(live_surrealdb, eid)
    assert rij["properties"]["grounding"][bron]["classification"] == "intern"
    assert rij["classification"] == "intern"


async def test_mentions_volgt_haar_bron(live_surrealdb) -> None:
    extern = await _bron(live_surrealdb, "extern")
    intern = await _bron(live_surrealdb, "intern")
    eid = await _entiteit(live_surrealdb, [extern], {})
    for b in (extern, intern):
        await _q(live_surrealdb, "RELATE $b->mentions->$e SET weight = 1.0",
                 {"b": ensure_record_id(b), "e": ensure_record_id(eid)})
    rows = await _q(live_surrealdb, "SELECT <string> in AS bron, classification FROM mentions WHERE out = $e",
                    {"e": ensure_record_id(eid)})
    assert {r["bron"]: r["classification"] for r in rows} == {extern: "extern", intern: "intern"}
    # Een meegegeven waarde telt niet.
    await _q(live_surrealdb, "UPDATE mentions SET classification = 'extern' WHERE out = $e",
             {"e": ensure_record_id(eid)})
    await _q(live_surrealdb, "UPDATE type::thing($id) SET classification = 'intern'", {"id": extern})
    rows = await _q(live_surrealdb, "SELECT VALUE classification FROM mentions WHERE out = $e",
                    {"e": ensure_record_id(eid)})
    assert rows == ["intern", "intern"]


# --------------------------------------------------------------------------
# Heen en terug
# --------------------------------------------------------------------------


async def test_migratie_twee_keer_na_elkaar(live_surrealdb) -> None:
    await _opnieuw_89_en_90(live_surrealdb)
    await _opnieuw_89_en_90(live_surrealdb)
    bron = await _bron(live_surrealdb)
    run = await _run(live_surrealdb)
    await _versie(live_surrealdb, bron, run)


async def test_down_draait_terug(live_surrealdb) -> None:
    bron = await _bron(live_surrealdb)
    run = await _run(live_surrealdb)
    versie = await _versie(live_surrealdb, bron, run)
    # Herverwerking: twee runs met dezelfde structuur. Terugdraaien lukt toch.
    for r in (run, await _run(live_surrealdb)):
        await _q(live_surrealdb, "CREATE doc_node CONTENT $d", {"d": {
            "source": ensure_record_id(bron), "element_type": "paragraph",
            "self_ref": "#/texts/0", "sequence": 0,
            "source_version": ensure_record_id(versie), "run": ensure_record_id(r)}})
    try:
        # Zoals de migratieloader: eerst de latere migratie met een event op
        # `source_version` (97 en 96, F2-14), dan 90; zonder de herclassificatievlag.
        await execute_transaction(_migratie(97, down=True), config=live_surrealdb)
        await execute_transaction(_migratie(96, down=True), config=live_surrealdb)
        await execute_transaction(_migratie(90, down=True), config=live_surrealdb)
        await execute_transaction(_migratie(90, down=True), config=live_surrealdb)
        db = await _q(live_surrealdb, "INFO FOR DB")
        db = db[0] if isinstance(db, list) else db
        # De herkomst blijft bewaard.
        assert "processing_run" in db["tables"] and "source_version" in db["tables"]
        assert await _q(live_surrealdb, "SELECT * FROM type::thing($id)", {"id": versie})
        assert await _q(live_surrealdb, "SELECT * FROM type::thing($id)", {"id": run})
        sv = await _q(live_surrealdb, "INFO FOR TABLE source_version")
        assert (sv[0] if isinstance(sv, list) else sv)["events"] == {}
        assert not [f for f in db["functions"] if f.startswith("vermelding_waarde")
                    or f.startswith("vermeldingen_") or f.startswith("fragment_hoort")]
        # De herkomstvelden blijven, optioneel en zonder ASSERT.
        for tabel, velden in {
            "chunk": ["source_version", "run", "extractor"],
            "doc_node": ["source_version", "run"],
            "source_embedding": ["run", "embedding_model"],
            "job": ["run"],
            "source": ["current_version"],
        }.items():
            info = await _q(live_surrealdb, f"INFO FOR TABLE {tabel}")
            info = info[0] if isinstance(info, list) else info
            for veld in velden:
                definitie = info["fields"][veld]
                assert "option<" in definitie and "ASSERT" not in definitie, (tabel, veld)
        mentions = await _q(live_surrealdb, "INFO FOR TABLE mentions")
        assert "classification" not in (mentions[0] if isinstance(mentions, list) else mentions)["fields"]
        dn = await _q(live_surrealdb, "INFO FOR TABLE doc_node")
        dn = dn[0] if isinstance(dn, list) else dn
        assert "dn_ref" in dn["indexes"] and "dn_run_ref" not in dn["indexes"]
        assert "UNIQUE" not in dn["indexes"]["dn_ref"]
        for tabel in ("chunk", "doc_node", "source_embedding"):
            info = await _q(live_surrealdb, f"INFO FOR TABLE {tabel}")
            assert "READONLY" not in (info[0] if isinstance(info, list) else info)["fields"]["source"]
        # Zonder 90 kan een fragment weer zonder herkomst; een bestaand
        # fragment houdt bij een schrijfactie zijn herkomst.
        b = await _bron(live_surrealdb)
        await _fragment(live_surrealdb, b)
        frag = await _fragment(live_surrealdb, bron, source_version=versie, run=run)
        await _q(live_surrealdb, "UPDATE type::thing($id) SET text = 'y'", {"id": frag})
        assert str((await _lees(live_surrealdb, frag))["source_version"]) == versie
    finally:
        await _opnieuw_89_en_90(live_surrealdb)
        await execute_transaction(_migratie(96), config=live_surrealdb)
        await execute_transaction(_migratie(97), config=live_surrealdb)
