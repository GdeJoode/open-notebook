"""Migratie 93: bronkaart, innamebatch en innamewachtrij (F2-13, ADR 0020).

Schemacontrole, ook op de standaardwaarden. Onder SurrealDB v2.7.0 laat een
SCHEMAFULL-tabel een onbekend veld stil weg (P10 K6); daarom schrijft elke
test een waarde én leest die terug. Alle testdata is synthetisch.
"""

from __future__ import annotations

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

TABELLEN = ("source_map_rule", "intake_batch", "intake_item")


def _migratie(versie: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}{suffix}.surrealql").sql


@pytest.fixture(autouse=True, scope="module")
def _migratie_93_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    """De database is per sessie gedeeld; de tests van migratie 89 voeren die
    opnieuw uit en zetten `classification_log.source` terug. 93 is idempotent."""
    import asyncio

    import surrealdb_service.connection as conn

    asyncio.run(execute_transaction(_migratie(93), config=live_surrealdb))
    conn._pool = None


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _info(config: SurrealDBConfig, sql: str) -> dict:
    rij = await _q(config, sql)
    return rij[0] if isinstance(rij, list) else rij


def _regel(**extra) -> dict:
    return {"name": "Openbare raadsstukken", "route_type": "api", "source_type": "api",
            "location": "https://example.org/api/moties", "pipeline_start": "automatisch",
            "route": "inname_api", "fetch_content": "volledig_document", **extra}


async def test_regel_met_standaardwaarden(live_surrealdb) -> None:
    [rij] = await _q(live_surrealdb, "CREATE source_map_rule CONTENT $d RETURN AFTER",
                     {"d": _regel(schedule={"cron": "0 6 * * 1"},
                                  label_schema={"schema": "iv3", "version": "2025"},
                                  document_type=["motie"], selection=[])})
    assert rij["classification"] == "intern"
    assert rij["active"] is True
    assert rij["owner"] == "beheerder"
    assert rij["rev"] == 0
    assert rij["schedule"] == {"cron": "0 6 * * 1", "timezone": "Europe/Amsterdam",
                               "catch_up": "coalesce"}
    assert rij["label_schema"] == {"schema": "iv3", "version": "2025"}
    assert rij["document_type"] == ["motie"] and rij["created_at"]
    assert "classification_confirmed_by" not in rij

    # Elke schrijfactie telt `rev` op; een meegestuurde waarde telt niet.
    [rij] = await _q(live_surrealdb, "UPDATE $id SET rev = 42, active = false RETURN AFTER",
                     {"id": ensure_record_id(str(rij["id"]))})
    assert (rij["rev"], rij["active"]) == (1, False)


@pytest.mark.parametrize("fout", [
    {"route_type": "ftp"}, {"source_type": "fax"}, {"classification": "onbekend"},
    {"location": " "}, {"name": ""}, {"route": ""}, {"fetch_content": "alles"},
    {"pipeline_start": "wachtrij"}, {"schedule": {"cron": "x", "catch_up": "altijd"}},
    {"classification_confirmed_by": " "},
])
async def test_regel_weigert_ongeldige_waarden(live_surrealdb, fout) -> None:
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE source_map_rule CONTENT $d", {"d": _regel(**fout)})


@pytest.mark.parametrize("route_type,start", [
    ("api", "automatisch"), ("mcp", "automatisch"), ("feed", "automatisch"),
    ("website", "automatisch"), ("bulkupload", "na_bevestiging"),
    ("upload", "na_bevestiging"), ("watchfolder", "wachtrij"),
])
async def test_pipeline_start_volgt_uit_de_route(live_surrealdb, route_type, start) -> None:
    [rij] = await _q(live_surrealdb, "CREATE source_map_rule CONTENT $d RETURN pipeline_start",
                     {"d": _regel(route_type=route_type, pipeline_start=start)})
    assert rij["pipeline_start"] == start
    anders = "wachtrij" if start != "wachtrij" else "automatisch"
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE source_map_rule CONTENT $d",
                 {"d": _regel(route_type=route_type, pipeline_start=anders)})


async def test_bevestiging_teruggelezen(live_surrealdb) -> None:
    nu = datetime.now(timezone.utc)
    [rij] = await _q(live_surrealdb, "CREATE source_map_rule CONTENT $d RETURN AFTER", {"d": _regel(
        classification="extern", classification_confirmed_by="Beheerder Test",
        classification_confirmed_at=nu)})
    assert (rij["classification"], rij["classification_confirmed_by"]) == ("extern", "Beheerder Test")
    assert rij["classification_confirmed_at"]
    # Samen leeg mag ook; dan geldt de regel als intern.
    [rij] = await _q(live_surrealdb, "UPDATE $id SET classification_confirmed_by = NONE,"
                     " classification_confirmed_at = NONE RETURN AFTER",
                     {"id": ensure_record_id(str(rij["id"]))})
    assert "classification_confirmed_by" not in rij


@pytest.mark.parametrize("fout", [
    {"classification": "extern", "classification_confirmed_by": "Beheerder Test"},
    {"classification": "extern", "classification_confirmed_at": "nu"},
    {"classification": "intern", "classification_confirmed_by": "Beheerder Test",
     "classification_confirmed_at": "nu"},
    {"classification": "extern", "route_type": "watchfolder", "pipeline_start": "wachtrij"},
    {"classification": "extern", "route_type": "upload", "pipeline_start": "na_bevestiging"},
    {"classification": "extern", "route_type": "bulkupload", "pipeline_start": "na_bevestiging"},
])
async def test_bevestiging_samen_en_alleen_extern_op_openbare_route(live_surrealdb, fout) -> None:
    if fout.get("classification_confirmed_at") == "nu":
        fout = {**fout, "classification_confirmed_at": datetime.now(timezone.utc)}
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE source_map_rule CONTENT $d", {"d": _regel(**fout)})
    if "route_type" not in fout:
        batch = {k: v for k, v in fout.items()}
        with pytest.raises(RuntimeError):
            await _q(live_surrealdb, "CREATE intake_batch CONTENT $d", {"d": {
                "route": "inname_pdf", "created_by": "Beheerder Test", **batch}})


@pytest.mark.parametrize("sql", [
    "UPDATE $id SET route_type = 'watchfolder', pipeline_start = 'wachtrij'",
    "UPDATE $id MERGE {route_type: 'watchfolder', pipeline_start: 'wachtrij'}",
])
async def test_externe_regel_niet_naar_watchfolder(live_surrealdb, sql) -> None:
    [rij] = await _q(live_surrealdb, "CREATE source_map_rule CONTENT $d RETURN id", {"d": _regel(
        classification="extern", classification_confirmed_by="Beheerder Test",
        classification_confirmed_at=datetime.now(timezone.utc))})
    rid = ensure_record_id(str(rij["id"]))
    with pytest.raises(RuntimeError, match="classification"):
        await _q(live_surrealdb, sql, {"id": rid})
    [terug] = await _q(live_surrealdb, "SELECT route_type, classification FROM $id", {"id": rid})
    assert (terug["route_type"], terug["classification"]) == ("api", "extern")

    # Tegenproef: dezelfde UPDATE op een interne regel slaagt.
    [intern] = await _q(live_surrealdb, "CREATE source_map_rule CONTENT $d RETURN id", {"d": _regel()})
    iid = ensure_record_id(str(intern["id"]))
    await _q(live_surrealdb, sql, {"id": iid})
    [terug] = await _q(live_surrealdb, "SELECT route_type, classification FROM $id", {"id": iid})
    assert (terug["route_type"], terug["classification"]) == ("watchfolder", "intern")


async def test_batch_standaard_intern(live_surrealdb) -> None:
    [rij] = await _q(live_surrealdb, "CREATE intake_batch CONTENT $d RETURN AFTER", {"d": {
        "route": "inname_pdf", "created_by": "Beheerder Test", "files": ["/data/batch/a.pdf"]}})
    assert (rij["classification"], rij["rev"], rij["files"]) == ("intern", 0, ["/data/batch/a.pdf"])
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE intake_batch CONTENT {route: 'x'}")
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "UPDATE $id SET created_by = 'iemand anders'",
                 {"id": ensure_record_id(str(rij["id"]))})


def _item(**extra) -> dict:
    return {"origin": "watchfolder", "path": "/data/inbox/verslag.pdf",
            "content_hash": "sha256:" + "a" * 64, "registered_by": "langflow", **extra}


async def test_wachtrij_item_statusovergangen(live_surrealdb) -> None:
    [rij] = await _q(live_surrealdb, "CREATE intake_item CONTENT $d RETURN AFTER",
                     {"d": _item(path="/data/inbox/status.pdf")})
    assert (rij["status"], rij["rev"]) == ("wacht", 0)
    assert "proposed_classification" not in rij and "decision" not in rij
    rid = ensure_record_id(str(rij["id"]))

    # Besloten zonder besluit, of meteen verwerkt: geweigerd.
    for sql in ("UPDATE $id SET status = 'besloten'", "UPDATE $id SET status = 'verwerkt'"):
        with pytest.raises(RuntimeError):
            await _q(live_surrealdb, sql, {"id": rid})
    besluit = {"route": "inname_pdf", "classification": "extern",
               "decided_by": "Beheerder Test", "decided_at": datetime.now(timezone.utc)}
    [rij] = await _q(live_surrealdb, "UPDATE $id SET status = 'besloten', decision = $b RETURN AFTER",
                     {"id": rid, "b": besluit})
    assert rij["status"] == "besloten" and rij["decision"]["classification"] == "extern"
    # Het besluit ligt vast: niet wijzigen en niet wissen.
    for sql, params in (("UPDATE $id SET decision.classification = 'intern'", {}),
                        ("UPDATE $id SET decision = $b", {"b": {**besluit, "route": "anders"}}),
                        ("UPDATE $id SET decision = NONE", {})):
        with pytest.raises(RuntimeError):
            await _q(live_surrealdb, sql, {"id": rid, **params})
    [rij] = await _q(live_surrealdb, "UPDATE $id SET status = 'verwerkt' RETURN AFTER", {"id": rid})
    assert rij["decision"]["classification"] == "extern"
    assert rij["status"] == "verwerkt"
    # Terug kan niet.
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "UPDATE $id SET status = 'wacht'", {"id": rid})


async def test_wachtrij_item_met_voorstel_en_uniek(live_surrealdb) -> None:
    [rij] = await _q(live_surrealdb, "CREATE intake_item CONTENT $d RETURN AFTER", {"d": _item(
        origin="upload", path="/data/upload/b.pdf", proposed_classification="extern",
        gatekeeper={"decision": "extern", "method": "regel", "text_source": "pymupdf"})})
    assert rij["proposed_classification"] == "extern"
    assert rij["gatekeeper"] == {"decision": "extern", "method": "regel", "text_source": "pymupdf"}
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE intake_item CONTENT $d", {"d": _item(
            origin="upload", path="/data/upload/c.pdf", gatekeeper={"method": "regel"})})
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE intake_item CONTENT $d", {"d": _item(
            origin="upload", path="/data/upload/b.pdf")})
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE intake_item CONTENT $d", {"d": _item(origin="mail")})


async def test_classificatielog_precies_een_onderwerp(live_surrealdb) -> None:
    [regel] = await _q(live_surrealdb, "CREATE source_map_rule CONTENT $d RETURN id", {"d": _regel()})
    [bron] = await _q(live_surrealdb, "CREATE source SET title = 'Openbare testbron' RETURN id")
    log = {"from_classification": "intern", "to_classification": "extern",
           "reason": "test", "agent_key": "agent_keys:test", "person": "Beheerder Test"}
    [rij] = await _q(live_surrealdb, "CREATE classification_log CONTENT $d RETURN AFTER",
                     {"d": {**log, "source_map_rule": ensure_record_id(str(regel["id"]))}})
    assert str(rij["source_map_rule"]) == str(regel["id"]) and rij["person"] == "Beheerder Test"
    # De bestaande vorm met een bron blijft werken.
    await _q(live_surrealdb, "CREATE classification_log CONTENT $d",
             {"d": {**log, "source": ensure_record_id(str(bron["id"]))}})
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE classification_log CONTENT $d", {"d": log})
    with pytest.raises(RuntimeError):
        await _q(live_surrealdb, "CREATE classification_log CONTENT $d", {"d": {
            **log, "source": ensure_record_id(str(bron["id"])),
            "source_map_rule": ensure_record_id(str(regel["id"]))}})


async def test_twee_keer_uitvoeren_en_down(live_surrealdb) -> None:
    await execute_transaction(_migratie(93), config=live_surrealdb)
    await execute_transaction(_migratie(93), config=live_surrealdb)
    [regel] = await _q(live_surrealdb, "CREATE source_map_rule CONTENT $d RETURN id", {"d": _regel()})
    [log] = await _q(live_surrealdb, "CREATE classification_log CONTENT $d RETURN id", {"d": {
        "from_classification": "intern", "to_classification": "intern", "reason": "test",
        "agent_key": "agent_keys:test", "person": "Beheerder Test",
        "source_map_rule": ensure_record_id(str(regel["id"]))}})
    try:
        await execute_transaction(_migratie(93, down=True), config=live_surrealdb)
        await execute_transaction(_migratie(93, down=True), config=live_surrealdb)
        tabellen = (await _info(live_surrealdb, "INFO FOR DB"))["tables"]
        assert not set(TABELLEN) & set(tabellen)
        velden = (await _info(live_surrealdb, "INFO FOR TABLE classification_log"))["fields"]
        assert "option" not in velden["source"]
        # Het logboek wordt niet gewist.
        [rij] = await _q(live_surrealdb, "SELECT source_map_rule FROM $id",
                         {"id": ensure_record_id(str(log["id"]))})
        assert str(rij["source_map_rule"]) == str(regel["id"])
    finally:
        await execute_transaction(_migratie(93), config=live_surrealdb)
    assert set(TABELLEN) <= set((await _info(live_surrealdb, "INFO FOR DB"))["tables"])
