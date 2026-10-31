"""Bronkaart, innamebatch en innamewachtrij (F2-13, ADR 0019, 0020, 0004).

Tegen een echte SurrealDB, via de API met een sleutel ``beheerder`` (persoon
"Testpersoon", conftest) en, voor de wachtrij, ``pijplijn``:

* een regel voor route API, MCP, feed of website stelt ``extern`` voor en
  vraagt bevestiging; zonder bevestiging geldt ``intern``;
* bevestigen of wijzigen van een classificatie geeft één regel in
  ``classification_log`` met de persoon;
* een batch zonder keuze is ``intern``;
* een watchfolder-item heeft geen standaard en staat op ``wacht``;
* een wijziging met een verouderde ``rev`` geeft 409;
* een run bevat id en ``rev`` van zijn regel;
* een regel met een locatie in de vault, in de exportmap of via een
  symbolische koppeling daarnaartoe geeft 422.

Alle testdata is synthetisch; paden liggen in ``tmp_path``.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.dependencies import (
    get_intake_service,
    get_processing_run_service,
    get_source_map_service,
)
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.intake_service import IntakeService
from app_main.services.processing_run_service import ProcessingRunService
from app_main.services.source_map_service import (
    Locatiecontrole,
    SourceMapService,
    ligt_in_verboden_map,
)
from shared.models.agents import Right

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

BEHEERDER = agent_auth.AgentKeyContext(
    key_id="agent_keys:beheer", agent_id="werkbak", permission=Right.BEHEERDER,
    person="Testpersoon")
PIJPLIJN = agent_auth.AgentKeyContext(
    key_id="agent_keys:langflow", agent_id="langflow", permission=Right.PIJPLIJN)


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
def mappen(tmp_path):
    vault = tmp_path / "vault"
    export = tmp_path / "export"
    buiten = tmp_path / "bronnen"
    for m in (vault / "20 Atlas", export, buiten):
        m.mkdir(parents=True)
    return vault, export, buiten


@pytest.fixture
async def env(live_surrealdb, monkeypatch, app, mappen, git_flows):
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    sleutel = {"ctx": BEHEERDER}

    async def _sleutel(_request):
        return sleutel["ctx"]

    monkeypatch.setattr(agent_auth, "authenticate_request", _sleutel)
    vault, export, _ = mappen
    controle = Locatiecontrole([str(vault)], [str(export)])
    app.dependency_overrides[get_source_map_service] = lambda: SourceMapService(
        locatiecontrole=controle)
    app.dependency_overrides[get_intake_service] = lambda: IntakeService(locatiecontrole=controle)
    app.dependency_overrides[get_processing_run_service] = lambda: ProcessingRunService(
        flow_register=git_flows.register)
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    yield sleutel
    app.dependency_overrides.clear()
    await conn._pool.close_all()
    conn._pool = original or conn.ConnectionPool(config=live_surrealdb)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                             headers={"Authorization": "Bearer ak_test"})


def _regel(route_type: str = "api", **extra) -> dict:
    return {"name": "Openbare raadsstukken", "route_type": route_type, "source_type": "api",
            "location": "https://example.org/api/moties", "route": "inname_api", **extra}


async def _log(onderwerp: str, rid: str) -> list[dict]:
    return await conn.execute_query(
        f"SELECT * FROM classification_log WHERE {onderwerp} = $id ORDER BY created_at",
        {"id": conn.ensure_record_id(rid)})


# --------------------------------------------------------------------------
# Standaard per route en bevestiging
# --------------------------------------------------------------------------


@pytest.mark.parametrize("route_type", ["api", "mcp", "feed", "website"])
async def test_openbare_route_stelt_extern_voor_en_vraagt_bevestiging(env, app, route_type) -> None:
    async with _client(app) as c:
        r = await c.post("/api/v1/source-map-rules", json=_regel(route_type))
        assert r.status_code == 201, r.text
        regel = r.json()
        assert regel["classification"] == "extern"
        assert regel["classification_confirmed_by"] is None
        assert regel["effective_classification"] == "intern"
        assert regel["pipeline_start"] == "automatisch"
        assert await _log("source_map_rule", regel["id"]) == []

        r = await c.post(f"/api/v1/source-map-rules/{regel['id']}/confirm",
                         json={"rev": regel["rev"]})
        assert r.status_code == 200, r.text
        bevestigd = r.json()
        assert bevestigd["effective_classification"] == "extern"
        assert bevestigd["classification_confirmed_by"] == "Testpersoon"
        assert bevestigd["classification_confirmed_at"]
        assert bevestigd["rev"] == regel["rev"] + 1
        [log] = await _log("source_map_rule", regel["id"])
        assert (log["from_classification"], log["to_classification"], log["person"],
                log["agent_key"]) == ("intern", "extern", "Testpersoon", "agent_keys:beheer")


@pytest.mark.parametrize("route_type,start", [
    ("bulkupload", "na_bevestiging"), ("watchfolder", "wachtrij"), ("upload", "na_bevestiging")])
async def test_andere_routes_zijn_intern(env, app, route_type, start) -> None:
    async with _client(app) as c:
        r = await c.post("/api/v1/source-map-rules", json=_regel(
            route_type, source_type="pdf", location="/data/inbox"))
        assert r.status_code == 201, r.text
        assert (r.json()["classification"], r.json()["pipeline_start"]) == ("intern", start)
        # Bevestigen geldt alleen extern.
        r = await c.post(f"/api/v1/source-map-rules/{r.json()['id']}/confirm", json={"rev": 0})
        assert r.status_code == 422


async def test_aanmaken_met_bevestiging_en_wijziging_trekt_bevestiging_in(env, app) -> None:
    async with _client(app) as c:
        r = await c.post("/api/v1/source-map-rules", json=_regel(confirm=True))
        regel = r.json()
        assert r.status_code == 201 and regel["effective_classification"] == "extern"
        assert len(await _log("source_map_rule", regel["id"])) == 1

        # Een andere locatie: de bevestiging gold de oude regel.
        r = await c.patch(f"/api/v1/source-map-rules/{regel['id']}", json={
            "rev": regel["rev"], "location": "https://example.org/api/besluiten"})
        assert r.status_code == 200, r.text
        assert r.json()["classification_confirmed_by"] is None
        assert r.json()["effective_classification"] == "intern"

        # Naar intern zetten is ook een vastlegging: één logregel erbij.
        r = await c.patch(f"/api/v1/source-map-rules/{regel['id']}", json={
            "rev": r.json()["rev"], "classification": "intern", "reason": "toch niet openbaar"})
        assert r.status_code == 200 and r.json()["classification"] == "intern"
        logs = await _log("source_map_rule", regel["id"])
        assert len(logs) == 2 and logs[-1]["reason"] == "toch niet openbaar"
        # Teruglezen: de waarden staan echt in de database.
        r = await c.get(f"/api/v1/source-map-rules/{regel['id']}")
        assert (r.json()["classification"], r.json()["location"]) == (
            "intern", "https://example.org/api/besluiten")


async def test_velden_teruggelezen(env, app) -> None:
    async with _client(app) as c:
        r = await c.post("/api/v1/source-map-rules", json=_regel(
            "feed", source_type="video", location="https://example.org/feeds/videos.xml",
            schedule={"cron": "0 * * * *"}, document_type=["debat"], selection=["agenda"],
            label_schema={"schema": "iv3", "version": "2025"}))
        assert r.status_code == 201, r.text
        r = await c.get(f"/api/v1/source-map-rules/{r.json()['id']}")
        regel = r.json()
        assert regel["schedule"] == {"cron": "0 * * * *", "timezone": "Europe/Amsterdam",
                                     "catch_up": "coalesce"}
        assert regel["label_schema"] == {"schema": "iv3", "version": "2025"}
        assert (regel["document_type"], regel["selection"]) == (["debat"], ["agenda"])
        assert regel["fetch_content"] == "metadata"  # video: standaard alleen metadata
        assert (regel["owner"], regel["active"], regel["modified_by"]) == (
            "beheerder", True, "Testpersoon")


# --------------------------------------------------------------------------
# Vergrendeling en run
# --------------------------------------------------------------------------


async def test_verouderde_rev_geeft_409(env, app) -> None:
    async with _client(app) as c:
        regel = (await c.post("/api/v1/source-map-rules", json=_regel())).json()
        pad = f"/api/v1/source-map-rules/{regel['id']}"
        assert (await c.patch(pad, json={"rev": 0, "name": "Eerste"})).status_code == 200
        r = await c.patch(pad, json={"rev": 0, "name": "Tweede"})
        assert r.status_code == 409
        assert (await c.post(f"{pad}/confirm", json={"rev": 0})).status_code == 409
        assert (await c.get(pad)).json()["name"] == "Eerste"
        assert await _log("source_map_rule", regel["id"]) == []

        batch = (await c.post("/api/v1/intake/batches", json={"route": "inname_pdf"})).json()
        assert (await c.patch(f"/api/v1/intake/batches/{batch['id']}",
                              json={"rev": batch["rev"] + 1, "route": "x"})).status_code == 409


async def test_run_bevat_id_en_rev_van_zijn_regel(env, app, git_flows) -> None:
    async with _client(app) as c:
        regel = (await c.post("/api/v1/source-map-rules", json=_regel())).json()
        regel = (await c.patch(f"/api/v1/source-map-rules/{regel['id']}",
                               json={"rev": regel["rev"], "name": "Gewijzigd"})).json()
        regel = (await c.post(f"/api/v1/source-map-rules/{regel['id']}/confirm",
                              json={"rev": regel["rev"]})).json()
        assert regel["rev"] == 2
        env["ctx"] = PIJPLIJN
        r = await c.post("/api/v1/runs", json={
            "trigger": "schema", "pipeline_id": "inname_pdf",
            "pipeline_version": {"git_commit": git_flows.head, "flow_hash": git_flows.hash},
            "source_map_rule_id": regel["id"]})
        assert r.status_code == 201, r.text
        run = (await c.get(f"/api/v1/runs/{r.json()['run_id']}")).json()
        assert (run["source_map_rule_id"], run["source_map_rule_rev"]) == (regel["id"], 2)
        assert run["input_classification"] == "extern"


async def test_onbevestigde_regel_geeft_interne_invoer(env, app, git_flows) -> None:
    async with _client(app) as c:
        regel = (await c.post("/api/v1/source-map-rules", json=_regel())).json()
        env["ctx"] = PIJPLIJN
        r = await c.post("/api/v1/runs", json={
            "trigger": "schema", "pipeline_id": "inname_pdf",
            "pipeline_version": {"git_commit": git_flows.head, "flow_hash": git_flows.hash},
            "source_map_rule_id": regel["id"]})
        run = (await c.get(f"/api/v1/runs/{r.json()['run_id']}")).json()
        assert (run["input_classification"], run["source_map_rule_rev"]) == ("intern", 0)


# --------------------------------------------------------------------------
# Locatie: niet in de vault of de exportmap (ADR 0004)
# --------------------------------------------------------------------------


async def test_locatie_in_vault_exportmap_of_via_koppeling_geeft_422(env, app, mappen) -> None:
    vault, export, buiten = mappen
    koppeling = buiten / "koppeling"
    os.symlink(vault / "20 Atlas", koppeling)
    verboden = [str(vault), str(vault / "20 Atlas" / "x.md"), str(export),
                str(export / "sub"), str(koppeling), str(koppeling / "notitie.md"),
                f"{buiten}/../vault/20 Atlas", f"file://{vault}/20%20Atlas",
                "obsidian://open?vault=Test"]
    async with _client(app) as c:
        for locatie in verboden:
            r = await c.post("/api/v1/source-map-rules", json=_regel(
                "watchfolder", source_type="map", location=locatie))
            assert r.status_code == 422, locatie
            assert str(vault) not in r.text
        r = await c.post("/api/v1/source-map-rules", json=_regel(
            "watchfolder", source_type="map", location=str(buiten)))
        assert r.status_code == 201
        regel = r.json()
        r = await c.patch(f"/api/v1/source-map-rules/{regel['id']}",
                          json={"rev": regel["rev"], "location": str(koppeling)})
        assert r.status_code == 422
        assert (await c.get(f"/api/v1/source-map-rules/{regel['id']}")).json()["location"] == str(buiten)
        # Ook een batch of wachtrij-item wijst niet in de vault.
        r = await c.post("/api/v1/intake/batches", json={
            "route": "inname_pdf", "files": [str(vault / "a.pdf")]})
        assert r.status_code == 422


async def test_padvergelijking_windows_en_wsl() -> None:
    vault = [r"D:\# Vault"]
    for locatie in (r"D:\# Vault", r"d:\# vault\20 Atlas\x.md", "D:/# Vault/50 Resources",
                    "/mnt/d/# Vault/x", r"D:\Data\..\# Vault\x", "file:///D:/%23%20Vault/x"):
        assert ligt_in_verboden_map(locatie, vault), locatie
    for locatie in (r"D:\# Vault2", r"D:\Data\inbox", "https://example.org/D:/# Vault",
                    "/mnt/e/# Vault"):
        assert not ligt_in_verboden_map(locatie, vault), locatie


# --------------------------------------------------------------------------
# Batch en wachtrij
# --------------------------------------------------------------------------


async def test_batch_zonder_keuze_is_intern(env, app) -> None:
    async with _client(app) as c:
        r = await c.post("/api/v1/intake/batches", json={
            "route": "inname_pdf", "files": ["/data/batch/a.pdf", "/data/batch/b.pdf"]})
        assert r.status_code == 201, r.text
        batch = r.json()
        assert (batch["classification"], batch["effective_classification"]) == ("intern", "intern")
        assert (batch["created_by"], batch["files"]) == (
            "Testpersoon", ["/data/batch/a.pdf", "/data/batch/b.pdf"])

        r = await c.post("/api/v1/intake/batches", json={
            "route": "inname_pdf", "classification": "extern"})
        assert r.json()["effective_classification"] == "intern"
        r = await c.post(f"/api/v1/intake/batches/{r.json()['id']}/confirm",
                         json={"rev": r.json()["rev"]})
        assert r.status_code == 200 and r.json()["effective_classification"] == "extern"
        assert len(await _log("intake_batch", r.json()["id"])) == 1


async def test_watchfolder_item_wacht_zonder_standaard(env, app) -> None:
    env["ctx"] = PIJPLIJN
    async with _client(app) as c:
        item = {"origin": "watchfolder", "path": "/data/inbox/verslag.pdf",
                "content_hash": "sha256:" + "b" * 64}
        r = await c.post("/api/v1/intake/items", json=item)
        assert r.status_code == 201, r.text
        nieuw = r.json()
        assert nieuw["status"] == "wacht"
        assert nieuw["proposed_classification"] is None
        assert nieuw["effective_classification"] == "intern"
        # Nog eens aanmelden geeft hetzelfde item.
        r = await c.post("/api/v1/intake/items", json=item)
        assert r.status_code == 200 and r.json()["id"] == nieuw["id"]

        # Een losse upload krijgt het voorstel van de poortwachter, maar geldt intern.
        r = await c.post("/api/v1/intake/items", json={
            "origin": "upload", "path": "/data/upload/c.pdf", "content_hash": "sha256:" + "c" * 64,
            "gatekeeper": {"decision": "extern", "method": "regel", "text_source": "pymupdf"}})
        assert (r.json()["proposed_classification"], r.json()["effective_classification"]) == (
            "extern", "intern")

        # Langflow haalt alleen besloten items op; dit item staat er nog niet bij.
        r = await c.get("/api/v1/intake/items?status=besloten")
        assert nieuw["id"] not in {i["id"] for i in r.json()}
        # Verwerken vóór het besluit kan niet.
        r = await c.patch(f"/api/v1/intake/items/{nieuw['id']}",
                          json={"rev": nieuw["rev"], "status": "verwerkt"})
        assert r.status_code == 409

        env["ctx"] = BEHEERDER
        r = await c.post(f"/api/v1/intake/items/{nieuw['id']}/decision", json={
            "rev": nieuw["rev"], "route": "inname_pdf", "classification": "extern"})
        assert r.status_code == 200, r.text
        besloten = r.json()
        assert besloten["status"] == "besloten"
        assert besloten["decision"]["decided_by"] == "Testpersoon"
        assert besloten["effective_classification"] == "extern"
        [log] = await _log("intake_item", nieuw["id"])
        assert (log["to_classification"], log["person"]) == ("extern", "Testpersoon")

        env["ctx"] = PIJPLIJN
        r = await c.get("/api/v1/intake/items?status=besloten")
        assert nieuw["id"] in {i["id"] for i in r.json()}
        r = await c.patch(f"/api/v1/intake/items/{nieuw['id']}",
                          json={"rev": besloten["rev"], "status": "verwerkt"})
        assert r.status_code == 200 and r.json()["status"] == "verwerkt"
