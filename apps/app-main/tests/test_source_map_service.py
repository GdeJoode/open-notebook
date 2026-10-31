"""Bronkaart, innamebatch en innamewachtrij (F2-13, ADR 0019, 0020, 0004).

Tegen een echte SurrealDB, via de API met een sleutel ``beheerder`` (persoon
"Testpersoon") en, voor de wachtrij, ``pijplijn``:

* een regel voor route API, MCP, feed of website stelt ``extern`` voor en
  vraagt bevestiging; zonder bevestiging geldt ``intern``; andere routes
  kunnen niet extern zijn (422);
* een logregel met de persoon bij elke verandering van de effectieve
  classificatie: bevestigen, en elke intrekking van extern naar intern;
* elke inhoudelijke wijziging trekt de bevestiging in, ook ``active`` van
  false naar true; een inactieve regel wordt niet bevestigd;
* een batch zonder keuze is ``intern``;
* een watchfolder-item heeft geen standaard en staat op ``wacht``; een besluit
  ligt vast;
* een wijziging met een verouderde ``rev`` geeft 409;
* een run bevat id en ``rev`` van zijn regel;
* een locatie buiten de innamemappen, in de vault, in de exportmap, via een
  symbolische koppeling of via een Windows-omweg geeft 422.

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
    VerbodenLocatieError,
)
from shared.models.agents import Right

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

BEHEERDER = agent_auth.AgentKeyContext(
    key_id="agent_keys:beheer", agent_id="werkbak", permission=Right.BEHEERDER,
    person="Testpersoon")
PIJPLIJN = agent_auth.AgentKeyContext(
    key_id="agent_keys:langflow", agent_id="langflow", permission=Right.PIJPLIJN)
HASH = "sha256:" + "b" * 64


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
def mappen(tmp_path):
    vault = tmp_path / "vault"
    export = tmp_path / "export"
    buiten = tmp_path / "bronnen"
    for m in (vault / "20 Atlas", export, buiten / "inbox"):
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
    vault, export, buiten = mappen
    controle = Locatiecontrole([str(vault)], [str(export)], [str(buiten)], [])
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


async def _bevestigde_regel(c, **extra) -> dict:
    r = await c.post("/api/v1/source-map-rules", json=_regel(confirm=True, **extra))
    assert r.status_code == 201, r.text
    assert r.json()["effective_classification"] == "extern"
    return r.json()


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
async def test_andere_routes_zijn_intern_en_nooit_extern(env, app, mappen, route_type, start) -> None:
    pad = str(mappen[2] / "inbox")
    async with _client(app) as c:
        r = await c.post("/api/v1/source-map-rules", json=_regel(
            route_type, source_type="pdf", location=pad))
        assert r.status_code == 201, r.text
        regel = r.json()
        assert (regel["classification"], regel["pipeline_start"]) == ("intern", start)
        # Bevestigen geldt alleen extern; extern kan op deze route niet.
        r = await c.post(f"/api/v1/source-map-rules/{regel['id']}/confirm", json={"rev": 0})
        assert r.status_code == 422
        r = await c.patch(f"/api/v1/source-map-rules/{regel['id']}",
                          json={"rev": 0, "classification": "extern"})
        assert r.status_code == 422
        r = await c.post("/api/v1/source-map-rules", json=_regel(
            route_type, source_type="pdf", location=pad, classification="extern"))
        assert r.status_code == 422
    # Een externe regel naar zo'n route verplaatsen kan ook niet.
    async with _client(app) as c:
        regel = await _bevestigde_regel(c)
        r = await c.patch(f"/api/v1/source-map-rules/{regel['id']}", json={
            "rev": regel["rev"], "route_type": route_type, "location": pad})
        assert r.status_code == 422
        assert (await c.get(f"/api/v1/source-map-rules/{regel['id']}")).json()[
            "effective_classification"] == "extern"


# --------------------------------------------------------------------------
# Logregel bij een verschil in effectieve classificatie (blokkerend punt 1)
# --------------------------------------------------------------------------


async def test_inactieve_regel_wordt_niet_bevestigd(env, app) -> None:
    async with _client(app) as c:
        r = await c.post("/api/v1/source-map-rules", json=_regel(active=False, confirm=True))
        assert r.status_code == 422
        r = await c.post("/api/v1/source-map-rules", json=_regel(active=False))
        regel = r.json()
        r = await c.post(f"/api/v1/source-map-rules/{regel['id']}/confirm",
                         json={"rev": regel["rev"]})
        assert r.status_code == 422
        assert await _log("source_map_rule", regel["id"]) == []


async def test_activeren_trekt_bevestiging_in(env, app) -> None:
    async with _client(app) as c:
        regel = await _bevestigde_regel(c)
        pad = f"/api/v1/source-map-rules/{regel['id']}"
        # Uitzetten: extern -> intern, gelogd, bevestiging weg.
        r = await c.patch(pad, json={"rev": regel["rev"], "active": False})
        assert r.status_code == 200, r.text
        assert r.json()["effective_classification"] == "intern"
        assert r.json()["classification_confirmed_by"] is None
        # Weer aanzetten maakt de regel niet extern zonder nieuwe bevestiging.
        r = await c.patch(pad, json={"rev": r.json()["rev"], "active": True})
        assert r.json()["effective_classification"] == "intern"
        assert r.json()["classification_confirmed_by"] is None
        logs = await _log("source_map_rule", regel["id"])
        assert [(lg["from_classification"], lg["to_classification"]) for lg in logs] == [
            ("intern", "extern"), ("extern", "intern")]
        assert all(lg["person"] == "Testpersoon" for lg in logs)
        # Opnieuw bevestigen: weer één logregel.
        r = await c.post(f"{pad}/confirm", json={"rev": r.json()["rev"]})
        assert r.json()["effective_classification"] == "extern"
        assert len(await _log("source_map_rule", regel["id"])) == 3


@pytest.mark.parametrize("wijziging", [
    {"selection": ["paragraaf weerstandsvermogen"]},
    {"route": "inname_api_v2"},
    {"schedule": {"cron": "0 6 * * 1"}},
    {"location": "https://example.org/api/besluiten"},
    {"classification": "intern"},
    {"fetch_content": "metadata"},
])
async def test_inhoudelijke_wijziging_trekt_bevestiging_in(env, app, wijziging) -> None:
    async with _client(app) as c:
        regel = await _bevestigde_regel(c)
        r = await c.patch(f"/api/v1/source-map-rules/{regel['id']}",
                          json={"rev": regel["rev"], **wijziging})
        assert r.status_code == 200, r.text
        assert r.json()["classification_confirmed_by"] is None
        assert r.json()["effective_classification"] == "intern"
        [_, intrekking] = await _log("source_map_rule", regel["id"])
        assert (intrekking["from_classification"], intrekking["to_classification"],
                intrekking["person"]) == ("extern", "intern", "Testpersoon")


async def _overgangen(onderwerp: str, rid: str) -> list[tuple[str, str]]:
    return sorted((lg["from_classification"], lg["to_classification"])
                  for lg in await _log(onderwerp, rid))


@pytest.mark.parametrize("wijziging", [
    {"location": "https://example.org/api/besluiten"},
    {"route_type": "feed", "location": "https://example.org/feed.xml"},
])
async def test_intrekken_en_opnieuw_bevestigen_geeft_twee_logregels(env, app, wijziging) -> None:
    async with _client(app) as c:
        regel = await _bevestigde_regel(c)
        r = await c.patch(f"/api/v1/source-map-rules/{regel['id']}",
                          json={"rev": regel["rev"], "confirm": True, **wijziging})
        assert r.status_code == 200, r.text
        assert r.json()["effective_classification"] == "extern"
        # Aanmaken (bevestiging), dan in een verzoek intrekking en bevestiging.
        assert await _overgangen("source_map_rule", regel["id"]) == [
            ("extern", "intern"), ("intern", "extern"), ("intern", "extern")]
        assert {lg["person"] for lg in await _log("source_map_rule", regel["id"])} == {"Testpersoon"}


async def test_activeren_en_bevestigen_in_een_verzoek(env, app) -> None:
    async with _client(app) as c:
        regel = await _bevestigde_regel(c)
        pad = f"/api/v1/source-map-rules/{regel['id']}"
        r = await c.patch(pad, json={"rev": regel["rev"], "active": False})
        r = await c.patch(pad, json={"rev": r.json()["rev"], "active": True, "confirm": True})
        assert r.status_code == 200, r.text
        assert r.json()["effective_classification"] == "extern"
        assert await _overgangen("source_map_rule", regel["id"]) == [
            ("extern", "intern"), ("intern", "extern"), ("intern", "extern")]


async def test_bevestigen_van_bevestigde_regel_geeft_409(env, app) -> None:
    async with _client(app) as c:
        regel = await _bevestigde_regel(c)
        pad = f"/api/v1/source-map-rules/{regel['id']}"
        r = await c.post(f"{pad}/confirm", json={"rev": regel["rev"]})
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "al_bevestigd"
        # Ook met een beschrijvende wijziging: niets ingetrokken, dus al bevestigd.
        r = await c.patch(pad, json={"rev": regel["rev"], "name": "x", "confirm": True})
        assert r.status_code == 409
        r = await c.get(pad)
        assert (r.json()["rev"], r.json()["name"]) == (regel["rev"], regel["name"])
        assert len(await _log("source_map_rule", regel["id"])) == 1


async def test_gelijktijdig_bevestigen_en_wijzigen(env, app) -> None:
    import asyncio

    async with _client(app) as c:
        regel = (await c.post("/api/v1/source-map-rules", json=_regel())).json()
        pad = f"/api/v1/source-map-rules/{regel['id']}"
        bevestig, wijzig = await asyncio.gather(
            c.post(f"{pad}/confirm", json={"rev": regel["rev"]}),
            c.patch(pad, json={"rev": regel["rev"], "location": "https://example.org/ander"}),
        )
        assert sorted([bevestig.status_code, wijzig.status_code]) == [200, 409]
        na = (await c.get(pad)).json()
        assert na["rev"] == regel["rev"] + 1
        # Een PATCH met de rev van voor de bevestiging, na afloop: 409.
        r = await c.patch(pad, json={"rev": regel["rev"], "location": "https://example.org/x"})
        assert r.status_code == 409
        if bevestig.status_code == 200:
            assert na["effective_classification"] == "extern"
            assert len(await _log("source_map_rule", regel["id"])) == 1


@pytest.mark.parametrize("wijziging", [
    {"name": "Nieuwe naam"}, {"owner": "Testpersoon"}, {"document_type": ["motie"]}])
async def test_beschrijvende_wijziging_houdt_bevestiging(env, app, wijziging) -> None:
    async with _client(app) as c:
        regel = await _bevestigde_regel(c)
        r = await c.patch(f"/api/v1/source-map-rules/{regel['id']}",
                          json={"rev": regel["rev"], **wijziging})
        assert r.json()["effective_classification"] == "extern"
        assert len(await _log("source_map_rule", regel["id"])) == 1


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
        assert r.json()["detail"]["code"] == "verouderde_rev"
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
# Locatie: allowlist, niet in de vault of de exportmap (ADR 0004)
# --------------------------------------------------------------------------


async def test_locatie_in_vault_exportmap_of_via_koppeling_geeft_422(env, app, mappen) -> None:
    vault, export, buiten = mappen
    koppeling = buiten / "koppeling"
    os.symlink(vault / "20 Atlas", koppeling)
    verboden = [str(vault), str(vault / "20 Atlas" / "x.md"), str(export),
                str(export / "sub"), str(koppeling), str(koppeling / "notitie.md"),
                f"{buiten}/../vault/20 Atlas", f"file://{vault}/20%20Atlas",
                "obsidian://open?vault=Test", "/elders/buiten/de/innamemappen",
                f"http://localhost{vault}/20%20Atlas/x.md",
                f"https://127.0.0.1:8443{vault}/x.md"]
    async with _client(app) as c:
        for locatie in verboden:
            r = await c.post("/api/v1/source-map-rules", json=_regel(
                "watchfolder", source_type="map", location=locatie))
            assert r.status_code == 422, locatie
            assert str(vault) not in r.text
        r = await c.post("/api/v1/source-map-rules", json=_regel(
            "watchfolder", source_type="map", location=str(buiten / "inbox")))
        assert r.status_code == 201
        regel = r.json()
        r = await c.patch(f"/api/v1/source-map-rules/{regel['id']}",
                          json={"rev": regel["rev"], "location": str(koppeling)})
        assert r.status_code == 422
        assert (await c.get(f"/api/v1/source-map-rules/{regel['id']}")).json()[
            "location"] == str(buiten / "inbox")
        # Ook een batch of wachtrij-item wijst niet in de vault.
        r = await c.post("/api/v1/intake/batches", json={
            "route": "inname_pdf", "files": [str(vault / "a.pdf")]})
        assert r.status_code == 422
        env["ctx"] = PIJPLIJN
        r = await c.post("/api/v1/intake/items", json={
            "origin": "watchfolder", "path": str(vault / "20 Atlas" / "x.md"), "content_hash": HASH})
        assert r.status_code == 422


@pytest.mark.parametrize("locatie", [
    r"\\?\D:\# Vault\x.md", r"\\?\D:\Data\inbox\x.pdf", r"\\.\D:\Data\inbox\x.pdf",
    r"\\server\share\x.pdf", "//server/share/x.pdf",
    r"D:\#VAULT~1\x.md", r"D:\Data\INBOX~1\x.pdf",
    r"D:\# Vault.\x.md", r"D:\# Vault \x.md", r"D:\Data\inbox.\x.pdf",
    r"D:\# Vault\x.md::$DATA", r"D:\Data\inbox\x.pdf::$DATA", r"D:\Data\inbox\x:y",
    "D:#", "D:# Vault", "D:", "file:/D:/# Vault/x.md", "file:/D:/Data/inbox/x.pdf",
    "file:///D:/Data/inbox/x.pdf",
    r"D:\# Vault", r"d:\# vault\20 Atlas\x.md", "/mnt/d/# Vault/x", r"D:\Data\..\# Vault\x",
    r"E:\Data\inbox\x.pdf", "http://localhost/D:/%23%20Vault/x.md",
])
async def test_windows_omwegen_geweigerd(locatie) -> None:
    controle = Locatiecontrole([r"D:\# Vault"], [], [r"D:\Data\inbox"], [])
    with pytest.raises(VerbodenLocatieError):
        controle.controleer(locatie)


@pytest.mark.parametrize("locatie", [
    r"D:\Data\inbox", r"D:\Data\inbox\verslag.pdf", "d:/data/INBOX/sub/x.pdf",
    "/mnt/d/Data/inbox/x.pdf", "https://example.org/D:/# Vault", "https://example.org/api",
    "http://8.8.8.8/feed.xml",
])
async def test_toegestane_locaties(locatie) -> None:
    Locatiecontrole([r"D:\# Vault"], [], [r"D:\Data\inbox"], []).controleer(locatie)


@pytest.mark.parametrize("locatie", [
    "http://127.0.0.2/api", "http://127.1/api", "http://2130706433/api",
    "http://host.docker.internal/api", "http://localhost./api", "http://localhost/api",
    "http://0.0.0.0/api", "http://[0:0:0:0:0:0:0:1]/api", "http://[::1]/api",
    "http://[::ffff:127.0.0.1]/api", "http://10.1.2.3/api", "http://172.16.0.1/api",
    "http://192.168.1.1/api", "http://169.254.169.254/latest", "http://[fe80::1]/api",
    "http://[fc00::1]/api", "http://ollama:8080/api", "http://open-notebook/api",
    "http://api.localhost/x", "smb://server/share/x.pdf", "ftp://example.org/x.pdf",
    "ws://example.org/x", "http:///geen-host",
    "http://１２７.０.０.１/", "http://①②⑦.0.0.1/", "http://a.localhost。/",
    "http://100.64.0.1/api", "http://100.127.255.254/api", "http://printer.local/x",
    "http://nas.lan/x", "http://router.home.arpa/x", "http://db.internal/x",
    "http://224.0.0.1/x", "http://exämple.org/x",
])
async def test_url_naar_interne_host_of_ander_schema_geweigerd(locatie) -> None:
    with pytest.raises(VerbodenLocatieError):
        Locatiecontrole([], [], [], []).controleer(locatie)


async def test_url_allowlist(tmp_path) -> None:
    vault = tmp_path / "vault"
    controle = Locatiecontrole([str(vault)], [], [], ["localhost", "ollama"])
    controle.controleer("http://localhost:8080/api")
    controle.controleer("http://localhost./api")
    controle.controleer("http://ollama:8080/api")
    with pytest.raises(VerbodenLocatieError):
        controle.controleer(f"http://localhost{vault}/x.md")
    with pytest.raises(VerbodenLocatieError):
        controle.controleer("http://127.0.0.2/api")


async def test_zonder_innamemappen_geen_pad() -> None:
    with pytest.raises(VerbodenLocatieError):
        Locatiecontrole([], [], [], []).controleer("/data/inbox/x.pdf")
    Locatiecontrole([], [], [], []).controleer("https://example.org/api")


# --------------------------------------------------------------------------
# Batch en wachtrij
# --------------------------------------------------------------------------


async def test_batch_zonder_keuze_is_intern(env, app, mappen) -> None:
    buiten = mappen[2]
    bestanden = [str(buiten / "inbox" / "a.pdf"), str(buiten / "inbox" / "b.pdf")]
    async with _client(app) as c:
        r = await c.post("/api/v1/intake/batches", json={"route": "inname_pdf", "files": bestanden})
        assert r.status_code == 201, r.text
        batch = r.json()
        assert (batch["classification"], batch["effective_classification"]) == ("intern", "intern")
        assert (batch["created_by"], batch["files"]) == ("Testpersoon", bestanden)

        r = await c.post("/api/v1/intake/batches", json={
            "route": "inname_pdf", "classification": "extern"})
        assert r.json()["effective_classification"] == "intern"
        r = await c.post(f"/api/v1/intake/batches/{r.json()['id']}/confirm",
                         json={"rev": r.json()["rev"]})
        assert r.status_code == 200 and r.json()["effective_classification"] == "extern"
        assert len(await _log("intake_batch", r.json()["id"])) == 1
        # Een andere route trekt de bevestiging in, met een logregel.
        r = await c.patch(f"/api/v1/intake/batches/{r.json()['id']}",
                          json={"rev": r.json()["rev"], "route": "inname_pdf_v2"})
        assert r.json()["effective_classification"] == "intern"
        assert [lg["to_classification"] for lg in await _log("intake_batch", r.json()["id"])] == [
            "extern", "intern"]


async def test_batch_bestanden_wijzigen_en_bevestigen(env, app, mappen) -> None:
    inbox = mappen[2] / "inbox"
    # De fork hasht de bestanden zelf bij het bevestigen (F2-14).
    (inbox / "a.pdf").write_bytes(b"%PDF-1.7 a")
    (inbox / "b.pdf").write_bytes(b"%PDF-1.7 b")
    async with _client(app) as c:
        r = await c.post("/api/v1/intake/batches", json={
            "route": "inname_pdf", "classification": "extern", "confirm": True,
            "files": [str(inbox / "a.pdf")]})
        batch = r.json()
        assert r.status_code == 201 and batch["effective_classification"] == "extern"
        r = await c.patch(f"/api/v1/intake/batches/{batch['id']}", json={
            "rev": batch["rev"], "files": [str(inbox / "a.pdf"), str(inbox / "b.pdf")],
            "confirm": True})
        assert r.status_code == 200, r.text
        assert r.json()["effective_classification"] == "extern"
        assert await _overgangen("intake_batch", batch["id"]) == [
            ("extern", "intern"), ("intern", "extern"), ("intern", "extern")]
        r = await c.post(f"/api/v1/intake/batches/{batch['id']}/confirm",
                         json={"rev": r.json()["rev"]})
        assert r.status_code == 409


async def test_gelijktijdige_besluiten_over_een_item(env, app, mappen) -> None:
    import asyncio

    env["ctx"] = PIJPLIJN
    async with _client(app) as c:
        item = (await c.post("/api/v1/intake/items", json={
            "origin": "watchfolder", "path": str(mappen[2] / "inbox" / "gelijk.pdf"),
            "content_hash": "sha256:" + "e" * 64})).json()
        env["ctx"] = BEHEERDER
        pad = f"/api/v1/intake/items/{item['id']}/decision"
        een, twee = await asyncio.gather(
            c.post(pad, json={"rev": item["rev"], "route": "inname_pdf", "classification": "extern"}),
            c.post(pad, json={"rev": item["rev"], "route": "inname_pdf", "classification": "intern"}),
        )
        assert sorted([een.status_code, twee.status_code]) == [200, 409]
        winnaar = een if een.status_code == 200 else twee
        [log] = await _log("intake_item", item["id"])
        assert log["to_classification"] == winnaar.json()["decision"]["classification"]


async def test_batch_alleen_bij_een_regel_voor_bulkupload(env, app, mappen) -> None:
    async with _client(app) as c:
        api = (await c.post("/api/v1/source-map-rules", json=_regel())).json()
        r = await c.post("/api/v1/intake/batches", json={
            "route": "inname_pdf", "source_map_rule_id": api["id"]})
        assert r.status_code == 422
        bulk = (await c.post("/api/v1/source-map-rules", json=_regel(
            "bulkupload", source_type="pdf", location=str(mappen[2])))).json()
        r = await c.post("/api/v1/intake/batches", json={
            "route": "inname_pdf", "source_map_rule_id": bulk["id"]})
        assert r.status_code == 201 and r.json()["source_map_rule"] == bulk["id"]


async def test_ongeldige_invoer_geeft_422(env, app, mappen) -> None:
    pad = str(mappen[2] / "inbox" / "x.pdf")
    async with _client(app) as c:
        r = await c.post("/api/v1/source-map-rules", json=_regel(route="flow; DROP"))
        assert r.status_code == 422
        r = await c.get("/api/v1/intake/items?status=onzin")
        assert r.status_code == 422
        env["ctx"] = PIJPLIJN
        for fout in ("b" * 64, "sha256:" + "B" * 64, "sha256:abc", "md5:" + "b" * 32):
            r = await c.post("/api/v1/intake/items", json={
                "origin": "watchfolder", "path": pad, "content_hash": fout})
            assert r.status_code == 422, fout
        item = (await c.post("/api/v1/intake/items", json={
            "origin": "watchfolder", "path": pad, "content_hash": HASH})).json()
        env["ctx"] = BEHEERDER
        r = await c.post(f"/api/v1/intake/items/{item['id']}/decision", json={
            "rev": item["rev"], "route": "../vault", "classification": "intern"})
        assert r.status_code == 422


async def test_watchfolder_item_wacht_zonder_standaard(env, app, mappen) -> None:
    inbox = mappen[2] / "inbox"
    env["ctx"] = PIJPLIJN
    async with _client(app) as c:
        item = {"origin": "watchfolder", "path": str(inbox / "verslag.pdf"), "content_hash": HASH}
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
            "origin": "upload", "path": str(inbox / "c.pdf"), "content_hash": "sha256:" + "c" * 64,
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

        # Het besluit ligt vast: een tweede besluit geeft 409, ook met de juiste rev.
        r = await c.post(f"/api/v1/intake/items/{nieuw['id']}/decision", json={
            "rev": besloten["rev"], "route": "inname_pdf", "classification": "intern"})
        assert r.status_code == 409
        assert len(await _log("intake_item", nieuw["id"])) == 1

        env["ctx"] = PIJPLIJN
        r = await c.get("/api/v1/intake/items?status=besloten")
        assert nieuw["id"] in {i["id"] for i in r.json()}
        r = await c.patch(f"/api/v1/intake/items/{nieuw['id']}",
                          json={"rev": besloten["rev"], "status": "verwerkt"})
        assert r.status_code == 200 and r.json()["status"] == "verwerkt"
        assert r.json()["decision"]["classification"] == "extern"

        env["ctx"] = BEHEERDER
        r = await c.post(f"/api/v1/intake/items/{nieuw['id']}/decision", json={
            "rev": r.json()["rev"], "route": "inname_pdf", "classification": "intern"})
        assert r.status_code == 409
