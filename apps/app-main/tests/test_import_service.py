"""Het importeindpunt ``POST /api/v1/import`` (F2-14, ADR 0010, plan §5.1).

Tegen een echte SurrealDB, met de sleutel ``pijplijn``. Het bronarchief en de
innamemap zijn tijdelijke mappen; alle testdata is synthetisch.

Stopcondities uit het plan van aanpak:

* hetzelfde verzoek twee keer geeft hetzelfde resultaat; een bekende hash
  maakt geen nieuwe versie; A -> B -> A volgt B26;
* 409 bij een lagere classificatie; elk fragment en elke ``doc_node`` heeft
  ``source_version``, ``run`` en de classificatie van de versie, en elk
  fragment ``extractor``;
* de markering van het gegenereerde blok geeft 422 en een logregel;
* een niet-vastgelegde flow met interne invoer geeft 409;
* de classificatie komt uit de regel van de werkelijke bron, de batch of het
  actuele besluit over het item, nooit uit de aanroeper;
* de inhoudshash van een URL-bron is die van de ruwe inhoud;
* alleen een run uit ``POST /api/v1/runs`` van dezelfde sleutel; ``PATCH``
  alleen door de aanmaker; een run die te lang ``bezig`` staat, wordt
  ``mislukt``; batch en item horen bij de regel en de run.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.dependencies import (
    get_import_service,
    get_intake_service,
    get_processing_run_service,
)
from app_main.services import import_service
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.import_service import ImportService
from app_main.services.intake_service import IntakeService
from app_main.services.processing_run_service import VERLOPEN, ProcessingRunService
from app_main.services.source_map_service import Locatiecontrole
from shared.models.agents import Right

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

LANGFLOW = agent_auth.AgentKeyContext(
    key_id="agent_keys:langflow", agent_id="langflow", permission=Right.PIJPLIJN)
ANDERE = agent_auth.AgentKeyContext(
    key_id="agent_keys:andere", agent_id="andere", permission=Right.PIJPLIJN)
MARKERING = "<!-- signaalsysteem:begin -->"


@pytest.fixture(scope="module")
def app():
    return create_app()


class Omgeving:
    def __init__(self, monkeypatch, archief, inname, flows) -> None:
        self.monkeypatch = monkeypatch
        self.archief = archief
        self.inname = inname
        self.flows = flows
        self.sleutel = LANGFLOW

    def als(self, sleutel) -> None:
        self.sleutel = sleutel

    def origineel(self, inhoud: bytes) -> tuple[str, str]:
        naam = f"{uuid.uuid4().hex}.bin"
        (self.archief / naam).write_bytes(inhoud)
        return f"bronarchief/{naam}", "sha256:" + hashlib.sha256(inhoud).hexdigest()


@pytest.fixture
async def env(live_surrealdb, monkeypatch, app, git_flows, tmp_path):
    async for omgeving in _omgeving(live_surrealdb, monkeypatch, app, git_flows, tmp_path):
        yield omgeving


@pytest.fixture
async def env_rocksdb(live_surrealdb_rocksdb, monkeypatch, app, git_flows, tmp_path):
    """Als ``env``, op de opslag van productie (rocksdb), voor gelijktijdigheid."""
    async for omgeving in _omgeving(live_surrealdb_rocksdb, monkeypatch, app, git_flows,
                                    tmp_path):
        yield omgeving


async def _omgeving(live_surrealdb, monkeypatch, app, git_flows, tmp_path):
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    archief = tmp_path / "bronarchief"
    inname = tmp_path / "inname"
    archief.mkdir()
    inname.mkdir()
    omgeving = Omgeving(monkeypatch, archief, inname, git_flows)

    async def _sleutel(_request):
        return omgeving.sleutel

    monkeypatch.setattr(agent_auth, "authenticate_request", _sleutel)
    controle = Locatiecontrole(vault_paden=[str(tmp_path / "vault")], exportmap_paden=[],
                               inname_paden=[str(inname)], inname_urls=[])

    def _runs():
        return ProcessingRunService(flow_register=git_flows.register)

    app.dependency_overrides[get_processing_run_service] = _runs
    omgeving.forkarchief = tmp_path / "forkarchief"
    app.dependency_overrides[get_import_service] = lambda: ImportService(
        run_service=_runs(), locatiecontrole=controle, bronarchief=str(archief),
        fork_archief=str(omgeving.forkarchief))
    app.dependency_overrides[get_intake_service] = lambda: IntakeService(locatiecontrole=controle)
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    yield omgeving
    for dep in (get_processing_run_service, get_import_service, get_intake_service):
        app.dependency_overrides.pop(dep, None)
    await conn._pool.close_all()
    # De vorige pool terug, of geen: een nieuwe pool naar deze database (bij
    # ``env_rocksdb`` de rocksdb-container) zou de volgende tests stil naar een
    # andere database laten schrijven.
    conn._pool = original


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                             headers={"Authorization": "Bearer ak_test"})


async def _q(sql: str, params: dict | None = None):
    return await conn.execute_query(sql, params or {})


async def _een(sql: str, params: dict | None = None):
    rijen = await _q(sql, params)
    return rijen[0] if isinstance(rijen, list) else rijen


_SOORT = {"api": "api", "website": "website", "watchfolder": "map", "upload": "pdf",
          "bulkupload": "pdf"}
_START = {"api": "automatisch", "website": "automatisch", "watchfolder": "wachtrij"}


async def _regel(route_type: str = "api", classification: str = "extern", *,
                 bevestigd: bool = True, location: str = "https://example.org/api") -> str:
    data = {"name": "Openbare testbron", "route_type": route_type,
            "source_type": _SOORT[route_type], "location": location,
            "pipeline_start": _START.get(route_type, "na_bevestiging"),
            "route": "inname_pdf", "fetch_content": "volledig_document",
            "classification": classification}
    if bevestigd and classification == "extern":
        data.update(classification_confirmed_by="Testpersoon",
                    classification_confirmed_at=datetime.now(timezone.utc))
    return str((await _een("CREATE source_map_rule CONTENT $d RETURN id", {"d": data}))["id"])


async def _batch(regel: str, files: list[str], classification: str = "extern",
                 hashes: list[str] | None = None) -> str:
    data = {"route": "inname_pdf", "files": files, "classification": classification,
            "created_by": "Testpersoon", "source_map_rule": conn.ensure_record_id(regel),
            "file_hashes": [{"path": f, "content_hash": h} for f, h in zip(files, hashes or [])]}
    if classification == "extern":
        data.update(classification_confirmed_by="Testpersoon",
                    classification_confirmed_at=datetime.now(timezone.utc))
    return str((await _een("CREATE intake_batch CONTENT $d RETURN id", {"d": data}))["id"])


async def _item(path: str, content_hash: str, besluit: str | None = "extern",
                origin: str = "watchfolder") -> str:
    rij = await _een("CREATE intake_item CONTENT $d RETURN id", {"d": {
        "origin": origin, "path": path, "content_hash": content_hash,
        "registered_by": "langflow"}})
    if besluit:
        await _q("UPDATE $id SET status = 'besloten', decision = $b", {
            "id": conn.ensure_record_id(str(rij["id"])),
            "b": {"route": "inname_pdf", "classification": besluit,
                  "decided_by": "Testpersoon", "decided_at": datetime.now(timezone.utc)}})
    return str(rij["id"])


async def _start(client, env, *, regel=None, batch=None, item=None, flow_hash=None,
                 status: int = 201):
    body = {"trigger": "schema", "pipeline_id": "inname_pdf",
            "pipeline_version": {"git_commit": env.flows.head,
                                 "flow_hash": flow_hash or env.flows.hash},
            "source_map_rule_id": regel, "intake_batch_id": batch, "intake_item_id": item}
    resp = await client.post("/api/v1/runs", json=body)
    assert resp.status_code == status, resp.text
    return resp.json().get("run_id") if status == 201 else resp.json()


EXTERN = {"value": "extern", "basis": "bronkaartregel",
          "gatekeeper": {"decision": "extern", "internal_markers": [], "method": "regel"}}


def _verzoek(run_id, *, original, content_hash, regel=None, batch=None, item=None,
             external_id=None, location="https://example.org/api/stuk-1", teksten=None,
             title="Openbaar stuk", classification=EXTERN, **extra) -> dict:
    teksten = teksten or ["Inleiding van het openbare stuk.", "Tweede alinea."]
    return {
        "run_id": run_id, "source_map_rule_id": regel, "batch_id": batch,
        "intake_item_id": item, "classification": copy.deepcopy(classification),
        "source": {"external_id": external_id or f"stuk-{uuid.uuid4().hex}", "type": "pdf",
                   "location": location, "title": title},
        "source_version": {"content_hash": content_hash, "original": original,
                           "fetched_at": "2026-11-02T09:00:00+01:00"},
        "extraction": {"extractor": "docling-serve", "version": "v1.38.2"},
        "structure": {"format": "docling_document", "content": {"texts": []}},
        "chunks": [
            {"seq": i, "text": t, "section_path": ["Hoofdstuk 1", "Paragraaf 1.1"],
             "page": 1, "printed_page": "3", "element_type": "paragraph",
             "positions": [{"page": 1, "bbox": [0.1, 0.2, 0.3, 0.4]}],
             "doc_ref": f"#/texts/{i}", "labels": []}
            for i, t in enumerate(teksten)
        ],
        **extra,
    }


async def _run(run_id: str) -> dict:
    return await _een("SELECT * FROM type::thing($id)", {"id": run_id})


async def _versies(bron: str) -> list:
    return await _q("SELECT * FROM source_version WHERE source = $b ORDER BY version_no",
                    {"b": conn.ensure_record_id(bron)})


# --------------------------------------------------------------------------
# Idempotent, bekende hash, B26
# --------------------------------------------------------------------------


async def test_hetzelfde_verzoek_geeft_hetzelfde_resultaat(env, app) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 openbaar stuk A")
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        verzoek = _verzoek(run, original=original, content_hash=h, regel=regel)
        eerste = await client.post("/api/v1/import", json=verzoek)
        tweede = await client.post("/api/v1/import", json=verzoek)
    assert eerste.status_code == 201, eerste.text
    assert tweede.status_code == 200, tweede.text
    assert eerste.json() == tweede.json()
    body = eerste.json()
    assert body["status"] == "nieuw" and body["version_no"] == 1
    assert body["classification"] == "extern" and len(body["chunk_ids"]) == 2
    assert len(await _versies(body["source_id"])) == 1
    [aantal] = await _q("SELECT count() AS n FROM chunk WHERE source = $b GROUP ALL",
                        {"b": conn.ensure_record_id(body["source_id"])})
    assert aantal["n"] == 2


async def test_ander_verzoek_in_dezelfde_run_geeft_409(env, app) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 stuk met twee versies van fragmenten")
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        verzoek = _verzoek(run, original=original, content_hash=h, regel=regel)
        assert (await client.post("/api/v1/import", json=verzoek)).status_code == 201
        verzoek["chunks"][0]["text"] = "Een andere tekst."
        resp = await client.post("/api/v1/import", json=verzoek)
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "ander_verzoek"


async def test_bekende_hash_maakt_geen_nieuwe_versie(env, app) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 ongewijzigd stuk")
    extern_id = f"stuk-{uuid.uuid4().hex}"
    async with _client(app) as client:
        eerste = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            eerste, original=original, content_hash=h, regel=regel, external_id=extern_id))
        assert resp.status_code == 201
        versie = resp.json()["source_version_id"]
        tweede = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            tweede, original=original, content_hash=h, regel=regel, external_id=extern_id))
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "ongewijzigd"
        assert resp.json()["source_version_id"] == versie and resp.json()["chunk_ids"] == []
        gesloten = await client.patch(f"/api/v1/runs/{tweede}", json={"status": "geslaagd"})
    assert gesloten.json()["status"] == "niets_nieuws"
    assert len(await _versies(resp.json()["source_id"])) == 1


async def test_inhoud_a_b_a_volgt_b26(env, app) -> None:
    regel = await _regel()
    a, ha = env.origineel(b"%PDF-1.7 inhoud A")
    b, hb = env.origineel(b"%PDF-1.7 inhoud B")
    extern_id = f"stuk-{uuid.uuid4().hex}"
    async with _client(app) as client:
        ids = []
        for original, h in ((a, ha), (b, hb), (a, ha)):
            run = await _start(client, env, regel=regel)
            resp = await client.post("/api/v1/import", json=_verzoek(
                run, original=original, content_hash=h, regel=regel, external_id=extern_id))
            assert resp.status_code in (200, 201), resp.text
            ids.append((run, resp.json()))
            await client.patch(f"/api/v1/runs/{run}", json={"status": "geslaagd"})
    (_, v1), (_, v2), (derde, v3) = ids
    assert (v1["version_no"], v2["version_no"]) == (1, 2)
    assert v3["status"] == "ongewijzigd" and v3["source_version_id"] == v1["source_version_id"]
    bron = await _een("SELECT current_version FROM type::thing($id)", {"id": v1["source_id"]})
    assert str(bron["current_version"]) == v1["source_version_id"]
    assert len(await _versies(v1["source_id"])) == 2
    regels = [r for r in (await _run(derde))["log"] if r["status"] == "versie_teruggezet"]
    assert len(regels) == 1
    assert (await _run(derde))["status"] == "niets_nieuws"


# --------------------------------------------------------------------------
# Herkomst en classificatie
# --------------------------------------------------------------------------


async def test_fragment_en_doc_node_dragen_herkomst(env, app) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 stuk met structuur")
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel,
            classification={"value": "extern", "gatekeeper": {"decision": "intern",
                                                              "internal_markers": ["rubricering"]}}))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # De poortwachter vond interne kenmerken: strenger dan de regel.
    assert body["classification"] == "intern"
    versie = await _een("SELECT * FROM type::thing($id)", {"id": body["source_version_id"]})
    assert versie["classification"] == "intern"
    assert versie["content_hash"] == h.removeprefix("sha256:")
    fragmenten = await _q("SELECT * FROM chunk WHERE source_version = $v",
                          {"v": conn.ensure_record_id(body["source_version_id"])})
    knopen = await _q("SELECT * FROM doc_node WHERE source_version = $v",
                      {"v": conn.ensure_record_id(body["source_version_id"])})
    assert len(fragmenten) == 2 and len(knopen) >= 2
    for rij in fragmenten + knopen:
        assert str(rij["source_version"]) == body["source_version_id"]
        assert str(rij["run"]) == run
        assert rij["classification"] == versie["classification"]
    for rij in fragmenten:
        assert rij["extractor"] == "docling-serve@v1.38.2"
        assert rij["section_path"] == ["Hoofdstuk 1", "Paragraaf 1.1"]
        assert rij["positions"] == [[1.0, 0.1, 0.3, 0.2, 0.4]]
        assert rij["printed_page"] == 3 and rij["metadata"]["doc_ref"].startswith("#/texts/")


async def test_lagere_classificatie_dan_de_bestaande_geeft_409(env, app) -> None:
    regel = await _regel()
    a, ha = env.origineel(b"%PDF-1.7 eerst intern")
    b, hb = env.origineel(b"%PDF-1.7 daarna extern")
    extern_id = f"stuk-{uuid.uuid4().hex}"
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=a, content_hash=ha, regel=regel, external_id=extern_id,
            classification={"value": "intern"}))
        assert resp.status_code == 201 and resp.json()["classification"] == "intern"
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=b, content_hash=hb, regel=regel, external_id=extern_id))
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "lagere_classificatie"
    assert len(await _versies((await _q("SELECT VALUE id FROM source WHERE external_id = $e",
                                         {"e": extern_id}))[0])) == 1


async def test_strengere_import_maakt_externe_bron_intern(env, app) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 bekend en toch intern")
    extern_id = f"stuk-{uuid.uuid4().hex}"
    async with _client(app) as client:
        for klasse in (EXTERN, {"value": "intern"}):
            run = await _start(client, env, regel=regel)
            resp = await client.post("/api/v1/import", json=_verzoek(
                run, original=original, content_hash=h, regel=regel, external_id=extern_id,
                classification=klasse))
            assert resp.status_code in (200, 201)
    assert resp.json()["status"] == "ongewijzigd" and resp.json()["classification"] == "intern"
    waarden = await _q("SELECT VALUE classification FROM chunk WHERE source = type::thing($b)",
                       {"b": resp.json()["source_id"]})
    assert set(waarden) == {"intern"}


@pytest.mark.parametrize("plek", ["fragment", "structuur", "titel", "ruwe_inhoud"])
async def test_markering_van_het_blok_geeft_422_en_een_logregel(env, app, plek) -> None:
    regel = await _regel()
    inhoud = b"%PDF-1.7 " + (MARKERING.encode() if plek == "ruwe_inhoud" else b"schoon")
    original, h = env.origineel(inhoud + uuid.uuid4().hex.encode())
    extern_id = f"stuk-{uuid.uuid4().hex}"
    verzoek = None
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        verzoek = _verzoek(run, original=original, content_hash=h, regel=regel,
                           external_id=extern_id)
        if plek == "fragment":
            verzoek["chunks"][1]["text"] = f"Notitie {MARKERING} afgeleid"
        elif plek == "structuur":
            verzoek["structure"]["content"] = {"texts": [{"text": "<!-- signaalsysteem:end -->"}]}
        elif plek == "titel":
            verzoek["source"]["title"] = MARKERING
        resp = await client.post("/api/v1/import", json=verzoek)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "afgeleide_tekst"
    assert MARKERING not in resp.text
    assert [r["status"] for r in (await _run(run))["log"]].count("geweigerd") == 1
    assert await _q("SELECT id FROM source WHERE external_id = $e", {"e": extern_id}) == []


async def test_flow_niet_vastgelegd_met_interne_invoer_geeft_409(env, app) -> None:
    regel = await _regel()  # extern: de run start met een waarschuwing
    original, h = env.origineel(b"%PDF-1.7 flow buiten git")
    async with _client(app) as client:
        run = await _start(client, env, regel=regel, flow_hash="0" * 64)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel,
            classification={"gatekeeper": {"decision": "intern"}}))
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "flow_niet_vastgelegd"
    gelezen = await _run(run)
    assert gelezen["status"] == "mislukt" and gelezen["error"]


async def test_classificatie_uit_de_regel_van_de_werkelijke_bron(env, app) -> None:
    extern = await _regel()
    onbevestigd = await _regel(bevestigd=False)
    original, h = env.origineel(b"%PDF-1.7 bron van regel A")
    b, hb = env.origineel(b"%PDF-1.7 bron van regel A, nieuwe inhoud")
    extern_id = f"stuk-{uuid.uuid4().hex}"
    async with _client(app) as client:
        # Een onbevestigde externe regel is intern, ook als de aanroeper extern zegt.
        run = await _start(client, env, regel=onbevestigd)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=onbevestigd, external_id=extern_id,
            classification={"value": "extern", "basis": "bronkaartregel"}))
        assert resp.status_code == 201 and resp.json()["classification"] == "intern"

        # Dezelfde bron via een andere (externe) regel: geweigerd.
        run = await _start(client, env, regel=extern)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=b, content_hash=hb, regel=extern, external_id=extern_id))
        assert resp.status_code == 409 and resp.json()["detail"]["code"] == "niet_gebonden"

        # Een andere regel in het verzoek dan die van de run: geweigerd.
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=b, content_hash=hb, regel=onbevestigd))
        assert resp.status_code == 409 and resp.json()["detail"]["code"] == "niet_gebonden"
    [klasse] = await _q("SELECT VALUE classification FROM source WHERE external_id = $e",
                        {"e": extern_id})
    assert klasse == "intern"


async def test_inhoudshash_van_een_url_bron_is_die_van_de_ruwe_inhoud(env, app) -> None:
    regel = await _regel("website", location="https://example.org/nieuws")
    ruw = b"<html><body><p>Openbaar nieuwsbericht.</p></body></html>"
    original, h = env.origineel(ruw)
    tekst = "Openbaar nieuwsbericht."
    van_tekst = "sha256:" + hashlib.sha256(tekst.encode()).hexdigest()
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=van_tekst, regel=regel,
            location="https://example.org/nieuws/1", teksten=[tekst]))
        assert resp.status_code == 422 and resp.json()["detail"]["code"] == "inhoudshash"
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel,
            location="https://example.org/nieuws/1", teksten=[tekst]))
    assert resp.status_code == 201, resp.text
    versie = await _een("SELECT content_hash FROM type::thing($id)",
                        {"id": resp.json()["source_version_id"]})
    assert versie["content_hash"] == hashlib.sha256(ruw).hexdigest()


@pytest.mark.parametrize("original", ["../buiten.bin", "/etc/hostname", "bronarchief/ontbreekt.bin"])
async def test_origineel_buiten_het_bronarchief_geeft_422(env, app, original) -> None:
    regel = await _regel()
    _, h = env.origineel(b"%PDF-1.7 los")
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel))
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "origineel"


async def test_locatie_in_de_vault_geeft_422(env, app, tmp_path) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 vault")
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel,
            location=str(tmp_path / "vault" / "notitie.md")))
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "locatie"


# --------------------------------------------------------------------------
# Runs: aanmaker, herkomst van de run, verlopen
# --------------------------------------------------------------------------


async def test_import_alleen_in_een_run_uit_post_runs_van_dezelfde_sleutel(env, app) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 run van de fork")
    fork = await _een("CREATE processing_run CONTENT $d RETURN id", {"d": {
        "trigger": "inname", "pipeline_id": "source_processor", "flow_vastgelegd": True,
        "source_map_rule_id": regel}})
    async with _client(app) as client:
        resp = await client.post("/api/v1/import", json=_verzoek(
            str(fork["id"]), original=original, content_hash=h, regel=regel))
        assert resp.status_code == 409 and resp.json()["detail"]["code"] == "niet_gebonden"
        env.als(ANDERE)
        run = await _start(client, env, regel=regel)
        env.als(LANGFLOW)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel))
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "niet_gebonden"


@pytest.mark.echte_sleutels
async def test_patch_alleen_door_de_aanmaker(env, app) -> None:
    async with _client(app) as client:
        run = await _start(client, env)
        env.als(ANDERE)
        resp = await client.patch(f"/api/v1/runs/{run}", json={"status": "mislukt"})
        assert resp.status_code == 403
        env.als(LANGFLOW)
        resp = await client.patch(f"/api/v1/runs/{run}", json={"status": "mislukt"})
    assert resp.status_code == 200 and resp.json()["status"] == "mislukt"


async def test_een_run_die_te_lang_bezig_staat_wordt_mislukt(env, app, monkeypatch) -> None:
    monkeypatch.setenv("SIGNAAL_RUN_MAX_MINUTEN", "60")
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 oude run")
    oud = await _een("CREATE processing_run CONTENT $d RETURN id", {"d": {
        "trigger": "schema", "pipeline_id": "inname_pdf", "flow_vastgelegd": True,
        "source_map_rule_id": regel, "created_by_key": LANGFLOW.key_id,
        "started_at": datetime.now(timezone.utc) - timedelta(hours=2)}})
    async with _client(app) as client:
        resp = await client.post("/api/v1/import", json=_verzoek(
            str(oud["id"]), original=original, content_hash=h, regel=regel))
        assert resp.status_code == 409 and resp.json()["detail"]["code"] == "verkeerde_status"
        # Een verse run blijft lopen.
        vers = await _start(client, env, regel=regel)
    gelezen = await _run(str(oud["id"]))
    assert gelezen["status"] == "mislukt" and gelezen["error"] == VERLOPEN
    assert (await _run(vers))["status"] == "bezig"


# --------------------------------------------------------------------------
# Batch en wachtrij-item
# --------------------------------------------------------------------------


async def test_bulkupload_krijgt_de_classificatie_van_de_batch(env, app) -> None:
    regel = await _regel("bulkupload", "intern", location=str(env.inname))
    uitkomst = {}
    async with _client(app) as client:
        for klasse in ("intern", "extern"):
            pad = str(env.inname / f"jaarstukken-{klasse}.pdf")
            original, h = env.origineel(f"%PDF-1.7 jaarstukken {klasse}".encode())
            batch = await _batch(regel, [pad], klasse, [h])
            run = await _start(client, env, regel=regel, batch=batch)
            # De aanroeper zegt extern; alleen de batch telt.
            resp = await client.post("/api/v1/import", json=_verzoek(
                run, original=original, content_hash=h, regel=regel, batch=batch,
                location=pad))
            assert resp.status_code == 201, resp.text
            uitkomst[klasse] = resp.json()["classification"]
            # Een bestand buiten de batch hoort er niet bij.
            resp = await client.post("/api/v1/import", json=_verzoek(
                run, original=original, content_hash=h, regel=regel, batch=batch,
                location=str(env.inname / "ander.pdf")))
            assert resp.status_code == 409
    assert uitkomst == {"intern": "intern", "extern": "extern"}


async def test_import_toetst_aan_de_actuele_batch(env, app) -> None:
    regel = await _regel("bulkupload", "intern", location=str(env.inname))
    pad = str(env.inname / "notulen.pdf")
    original, h = env.origineel(b"%PDF-1.7 notulen")
    batch = await _batch(regel, [pad], "extern", [h])
    async with _client(app) as client:
        run = await _start(client, env, regel=regel, batch=batch)
        assert (await _run(run))["input_classification"] == "extern"
        # De beheerder trekt de bevestiging in nadat de run begon.
        await _q("UPDATE type::thing($id) SET classification_confirmed_by = NONE,"
                 " classification_confirmed_at = NONE", {"id": batch})
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel, batch=batch, location=pad))
    assert resp.status_code == 201 and resp.json()["classification"] == "intern"


async def test_watchfolder_krijgt_het_besluit_van_het_item(env, app) -> None:
    regel = await _regel("watchfolder", "intern", location=str(env.inname))
    uitkomst = {}
    async with _client(app) as client:
        for besluit in ("intern", "extern"):
            pad = str(env.inname / f"{besluit}-{uuid.uuid4().hex}.pdf")
            original, h = env.origineel(f"%PDF-1.7 {pad}".encode())
            item = await _item(pad, h, besluit)
            run = await _start(client, env, regel=regel, item=item)
            resp = await client.post("/api/v1/import", json=_verzoek(
                run, original=original, content_hash=h, regel=regel, item=item, location=pad))
            assert resp.status_code == 201, resp.text
            uitkomst[besluit] = resp.json()["classification"]
            # Andere inhoud dan waarover is besloten: geweigerd.
            ander, h2 = env.origineel(b"%PDF-1.7 andere inhoud" + uuid.uuid4().bytes)
            resp = await client.post("/api/v1/import", json=_verzoek(
                run, original=ander, content_hash=h2, regel=regel, item=item, location=pad))
            assert resp.status_code == 409
    assert uitkomst == {"intern": "intern", "extern": "extern"}


async def test_batch_of_item_die_niet_bij_de_regel_hoort_wordt_geweigerd(env, app, tmp_path) -> None:
    bulk = await _regel("bulkupload", "intern", location=str(env.inname))
    andere_bulk = await _regel("bulkupload", "intern", location=str(env.inname))
    watch = await _regel("watchfolder", "intern", location=str(env.inname))
    api = await _regel()
    batch = await _batch(andere_bulk, [str(env.inname / "x.pdf")])
    _, h = env.origineel(b"%PDF-1.7 item")
    item = await _item(str(env.inname / "item.pdf"), h)
    elders = await _item(str(tmp_path / "elders" / "item.pdf"), h)
    wacht = await _item(str(env.inname / "wacht.pdf"), h, besluit=None)
    async with _client(app) as client:
        for regel, kwargs in ((bulk, {"batch": batch}), (api, {"batch": batch}),
                              (api, {"item": item}), (bulk, {"item": item}),
                              (watch, {"item": elders}), (watch, {"batch": batch})):
            body = await _start(client, env, regel=regel, status=409, **kwargs)
            assert body["detail"]["code"] == "niet_gebonden", (regel, kwargs)
        body = await _start(client, env, regel=watch, item=wacht, status=409)
        assert body["detail"]["code"] == "verkeerde_status"
        # Een import zonder batch op een regel voor bulkupload: geweigerd.
        run = await _start(client, env, regel=bulk)
        original, h2 = env.origineel(b"%PDF-1.7 zonder batch")
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h2, regel=bulk,
            location=str(env.inname / "x.pdf")))
        assert resp.status_code == 409
        # Een batch in het verzoek die niet die van de run is: geweigerd.
        eigen = await _batch(bulk, [str(env.inname / "x.pdf")])
        run = await _start(client, env, regel=bulk, batch=eigen)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h2, regel=bulk, batch=batch,
            location=str(env.inname / "x.pdf")))
        assert resp.status_code == 409 and resp.json()["detail"]["code"] == "niet_gebonden"


async def test_pijplijn_koppelt_een_run_niet_aan_een_ander_item(env, app) -> None:
    regel = await _regel("watchfolder", "intern", location=str(env.inname))
    _, h = env.origineel(b"%PDF-1.7 twee items")
    eerste = await _item(str(env.inname / f"{uuid.uuid4().hex}.pdf"), h)
    tweede = await _item(str(env.inname / f"{uuid.uuid4().hex}.pdf"), h)
    async with _client(app) as client:
        run = await _start(client, env, regel=regel, item=eerste)
        rev = (await _een("SELECT VALUE rev FROM type::thing($id)", {"id": tweede}))
        resp = await client.patch(f"/api/v1/intake/items/{tweede}",
                                  json={"status": "verwerkt", "rev": rev, "run_id": run})
        assert resp.status_code == 409 and resp.json()["detail"]["code"] == "niet_gebonden"
        rev = (await _een("SELECT VALUE rev FROM type::thing($id)", {"id": eerste}))
        resp = await client.patch(f"/api/v1/intake/items/{eerste}",
                                  json={"status": "verwerkt", "rev": rev, "run_id": run})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "verwerkt" and resp.json()["run"] == run


# --------------------------------------------------------------------------
# Review F2-14
# --------------------------------------------------------------------------

BEHEERDER = agent_auth.AgentKeyContext(
    key_id="agent_keys:beheer", agent_id="mens", permission=Right.BEHEERDER, person="Testpersoon")


@pytest.mark.parametrize("locatie", [
    "https://ander-domein.example.net/geheim", "https://example.org/apix/1",
    "http://example.org/api/1", "https://example.org/api/../geheim",
    "https://example.org:8443/api/1", "INNAME"])
async def test_bron_buiten_de_locatie_van_de_regel_wordt_geweigerd(env, app, locatie) -> None:
    regel = await _regel()  # https://example.org/api
    if locatie == "INNAME":
        locatie = str(env.inname / "watch" / "intern.pdf")
    original, h = env.origineel(b"%PDF-1.7 elders " + uuid.uuid4().hex.encode())
    extern_id = f"stuk-{uuid.uuid4().hex}"
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel, location=locatie,
            external_id=extern_id))
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "niet_gebonden"
    assert await _q("SELECT id FROM source WHERE external_id = $e", {"e": extern_id}) == []


@pytest.mark.parametrize("klasse", [
    None, {}, {"value": "extern"},
    {"value": "extern", "gatekeeper": {"decision": "extern", "internal_markers": ["rubricering"]}},
    {"value": "extern", "gatekeeper": {"internal_markers": []}},
    {"gatekeeper": {"decision": "extern"}},
])
async def test_extern_alleen_met_waarde_en_poortwachter(env, app, klasse) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 zonder poortwachter " + uuid.uuid4().hex.encode())
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        verzoek = _verzoek(run, original=original, content_hash=h, regel=regel,
                           classification=klasse)
        if klasse is None:
            verzoek.pop("classification")
        resp = await client.post("/api/v1/import", json=verzoek)
    assert resp.status_code == 201 and resp.json()["classification"] == "intern"


async def test_externe_batch_vraagt_het_bestand_van_de_batch(env, app) -> None:
    regel = await _regel("bulkupload", "intern", location=str(env.inname))
    pad = str(env.inname / f"{uuid.uuid4().hex}.pdf")
    _, eigen = env.origineel(b"%PDF-1.7 bestand van de batch " + uuid.uuid4().hex.encode())
    ander, h = env.origineel(b"%PDF-1.7 ander intern document " + uuid.uuid4().hex.encode())
    async with _client(app) as client:
        for hashes in ([eigen], None):
            batch = await _batch(regel, [pad], "extern", hashes)
            run = await _start(client, env, regel=regel, batch=batch)
            resp = await client.post("/api/v1/import", json=_verzoek(
                run, original=ander, content_hash=h, regel=regel, batch=batch, location=pad))
            assert resp.status_code == 409 and resp.json()["detail"]["code"] == "niet_gebonden"


async def test_batch_legt_bestandshashes_vast(env, app) -> None:
    leesbaar = env.inname / f"{uuid.uuid4().hex}.pdf"
    leesbaar.write_bytes(b"%PDF-1.7 batchbestand")
    ontbreekt = str(env.inname / "ontbreekt.pdf")
    env.als(BEHEERDER)
    async with _client(app) as client:
        resp = await client.post("/api/v1/intake/batches", json={
            "route": "inname_pdf", "files": [str(leesbaar), ontbreekt],
            "classification": "extern", "confirm": True})
        assert resp.status_code == 422
        resp = await client.post("/api/v1/intake/batches", json={
            "route": "inname_pdf", "files": [str(leesbaar)], "classification": "extern",
            "confirm": True})
    assert resp.status_code == 201, resp.text
    verwacht = "sha256:" + hashlib.sha256(b"%PDF-1.7 batchbestand").hexdigest()
    assert resp.json()["file_hashes"] == [{"path": str(leesbaar), "content_hash": verwacht}]


async def test_extern_geweigerd_als_de_inhoud_intern_bekend_is(env, app) -> None:
    intern = await _regel(bevestigd=False)
    extern = await _regel()
    original, h = env.origineel(b"%PDF-1.7 gedeeld " + uuid.uuid4().hex.encode())
    wacht, hw = env.origineel(b"%PDF-1.7 in de wachtrij " + uuid.uuid4().hex.encode())
    await _item(str(env.inname / f"{uuid.uuid4().hex}.pdf"), hw, besluit=None)
    async with _client(app) as client:
        run = await _start(client, env, regel=intern)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=intern))
        assert resp.status_code == 201 and resp.json()["classification"] == "intern"
        for o, hh in ((original, h), (wacht, hw)):
            run = await _start(client, env, regel=extern)
            resp = await client.post("/api/v1/import", json=_verzoek(
                run, original=o, content_hash=hh, regel=extern))
            assert resp.status_code == 409
            assert resp.json()["detail"]["code"] == "inhoud_intern_bekend"


@pytest.mark.parametrize("variant", [
    "<!--signaalsysteem:begin-->", "<!-- SIGNAALSYSTEEM:BEGIN -->",
    "&lt;!-- signaalsysteem:begin --&gt;", "<!--  signaalsysteem :  end  -->",
    "＜!-- signaalsysteem:begin --＞", "split"])
async def test_markering_in_varianten(env, app, variant) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 schoon " + uuid.uuid4().hex.encode())
    teksten = (["tekst <!-- signaal", "systeem:begin --> rest"] if variant == "split"
               else ["a", f"b {variant} c"])
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel, teksten=teksten))
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "afgeleide_tekst"


@pytest.mark.parametrize("veld", ["labels", "doc_ref", "element_type", "external_id", "location"])
async def test_markering_in_elk_tekstveld(env, app, veld) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 schoon " + uuid.uuid4().hex.encode())
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        verzoek = _verzoek(run, original=original, content_hash=h, regel=regel)
        if veld == "labels":
            verzoek["chunks"][0]["labels"] = [{"term": MARKERING}]
        elif veld in ("doc_ref", "element_type"):
            verzoek["chunks"][0][veld] = MARKERING
        elif veld == "external_id":
            verzoek["source"]["external_id"] = MARKERING
        else:
            verzoek["source"]["location"] = "https://example.org/api/" + MARKERING
        resp = await client.post("/api/v1/import", json=verzoek)
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "afgeleide_tekst"


async def test_markering_in_ruwe_inhoud_als_utf16(env, app) -> None:
    regel = await _regel()
    inhoud = ("Notitie " + MARKERING + uuid.uuid4().hex).encode("utf-16")
    original, h = env.origineel(inhoud)
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel))
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "afgeleide_tekst"


async def test_origineel_wordt_bewaard_in_het_archief_van_de_fork(env, app) -> None:
    regel = await _regel()
    inhoud = b"%PDF-1.7 origineel " + uuid.uuid4().hex.encode()
    original, h = env.origineel(inhoud)
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel))
    assert resp.status_code == 201
    # Langflow vervangt het origineel daarna.
    (env.archief / original.removeprefix("bronarchief/")).write_bytes(b"vervangen")
    hexhash = h.removeprefix("sha256:")
    versie = await _een("SELECT original FROM type::thing($id)",
                        {"id": resp.json()["source_version_id"]})
    assert versie["original"] == f"forkarchief/{hexhash}"
    kopie = env.forkarchief / hexhash
    assert kopie.read_bytes() == inhoud and not os.access(kopie, os.W_OK)
    assert [p.name for p in env.forkarchief.iterdir()] == [hexhash]


@pytest.mark.echte_sleutels
async def test_beheerder_importeert_niet(env, app) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 beheerder " + uuid.uuid4().hex.encode())
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        env.als(BEHEERDER)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel))
    assert resp.status_code == 403


async def test_tweede_versie_in_dezelfde_run_krijgt_eigen_structuur(env, app) -> None:
    regel = await _regel()
    extern_id = f"stuk-{uuid.uuid4().hex}"
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        versies = []
        for _ in range(2):
            original, h = env.origineel(b"%PDF-1.7 versie " + uuid.uuid4().hex.encode())
            resp = await client.post("/api/v1/import", json=_verzoek(
                run, original=original, content_hash=h, regel=regel, external_id=extern_id))
            assert resp.status_code == 201, resp.text
            versies.append(resp.json()["source_version_id"])
    for versie in versies:
        [rij] = await _q("SELECT count() AS n FROM doc_node WHERE source_version = $v GROUP ALL",
                         {"v": conn.ensure_record_id(versie)})
        assert rij["n"] >= 2


async def test_herhaling_met_andere_structuur_of_classificatie(env, app) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 herhaling " + uuid.uuid4().hex.encode())
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        verzoek = _verzoek(run, original=original, content_hash=h, regel=regel)
        assert (await client.post("/api/v1/import", json=verzoek)).status_code == 201
        anders = _verzoek(run, original=original, content_hash=h, regel=regel,
                          external_id=verzoek["source"]["external_id"])
        anders["chunks"][0]["section_path"] = ["Ander hoofdstuk"]
        resp = await client.post("/api/v1/import", json=anders)
        assert resp.status_code == 409 and resp.json()["detail"]["code"] == "ander_verzoek"
        strenger = _verzoek(run, original=original, content_hash=h, regel=regel,
                            external_id=verzoek["source"]["external_id"],
                            classification={"value": "intern"})
        resp = await client.post("/api/v1/import", json=strenger)
    # Strenger bij herhaling: de bron wordt intern, daarna is het hetzelfde verzoek.
    assert resp.status_code == 200 and resp.json()["classification"] == "intern"


async def _gelijktijdige_versies(env, app, n: int = 12) -> None:
    regel = await _regel()
    extern_id = f"stuk-{uuid.uuid4().hex}"
    o0, h0 = env.origineel(b"%PDF-1.7 v0 " + uuid.uuid4().hex.encode())
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=o0, content_hash=h0, regel=regel, external_id=extern_id))
        assert resp.status_code == 201
        versies = [env.origineel(b"%PDF-1.7 v " + uuid.uuid4().hex.encode()) for _ in range(n)]
        antwoorden = await asyncio.gather(*[client.post("/api/v1/import", json=_verzoek(
            run, original=o, content_hash=h, regel=regel, external_id=extern_id))
            for o, h in versies])
    codes = [r.status_code for r in antwoorden]
    assert set(codes) <= {201, 409}, codes
    rijen = await _versies(resp.json()["source_id"])
    # Unieke nummers 1..n zonder gaten en een lineaire keten van `predecessor`.
    assert [v["version_no"] for v in rijen] == list(range(1, 2 + codes.count(201)))
    assert rijen[0].get("predecessor") is None
    for vorige, versie in zip(rijen, rijen[1:]):
        assert str(versie["predecessor"]) == str(vorige["id"])
    bron = await _een("SELECT current_version, version_seq FROM type::thing($b)",
                      {"b": resp.json()["source_id"]})
    assert str(bron["current_version"]) == str(rijen[-1]["id"])
    assert bron["version_seq"] == len(rijen)
    for versie in rijen:
        for tabel in ("chunk", "doc_node"):
            [n] = await _q(f"SELECT count() AS n FROM {tabel} WHERE source_version = $v GROUP ALL",
                           {"v": conn.ensure_record_id(str(versie["id"]))})
            assert n["n"] >= 2


async def test_gelijktijdige_nieuwe_versies(env, app) -> None:
    await _gelijktijdige_versies(env, app)


async def test_gelijktijdige_nieuwe_versies_rocksdb_zonder_slot(env_rocksdb, app,
                                                                 monkeypatch) -> None:
    """Op rocksdb en zonder het slot in het proces: de database serialiseert."""
    monkeypatch.setattr(import_service, "_slot", lambda _e: asyncio.Lock())
    for _ in range(3):
        await _gelijktijdige_versies(env_rocksdb, app, 16)


# --------------------------------------------------------------------------
# Herreview F2-14
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batchsoort", ["intern", "extern_onbevestigd"])
async def test_inhoud_in_een_interne_batch_telt_als_intern(env, app, batchsoort) -> None:
    bulk = await _regel("bulkupload", "intern", location=str(env.inname))
    extern = await _regel()
    original, h = env.origineel(b"%PDF-1.7 in interne batch " + uuid.uuid4().hex.encode())
    pad = str(env.inname / f"{uuid.uuid4().hex}.pdf")
    if batchsoort == "intern":
        await _batch(bulk, [pad], "intern", [h])
    else:
        await _q("CREATE intake_batch CONTENT $d", {"d": {
            "route": "inname_pdf", "files": [pad], "classification": "extern",
            "created_by": "Testpersoon", "source_map_rule": conn.ensure_record_id(bulk),
            "file_hashes": [{"path": pad, "content_hash": h}]}})
    async with _client(app) as client:
        run = await _start(client, env, regel=extern)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=extern))
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "inhoud_intern_bekend"


async def _meldingen(bron: str) -> list:
    return await _q("SELECT * FROM beheer_melding WHERE source = type::thing($b)", {"b": bron})


async def test_interne_import_maakt_externe_bron_met_dezelfde_inhoud_intern(env, app) -> None:
    intern = await _regel(bevestigd=False)
    extern = await _regel()
    original, h = env.origineel(b"%PDF-1.7 eerst extern " + uuid.uuid4().hex.encode())
    async with _client(app) as client:
        run = await _start(client, env, regel=extern)
        eerste = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=extern))
        assert eerste.status_code == 201 and eerste.json()["classification"] == "extern"
        run = await _start(client, env, regel=intern)
        tweede = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=intern))
    assert tweede.status_code == 201 and tweede.json()["classification"] == "intern"
    bron = eerste.json()["source_id"]
    assert (await _een("SELECT VALUE classification FROM type::thing($b)", {"b": bron})) == "intern"
    waarden = await _q("SELECT VALUE classification FROM chunk WHERE source = type::thing($b)",
                       {"b": bron})
    assert set(waarden) == {"intern"}
    assert await _q("SELECT id FROM chunk_extern WHERE chunk.source = type::thing($b)",
                    {"b": bron}) == []
    [melding] = await _meldingen(bron)
    assert melding["soort"] == "inhoud_intern"


async def test_intern_besluit_over_een_item_maakt_externe_bron_intern(env, app) -> None:
    extern = await _regel()
    original, h = env.origineel(b"%PDF-1.7 item later intern " + uuid.uuid4().hex.encode())
    async with _client(app) as client:
        run = await _start(client, env, regel=extern)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=extern))
    assert resp.status_code == 201 and resp.json()["classification"] == "extern"
    await _item(str(env.inname / f"{uuid.uuid4().hex}.pdf"), h, besluit="intern")
    bron = resp.json()["source_id"]
    assert (await _een("SELECT VALUE classification FROM type::thing($b)", {"b": bron})) == "intern"
    assert [m["soort"] for m in await _meldingen(bron)] == ["inhoud_intern"]


@pytest.mark.parametrize("variant", [
    "<!-- signaal​systeem:begin -->", "<!--​signaalsysteem:begin-->",
    "<!-- signaalsysteem:begin­ -->", "<!-- signaalsysteem:begin --!>",
    "&amp;amp;amp;amp;lt;!-- signaalsysteem:begin --&gt;", "LAGEN"])
async def test_markering_na_opmaaktekens_en_lagen(env, app, variant) -> None:
    if variant == "LAGEN":
        variant = "&" + "amp;" * 11 + "lt;!-- signaalsysteem:begin --&gt;"
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 schoon " + uuid.uuid4().hex.encode())
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel, teksten=["a", f"b {variant} c"]))
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "afgeleide_tekst"


@pytest.mark.parametrize("plek", ["titel_tekst", "extractor", "poortwachter", "type",
                                  "feed_item_id", "sectie_gesplitst"])
async def test_markering_in_overige_velden(env, app, plek) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 schoon " + uuid.uuid4().hex.encode())
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        v = _verzoek(run, original=original, content_hash=h, regel=regel)
        if plek == "titel_tekst":
            v["source"]["title"] = "x <!-- signaal"
            v["chunks"][0]["text"] = "systeem:begin --> y"
        elif plek == "extractor":
            v["extraction"]["version"] = MARKERING
        elif plek == "poortwachter":
            v["classification"]["gatekeeper"]["method"] = MARKERING
        elif plek == "type":
            v["source"]["type"] = MARKERING
        elif plek == "feed_item_id":
            v["source_version"]["feed_item_id"] = MARKERING
        else:
            v["chunks"][0]["section_path"] = ["<!-- signaal", "systeem:begin -->"]
        resp = await client.post("/api/v1/import", json=v)
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "afgeleide_tekst"


async def test_markering_in_ruwe_inhoud_utf16_zonder_bom(env, app) -> None:
    regel = await _regel()
    original, h = env.origineel(("Notitie " + MARKERING + uuid.uuid4().hex).encode("utf-16-le"))
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel))
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "afgeleide_tekst"


@pytest.mark.parametrize("locatie", [
    "https://evil.net\\@example.org/api/x", "https://user:pw@example.org/api/x",
    "https://example.org/api/..;/geheim", "https://example.org/api/%252e%252e/geheim"])
async def test_url_omwegen_worden_geweigerd(env, app, locatie) -> None:
    regel = await _regel()
    original, h = env.origineel(b"%PDF-1.7 omweg " + uuid.uuid4().hex.encode())
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel, location=locatie))
    assert resp.status_code in (409, 422)
    assert resp.json()["detail"]["code"] in ("locatie", "niet_gebonden")


@pytest.mark.parametrize("locatie, regel, verwacht", [
    ("https://example.org/api/x", "https://example.org/api", True),
    ("https://example.org/api%2Fx", "https://example.org/api", False),
    ("https://example.org/api?feed=geheim", "https://example.org/api?feed=open", False),
    ("https://example.org/api?feed=open", "https://example.org/api?feed=open", True),
    ("https://example.org/api/x?y=1", "https://example.org/api", True),
    ("https://user@example.org/api/x", "https://example.org/api", False),
])
async def test_url_binding_aan_de_regel(locatie, regel, verwacht) -> None:
    from app_main.services.inname_binding import locatie_hoort_bij_regel

    assert locatie_hoort_bij_regel(locatie, regel) is verwacht


@pytest.mark.parametrize("geval", ["andere_inhoud", "schrijfbaar"])
async def test_bestaand_archiefbestand_wordt_gecontroleerd(env, app, geval) -> None:
    regel = await _regel()
    inhoud = b"%PDF-1.7 echt " + uuid.uuid4().hex.encode()
    original, h = env.origineel(inhoud)
    env.forkarchief.mkdir(parents=True, exist_ok=True)
    bestaand = env.forkarchief / h.removeprefix("sha256:")
    bestaand.write_bytes(b"geknoeid" if geval == "andere_inhoud" else inhoud)
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=regel))
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "archief_onjuist"
    assert any(r["status"] == "archief_onjuist" for r in (await _run(run))["log"])


async def test_origineel_via_een_gekoppelde_map_wordt_geweigerd(env, app, tmp_path) -> None:
    regel = await _regel()
    buiten = tmp_path / "buiten"
    buiten.mkdir()
    (buiten / "stuk.bin").write_bytes(b"buiten het archief")
    os.symlink(buiten, env.archief / "map")
    h = "sha256:" + hashlib.sha256(b"buiten het archief").hexdigest()
    async with _client(app) as client:
        run = await _start(client, env, regel=regel)
        for original in ("bronarchief/map/stuk.bin",):
            resp = await client.post("/api/v1/import", json=_verzoek(
                run, original=original, content_hash=h, regel=regel))
            assert resp.status_code == 422 and resp.json()["detail"]["code"] == "origineel"
        os.symlink(buiten / "stuk.bin", env.archief / "koppeling.bin")
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original="bronarchief/koppeling.bin", content_hash=h, regel=regel))
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "origineel"


async def test_item_verwerkt_alleen_met_een_run_van_dezelfde_sleutel(env, app) -> None:
    regel = await _regel("watchfolder", "intern", location=str(env.inname))
    _, h = env.origineel(b"%PDF-1.7 sleutel " + uuid.uuid4().hex.encode())
    item = await _item(str(env.inname / f"{uuid.uuid4().hex}.pdf"), h)
    async with _client(app) as client:
        env.als(ANDERE)
        run = await _start(client, env, regel=regel, item=item)
        env.als(LANGFLOW)
        rev = await _een("SELECT VALUE rev FROM type::thing($id)", {"id": item})
        resp = await client.patch(f"/api/v1/intake/items/{item}",
                                  json={"status": "verwerkt", "rev": rev, "run_id": run})
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "niet_gebonden"


# --------------------------------------------------------------------------
# Derde review F2-14
# --------------------------------------------------------------------------


async def _race_extern_tegen_intern(env, app, monkeypatch) -> None:
    """De externe import heeft de controle buiten de transactie al gehaald als de
    interne import van dezelfde inhoud slaagt; daarna mag ze niet extern
    schrijven (de controle in de transactie)."""
    from surrealdb_service.repositories.source import SourceRepository

    monkeypatch.setattr(import_service, "_slot", lambda _e: asyncio.Lock())
    extern = await _regel()
    intern = await _regel(bevestigd=False)
    original, h = env.origineel(b"%PDF-1.7 race " + uuid.uuid4().hex.encode())
    origineel = SourceRepository.inhoud_intern_bekend
    gecontroleerd, doorgaan = asyncio.Event(), asyncio.Event()

    async def traag(self, hexhash):
        uit = await origineel(self, hexhash)
        if not gecontroleerd.is_set():
            gecontroleerd.set()
            await doorgaan.wait()
        return uit

    monkeypatch.setattr(SourceRepository, "inhoud_intern_bekend", traag)
    async with _client(app) as client:
        run_e = await _start(client, env, regel=extern)
        run_i = await _start(client, env, regel=intern)
        taak = asyncio.create_task(client.post("/api/v1/import", json=_verzoek(
            run_e, original=original, content_hash=h, regel=extern)))
        await gecontroleerd.wait()
        intern_resp = await client.post("/api/v1/import", json=_verzoek(
            run_i, original=original, content_hash=h, regel=intern))
        doorgaan.set()
        extern_resp = await taak
    assert intern_resp.status_code == 201 and intern_resp.json()["classification"] == "intern"
    assert extern_resp.status_code == 409, extern_resp.text
    assert extern_resp.json()["detail"]["code"] == "inhoud_intern_bekend"


async def test_race_extern_tegen_intern(env, app, monkeypatch) -> None:
    await _race_extern_tegen_intern(env, app, monkeypatch)


async def test_race_extern_tegen_intern_rocksdb(env_rocksdb, app, monkeypatch) -> None:
    await _race_extern_tegen_intern(env_rocksdb, app, monkeypatch)


async def test_gelijktijdig_extern_en_intern_rocksdb(env_rocksdb, app, monkeypatch) -> None:
    """Echt gelijktijdig: nooit een externe bron naast een interne met dezelfde inhoud."""
    monkeypatch.setattr(import_service, "_slot", lambda _e: asyncio.Lock())
    extern = await _regel()
    intern = await _regel(bevestigd=False)
    async with _client(app) as client:
        for _ in range(5):
            original, h = env_rocksdb.origineel(b"%PDF-1.7 tegelijk " + uuid.uuid4().hex.encode())
            runs = [(await _start(client, env_rocksdb, regel=r), r)
                    for r in (extern, intern, extern, intern)]
            antwoorden = await asyncio.gather(*[client.post("/api/v1/import", json=_verzoek(
                run, original=original, content_hash=h, regel=r)) for run, r in runs])
            assert {a.status_code for a in antwoorden} <= {201, 409}
            klassen = await _q("SELECT VALUE classification FROM source WHERE"
                               " current_version.content_hash = $h", {"h": h.removeprefix("sha256:")})
            assert set(klassen) == {"intern"}, klassen


async def test_interne_import_maakt_ook_oudere_versie_intern(env, app) -> None:
    extern = await _regel()
    intern = await _regel(bevestigd=False)
    woord = f"Oudeversiewoord{uuid.uuid4().hex[:8]}"
    o1, h1 = env.origineel(b"%PDF-1.7 oud " + uuid.uuid4().hex.encode())
    o2, h2 = env.origineel(b"%PDF-1.7 nieuw " + uuid.uuid4().hex.encode())
    extern_id = f"stuk-{uuid.uuid4().hex}"
    async with _client(app) as client:
        run = await _start(client, env, regel=extern)
        v1 = await client.post("/api/v1/import", json=_verzoek(
            run, original=o1, content_hash=h1, regel=extern, external_id=extern_id,
            teksten=[f"{woord} eerste.", "twee"]))
        v2 = await client.post("/api/v1/import", json=_verzoek(
            run, original=o2, content_hash=h2, regel=extern, external_id=extern_id))
        assert v1.status_code == 201 and v2.status_code == 201
        assert await _q("RETURN fn::text_search($w, 10, ['extern'], true)", {"w": woord})
        run2 = await _start(client, env, regel=intern)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run2, original=o1, content_hash=h1, regel=intern))
    assert resp.status_code == 201 and resp.json()["classification"] == "intern"
    oud = await _q("SELECT VALUE classification FROM chunk WHERE source_version = type::thing($v)",
                   {"v": v1.json()["source_version_id"]})
    assert set(oud) == {"intern"}
    assert await _q("RETURN fn::text_search($w, 10, ['extern'], true)", {"w": woord}) == []
    assert len(await _meldingen(v1.json()["source_id"])) == 1


async def test_een_melding_per_geraakte_bron(env, app) -> None:
    extern = await _regel()
    original, h = env.origineel(b"%PDF-1.7 twee bronnen " + uuid.uuid4().hex.encode())
    bronnen = []
    async with _client(app) as client:
        for _ in range(2):
            run = await _start(client, env, regel=extern)
            resp = await client.post("/api/v1/import", json=_verzoek(
                run, original=original, content_hash=h, regel=extern))
            assert resp.status_code == 201, resp.text
            bronnen.append(resp.json()["source_id"])
    await _item(str(env.inname / f"{uuid.uuid4().hex}.pdf"), h, besluit="intern")
    for bron in bronnen:
        assert [m["soort"] for m in await _meldingen(bron)] == ["inhoud_intern"]


async def test_batch_die_later_intern_wordt(env, app) -> None:
    bulk = await _regel("bulkupload", "intern", location=str(env.inname))
    original, h = env.origineel(b"%PDF-1.7 batch later " + uuid.uuid4().hex.encode())
    pad = str(env.inname / f"{uuid.uuid4().hex}.pdf")
    batch = await _batch(bulk, [pad], "extern", [h])
    async with _client(app) as client:
        run = await _start(client, env, regel=bulk, batch=batch)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=bulk, batch=batch, location=pad))
    assert resp.status_code == 201 and resp.json()["classification"] == "extern"
    await _q("UPDATE type::thing($b) SET classification_confirmed_by = NONE,"
             " classification_confirmed_at = NONE", {"b": batch})
    bron = resp.json()["source_id"]
    assert (await _een("SELECT VALUE classification FROM type::thing($b)", {"b": bron})) == "intern"
    assert [m["soort"] for m in await _meldingen(bron)] == ["inhoud_intern"]


async def test_interne_batch_zonder_passende_hash_geeft_een_melding(env, app) -> None:
    bulk = await _regel("bulkupload", "intern", location=str(env.inname))
    original, h = env.origineel(b"%PDF-1.7 onleesbaar " + uuid.uuid4().hex.encode())
    pad = str(env.inname / f"{uuid.uuid4().hex}.pdf")  # bestaat niet
    batch = await _batch(bulk, [pad], "intern")
    async with _client(app) as client:
        run = await _start(client, env, regel=bulk, batch=batch)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=original, content_hash=h, regel=bulk, batch=batch, location=pad))
    assert resp.status_code == 201 and resp.json()["classification"] == "intern"
    meldingen = await _q("SELECT VALUE soort FROM beheer_melding WHERE intake_batch = type::thing($b)",
                         {"b": batch})
    assert meldingen == ["batchhash_ontbreekt"]
    # Is het bestand er wel en past het, dan geen melding.
    (env.inname / "leesbaar.pdf").write_bytes(b"%PDF-1.7 leesbaar")
    o2, h2 = env.origineel(b"%PDF-1.7 leesbaar")
    pad2 = str(env.inname / "leesbaar.pdf")
    batch2 = await _batch(bulk, [pad2], "intern")
    async with _client(app) as client:
        run = await _start(client, env, regel=bulk, batch=batch2)
        resp = await client.post("/api/v1/import", json=_verzoek(
            run, original=o2, content_hash=h2, regel=bulk, batch=batch2, location=pad2))
    assert resp.status_code == 201
    assert await _q("SELECT id FROM beheer_melding WHERE intake_batch = type::thing($b)",
                    {"b": batch2}) == []
