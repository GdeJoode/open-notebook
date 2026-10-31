"""Auditlog van modelaanroepen (F2-07, ADR 0003, 0007, plan §5.4).

Tegen een echte SurrealDB:

* ``POST /api/v1/model-calls`` schrijft één regel; prompt- of antwoordtekst
  geeft 422 en schrijft niets;
* een uitgevoerde aanroep moet een model uit het register noemen;
* alleen de aanmaker van de run schrijft regels bij die run (F2-14);
* ``processing_run.models`` volgt ``model_call``, ook na nieuwe regels en ook
  als Langflow een andere lijst meestuurt;
* ``ModelGate`` in de fork schrijft via :class:`ModelCallTelemetry`;
* gelijktijdige regels bij één run gaan niet verloren (rocksdb).

Alle testdata is synthetisch.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.api.routers.model_calls import get_model_call_telemetry
from app_main.dependencies import get_processing_run_service
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.model_routing.telemetry import ModelCallTelemetry
from app_main.services.processing_run_service import ProcessingRunService
from llm_manager.gate import ModelCallRegel, ModelGate, ModelGeweigerd
from llm_manager.register import ModelRegister, RegisterModel, get_register
from shared.models.agents import Right
from surrealdb_service.repositories import ProcessingRunRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

GEHEIM = "Interne tekst die nooit in het auditlog mag"


@pytest.fixture(scope="module")
def app():
    return create_app()


class Sleutel:
    def __init__(self) -> None:
        self.ctx = agent_auth.AgentKeyContext(
            key_id="agent_keys:langflow", agent_id="langflow", permission=Right.PIJPLIJN)


async def _omgeving(db, monkeypatch, app, git_flows):
    monkeypatch.setattr(surreal_config, "_config", db)
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    sleutel = Sleutel()

    async def _pijplijn(_request):
        return sleutel.ctx

    monkeypatch.setattr(agent_auth, "authenticate_request", _pijplijn)
    reg = git_flows.register
    app.dependency_overrides[get_processing_run_service] = lambda: ProcessingRunService(
        flow_register=reg)
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=db)
    yield sleutel
    app.dependency_overrides.pop(get_processing_run_service, None)
    await conn._pool.close_all()
    conn._pool = original or conn.ConnectionPool(config=db)


@pytest.fixture
async def env(live_surrealdb, monkeypatch, app, git_flows):
    async for sleutel in _omgeving(live_surrealdb, monkeypatch, app, git_flows):
        yield sleutel


@pytest.fixture
async def env_rocksdb(live_surrealdb_rocksdb, monkeypatch, app, git_flows):
    async for sleutel in _omgeving(live_surrealdb_rocksdb, monkeypatch, app, git_flows):
        yield sleutel


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                             headers={"Authorization": "Bearer ak_test"})


async def _start(client, flow) -> str:
    resp = await client.post("/api/v1/runs", json={
        "trigger": "schema", "pipeline_id": "inname_pdf",
        "pipeline_version": {"git_commit": flow.head, "flow_hash": flow.hash}})
    assert resp.status_code == 201, resp.text
    return resp.json()["run_id"]


def _regel(**velden) -> dict:
    regel = {"model": "werkmodel", "invocation": "llama-swap", "location": "lokaal",
             "task": "entity_extraction", "classification": "intern", "status": "geslaagd",
             "input_tokens": 100, "output_tokens": 20, "duration_ms": 900}
    regel.update(velden)
    return regel


async def _regels(run_id: str) -> list[dict]:
    return await conn.execute_query("SELECT * FROM model_call WHERE run = $r",
                                    {"r": conn.ensure_record_id(run_id)})


async def test_regel_van_langflow_en_models_van_de_run(env, app, git_flows) -> None:
    async with _client(app) as client:
        run_id = await _start(client, git_flows)
        resp = await client.post("/api/v1/model-calls", json=_regel(run=run_id))
        assert resp.status_code == 201, resp.text
        [regel] = await _regels(run_id)
        assert regel["origin"] == "langflow" and regel["task"] == "entity_extraction"
        assert regel["input_tokens"] == 100 and regel["status"] == "geslaagd"

        # `models` volgt elke nieuwe regel, ook vóór het afsluiten.
        run = (await client.get(f"/api/v1/runs/{run_id}")).json()
        assert [m["model"] for m in run["models"]] == ["werkmodel"]

        # Bij het afsluiten telt model_call, niet de opgave van Langflow.
        resp = await client.patch(f"/api/v1/runs/{run_id}", json={
            "status": "mislukt", "models": [{"name": "ander-model", "version": "x"}]})
        run = resp.json()
        assert [(m["model"], m["invocation"]) for m in run["models"]] == [
            ("werkmodel", "llama-swap")]
        assert any(r["step"] == "models" and r["status"] == "waarschuwing" for r in run["log"])

        # Een regel na het afsluiten werkt `models` ook bij (ADR 0003).
        resp = await client.post("/api/v1/model-calls", json=_regel(
            run=run_id, model="embed", task="embedding"))
        assert resp.status_code == 201
        run = (await client.get(f"/api/v1/runs/{run_id}")).json()
        assert sorted(m["model"] for m in run["models"]) == ["embed", "werkmodel"]


@pytest.mark.parametrize("veld", ["prompt", "response", "text", "messages"])
async def test_tekst_geweigerd(env, app, git_flows, veld) -> None:
    async with _client(app) as client:
        run_id = await _start(client, git_flows)
        resp = await client.post("/api/v1/model-calls", json=_regel(run=run_id, **{veld: GEHEIM}))
    assert resp.status_code == 422
    assert await _regels(run_id) == []


async def test_vrije_tekst_in_codevelden_geweigerd(env, app) -> None:
    async with _client(app) as client:
        for fout in ({"task": GEHEIM}, {"error_code": GEHEIM}, {"model": GEHEIM}):
            resp = await client.post("/api/v1/model-calls", json=_regel(**fout))
            assert resp.status_code == 422


async def test_uitgevoerde_aanroep_vraagt_een_model_uit_het_register(env, app) -> None:
    async with _client(app) as client:
        resp = await client.post("/api/v1/model-calls", json=_regel(model="bestaat-niet"))
        assert resp.status_code == 422 and resp.json()["detail"]["code"] == "model_niet_in_register"
        resp = await client.post("/api/v1/model-calls", json=_regel(invocation="api",
                                                                     location="cloud"))
        assert resp.status_code == 422
        # Een weigering van een onbekend model wordt wel vastgelegd.
        resp = await client.post("/api/v1/model-calls", json=_regel(
            model="bestaat-niet", status="geweigerd", attempts=0, error_code="model_onbekend"))
        assert resp.status_code == 201
        [rij] = await conn.execute_query("SELECT model FROM $id",
                                         {"id": conn.ensure_record_id(resp.json()["id"])})
        assert rij["model"] == "ongeldige_naam"
        # Ook een weigering moet aanroepwijze en locatie uit het register noemen.
        resp = await client.post("/api/v1/model-calls", json=_regel(
            invocation="api", location="cloud", status="geweigerd", attempts=0,
            error_code="niet_toegestaan_intern"))
        assert resp.status_code == 422


async def test_versie_komt_uit_het_register(env, app, git_flows) -> None:
    async with _client(app) as client:
        run_id = await _start(client, git_flows)
        resp = await client.post("/api/v1/model-calls", json=_regel(
            run=run_id, model_version="een-andere-versie"))
        assert resp.status_code == 201
        resp = await client.post("/api/v1/model-calls", json=_regel(
            run=run_id, model_version="v1 met vrije tekst"))
        assert resp.status_code == 422
        run = (await client.get(f"/api/v1/runs/{run_id}")).json()
    [regel] = await _regels(run_id)
    assert regel["model_version"] == get_register().zoek("werkmodel").versie
    assert run["models"][0]["model_version"] == regel["model_version"]
    assert any(r["step"] == "model_call" and r["status"] == "waarschuwing" for r in run["log"])


async def test_cloudaanroep_met_interne_invoer_geeft_een_melding(env, app, git_flows) -> None:
    register = ModelRegister(versie=1, modellen=[*get_register().modellen, RegisterModel(
        naam="cloudmodel", versie="2026-10", aanroepwijze="api", locatie="cloud",
        taken=["chat"])])
    app.dependency_overrides[get_model_call_telemetry] = lambda: ModelCallTelemetry(
        register=register)
    try:
        async with _client(app) as client:
            run_id = await _start(client, git_flows)
            for klasse, status in (("intern", "geslaagd"), ("onbekend", "mislukt"),
                                   ("extern", "geslaagd"), ("intern", "geweigerd")):
                resp = await client.post("/api/v1/model-calls", json=_regel(
                    run=run_id, model="cloudmodel", invocation="api", location="cloud",
                    task="chat", classification=klasse, status=status))
                assert resp.status_code == 201
    finally:
        app.dependency_overrides.pop(get_model_call_telemetry, None)
    assert len(await _regels(run_id)) == 4
    meldingen = await conn.execute_query(
        "SELECT soort, model_call FROM beheer_melding WHERE run = $r",
        {"r": conn.ensure_record_id(run_id)})
    assert [m["soort"] for m in meldingen] == ["cloudaanroep_intern"] * 2


@pytest.mark.echte_sleutels
async def test_alleen_de_aanmaker_van_de_run(env, app, git_flows) -> None:
    async with _client(app) as client:
        run_id = await _start(client, git_flows)
        env.ctx = agent_auth.AgentKeyContext(
            key_id="agent_keys:andere", agent_id="ander", permission=Right.PIJPLIJN)
        resp = await client.post("/api/v1/model-calls", json=_regel(run=run_id))
        assert resp.status_code == 403
        resp = await client.post("/api/v1/model-calls",
                                 json=_regel(run="processing_run:bestaat_niet"))
        assert resp.status_code == 404
    assert await _regels(run_id) == []


async def test_modelgate_in_de_fork_schrijft_de_auditregel(env) -> None:
    [run] = await conn.execute_query("CREATE processing_run CONTENT {trigger: 'inname',"
                                     " pipeline_id: 'p', flow_vastgelegd: true} RETURN id")
    run_id = str(run["id"])
    gate = ModelGate(get_register(), ModelCallTelemetry())

    async def antwoord(_model):
        return {"tekst": GEHEIM, "usage": {"prompt_tokens": 5, "completion_tokens": 1}}

    await gate.roep_aan(model="werkmodel", classificatie="intern", taak="chat",
                        uitvoer=antwoord, run_id=run_id)
    with pytest.raises(ModelGeweigerd):
        await gate.roep_aan(model="bestaat-niet", classificatie="intern", taak="chat",
                            uitvoer=antwoord, run_id=run_id)
    regels = await _regels(run_id)
    assert sorted(r["status"] for r in regels) == ["geslaagd", "geweigerd"]
    assert all(r["origin"] == "fork" for r in regels)
    assert GEHEIM not in str(regels)
    models = (await ProcessingRunRepository().get(run_id)).models
    assert [m["model"] for m in models] == ["werkmodel"]


async def test_gelijktijdige_regels_bij_een_run(env_rocksdb) -> None:
    [run] = await conn.execute_query("CREATE processing_run CONTENT {trigger: 'inname',"
                                     " pipeline_id: 'p', flow_vastgelegd: true} RETURN id")
    run_id = str(run["id"])
    telemetry = ModelCallTelemetry()
    modellen = ["werkmodel", "embed"] * 6
    await asyncio.gather(*(
        telemetry.schrijf(ModelCallRegel(**_regel(run=run_id, model=m, origin="fork")))
        for m in modellen))
    assert len(await _regels(run_id)) == len(modellen)
    # Een volgende regel brengt `models` in elk geval gelijk met model_call.
    await telemetry.schrijf(ModelCallRegel(**_regel(run=run_id, origin="fork")))
    models = (await ProcessingRunRepository().get(run_id)).models
    assert sorted(m["model"] for m in models) == ["embed", "werkmodel"]
