"""Stap-eindpunten en ``RunStepOrchestrator`` (F2-15, ADR 0011, 0003, 0008, plan §5.4).

Stopconditie:

* een stap maakt een job met ``run``;
* een mislukte job wordt opnieuw geprobeerd tot ``max_retries`` en zet daarna
  run en stap op ``mislukt`` met een melding (zonder inhoud);
* een stap op een onbekende run geeft 404;
* embeddings van de stap hebben ``run`` en ``embedding_model`` (en ``afgekapt``);
* een stapaanroep zet een run die te lang ``bezig`` staat op ``mislukt``, maar
  niet een run met een stap in ``wacht_op_review``.

Tegen een echte SurrealDB (migraties tot en met 104), met de sleutel
``pijplijn``. De jobs draaien in een eigen wachtrij met een eigen worker; er
draait geen llama-swap: een ``httpx.MockTransport`` speelt het na, en de
auditregels gaan via ``ModelCallTelemetry`` naar ``model_call``. Onder
SurrealDB v2.7.0 laat een SCHEMAFULL-tabel een onbekend veld stil weg; daarom
leest elke test terug wat er is geschreven. Alle testdata is synthetisch.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.dependencies import get_processing_run_service, get_run_step_orchestrator
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.model_routing.telemetry import ModelCallTelemetry
from app_main.services.processing_run_service import ProcessingRunService
from app_main.services.run_step_orchestrator import MAX_HERHALINGEN, RunStepOrchestrator
from job_queue import HandlerRegistry, JobPausedForReviewError, JobQueue, JobRepository
from job_queue import JobService, JobWorker
from llm_manager import aanroep
from llm_manager.providers.factory import LlamaSwapClient
from llm_manager.register import get_register
from shared.models.agents import Right
from shared.types.enums import JobType
from surrealdb_service.repositories.source_version import SourceVersionRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

LANGFLOW = agent_auth.AgentKeyContext(
    key_id="agent_keys:langflow", agent_id="langflow", permission=Right.PIJPLIJN)
ANDERE = agent_auth.AgentKeyContext(
    key_id="agent_keys:andere", agent_id="andere", permission=Right.PIJPLIJN)
GEHEIM = "Interne zin die nooit in een melding mag"


@pytest.fixture(scope="module")
def app():
    return create_app()


class LlamaSwap:
    """Speelt ``/v1/embeddings`` van llama-swap na: 1 024 dimensies."""

    def __init__(self) -> None:
        self.invoer: List[List[str]] = []

    def __call__(self, verzoek: httpx.Request) -> httpx.Response:
        teksten = json.loads(verzoek.content)["input"]
        self.invoer.append(teksten)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [0.25] * 1024} for i in range(len(teksten))]})


@pytest.fixture
async def env(live_surrealdb, monkeypatch, app, git_flows):
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    sleutel = SimpleNamespace(ctx=LANGFLOW)

    async def _auth(_request):
        return sleutel.ctx

    monkeypatch.setattr(agent_auth, "authenticate_request", _auth)
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)

    server = LlamaSwap()
    aanroep.configureer(ModelCallTelemetry(), client=LlamaSwapClient(
        transport=httpx.MockTransport(server)))

    # Een eigen wachtrij en worker, zodat de test de pogingen zelf draait.
    queue, registry = JobQueue(), HandlerRegistry()
    repo = JobRepository(live_surrealdb)
    worker = JobWorker(queue, repo, registry)
    jobs = JobService(repo, queue, worker)
    runs = ProcessingRunService(flow_register=git_flows.register)
    orch = RunStepOrchestrator(run_service=runs, jobs=jobs)
    registry.register_handler(JobType.RUN_STEP, orch.voer_uit)
    registry.register_final_failure(JobType.RUN_STEP, orch.bij_definitief_mislukt)
    app.dependency_overrides[get_run_step_orchestrator] = lambda: orch
    app.dependency_overrides[get_processing_run_service] = lambda: runs

    async def draai() -> None:
        while not queue.is_empty():
            await worker._execute_job(await queue.dequeue())

    def als(ctx) -> None:
        sleutel.ctx = ctx

    yield SimpleNamespace(orch=orch, draai=draai, als=als, server=server,
                          flows=git_flows)
    app.dependency_overrides.pop(get_run_step_orchestrator, None)
    app.dependency_overrides.pop(get_processing_run_service, None)
    aanroep.reset()
    await conn._pool.close_all()
    conn._pool = original or conn.ConnectionPool(config=live_surrealdb)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                             headers={"Authorization": "Bearer ak_test"})


async def _q(sql: str, params: dict | None = None):
    return await conn.execute_query(sql, params or {})


async def _een(sql: str, params: dict | None = None) -> dict:
    rows = await _q(sql, params)
    return rows[0] if isinstance(rows, list) else rows


async def _start_run(client, env) -> str:
    resp = await client.post("/api/v1/runs", json={
        "trigger": "schema", "pipeline_id": "inname_pdf",
        "pipeline_version": {"git_commit": env.flows.head, "flow_hash": env.flows.hash}})
    assert resp.status_code == 201, resp.text
    return resp.json()["run_id"]


async def _versie(run: str, teksten: list[str], classificatie: str = "intern") -> tuple[str, str]:
    """Een bron met een versie en fragmenten, geschreven in ``run``."""
    [bron] = await _q("CREATE source SET title = 'Openbare testbron', classification = $c"
                      " RETURN id", {"c": classificatie})
    bron = str(bron["id"])
    versie, _ = await SourceVersionRepository().register(
        bron, hashlib.sha256(uuid.uuid4().bytes).hexdigest(), run)
    for i, tekst in enumerate(teksten):
        await _q("CREATE chunk CONTENT $c", {"c": {
            "source": conn.ensure_record_id(bron), "text": tekst, "order": i,
            "physical_page": 1, "element_type": "paragraph", "positions": [], "metadata": {},
            "source_version": conn.ensure_record_id(str(versie.id)),
            "run": conn.ensure_record_id(run)}})
    return bron, str(versie.id)


async def _post(client, run: str, stap: str, versie: str) -> httpx.Response:
    return await client.post(f"/api/v1/runs/{run}/steps/{stap}",
                             json={"source_version_id": versie})


async def _run(run: str) -> dict:
    return await _een("SELECT * FROM ONLY $id", {"id": conn.ensure_record_id(run)})


# --------------------------------------------------------------------------
# Stopconditie
# --------------------------------------------------------------------------


async def test_een_stap_maakt_een_job_met_run(env, app) -> None:
    async with _client(app) as client:
        run = await _start_run(client, env)
        _, versie = await _versie(run, ["Openbare tekst over de gemeenteraad."])
        resp = await _post(client, run, "embed", versie)
        assert resp.status_code == 202, resp.text
        stap = resp.json()
        # Dezelfde stap op dezelfde versie in dezelfde run: dezelfde stap en job.
        weer = (await _post(client, run, "embed", versie)).json()

    assert stap["status"] == "wachtrij" and stap["job_id"]
    assert (weer["step_id"], weer["job_id"]) == (stap["step_id"], stap["job_id"])
    job = await _een("SELECT * FROM ONLY $id", {"id": conn.ensure_record_id(stap["job_id"])})
    assert str(job["run"]) == run
    assert job["job_type"] == "run_step" and job["max_retries"] == MAX_HERHALINGEN
    assert len(await _q("SELECT id FROM job WHERE run = $r",
                        {"r": conn.ensure_record_id(run)})) == 1


async def test_embeddings_van_de_stap_hebben_run_en_model(env, app) -> None:
    kort, lang = "Openbare tekst over de begroting.", "Lange openbare tekst. " * 100
    async with _client(app) as client:
        run = await _start_run(client, env)
        bron, versie = await _versie(run, [kort, lang])
        stap = (await _post(client, run, "embed", versie)).json()
        await env.draai()
        gelezen = (await client.get(f"/api/v1/runs/{run}/steps/embed")).json()

    assert gelezen["status"] == "geslaagd" and gelezen["attempts"] == 1
    assert gelezen["step_id"] == stap["step_id"]
    model = get_register().voor_taak("embedding")
    rijen = await _q("SELECT run, embedding_model, afgekapt, content, chunk.order AS nr"
                     " FROM source_embedding WHERE source = $b ORDER BY nr",
                     {"b": conn.ensure_record_id(bron)})
    assert [(str(r["run"]), r["embedding_model"], r["afgekapt"]) for r in rijen] == [
        (run, f"{model.naam}:{model.versie}", False),
        (run, f"{model.naam}:{model.versie}", True)]
    assert rijen[1]["content"] == lang  # het fragment blijft heel; de invoer is afgekapt
    assert len(env.server.invoer[0][1]) < len(lang)

    # De run kreeg de modellen uit `model_call` (ADR 0003, 0007) en een logregel.
    gelezen_run = await _run(run)
    assert [m["model"] for m in gelezen_run["models"]] == [model.naam]
    stappen = [(r["step"], r["status"]) for r in gelezen_run["log"]]
    assert ("embed", "geslaagd") in stappen and ("embed", "afgekapt") in stappen
    regels = await _q("SELECT run, classification FROM model_call WHERE run = $r",
                      {"r": conn.ensure_record_id(run)})
    assert regels and all(r["classification"] == "intern" for r in regels)
    [stadium] = await _q("SELECT VALUE processing_stage FROM $b",
                         {"b": conn.ensure_record_id(bron)})
    assert stadium == "embedded"


async def test_mislukte_job_opnieuw_tot_het_maximum_daarna_mislukt(env, app) -> None:
    pogingen = []

    async def faalt(stap) -> str:
        pogingen.append(stap.attempts)
        raise RuntimeError(GEHEIM)

    env.orch._uitvoerders["graph"] = faalt
    async with _client(app) as client:
        run = await _start_run(client, env)
        bron, versie = await _versie(run, ["Openbare tekst."])
        stap = (await _post(client, run, "graph", versie)).json()
        await env.draai()
        gelezen = (await client.get(f"/api/v1/runs/{run}/steps/graph")).json()

    assert pogingen == [1, 2, 3] == list(range(1, MAX_HERHALINGEN + 2))
    assert gelezen["status"] == "mislukt" and gelezen["attempts"] == 3
    assert gelezen["error"] == "Stap graph mislukt na 3 pogingen (RuntimeError)"
    gelezen_run = await _run(run)
    assert gelezen_run["status"] == "mislukt" and gelezen_run["error"] == gelezen["error"]
    assert gelezen_run["log"][-1]["step"] == "graph"
    job = await _een("SELECT * FROM ONLY $id", {"id": conn.ensure_record_id(stap["job_id"])})
    assert job["status"] == "failed" and job["retry_count"] == MAX_HERHALINGEN
    [melding] = await _q("SELECT * FROM beheer_melding WHERE run_step = $s",
                         {"s": conn.ensure_record_id(stap["step_id"])})
    assert melding["soort"] == "stap_mislukt" and str(melding["source"]) == bron
    # Geen inhoud in stap, run, job of melding.
    for tekst in (json.dumps(gelezen), json.dumps(gelezen_run, default=str),
                  json.dumps(job, default=str), json.dumps(melding, default=str)):
        assert GEHEIM not in tekst


async def test_stap_op_een_onbekende_run_geeft_404(env, app) -> None:
    async with _client(app) as client:
        run = await _start_run(client, env)
        _, versie = await _versie(run, ["Openbare tekst."])
        for pad in ("processing_run:bestaatniet", "bestaatniet", "source:x"):
            assert (await _post(client, pad, "embed", versie)).status_code == 404
            assert (await client.get(f"/api/v1/runs/{pad}/steps/embed")).status_code == 404
        # Een bestaande run zonder deze stap.
        assert (await client.get(f"/api/v1/runs/{run}/steps/graph")).status_code == 404


async def test_bewaking_vanuit_de_stap_maar_niet_bij_wacht_op_review(
        env, app, monkeypatch) -> None:
    monkeypatch.setenv("SIGNAAL_RUN_MAX_MINUTEN", "60")
    oud = datetime.now(timezone.utc) - timedelta(hours=2)
    runs = []
    for _ in range(2):
        rij = await _een("CREATE ONLY processing_run CONTENT $d RETURN id", {"d": {
            "trigger": "schema", "pipeline_id": "inname_pdf", "flow_vastgelegd": True,
            "created_by_key": LANGFLOW.key_id, "started_at": oud}})
        runs.append(str(rij["id"]))
    te_lang, wacht = runs
    _, versie = await _versie(wacht, ["Openbare tekst."])

    async def pauzeert(stap) -> str:
        raise JobPausedForReviewError("twijfel")

    env.orch._uitvoerders["entity_dedup"] = pauzeert
    async with _client(app) as client:
        # Nog binnen de tijd: de stap komt op wacht_op_review (B5).
        monkeypatch.setenv("SIGNAAL_RUN_MAX_MINUTEN", "600")
        assert (await _post(client, wacht, "entity_dedup", versie)).status_code == 202
        await env.draai()
        stap = (await client.get(f"/api/v1/runs/{wacht}/steps/entity_dedup")).json()
        assert stap["status"] == "wacht_op_review"

        monkeypatch.setenv("SIGNAAL_RUN_MAX_MINUTEN", "60")
        # Een stapaanroep (hier GET) start de bewaking.
        assert (await client.get(f"/api/v1/runs/{wacht}/steps/entity_dedup")).status_code == 200

    assert (await _run(te_lang))["status"] == "mislukt"
    assert (await _run(wacht))["status"] == "bezig"
    job = await _een("SELECT * FROM ONLY $id", {"id": conn.ensure_record_id(stap["job_id"])})
    assert job["status"] == "paused_for_review"


# --------------------------------------------------------------------------
# Rechten, binding en contract
# --------------------------------------------------------------------------


@pytest.mark.echte_sleutels
async def test_alleen_de_aanmaker_van_de_run(env, app) -> None:
    async with _client(app) as client:
        run = await _start_run(client, env)
        _, versie = await _versie(run, ["Openbare tekst."])
        env.als(ANDERE)
        assert (await _post(client, run, "embed", versie)).status_code == 403
        assert (await client.get(f"/api/v1/runs/{run}/steps/embed")).status_code == 403
        env.als(LANGFLOW)
        assert (await _post(client, run, "embed", versie)).status_code == 202


async def test_alleen_een_versie_van_de_run_en_een_lopende_run(env, app) -> None:
    async with _client(app) as client:
        run = await _start_run(client, env)
        ander = await _start_run(client, env)
        _, eigen = await _versie(run, ["Openbare tekst."])
        _, vreemd = await _versie(ander, ["Andere openbare tekst."])
        resp = await _post(client, run, "embed", vreemd)
        assert resp.status_code == 409 and resp.json()["detail"]["code"] == "niet_gebonden"
        resp = await _post(client, run, "embed", "source_version:bestaatniet")
        assert resp.status_code == 409 and resp.json()["detail"]["code"] == "niet_gebonden"
        assert (await _post(client, run, "onbekend", eigen)).status_code == 422
        assert (await client.patch(f"/api/v1/runs/{run}",
                                   json={"status": "mislukt"})).status_code == 200
        resp = await _post(client, run, "embed", eigen)
        assert resp.status_code == 409 and resp.json()["detail"]["code"] == "run_afgesloten"


async def test_job_na_het_sluiten_van_de_run_wordt_niet_herhaald(env, app) -> None:
    async with _client(app) as client:
        run = await _start_run(client, env)
        _, versie = await _versie(run, ["Openbare tekst."])
        stap = (await _post(client, run, "embed", versie)).json()
        await client.patch(f"/api/v1/runs/{run}", json={"status": "mislukt"})
        await env.draai()
        gelezen = (await client.get(f"/api/v1/runs/{run}/steps/embed")).json()
    assert gelezen["status"] == "mislukt" and gelezen["attempts"] == 0
    job = await _een("SELECT * FROM ONLY $id", {"id": conn.ensure_record_id(stap["job_id"])})
    assert job["status"] == "failed" and job["retry_count"] == 0
    assert env.server.invoer == []
    assert not await _q("SELECT id FROM source_embedding WHERE run = $r",
                        {"r": conn.ensure_record_id(run)})


async def test_extractie_krijgt_alleen_de_fragmenten_van_versie_en_run(
        env, app, monkeypatch) -> None:
    ontvangen = {}

    class Extractie:
        async def run_extraction(self, **kw):
            ontvangen.update(kw)
            return {"entity_count": 0, "relation_count": 0}

    import app_main.dependencies as deps

    monkeypatch.setattr(deps, "get_entity_extraction_service", lambda: Extractie())
    async with _client(app) as client:
        oud = await _start_run(client, env)
        bron, _ = await _versie(oud, ["Oude openbare tekst."])
        await client.patch(f"/api/v1/runs/{oud}", json={"status": "geslaagd"})
        run = await _start_run(client, env)
        versie, _ = await SourceVersionRepository().register(
            bron, hashlib.sha256(b"nieuwe versie").hexdigest(), run)
        await _q("CREATE chunk CONTENT $c", {"c": {
            "source": conn.ensure_record_id(bron), "text": "Nieuwe openbare tekst.",
            "order": 0, "physical_page": 1, "element_type": "paragraph", "positions": [],
            "metadata": {}, "source_version": conn.ensure_record_id(str(versie.id)),
            "run": conn.ensure_record_id(run)}})
        await _post(client, run, "entity_extraction", str(versie.id))
        await env.draai()
        gelezen = (await client.get(f"/api/v1/runs/{run}/steps/entity_extraction")).json()

    assert gelezen["status"] == "geslaagd"
    assert ontvangen["run_id"] == run and ontvangen["source_id"] == bron
    assert [c.text for c in ontvangen["fragmenten"]] == ["Nieuwe openbare tekst."]
