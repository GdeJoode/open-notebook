"""Run-eindpunten en herkomst (F2-04, ADR 0003, plan §5.4).

Tegen een echte SurrealDB, met de sleutel ``pijplijn``:

* een run zonder nieuwe inhoud eindigt als ``niets_nieuws``, ook als Langflow
  ``geslaagd`` meldt; een bekende hash maakt geen nieuwe versie;
* ``PATCH`` op een afgesloten run geeft 409;
* ``models`` komt niet uit de opgave van Langflow; een verschil is een
  waarschuwing in het logboek;
* een ``mentions``-relatie van een interne bron is ``intern``, ook na
  herclassificatie van een externe bron naar intern.

Alle testdata is synthetisch.
"""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.dependencies import get_processing_run_service
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.classification_service import ClassificationService
from app_main.services.processing_run_service import ProcessingRunService
from shared.models.agents import Right
from surrealdb_service.repositories.source_version import SourceVersionRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
async def env(live_surrealdb, monkeypatch, app, git_flows):
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    ctx = agent_auth.AgentKeyContext(
        key_id="agent_keys:langflow", agent_id="langflow", permission=Right.PIJPLIJN
    )

    async def _pijplijn(_request):
        return ctx

    monkeypatch.setattr(agent_auth, "authenticate_request", _pijplijn)
    reg = git_flows.register
    app.dependency_overrides[get_processing_run_service] = lambda: ProcessingRunService(
        flow_register=reg
    )
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    yield
    app.dependency_overrides.pop(get_processing_run_service, None)
    await conn._pool.close_all()
    conn._pool = original or conn.ConnectionPool(config=live_surrealdb)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://t",
        headers={"Authorization": "Bearer ak_test"},
    )


async def _q(sql: str, params: dict | None = None):
    return await conn.execute_query(sql, params or {})


async def _start(client, flow) -> str:
    resp = await client.post("/api/v1/runs", json={
        "trigger": "schema", "pipeline_id": "inname_pdf",
        "pipeline_version": {"git_commit": flow.head, "flow_hash": flow.hash},
    })
    assert resp.status_code == 201, resp.text
    assert resp.json()["status"] == "bezig"
    return resp.json()["run_id"]


async def _bron(classification: str = "extern") -> str:
    [rij] = await _q("CREATE source SET title = 'Openbare testbron', classification = $c RETURN id",
                     {"c": classification})
    return str(rij["id"])


def _hash(tekst: str) -> str:
    return hashlib.sha256(tekst.encode()).hexdigest()


# --------------------------------------------------------------------------
# Afsluiten
# --------------------------------------------------------------------------


async def test_run_zonder_nieuwe_inhoud_eindigt_als_niets_nieuws(env, app, git_flows) -> None:
    bron = await _bron()
    async with _client(app) as client:
        eerste = await _start(client, git_flows)
        versie, nieuw = await SourceVersionRepository().register(bron, _hash(bron), eerste)
        assert nieuw
        resp = await client.patch(f"/api/v1/runs/{eerste}", json={"status": "geslaagd"})
        assert resp.status_code == 200 and resp.json()["status"] == "geslaagd"
        assert resp.json()["ended_at"]

        # Dezelfde inhoud nog eens: de bekende hash geeft de bestaande versie.
        tweede = await _start(client, git_flows)
        weer, nieuw = await SourceVersionRepository().register(bron, _hash(bron), tweede)
        assert not nieuw and weer.id == versie.id
        resp = await client.patch(f"/api/v1/runs/{tweede}", json={
            "status": "geslaagd",
            "log": [{"step": "registratie", "status": "ongewijzigd", "message": ""}],
        })
    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "niets_nieuws"
    assert [r["status"] for r in run["log"]][-2:] == ["ongewijzigd", "niets_nieuws"]
    assert len(await SourceVersionRepository().list_by_source(bron)) == 1


async def test_patch_op_een_afgesloten_run_geeft_409(env, app, git_flows) -> None:
    async with _client(app) as client:
        run_id = await _start(client, git_flows)
        assert (await client.patch(f"/api/v1/runs/{run_id}", json={"status": "mislukt",
                                                                    "error": "docling"})).status_code == 200
        for body in ({"status": "geslaagd"}, {"log": [{"step": "x", "status": "ok"}]}):
            resp = await client.patch(f"/api/v1/runs/{run_id}", json=body)
            assert resp.status_code == 409
        run = (await client.get(f"/api/v1/runs/{run_id}")).json()
    assert run["status"] == "mislukt" and run["error"] == "docling"


async def test_lopende_run_bijwerken_en_onbekende_run(env, app, git_flows) -> None:
    async with _client(app) as client:
        run_id = await _start(client, git_flows)
        resp = await client.patch(f"/api/v1/runs/{run_id}",
                                  json={"log": [{"step": "extractie", "status": "ok"}]})
        assert resp.status_code == 200 and resp.json()["status"] == "bezig"
        assert resp.json()["log"][-1]["step"] == "extractie"
        assert (await client.get("/api/v1/runs/processing_run:bestaatniet")).status_code == 404
        assert (await client.get("/api/v1/runs/source:x")).status_code == 404
        assert (await client.patch("/api/v1/runs/processing_run:bestaatniet",
                                   json={"status": "geslaagd"})).status_code == 404


async def test_patch_met_een_kale_run_id(env, app, git_flows) -> None:
    async with _client(app) as client:
        run_id = await _start(client, git_flows)
        kaal = run_id.split(":", 1)[1]
        resp = await client.patch(f"/api/v1/runs/{kaal}", json={
            "log": [{"step": "extractie", "status": "ok"}],
            "models": [{"name": "qwen3.5-9b"}],
        })
        assert resp.status_code == 200 and resp.json()["log"][-1]["status"] == "waarschuwing"
        resp = await client.patch(f"/api/v1/runs/{kaal}", json={"status": "mislukt"})
        assert resp.status_code == 200 and resp.json()["status"] == "mislukt"
        assert (await client.get(f"/api/v1/runs/{run_id}")).json()["status"] == "mislukt"


async def test_models_niet_overgenomen_van_langflow(env, app, git_flows) -> None:
    async with _client(app) as client:
        run_id = await _start(client, git_flows)
        resp = await client.patch(f"/api/v1/runs/{run_id}", json={
            "status": "mislukt",
            "models": [{"name": "qwen3.5-9b", "version": "Q4_K_M", "aanroepwijze": "lokaal"}],
        })
    run = resp.json()
    # `model_call` (ADR 0007) heeft geen regels voor deze run.
    assert run["models"] == []
    assert any(r["step"] == "models" and r["status"] == "waarschuwing" for r in run["log"])


# --------------------------------------------------------------------------
# `mentions` volgt haar bron
# --------------------------------------------------------------------------


async def _entiteit() -> str:
    naam = f"testentiteit {uuid.uuid4().hex[:8]}"
    [rij] = await _q("CREATE entity CONTENT $d RETURN id", {"d": {
        "canonical_name": naam, "name_key": naam, "hash_id": naam,
        "entity_type": "topic", "embedding": []}})
    return str(rij["id"])


async def _mentions(bron: str, entiteit: str) -> str:
    await _q("RELATE $b->mentions->$e SET weight = 1.0",
             {"b": conn.ensure_record_id(bron), "e": conn.ensure_record_id(entiteit)})
    [waarde] = await _q("SELECT VALUE classification FROM mentions WHERE in = $b AND out = $e",
                        {"b": conn.ensure_record_id(bron), "e": conn.ensure_record_id(entiteit)})
    return waarde


async def test_mentions_van_een_interne_bron_is_intern(env) -> None:
    entiteit = await _entiteit()
    assert await _mentions(await _bron("intern"), entiteit) == "intern"
    assert await _mentions(await _bron("extern"), entiteit) == "extern"


async def test_mentions_wordt_intern_na_herclassificatie_naar_intern(env) -> None:
    bron = await _bron("extern")
    entiteit = await _entiteit()
    assert await _mentions(bron, entiteit) == "extern"
    await ClassificationService().reclassify(
        bron, "intern", "Bevat een niet-openbare bijlage",
        permission=Right.BEHEERDER, person="Testpersoon", key_id="agent_keys:test",
    )
    [waarde] = await _q("SELECT VALUE classification FROM mentions WHERE in = $b",
                        {"b": conn.ensure_record_id(bron)})
    assert waarde == "intern"


# --------------------------------------------------------------------------
# Een handmatig fragment: 409 tot F2-29
# --------------------------------------------------------------------------


async def test_handmatig_fragment_toevoegen_geeft_409(app) -> None:
    from types import SimpleNamespace

    from app_main.dependencies import get_source_service

    bron = SimpleNamespace(id="source:s1")
    app.dependency_overrides[get_source_service] = lambda: SimpleNamespace(
        get=AsyncMock(return_value=bron)
    )
    try:
        async with _client(app) as client:
            resp = await client.post("/api/sources/source:s1/chunks", json={"text": "x", "physical_page": 0})
    finally:
        app.dependency_overrides.pop(get_source_service, None)
    assert resp.status_code == 409
    assert "handmatige run" in resp.json()["detail"]


# --------------------------------------------------------------------------
# Wat de afnemer niet zelf kan zetten
# --------------------------------------------------------------------------


@pytest.mark.echte_sleutels
async def test_trigger_handmatig_alleen_voor_beheerder(env, app, git_flows, monkeypatch) -> None:
    body = {"trigger": "handmatig", "pipeline_id": "inname_pdf",
            "pipeline_version": {"git_commit": git_flows.head, "flow_hash": git_flows.hash}}
    async with _client(app) as client:
        resp = await client.post("/api/v1/runs", json=body)
        assert resp.status_code == 403
        beheerder = agent_auth.AgentKeyContext(
            key_id="agent_keys:b", agent_id="mens", permission=Right.BEHEERDER,
            person="Testpersoon",
        )

        async def _beheerder(_request):
            return beheerder

        monkeypatch.setattr(agent_auth, "authenticate_request", _beheerder)
        resp = await client.post("/api/v1/runs", json=body)
        assert resp.status_code == 201
        # De persoon van de sleutel is de actor (migratie 92, F2-06).
        run = (await client.get(f"/api/v1/runs/{resp.json()['run_id']}")).json()
        assert run["actor"] == "Testpersoon"


async def test_flow_vastgelegd_in_de_body_telt_niet(env, app, git_flows) -> None:
    [regel] = await _q("CREATE source_map_rule CONTENT {classification: 'extern',"
                       " classification_confirmed_by: 'Testpersoon', classification_confirmed_at: time::now(),"
                       " active: true,"
                       " name: 'Openbare testbron', route_type: 'api', source_type: 'api',"
                       " location: 'https://example.org/api', pipeline_start: 'automatisch',"
                       " route: 'inname_pdf', fetch_content: 'volledig_document'} RETURN id")
    async with _client(app) as client:
        resp = await client.post("/api/v1/runs", json={
            "trigger": "schema", "pipeline_id": "inname_pdf",
            "pipeline_version": {"git_commit": git_flows.head, "flow_hash": "0" * 64},
            "source_map_rule_id": str(regel["id"]), "flow_vastgelegd": True,
        })
        assert resp.status_code == 201
        run_id = resp.json()["run_id"]
        resp = await client.patch(f"/api/v1/runs/{run_id}",
                                  json={"flow_vastgelegd": True, "log": []})
        assert resp.status_code == 200 and resp.json()["flow_vastgelegd"] is False
    [waarde] = await _q("SELECT VALUE flow_vastgelegd FROM type::thing($id)", {"id": run_id})
    assert waarde is False
