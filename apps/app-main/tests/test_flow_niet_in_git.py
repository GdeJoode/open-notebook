"""Controle van de flow bij ``POST /api/v1/runs`` (F2-04, ADR 0022 E, ADR 0017 B).

Een run met een flow die niet in Git staat (commit of hash van de flow-JSON
wijkt af van de gekoppelde flows) mislukt zodra de invoer intern of onbekend
is: HTTP 409 en de run staat als ``mislukt`` in de database. Bij externe invoer
(een bevestigde externe bronkaartregel) 201 met een logregel
``niet_vastgelegd``. Tegen een echte SurrealDB; de flows en de bronkaartregel
zijn synthetisch.
"""

from __future__ import annotations

import hashlib
import json
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
from app_main.services.flow_register import FlowRegister
from app_main.services.processing_run_service import ProcessingRunService
from shared.models.agents import Right

COMMIT = "0123456789abcdef0123456789abcdef01234567"
FLOW = {"id": "inname-pdf", "name": "Inname PDF", "data": {"nodes": [], "edges": []}}


@pytest.fixture
def register(tmp_path) -> tuple[FlowRegister, str]:
    pad = tmp_path / "inname_pdf.json"
    pad.write_text(json.dumps(FLOW), encoding="utf-8")
    return FlowRegister.from_dir(tmp_path, COMMIT), hashlib.sha256(pad.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Het register zelf (zonder database)
# --------------------------------------------------------------------------


def test_register_kent_de_flow_op_naam_en_id(register) -> None:
    reg, h = register
    assert reg.is_vastgelegd("inname_pdf", COMMIT, h)
    assert reg.is_vastgelegd("inname-pdf", COMMIT, h)


@pytest.mark.parametrize(
    "pipeline_id, commit, flow_hash",
    [
        ("inname_pdf", COMMIT, "0" * 64),   # gewijzigde flow
        ("inname_pdf", COMMIT, None),       # geen hash
        ("inname_pdf", "f" * 40, "HASH"),   # andere commit
        ("onbekend", COMMIT, "HASH"),       # flow niet in de map
        ("inname_pdf", None, "HASH"),       # geen commit
    ],
)
def test_register_weigert_wat_niet_in_git_staat(register, pipeline_id, commit, flow_hash) -> None:
    reg, h = register
    assert not reg.is_vastgelegd(pipeline_id, commit, h if flow_hash == "HASH" else flow_hash)


def test_register_zonder_commit_kent_niets(tmp_path, register) -> None:
    _, h = register
    (tmp_path / "x.json").write_text("{}", encoding="utf-8")
    assert not FlowRegister.from_dir(tmp_path, None).is_vastgelegd("x", COMMIT, h)
    assert FlowRegister.from_dir(tmp_path / "bestaat_niet", COMMIT).hashes == {}


# --------------------------------------------------------------------------
# De API, tegen een echte database
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
async def env(live_surrealdb, monkeypatch, app, register):
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    ctx = agent_auth.AgentKeyContext(
        key_id="agent_keys:langflow", agent_id="langflow", permission=Right.PIJPLIJN
    )

    async def _pijplijn(_request):
        return ctx

    monkeypatch.setattr(agent_auth, "authenticate_request", _pijplijn)
    reg, _ = register
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


async def _regel(classification: str, bevestigd: bool = True) -> str:
    """Een bronkaartregel volgens plan §5.2 (de tabel zelf komt met F2-13)."""
    rid = f"source_map_rule:t{uuid.uuid4().hex[:10]}"
    data = {"classification": classification, "active": True, "rev": 4}
    if bevestigd:
        data["classification_confirmed_by"] = "Testpersoon"
    await conn.execute_query("CREATE type::thing($id) CONTENT $d", {"id": rid, "d": data})
    return rid


async def _run(run_id: str) -> dict:
    [rij] = await conn.execute_query("SELECT * FROM type::thing($id)", {"id": run_id})
    return rij


def _verzoek(flow_hash: str, regel: str | None = None) -> dict:
    body = {
        "trigger": "schema",
        "pipeline_id": "inname_pdf",
        "pipeline_version": {"git_commit": COMMIT, "flow_hash": flow_hash},
    }
    if regel:
        body["source_map_rule_id"] = regel
    return body


@pytest.mark.requires_docker
@pytest.mark.asyncio
@pytest.mark.parametrize("invoer", ["onbekend", "intern", "extern_niet_bevestigd"])
async def test_flow_buiten_git_met_interne_of_onbekende_invoer_geeft_409(env, app, invoer) -> None:
    regel = None
    if invoer == "intern":
        regel = await _regel("intern")
    elif invoer == "extern_niet_bevestigd":
        regel = await _regel("extern", bevestigd=False)
    async with _client(app) as client:
        resp = await client.post("/api/v1/runs", json=_verzoek("0" * 64, regel))
    assert resp.status_code == 409
    body = resp.json()
    assert body["status"] == "mislukt"
    run = await _run(body["run_id"])
    assert run["status"] == "mislukt" and run["ended_at"] and run["error"]
    assert run["input_classification"] == ("onbekend" if invoer == "onbekend" else "intern")
    assert [r["status"] for r in run["log"]] == ["niet_vastgelegd"]


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_flow_buiten_git_met_externe_invoer_geeft_201_met_waarschuwing(env, app) -> None:
    regel = await _regel("extern")
    async with _client(app) as client:
        resp = await client.post("/api/v1/runs", json=_verzoek("0" * 64, regel))
    assert resp.status_code == 201
    assert resp.json()["status"] == "bezig"
    run = await _run(resp.json()["run_id"])
    assert run["status"] == "bezig" and run["input_classification"] == "extern"
    assert [r["status"] for r in run["log"]] == ["niet_vastgelegd"]
    # De fork legt de regel en haar `rev` vast (ADR 0020).
    assert (run["source_map_rule_id"], run["source_map_rule_rev"]) == (regel, 4)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_flow_in_git_met_interne_invoer_geeft_201(env, app, register) -> None:
    _, h = register
    regel = await _regel("intern")
    async with _client(app) as client:
        resp = await client.post("/api/v1/runs", json=_verzoek(h, regel))
    assert resp.status_code == 201
    run = await _run(resp.json()["run_id"])
    assert run["status"] == "bezig"
    assert run["pipeline_version"] == {"git_commit": COMMIT, "flow_hash": h}
    assert [r["status"] for r in run["log"]] == ["vastgelegd"]
