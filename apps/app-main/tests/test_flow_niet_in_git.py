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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.dependencies import get_processing_run_service
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.flow_register import STANDAARD_MAP, FlowRegister
from app_main.services.processing_run_service import ProcessingRunService
from shared.models.agents import Right
from surrealdb_service.repositories.source_version import SourceVersionRepository

# --------------------------------------------------------------------------
# Het register zelf (een echte Git-repository, zonder database)
# --------------------------------------------------------------------------


def test_register_kent_de_flow_op_naam_en_id(git_flows) -> None:
    reg = git_flows.register
    assert reg.is_vastgelegd("inname_pdf", git_flows.head, git_flows.hash)
    assert reg.is_vastgelegd("inname-pdf", git_flows.head, git_flows.hash)


@pytest.mark.parametrize(
    "pipeline_id, commit, flow_hash",
    [
        ("inname_pdf", "HEAD", "0" * 64),   # gewijzigde flow
        ("inname_pdf", "HEAD", None),       # geen hash
        ("inname_pdf", "f" * 40, "HASH"),   # andere commit
        ("onbekend", "HEAD", "HASH"),       # flow niet in Git
        ("inname_pdf", None, "HASH"),       # geen commit
    ],
)
def test_register_weigert_wat_niet_in_git_staat(git_flows, pipeline_id, commit, flow_hash) -> None:
    assert not git_flows.register.is_vastgelegd(
        pipeline_id,
        git_flows.head if commit == "HEAD" else commit,
        git_flows.hash if flow_hash == "HASH" else flow_hash,
    )


def test_hash_komt_uit_git_niet_uit_de_werkmap(git_flows) -> None:
    pad = git_flows.map / "inname_pdf.json"
    pad.write_text('{"id": "inname-pdf", "gewijzigd": true}', encoding="utf-8")
    nieuw = hashlib.sha256(pad.read_bytes()).hexdigest()
    reg = git_flows.register
    # De werkmap is vuil: niets is vastgelegd, ook niet de hash uit HEAD.
    assert not reg.is_vastgelegd("inname_pdf", git_flows.head, nieuw)
    assert not reg.is_vastgelegd("inname_pdf", git_flows.head, git_flows.hash)
    # Het register leest de hash uit HEAD.
    assert reg.hashes()["inname_pdf"] == git_flows.hash


def test_een_onbekend_bestand_maakt_de_map_vuil(git_flows) -> None:
    (git_flows.map / "los.json").write_text("{}", encoding="utf-8")
    assert not git_flows.register.is_vastgelegd("inname_pdf", git_flows.head, git_flows.hash)


def test_register_ververst_als_head_verandert(git_flows) -> None:
    reg = git_flows.register
    oud_head, oud = git_flows.head, git_flows.hash
    assert reg.is_vastgelegd("inname_pdf", oud_head, oud)
    nieuw = git_flows.commit("inname_pdf", {"id": "inname-pdf", "versie": 2})
    assert not reg.is_vastgelegd("inname_pdf", oud_head, oud)
    assert not reg.is_vastgelegd("inname_pdf", git_flows.head, oud)
    assert reg.is_vastgelegd("inname_pdf", git_flows.head, nieuw)


def test_opgegeven_commit_moet_head_zijn(git_flows) -> None:
    anders = FlowRegister(git_flows.map, "f" * 40)
    assert not anders.is_vastgelegd("inname_pdf", git_flows.head, git_flows.hash)
    gelijk = FlowRegister(git_flows.map, git_flows.head)
    assert gelijk.is_vastgelegd("inname_pdf", git_flows.head, git_flows.hash)


def test_zonder_git_is_niets_vastgelegd(tmp_path, git_flows) -> None:
    (tmp_path / "los").mkdir()
    (tmp_path / "los" / "inname_pdf.json").write_bytes(
        (git_flows.map / "inname_pdf.json").read_bytes()
    )
    reg = FlowRegister(tmp_path / "los")
    assert not reg.is_vastgelegd("inname_pdf", git_flows.head, git_flows.hash)
    assert FlowRegister(tmp_path / "bestaat_niet").hashes() == {}


def test_botsing_tussen_bestandsnaam_en_id_telt_niet(git_flows) -> None:
    # `andere.json` draagt als id de bestandsnaam van een andere flow.
    andere = git_flows.commit("andere", {"id": "inname_pdf", "data": {}})
    reg = git_flows.register
    for h in (git_flows.hash, andere):
        assert not reg.is_vastgelegd("inname_pdf", git_flows.head, h)
    # De naam zonder botsing blijft vastgelegd.
    assert reg.is_vastgelegd("andere", git_flows.head, andere)


def test_skip_worktree_verbergt_geen_wijziging(git_flows) -> None:
    git_flows.git("update-index", "--skip-worktree",
                  "services/langflow/flows/inname_pdf.json")
    (git_flows.map / "inname_pdf.json").write_text('{"id": "inname-pdf", "x": 1}',
                                                   encoding="utf-8")
    assert git_flows.git("status", "--porcelain") == ""  # git status ziet het niet
    assert not git_flows.register.schoon()
    assert not git_flows.register.is_vastgelegd("inname_pdf", git_flows.head, git_flows.hash)


def test_symbolische_link_telt_niet(git_flows) -> None:
    (git_flows.map / "link.json").symlink_to("inname_pdf.json")
    git_flows.git("add", "-A")
    git_flows.git("commit", "-q", "-m", "link")
    git_flows.head = git_flows.git("rev-parse", "HEAD")
    reg = git_flows.register
    assert "link" not in reg.hashes()
    assert not reg.is_vastgelegd("link", git_flows.head, git_flows.hash)
    assert reg.is_vastgelegd("inname_pdf", git_flows.head, git_flows.hash)


def test_symbolische_link_in_de_werkmap_is_niet_schoon(git_flows) -> None:
    pad = git_flows.map / "inname_pdf.json"
    kopie = git_flows.map.parent / "kopie.json"
    kopie.write_bytes(pad.read_bytes())
    git_flows.git("update-index", "--assume-unchanged",
                  "services/langflow/flows/inname_pdf.json")
    pad.unlink()
    pad.symlink_to(kopie)
    assert not git_flows.register.is_vastgelegd("inname_pdf", git_flows.head, git_flows.hash)


def test_standaardmap_ligt_in_de_root_van_de_repository() -> None:
    root = Path(__file__).resolve().parents[3]
    assert STANDAARD_MAP == root / "services" / "langflow" / "flows"


# --------------------------------------------------------------------------
# De API, tegen een echte database
# --------------------------------------------------------------------------


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


async def _regel(classification: str, bevestigd: bool = True) -> str:
    """Een bronkaartregel volgens plan §5.2 (migratie 93, F2-13)."""
    rid = f"source_map_rule:t{uuid.uuid4().hex[:10]}"
    data = {"name": "Openbare testbron", "route_type": "api", "source_type": "api",
            "location": "https://example.org/api", "pipeline_start": "automatisch",
            "route": "inname_pdf", "fetch_content": "volledig_document",
            "classification": classification, "active": True}
    if bevestigd and classification == "extern":
        data["classification_confirmed_by"] = "Testpersoon"
        data["classification_confirmed_at"] = datetime.now(timezone.utc)
    await conn.execute_query("CREATE type::thing($id) CONTENT $d", {"id": rid, "d": data})
    return rid


async def _bron(classification: str, run_id: str) -> str:
    """Een bron met een versie in ``run_id``; geeft het id van de bron."""
    [rij] = await conn.execute_query(
        "CREATE source SET title = 'Openbare testbron', classification = $c RETURN id",
        {"c": classification},
    )
    bron = str(rij["id"])
    await SourceVersionRepository().register(
        bron, hashlib.sha256(bron.encode()).hexdigest(), run_id
    )
    return bron


async def _fragment(bron: str) -> None:
    [rij] = await conn.execute_query(
        "SELECT current_version, current_version.current_run AS run FROM type::thing($b)",
        {"b": bron},
    )
    await conn.execute_query("CREATE chunk CONTENT $c", {"c": {
        "source": conn.ensure_record_id(bron), "text": "Openbare tekst", "order": 0,
        "physical_page": 0, "element_type": "paragraph", "positions": [], "metadata": {},
        "source_version": conn.ensure_record_id(str(rij["current_version"])),
        "run": conn.ensure_record_id(str(rij["run"]))}})


async def _run(run_id: str) -> dict:
    [rij] = await conn.execute_query("SELECT * FROM type::thing($id)", {"id": run_id})
    return rij


def _verzoek(commit: str, flow_hash: str, regel: str | None = None) -> dict:
    body = {
        "trigger": "schema",
        "pipeline_id": "inname_pdf",
        "pipeline_version": {"git_commit": commit, "flow_hash": flow_hash},
    }
    if regel:
        body["source_map_rule_id"] = regel
    return body


@pytest.mark.requires_docker
@pytest.mark.asyncio
@pytest.mark.parametrize("invoer", ["onbekend", "intern", "extern_niet_bevestigd"])
async def test_flow_buiten_git_met_interne_of_onbekende_invoer_geeft_409(
    env, app, git_flows, invoer
) -> None:
    regel = None
    if invoer == "intern":
        regel = await _regel("intern")
    elif invoer == "extern_niet_bevestigd":
        regel = await _regel("extern", bevestigd=False)
    async with _client(app) as client:
        resp = await client.post("/api/v1/runs", json=_verzoek(git_flows.head, "0" * 64, regel))
    assert resp.status_code == 409
    body = resp.json()
    assert body["status"] == "mislukt"
    run = await _run(body["run_id"])
    assert run["status"] == "mislukt" and run["ended_at"] and run["error"]
    assert run["input_classification"] == ("onbekend" if invoer == "onbekend" else "intern")
    assert [r["status"] for r in run["log"]] == ["niet_vastgelegd"]


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_flow_buiten_git_met_externe_invoer_geeft_201_met_waarschuwing(
    env, app, git_flows
) -> None:
    regel = await _regel("extern")
    # Een gewijzigde regel: de run neemt de `rev` van dat moment over.
    await conn.execute_query("UPDATE type::thing($id) SET name = 'Openbare testbron 2'",
                             {"id": regel})
    async with _client(app) as client:
        resp = await client.post("/api/v1/runs", json=_verzoek(git_flows.head, "0" * 64, regel))
    assert resp.status_code == 201
    assert resp.json()["status"] == "bezig"
    run = await _run(resp.json()["run_id"])
    assert run["status"] == "bezig" and run["input_classification"] == "extern"
    assert [r["status"] for r in run["log"]] == ["niet_vastgelegd"]
    assert run["flow_vastgelegd"] is False
    # De fork legt de regel en haar `rev` vast (ADR 0020).
    assert run["source_map_rule_id"] == regel and run["source_map_rule_rev"] >= 1

    # Zo'n run mag een versie en fragment van een externe bron schrijven, niet
    # van een interne (migratie 90).
    await _fragment(await _bron("extern", resp.json()["run_id"]))
    with pytest.raises(RuntimeError, match="created_by_run"):
        await _bron("intern", resp.json()["run_id"])


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_flow_in_git_met_interne_invoer_geeft_201(env, app, git_flows) -> None:
    h = git_flows.hash
    regel = await _regel("intern")
    async with _client(app) as client:
        resp = await client.post("/api/v1/runs", json=_verzoek(git_flows.head, h, regel))
    assert resp.status_code == 201
    run = await _run(resp.json()["run_id"])
    assert run["status"] == "bezig" and run["flow_vastgelegd"] is True
    assert run["pipeline_version"] == {"git_commit": git_flows.head, "flow_hash": h}
    assert [r["status"] for r in run["log"]] == ["vastgelegd"]
    # Een vastgelegde flow mag een fragment van een interne bron schrijven.
    await _fragment(await _bron("intern", resp.json()["run_id"]))
