"""Herclassificatie en de grenzen eromheen (F2-03, ADR 0001, 0002 B; FO 6, 14).

Tegen een echte SurrealDB (migratie 89):

* import of poortwachter (elk schrijfpad buiten de herclassificatie) kan een
  bron niet van intern naar extern zetten, en een fragment, embedding of
  ``doc_node`` niet losser maken dan de bron: ``ClassificationConflictError``,
  in de API HTTP 409;
* ``POST /api/v1/sources/{id}/classification`` alleen met ``beheerder`` en een
  reden; de bron, fragmenten, embeddings en ``doc_node`` gaan in één
  transactie om, met precies één logregel met persoon;
* er is geen pad om een logregel te wijzigen of te verwijderen.

Alle testdata is synthetisch.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from fastapi.routing import APIRoute
from shared.models.agents import Right
from surrealdb_service.connection import ClassificationConflictError
from surrealdb_service.repositories.source import ChunkRepository, SourceRepository

from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.classification_service import ClassificationService

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

REDEN = "Openbaar gepubliceerd op de website van de uitgever"


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
async def env(live_surrealdb, monkeypatch):
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    yield
    await conn._pool.close_all()
    # Zonder eerdere pool een nieuwe op de testdatabase achterlaten: een latere
    # aanroep zonder config zou anders een pool op de standaardinstelling maken.
    conn._pool = original or conn.ConnectionPool(config=live_surrealdb)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://t",
        headers={"Authorization": "Bearer ak_test"},
    )


async def _q(sql: str, params: dict | None = None):
    return await conn.execute_query(sql, params or {})


async def _bron(classification: str = "intern") -> str:
    [row] = await _q(
        "CREATE source SET title = 'Openbare testbron', classification = $c RETURN id",
        {"c": classification},
    )
    return str(row["id"])


async def _kinderen(bron: str) -> dict[str, str]:
    """Een fragment, embedding, ``doc_node`` en inzicht bij ``bron``."""
    b = conn.ensure_record_id(bron)
    [chunk] = await _q(
        "CREATE chunk CONTENT $c RETURN id",
        {"c": {"source": b, "text": "t", "order": 0, "physical_page": 0,
               "element_type": "paragraph", "positions": [], "metadata": {}}},
    )
    [emb] = await _q(
        "CREATE source_embedding CONTENT $c RETURN id",
        {"c": {"source": b, "chunk": conn.ensure_record_id(str(chunk["id"])),
               "content": "t", "order": 0,
               "embedding": []}},
    )
    [node] = await _q(
        "CREATE doc_node CONTENT $c RETURN id",
        {"c": {"source": b, "element_type": "paragraph", "self_ref": "#/texts/0",
               "sequence": 0}},
    )
    [insight] = await _q(
        "CREATE source_insight CONTENT $c RETURN id",
        {"c": {"source": b, "content": "t", "insight_type": "summary", "embedding": []}},
    )
    return {"chunk": str(chunk["id"]), "source_embedding": str(emb["id"]),
            "doc_node": str(node["id"]), "source_insight": str(insight["id"])}


async def _classificatie(rid: str) -> str:
    [row] = await _q("SELECT classification FROM type::thing($id)", {"id": rid})
    return row["classification"]


async def _log(bron: str) -> list[dict]:
    return await _q(
        "SELECT * FROM classification_log WHERE source = type::thing($b) ORDER BY created_at",
        {"b": bron},
    )


# --------------------------------------------------------------------------
# Import en poortwachter: intern blijft intern
# --------------------------------------------------------------------------


async def test_import_kan_intern_niet_naar_extern_zetten(env) -> None:
    bron = await _bron()
    repo = SourceRepository()
    with pytest.raises(ClassificationConflictError):
        await repo.update(bron, {"classification": "extern"})
    with pytest.raises(ClassificationConflictError):
        await repo.upsert(bron, {"classification": "extern"})
    assert await _classificatie(bron) == "intern"
    assert await _log(bron) == []


async def test_poortwachter_kan_geen_fragment_losser_maken_dan_de_bron(env) -> None:
    bron = await _bron()
    with pytest.raises(ClassificationConflictError):
        await ChunkRepository().bulk_create([{
            "source": bron, "text": "t", "order": 0, "physical_page": 0,
            "element_type": "paragraph", "positions": [], "metadata": {},
            "classification": "extern",
        }])
    kinderen = await _kinderen(bron)
    for tabel in ("chunk", "source_embedding", "doc_node", "source_insight"):
        with pytest.raises(ClassificationConflictError):
            await _q("UPDATE type::thing($id) SET classification = 'extern'",
                     {"id": kinderen[tabel]})
        assert await _classificatie(kinderen[tabel]) == "intern"


async def test_een_geweigerde_classificatie_geeft_409(env, app, monkeypatch) -> None:
    # Een herclassificatie op een verouderde waarde: de transactie weigert.
    bron = await _bron()
    monkeypatch.setattr(SourceRepository, "get_classification", AsyncMock(return_value="extern"))
    async with _client(app) as client:
        resp = await client.post(
            f"/api/v1/sources/{bron}/classification",
            json={"classification": "intern", "reason": REDEN},
        )
    assert resp.status_code == 409
    assert await _log(bron) == []


# --------------------------------------------------------------------------
# Herclassificatie
# --------------------------------------------------------------------------


async def test_herclassificatie_werkt_alles_bij_met_een_logregel(env, app) -> None:
    bron = await _bron("extern")
    kinderen = await _kinderen(bron)
    for rid in kinderen.values():
        assert await _classificatie(rid) == "extern"

    async with _client(app) as client:
        resp = await client.post(
            f"/api/v1/sources/{bron}/classification",
            json={"classification": "intern", "reason": REDEN},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["changed"] is True
        assert (body["from_classification"], body["to_classification"]) == ("extern", "intern")
        assert await _classificatie(bron) == "intern"
        for rid in kinderen.values():
            assert await _classificatie(rid) == "intern"
        [regel] = await _log(bron)
        assert regel["person"] == "Testpersoon"
        assert regel["agent_key"] == "agent_keys:test"
        assert regel["reason"] == REDEN
        assert (regel["from_classification"], regel["to_classification"]) == ("extern", "intern")
        assert str(regel["id"]) == body["log_id"]

        resp = await client.post(
            f"/api/v1/sources/{bron}/classification",
            json={"classification": "extern", "reason": REDEN},
        )
        assert resp.status_code == 200, resp.text
        assert await _classificatie(bron) == "extern"
        for tabel in ("chunk", "source_embedding", "doc_node"):
            assert await _classificatie(kinderen[tabel]) == "extern"
        # Een afgeleide tekst wordt niet losser.
        assert await _classificatie(kinderen["source_insight"]) == "intern"
        assert len(await _log(bron)) == 2

        # Geen wijziging: geen logregel.
        resp = await client.post(
            f"/api/v1/sources/{bron}/classification",
            json={"classification": "extern", "reason": REDEN},
        )
        assert resp.status_code == 200
        assert resp.json()["changed"] is False
        assert len(await _log(bron)) == 2


@pytest.mark.parametrize("body", [
    {"classification": "extern"},
    {"classification": "extern", "reason": ""},
    {"classification": "extern", "reason": "   "},
    {"classification": "openbaar", "reason": REDEN},
])
async def test_reden_en_geldige_waarde_zijn_verplicht(env, app, body) -> None:
    bron = await _bron()
    async with _client(app) as client:
        resp = await client.post(f"/api/v1/sources/{bron}/classification", json=body)
    assert resp.status_code == 422
    assert await _classificatie(bron) == "intern"
    assert await _log(bron) == []


async def test_onbekende_bron_geeft_404(env, app) -> None:
    async with _client(app) as client:
        resp = await client.post(
            "/api/v1/sources/source:bestaatniet/classification",
            json={"classification": "extern", "reason": REDEN},
        )
    assert resp.status_code == 404


@pytest.mark.echte_sleutels
@pytest.mark.parametrize("recht", ["pijplijn", "lezen_intern", "lezen_extern", "ophalen"])
async def test_alleen_beheerder_mag_herclassificeren(env, app, monkeypatch, recht) -> None:
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(return_value={"id": "agent_keys:x", "agent_id": "x",
                                "permission": recht, "person": "Testpersoon"}),
    )
    bron = await _bron()
    async with _client(app) as client:
        resp = await client.post(
            f"/api/v1/sources/{bron}/classification",
            json={"classification": "extern", "reason": REDEN},
        )
    assert resp.status_code == 403
    assert await _classificatie(bron) == "intern"
    assert await _log(bron) == []


async def test_sleutel_zonder_persoon_kan_geen_regel_schrijven(env) -> None:
    bron = await _bron()
    kinderen = await _kinderen(bron)
    service = ClassificationService()
    for person in (None, "", "  "):
        with pytest.raises(PermissionError):
            await service.reclassify(bron, "extern", REDEN, permission=Right.BEHEERDER,
                                     person=person, key_id="agent_keys:x")
    # Ook buiten de service om: zonder persoon faalt de hele transactie.
    with pytest.raises(RuntimeError):
        await SourceRepository().reclassify(
            bron, from_classification="intern", to_classification="extern",
            reason=REDEN, agent_key="agent_keys:x", person="",
        )
    assert await _classificatie(bron) == "intern"
    for rid in kinderen.values():
        assert await _classificatie(rid) == "intern"
    assert await _log(bron) == []


# --------------------------------------------------------------------------
# Het logboek is alleen aanvulbaar
# --------------------------------------------------------------------------


async def test_geen_pad_om_een_logregel_te_wijzigen_of_te_verwijderen(env, app) -> None:
    # Geen route: alleen POST op de herclassificatie.
    routes = [
        (m, r.path) for r in app.routes if isinstance(r, APIRoute)
        for m in r.methods if "classification" in r.path
    ]
    assert routes == [("POST", "/api/v1/sources/{source_id}/classification")]

    bron = await _bron()
    await ClassificationService().reclassify(
        bron, "extern", REDEN, permission=Right.BEHEERDER, person="Testpersoon",
        key_id="agent_keys:x",
    )
    [regel] = await _log(bron)
    rid = str(regel["id"])
    # Ook niet via de database of de generieke repository.
    pogingen = [
        _q("UPDATE type::thing($id) SET reason = 'anders'", {"id": rid}),
        _q("UPDATE type::thing($id) SET person = 'iemand anders'", {"id": rid}),
        _q("DELETE type::thing($id)", {"id": rid}),
        _q("DELETE classification_log WHERE source = type::thing($b)", {"b": bron}),
        SourceRepository().delete(rid),
    ]
    for poging in pogingen:
        with pytest.raises(Exception):
            await poging
    assert await _log(bron) == [regel]
