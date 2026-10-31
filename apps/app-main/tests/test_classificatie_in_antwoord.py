"""Classificatie in de antwoorden van de API (F2-05, ADR 0001).

Elk response-schema van een record met een classificatie geeft haar terug:
``SourceResponse``, ``SourceListResponse``, ``NotebookResponse``,
``NoteResponse`` en ``SourceInsightResponse``. Tegen een echte SurrealDB, met
de testsleutel ``beheerder`` (ziet intern en extern).

Alle testdata is synthetisch.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main.api.app import create_app
from app_main.api.schemas import (
    NotebookResponse,
    NoteResponse,
    SourceInsightResponse,
    SourceListResponse,
    SourceResponse,
)
from app_main.services.agents.audit_service import AgentAuditService

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
async def client(app, live_surrealdb, monkeypatch):
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t",
        headers={"Authorization": "Bearer ak_test"},
    ) as c:
        yield c
    await conn._pool.close_all()
    conn._pool = original or conn.ConnectionPool(config=live_surrealdb)


async def _q(sql: str, params: dict | None = None):
    return await conn.execute_query(sql, params or {})


@pytest.mark.parametrize(
    "schema",
    [SourceResponse, SourceListResponse, NotebookResponse, NoteResponse, SourceInsightResponse],
    ids=lambda s: s.__name__,
)
def test_schema_heeft_classification(schema) -> None:
    assert "classification" in schema.model_fields


@pytest.mark.parametrize("classification", ["intern", "extern"])
async def test_antwoorden_geven_classification_terug(client, classification) -> None:
    [bron] = await _q(
        "CREATE source CONTENT {title: 'Openbare testbron', classification: $c} RETURN id",
        {"c": classification},
    )
    bron = str(bron["id"])
    await _q(
        "CREATE source_insight CONTENT {source: type::thing($b), insight_type: 't',"
        " content: 'Openbare samenvatting', embedding: []}",
        {"b": bron},
    )
    [notebook] = await _q(
        "CREATE notebook CONTENT {name: 'Openbaar notebook', description: '', classification: $c}"
        " RETURN id",
        {"c": classification},
    )
    [notitie] = await _q(
        "CREATE note CONTENT {title: 'Openbare notitie', content: 'Tekst', embedding: [],"
        " classification: $c} RETURN id",
        {"c": classification},
    )
    notebook, notitie = str(notebook["id"]), str(notitie["id"])
    await _q("RELATE (type::thing($b))->reference->(type::thing($n))", {"b": bron, "n": notebook})

    r = await client.get(f"/api/sources/{bron}")
    assert r.status_code == 200 and r.json()["classification"] == classification
    r = await client.get(f"/api/sources/?notebook_id={notebook}")
    assert [s["classification"] for s in r.json()] == [classification]
    r = await client.get(f"/api/notebooks/{notebook}")
    assert r.json()["classification"] == classification
    r = await client.get("/api/notebooks?limit=100")
    assert {n["classification"] for n in r.json() if n["id"] == notebook} == {classification}
    r = await client.get(f"/api/notes/{notitie}")
    assert r.json()["classification"] == classification
    assert r.json()["is_derived"] is True
    r = await client.get(f"/api/sources/{bron}/insights")
    assert r.status_code == 200
    assert [i["classification"] for i in r.json()] == [classification]
