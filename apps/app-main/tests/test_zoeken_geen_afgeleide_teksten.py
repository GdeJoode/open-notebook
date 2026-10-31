"""Zoeken geeft geen afgeleide teksten terug (F2-05, ADR 0004, harde eis 3).

Een notitie en een inzicht met precies de zoektekst (en de zoekvector) komen
niet terug bij tekst-, vector- en hybride zoeken via ``POST /api/search``. Elk
resultaat is een bron en verwijst naar de fragmenten (``chunk``) waarop het
treft. Tegen een echte SurrealDB, met de sleutel ``beheerder`` (ziet alles).

Alle testdata is synthetisch.
"""

from __future__ import annotations

import hashlib
import math
import random
import string
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main import dependencies
from app_main.api.app import create_app
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.search_service import SearchService
from surrealdb_service.repositories.processing_run import ProcessingRunRepository
from surrealdb_service.repositories.search import SearchRepository
from surrealdb_service.repositories.source_version import SourceVersionRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


def _richting(dim: int = 7) -> list[float]:
    v = [random.uniform(-1, 1) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


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
    conn._pool = original or conn.ConnectionPool(config=live_surrealdb)


async def _q(sql: str, params: dict | None = None):
    return await conn.execute_query(sql, params or {})


async def test_notitie_en_inzicht_komen_niet_terug(app, env) -> None:
    woord = "".join(random.choices(string.ascii_lowercase, k=14))
    vector = _richting()

    # Een bron met een fragment en embedding: de positieve controle.
    [bron] = await _q("CREATE source CONTENT {title: 'Openbare testbron', classification: 'extern'}"
                      " RETURN id")
    bron = str(bron["id"])
    run = await ProcessingRunRepository().create(trigger="inname", pipeline_id="test",
                                                 flow_vastgelegd=True)
    versie, _ = await SourceVersionRepository().register(
        bron, hashlib.sha256(uuid.uuid4().bytes).hexdigest(), str(run.id))
    [chunk] = await _q("CREATE chunk CONTENT $d RETURN id", {"d": {
        "source": conn.ensure_record_id(bron), "source_version": conn.ensure_record_id(str(versie.id)),
        "run": conn.ensure_record_id(str(run.id)), "text": f"{woord} in de bron", "order": 0,
        "physical_page": 0, "element_type": "paragraph", "positions": [], "metadata": {}}})
    await _q("CREATE source_embedding CONTENT $d", {"d": {
        "source": conn.ensure_record_id(bron), "chunk": conn.ensure_record_id(str(chunk["id"])),
        "run": conn.ensure_record_id(str(run.id)), "content": woord, "order": 0,
        "embedding": vector, "embedding_model": "test"}})

    # Een notitie en een inzicht met dezelfde tekst en vector.
    [notitie] = await _q("CREATE note CONTENT {title: $w, content: $w, embedding: $v,"
                         " classification: 'extern'} RETURN id", {"w": woord, "v": vector})
    [inzicht] = await _q("CREATE source_insight CONTENT {source: type::thing($b),"
                         " insight_type: 't', content: $w, embedding: $v} RETURN id",
                         {"b": bron, "w": woord, "v": vector})
    afgeleid = {str(notitie["id"]), str(inzicht["id"])}

    manager = SimpleNamespace(get_defaults=lambda: SimpleNamespace(default_embedding_model="m"))
    app.dependency_overrides[dependencies.get_search_service] = lambda: SearchService(
        search_repo=SearchRepository(), model_manager=manager,
        embedding_model=SimpleNamespace(aembed=AsyncMock(return_value=[vector])))
    app.dependency_overrides[dependencies.get_model_service] = lambda: SimpleNamespace(
        model_manager=manager)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                     headers={"Authorization": "Bearer ak_test"}) as c:
            for soort in ("text", "vector", "hybrid"):
                for history in (False, True):
                    r = await c.post("/api/search", json={
                        "query": woord, "type": soort, "history": history,
                        "search_notes": True, "minimum_score": 0.9})
                    assert r.status_code == 200, r.text
                    hits = r.json()["results"]
                    assert bron in {h["id"] for h in hits}, soort
                    for hit in hits:
                        assert not {hit["id"], hit.get("parent_id")} & afgeleid, (soort, hit)
                        assert str(hit["id"]).startswith("source:"), (soort, hit)
                        assert hit["chunks"], (soort, hit)
                        assert all(str(c_).startswith("chunk:") for c_ in hit["chunks"]), hit
                    assert woord not in str([h for h in hits if h["id"] != bron])
    finally:
        app.dependency_overrides.clear()
