"""F2-08: de routers voor zoeken, ask en embed nemen het model uit het register.

Niet uit ``default_models`` en niet uit modelrijen: op een lege database
(geen standaardmodellen, geen modelrijen; hier: geen database bereikbaar en de
modeldienst niet aangesloten) werken ze zolang het register het model kent, en
zonder model in het register geven ze 400.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llm_manager.register import ModelRegister, RegisterModel

from app_main.api.routers import embedding as embedding_router
from app_main.api.routers import search as search_router
from app_main.dependencies import (
    get_model_service,
    get_note_service,
    get_search_service,
    get_source_service,
)


def _register(*taken: str) -> ModelRegister:
    return ModelRegister(versie=1, modellen=[RegisterModel(
        naam=f"m-{t}", versie="v1", aanroepwijze="llama-swap", locatie="lokaal",
        toegestaan_intern="ja", taken=[t]) for t in taken])


def _geen_database():
    raise AssertionError("de router las de database voor een model (F2-08)")


@pytest.fixture
def client(monkeypatch):
    def maak(*taken: str):
        monkeypatch.setattr("llm_manager.register.get_register", lambda: _register(*taken))
        app = FastAPI()
        app.include_router(search_router.router, prefix="/api")
        # Eigen prefix: het pad van de fork valt anders samen met een
        # Ollama-eindpunt in de stoptest.
        app.include_router(embedding_router.router, prefix="/fork")
        zoek = SimpleNamespace(vector_search=AsyncMock(return_value=[]))
        app.dependency_overrides[get_search_service] = lambda: zoek
        app.dependency_overrides[get_model_service] = _geen_database
        bron = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(id="source:s")))
        app.dependency_overrides[get_source_service] = lambda: bron
        app.dependency_overrides[get_note_service] = lambda: SimpleNamespace(get=AsyncMock())
        from app_main.api import agent_auth

        for recht in (agent_auth.RECHT_BEHEERDER, agent_auth.RECHT_LEZEN_EXTERN):
            app.dependency_overrides[recht.dependency] = lambda: None
        from app_main.api.agent_auth import current_scope

        app.dependency_overrides[current_scope] = lambda: None
        return TestClient(app), zoek

    return maak


def test_vectorzoeken_met_het_embeddingmodel_uit_het_register(client):
    c, zoek = client("embedding")
    resp = c.post("/api/search", json={"query": "q", "type": "vector"})
    assert resp.status_code == 200, resp.text
    zoek.vector_search.assert_awaited_once()


def test_vectorzoeken_zonder_embeddingmodel_in_het_register_400(client):
    c, zoek = client("chat")
    resp = c.post("/api/search", json={"query": "q", "type": "vector"})
    assert resp.status_code == 400
    zoek.vector_search.assert_not_called()


@pytest.mark.parametrize("pad", ["/api/search/ask", "/api/search/ask/simple"])
def test_ask_zonder_chatmodel_in_het_register_400(client, pad):
    c, _ = client("embedding")
    resp = c.post(pad, json={"question": "vraag"})
    assert resp.status_code == 400
    assert "chat" in resp.json()["detail"]


def test_ask_neemt_geen_model_id_uit_de_aanvraag(client):
    """Zonder model-id's in de aanvraag en zonder database start het antwoord."""
    c, _ = client("chat", "embedding")
    resp = c.post("/api/search/ask", json={"question": "vraag"})
    assert resp.status_code == 200


@pytest.mark.parametrize("taken,status", [(("embedding",), 200), (("chat",), 400)])
def test_embed_neemt_het_model_uit_het_register(client, taken, status):
    c, _ = client(*taken)
    resp = c.post("/fork/embed", json={"item_id": "source:s", "item_type": "source",
                                      "async_processing": False})
    assert resp.status_code == status, resp.text
