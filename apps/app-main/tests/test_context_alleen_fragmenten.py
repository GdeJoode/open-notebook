"""De context voor een model bestaat alleen uit fragmenten (F2-05, ADR 0004).

``POST /api/chat/context`` en ``POST /api/notebooks/{id}/context`` nemen geen
inzichten en notities meer mee, ook niet als de configuratie daarom vraagt: de
context bevat per bron alleen de tekst uit de fragmenten van haar huidige
versie.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app_main import dependencies
from app_main.api.app import create_app


def _diensten():
    bron = SimpleNamespace(id="source:a", title="Openbare bron")
    source_svc = AsyncMock()
    source_svc.get_scoped.return_value = bron
    source_svc.get_text.return_value = "tekst uit de fragmenten"
    source_svc.get_insights.return_value = [
        SimpleNamespace(insight_type="samenvatting", content="afgeleide tekst")
    ]
    notebook_svc = AsyncMock()
    notebook_svc.get.return_value = SimpleNamespace(id="notebook:n")
    notebook_svc.get_sources.return_value = [{"id": "source:a"}]
    notebook_svc.get_notes.return_value = [{"id": "note:x", "title": "t", "content": "afgeleide notitie"}]
    return source_svc, notebook_svc


@pytest.fixture
def client(monkeypatch):
    source_svc, notebook_svc = _diensten()
    app = create_app()
    app.dependency_overrides[dependencies.get_source_service] = lambda: source_svc
    app.dependency_overrides[dependencies.get_notebook_service] = lambda: notebook_svc
    # `chat.py` haalt de bronnendienst binnen de functie op.
    monkeypatch.setattr(dependencies, "get_source_service", lambda: source_svc)
    yield TestClient(app), source_svc, notebook_svc
    app.dependency_overrides.clear()


CONFIGS = [
    None,
    {"sources": {"source:a": "insights"}, "notes": {"note:x": "full content"}},
    {"sources": {"source:a": "full content"}, "notes": {"note:x": "full content"}},
]


@pytest.mark.parametrize("config", CONFIGS, ids=["standaard", "inzichten", "volledig"])
def test_chat_context_alleen_fragmenten(client, config) -> None:
    c, source_svc, notebook_svc = client
    r = c.post("/api/chat/context", json={"notebook_id": "notebook:n", "context_config": config or {}})
    assert r.status_code == 200, r.text
    ctx = r.json()["context"]
    assert ctx["notes"] == []
    assert ctx["sources"] == [{"id": "source:a", "title": "Openbare bron",
                               "full_text": "tekst uit de fragmenten"}]
    assert "afgeleid" not in r.text
    source_svc.get_insights.assert_not_called()


@pytest.mark.parametrize("config", CONFIGS, ids=["standaard", "inzichten", "volledig"])
def test_notebook_context_alleen_fragmenten(client, config) -> None:
    c, source_svc, notebook_svc = client
    body = {"notebook_id": "notebook:n"}
    if config:
        body["context_config"] = config
    r = c.post("/api/notebooks/notebook:n/context", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["notes"] == []
    assert r.json()["sources"] == [{"id": "source:a", "title": "Openbare bron",
                                    "full_text": "tekst uit de fragmenten"}]
    assert "afgeleid" not in r.text
    source_svc.get_insights.assert_not_called()
    notebook_svc.get_notes.assert_not_called()
