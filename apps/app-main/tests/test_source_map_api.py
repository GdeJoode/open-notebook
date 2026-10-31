"""Rechten op de bronkaart, batches en wachtrij (F2-13, ADR 0020, plan §5.3).

Een sleutel ``lezen_extern`` of ``lezen_intern`` krijgt op elke bronkaart-,
batch- en wachtrijroute 403: de bronkaart is configuratie, geen kennis, en de
naam of locatie van een bron hoeft geen MCP-client te zien. De routes worden
uit de app gelezen, zodat een nieuwe route vanzelf meedoet.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest
from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.dependencies import get_source_map_service
from app_main.services.agents.audit_service import AgentAuditService
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

pytestmark = pytest.mark.echte_sleutels

PREFIXEN = ("/api/v1/source-map-rules", "/api/v1/intake")


@pytest.fixture(scope="module")
def app():
    return create_app()


def _sleutel(monkeypatch, recht: str) -> None:
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    monkeypatch.setattr(
        agent_auth.AgentKeyService, "authenticate",
        AsyncMock(return_value={"id": f"agent_keys:{recht}", "agent_id": "test",
                                "permission": recht}),
    )


def _verzoeken(app) -> list[tuple[str, str]]:
    uit = []
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith(PREFIXEN):
            for m in route.methods - {"HEAD", "OPTIONS"}:
                uit.append((m, re.sub(r"\{[^}]+\}", "x", route.path)))
    return sorted(uit)


def test_alle_routes_gevonden(app) -> None:
    verzoeken = set(_verzoeken(app))
    assert {
        ("GET", "/api/v1/source-map-rules"), ("GET", "/api/v1/source-map-rules/x"),
        ("POST", "/api/v1/source-map-rules"), ("PATCH", "/api/v1/source-map-rules/x"),
        ("POST", "/api/v1/source-map-rules/x/confirm"),
        ("GET", "/api/v1/intake/batches"), ("GET", "/api/v1/intake/batches/x"),
        ("POST", "/api/v1/intake/batches"), ("PATCH", "/api/v1/intake/batches/x"),
        ("POST", "/api/v1/intake/batches/x/confirm"),
        ("POST", "/api/v1/intake/items"), ("GET", "/api/v1/intake/items"),
        ("PATCH", "/api/v1/intake/items/x"), ("POST", "/api/v1/intake/items/x/decision"),
    } <= verzoeken


@pytest.mark.parametrize("recht", ["lezen_extern", "lezen_intern"])
def test_leesrechten_krijgen_403_op_elke_route(app, monkeypatch, recht) -> None:
    _sleutel(monkeypatch, recht)
    client = TestClient(app, raise_server_exceptions=False)
    fouten = []
    for m, p in _verzoeken(app):
        r = client.request(m, p, json={}, headers={"Authorization": "Bearer ak_lezen"})
        if r.status_code != 403:
            fouten.append(f"{m} {p}: {r.status_code}")
        assert "location" not in r.text and "name" not in r.text
    assert not fouten, "\n".join(fouten)


def test_pijplijn_mag_regels_lezen(app, monkeypatch) -> None:
    # Positieve controle: de 403 hierboven komt van het recht, niet van de route.
    _sleutel(monkeypatch, "pijplijn")
    lijst = AsyncMock(return_value=[])
    app.dependency_overrides[get_source_map_service] = lambda: type(
        "Dienst", (), {"list": staticmethod(lijst)})()
    try:
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/api/v1/source-map-rules", headers={"Authorization": "Bearer ak_p"})
    finally:
        app.dependency_overrides.pop(get_source_map_service, None)
    assert r.status_code == 200 and r.json() == []
    lijst.assert_awaited_once()
