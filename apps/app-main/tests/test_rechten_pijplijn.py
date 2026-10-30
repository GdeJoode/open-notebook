"""Wat de sleutel ``pijplijn`` (Langflow) niet mag (F2-01, ADR 0002 B).

Groeit mee met F2-03 (herclassificatie), F2-13 en F2-30. Nu: sleutelbeheer
(``/api/agent-keys``) en de besluiten in de triage geven 403.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.services.agents.audit_service import AgentAuditService

pytestmark = pytest.mark.echte_sleutels


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture(autouse=True)
def _pijplijn(monkeypatch):
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(return_value={"id": "agent_keys:p", "agent_id": "langflow",
                                "permission": "pijplijn"}),
    )


def _routes_onder(app, prefix: str, methods: set[str]) -> list[tuple[str, str]]:
    uit = []
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith(prefix):
            for m in route.methods & methods:
                uit.append((m, re.sub(r"\{[^}]+\}", "x", route.path)))
    return uit


def _alle_403(app, verzoeken) -> None:
    client = TestClient(app, raise_server_exceptions=False)
    assert verzoeken
    fouten = [
        f"{m} {p}: {r.status_code}"
        for m, p in verzoeken
        if (r := client.request(m, p, headers={"Authorization": "Bearer ak_p"})).status_code
        != 403
    ]
    assert not fouten, "\n".join(fouten)


def test_pijplijn_krijgt_403_op_sleutelbeheer(app) -> None:
    verzoeken = _routes_onder(app, "/api/agent-keys", {"GET", "POST", "DELETE"})
    assert ("POST", "/api/agent-keys") in verzoeken
    _alle_403(app, verzoeken)


def test_pijplijn_krijgt_403_op_het_triagebesluit(app) -> None:
    verzoeken = _routes_onder(app, "/api/triage", {"POST", "PUT", "PATCH", "DELETE"})
    assert ("POST", "/api/triage/queue/x/decision") in verzoeken
    _alle_403(app, verzoeken)


def test_pijplijn_mag_de_leesroutes(app) -> None:
    # Positieve controle: de 403 hierboven komt van het recht, niet van de sleutel.
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/v1/auth/scope", headers={"Authorization": "Bearer ak_p"})
    assert resp.status_code == 200
    assert resp.json() == {"right": "pijplijn", "scope": "ALLES"}
