"""Elke route declareert haar minimale recht; standaard weigeren (F2-01, ADR 0002 B).

1. Elke route in ``create_app().routes`` heeft precies één ``require_right(...)``,
   behalve de twee paden zonder sleutel. Een route zonder declaratie faalt hier.
2. Alleen de routes op de lijst van ADR 0002 B hebben een lager recht dan
   ``beheerder``; ``ophalen`` mag alleen de ophaalcontrole.
3. Een sleutel ``ophalen`` krijgt op elke route behalve de ophaalcontrole 403.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from shared.models.agents import Right

import app_main.api.agent_rate_limit as agent_rate_limit
from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.services.agents.audit_service import AgentAuditService

pytestmark = pytest.mark.echte_sleutels

OPHAALCONTROLE = ("GET", "/api/v1/source-map-rules/{id}/ophaalcontrole")

#: ADR 0002 B: routes met een lager recht dan `beheerder`. Een latere taak die
#: hier een route aan toevoegt, past deze lijst aan.
LAGER_DAN_BEHEERDER = {
    ("GET", "/"): Right.LEZEN_EXTERN,
    ("GET", "/api/v1/auth/scope"): Right.LEZEN_EXTERN,
    ("GET", "/api/notebooks"): Right.LEZEN_EXTERN,
    ("GET", "/api/notebooks/{notebook_id}"): Right.LEZEN_EXTERN,
    ("GET", "/api/sources/"): Right.LEZEN_EXTERN,
    ("GET", "/api/sources/{source_id}"): Right.LEZEN_EXTERN,
    ("GET", "/api/notes"): Right.LEZEN_EXTERN,
    ("GET", "/api/notes/{note_id}"): Right.LEZEN_EXTERN,
    ("POST", "/api/search"): Right.LEZEN_EXTERN,
    # F2-04: run-eindpunten voor Langflow (plan §5.4).
    ("POST", "/api/v1/runs"): Right.PIJPLIJN,
    ("PATCH", "/api/v1/runs/{run_id}"): Right.PIJPLIJN,
    ("GET", "/api/v1/runs/{run_id}"): Right.PIJPLIJN,
    ("POST", "/api/v1/import"): Right.PIJPLIJN,
    # F2-07: auditregels van modelaanroepen uit Langflow (ADR 0007).
    ("POST", "/api/v1/model-calls"): Right.PIJPLIJN,
    # F2-13: bronkaart en batches lezen, wachtrij-items aanmelden, ophalen en
    # op `verwerkt` zetten (ADR 0020).
    ("GET", "/api/v1/source-map-rules"): Right.PIJPLIJN,
    ("GET", "/api/v1/source-map-rules/{rule_id}"): Right.PIJPLIJN,
    ("GET", "/api/v1/intake/batches"): Right.PIJPLIJN,
    ("GET", "/api/v1/intake/batches/{batch_id}"): Right.PIJPLIJN,
    ("POST", "/api/v1/intake/items"): Right.PIJPLIJN,
    ("GET", "/api/v1/intake/items"): Right.PIJPLIJN,
    ("PATCH", "/api/v1/intake/items/{item_id}"): Right.PIJPLIJN,
}


def _declaraties(route: APIRoute) -> list[Right]:
    """Alle ``require_right``-declaraties in de dependencyboom van een route."""
    gevonden: list[Right] = []

    def loop(dependant) -> None:
        for dep in dependant.dependencies:
            recht = getattr(dep.call, "required_right", None)
            if recht is not None:
                gevonden.append(recht)
            loop(dep)

    loop(route.dependant)
    return gevonden


def _routes(app) -> list[tuple[str, str, APIRoute]]:
    uit = []
    for route in app.routes:
        assert isinstance(route, APIRoute), f"route zonder rechten: {route!r}"
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            uit.append((method, route.path, route))
    return uit


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture(autouse=True)
def _geen_logboek_of_drempel(monkeypatch):
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    agent_rate_limit._ip_hits.clear()


def test_elke_route_declareert_precies_een_recht(app) -> None:
    fouten = []
    for method, path, route in _routes(app):
        rechten = _declaraties(route)
        if path in agent_auth.ZONDER_SLEUTEL:
            if rechten:
                fouten.append(f"{method} {path}: uitzondering met recht {rechten}")
        elif len(rechten) != 1:
            fouten.append(f"{method} {path}: {len(rechten)} declaraties {rechten}")
    assert not fouten, "\n".join(fouten)


def test_alleen_de_lijst_heeft_een_lager_recht_dan_beheerder(app) -> None:
    gevonden = {}
    for method, path, route in _routes(app):
        if path in agent_auth.ZONDER_SLEUTEL:
            continue
        (recht,) = _declaraties(route)
        if recht == Right.OPHALEN:
            assert (method, path) == OPHAALCONTROLE, f"ophalen op {method} {path}"
        elif recht != Right.BEHEERDER:
            gevonden[(method, path)] = recht
    assert gevonden == LAGER_DAN_BEHEERDER


def test_zonder_sleutel_alleen_health_en_auth_status(app) -> None:
    paden = {path for _, path, route in _routes(app) if not _declaraties(route)}
    assert paden == {"/health", "/api/auth/status"}


def _url(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x", path)


def test_ophalen_krijgt_403_op_elke_route_behalve_de_ophaalcontrole(
    app, monkeypatch
) -> None:
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(
            return_value={"id": "agent_keys:o", "agent_id": "ophalen", "permission": "ophalen"}
        ),
    )
    client = TestClient(app, raise_server_exceptions=False)
    fouten = []
    aantal = 0
    for method, path, _ in _routes(app):
        if path in agent_auth.ZONDER_SLEUTEL or (method, path) == OPHAALCONTROLE:
            continue
        resp = client.request(
            method, _url(path), headers={"Authorization": "Bearer ak_ophalen"}
        )
        aantal += 1
        if resp.status_code != 403:
            fouten.append(f"{method} {path}: {resp.status_code}")
    assert aantal > 200
    assert not fouten, "\n".join(fouten)


@pytest.mark.parametrize(
    "recht, verwacht",
    [
        (Right.LEZEN_EXTERN, 403),
        (Right.LEZEN_INTERN, 403),
        (Right.PIJPLIJN, 403),
    ],
)
def test_lagere_rechten_krijgen_403_op_een_beheerroute(
    app, monkeypatch, recht, verwacht
) -> None:
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(return_value={"id": "agent_keys:k", "agent_id": "k", "permission": recht.value}),
    )
    client = TestClient(app, raise_server_exceptions=False)
    for method, path in [("GET", "/api/settings"), ("POST", "/api/search/ask"),
                         ("GET", "/openapi.json")]:
        resp = client.request(method, path, headers={"Authorization": "Bearer ak_k"})
        assert resp.status_code == verwacht, (method, path, resp.status_code)
