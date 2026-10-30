"""Elk verzoek zonder geldige sleutel geeft 401, behalve twee paden (F2-01, ADR 0002 A).

Uitgezonderd zijn alleen ``/health`` en ``/api/auth/status``. Ook een onbekend
pad, de oude ``X-API-Key``, een ander schema dan Bearer en een sleutel met een
oud recht (``read``) geven 401. De opzoeking in de database is vervangen door
een mock; de middleware en de dependency zijn echt.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import app_main.api.agent_rate_limit as agent_rate_limit
from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.services.agents.audit_service import AgentAuditService

pytestmark = pytest.mark.echte_sleutels


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _geen_logboek_of_drempel(monkeypatch):
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    agent_rate_limit._ip_hits.clear()


def _verzoeken(client) -> list[tuple[str, str]]:
    uit = []
    for route in client.app.routes:
        assert isinstance(route, APIRoute)
        if route.path in agent_auth.ZONDER_SLEUTEL:
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            uit.append((method, re.sub(r"\{[^}]+\}", "x", route.path)))
    uit += [("GET", "/docs"), ("GET", "/redoc"), ("GET", "/bestaat-niet"),
            ("GET", "/api/bestaat-niet"), ("GET", "/health/"), ("GET", "/api/health")]
    return uit


def _alle_401(client, headers: dict) -> None:
    verzoeken = _verzoeken(client)
    assert len(verzoeken) > 200
    fouten = [
        f"{m} {p}: {r.status_code}"
        for m, p in verzoeken
        if (r := client.request(m, p, headers=headers)).status_code != 401
    ]
    assert not fouten, "\n".join(fouten)


def test_zonder_sleutel_overal_401(client) -> None:
    _alle_401(client, {})


@pytest.mark.parametrize(
    "headers",
    [
        {"X-API-Key": "ak_geldig"},
        {"Authorization": "Basic ak_geldig"},
        {"Authorization": "Bearer"},
        {"Authorization": "ak_geldig"},
    ],
    ids=["x-api-key", "basic", "bearer-leeg", "zonder-schema"],
)
def test_verkeerde_header_overal_401(client, monkeypatch, headers) -> None:
    # Ook als de sleutel zelf geldig zou zijn: alleen Bearer telt.
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(return_value={"id": "agent_keys:b", "agent_id": "b",
                                "permission": "beheerder", "person": "P"}),
    )
    _alle_401(client, headers)


@pytest.mark.parametrize(
    "rij",
    [
        None,  # onbekend of ingetrokken
        {"id": "agent_keys:r", "agent_id": "r", "permission": "read"},
        {"id": "agent_keys:a", "agent_id": "a", "permission": "admin"},
        {"id": "agent_keys:b", "agent_id": "b", "permission": "beheerder"},
    ],
    ids=["onbekend", "oud-read", "oud-admin", "beheerder-zonder-person"],
)
def test_ongeldige_sleutel_overal_401(client, monkeypatch, rij) -> None:
    monkeypatch.setattr(
        agent_auth.AgentKeyService, "authenticate", AsyncMock(return_value=rij)
    )
    _alle_401(client, {"Authorization": "Bearer ak_x"})


@pytest.mark.parametrize("path", ["/health", "/api/auth/status"])
def test_de_twee_uitzonderingen_zonder_sleutel(client, path) -> None:
    resp = client.get(path)
    assert resp.status_code == 200


def test_auth_status_meldt_alleen_dat_een_sleutel_nodig_is(client) -> None:
    assert client.get("/api/auth/status").json() == {
        "auth_enabled": True,
        "message": "Authentication is required",
    }


def test_geldige_sleutel_komt_door_de_middleware(client, monkeypatch) -> None:
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(return_value={"id": "agent_keys:e", "agent_id": "e",
                                "permission": "lezen_extern"}),
    )
    resp = client.get("/", headers={"Authorization": "Bearer ak_e"})
    assert resp.status_code == 200
