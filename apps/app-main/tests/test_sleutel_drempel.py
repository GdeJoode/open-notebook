"""Drempel voor mislukte sleutels per IP-adres, vóór de database (F2-01).

Boven de drempel krijgt een verzoek met een sleutel 429 zonder opzoeking in de
database en zonder regel in het auditlog. Een verzoek zonder sleutel kost geen
databasewerk en blijft 401.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app_main.api import agent_auth, auth
from app_main.api.app import create_app
from app_main.services.agents.audit_service import AgentAuditService

pytestmark = pytest.mark.echte_sleutels

DREMPEL = 3


@pytest.fixture
def opzet(monkeypatch):
    record = AsyncMock()
    lookup = AsyncMock(return_value=None)  # elke sleutel is ongeldig
    monkeypatch.setattr(AgentAuditService, "record", record)
    monkeypatch.setattr(agent_auth.AgentKeyService, "authenticate", lookup)
    monkeypatch.setattr(auth, "MAX_MISLUKT_PER_MINUUT", DREMPEL)
    auth._mislukt.clear()
    return TestClient(create_app(), raise_server_exceptions=False), lookup, record


def _met_sleutel(client):
    return client.get("/api/notebooks", headers={"Authorization": "Bearer ak_fout"})


def test_boven_de_drempel_429_zonder_databasewerk_en_zonder_logregel(opzet):
    client, lookup, record = opzet
    for _ in range(DREMPEL):
        assert _met_sleutel(client).status_code == 401
    assert lookup.await_count == DREMPEL

    resp = _met_sleutel(client)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "60"
    assert lookup.await_count == DREMPEL  # geen opzoeking meer
    statussen = [c.kwargs.get("status") for c in record.await_args_list]
    assert statussen == [401] * DREMPEL  # de 429 staat niet in het auditlog


def test_zonder_sleutel_blijft_401_boven_de_drempel(opzet):
    client, lookup, _ = opzet
    for _ in range(DREMPEL):
        _met_sleutel(client)
    assert client.get("/api/notebooks").status_code == 401
    assert lookup.await_count == DREMPEL


def test_verzoek_zonder_sleutel_telt_niet_mee(opzet):
    client, lookup, _ = opzet
    for _ in range(DREMPEL * 2):
        assert client.get("/api/notebooks").status_code == 401
    assert _met_sleutel(client).status_code == 401
    assert lookup.await_count == 1
