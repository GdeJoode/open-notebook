"""``GET /api/v1/auth/scope``: alleen het recht en het bereik van de sleutel (F2-01).

Voor de startcontrole van open-notebook-mcp (F3-10) en de gateway (F5-07): die
starten alleen bij precies ``EXTERN``. Het antwoord bevat niets anders, ook
geen id, persoon of label.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.services.agents.audit_service import AgentAuditService

pytestmark = pytest.mark.echte_sleutels


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), raise_server_exceptions=False)


def _sleutel(monkeypatch, permission: str) -> None:
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(
            return_value={
                "id": "agent_keys:geheim-id",
                "agent_id": "geheime-agent",
                "permission": permission,
                "person": "Geheime Persoon",
                "label": "geheim label",
            }
        ),
    )


@pytest.mark.parametrize(
    "recht, bereik",
    [
        ("lezen_extern", "EXTERN"),
        ("lezen_intern", "ALLES"),
        ("pijplijn", "ALLES"),
        ("beheerder", "ALLES"),
    ],
)
def test_bereik_per_recht_en_niets_anders(client, monkeypatch, recht, bereik) -> None:
    _sleutel(monkeypatch, recht)
    resp = client.get("/api/v1/auth/scope", headers={"Authorization": "Bearer ak_s"})
    assert resp.status_code == 200
    assert resp.json() == {"right": recht, "scope": bereik}
    assert "geheim" not in resp.text.lower()


def test_ophalen_krijgt_403(client, monkeypatch) -> None:
    # ADR 0002 B: `ophalen` mag alleen de ophaalcontrole; het bereik is leeg.
    _sleutel(monkeypatch, "ophalen")
    resp = client.get("/api/v1/auth/scope", headers={"Authorization": "Bearer ak_s"})
    assert resp.status_code == 403


def test_zonder_sleutel_401(client, monkeypatch) -> None:
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    assert client.get("/api/v1/auth/scope").status_code == 401
