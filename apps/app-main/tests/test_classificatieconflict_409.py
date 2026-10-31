"""Een classificatieconflict achter ``except Exception`` blijft 409 (F2-05, F2-03).

Een router met ``except Exception`` maakte van elke fout een 500. Sinds F2-05
laat ``fout_zonder_inhoud`` een ``ClassificationConflictError`` door naar de
afhandeling in ``app.py`` (409, zonder databasetekst), en een
``ClassificationValueError`` naar 422. Getoetst met een geforceerde fout achter
drie routers met ``except Exception``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from surrealdb_service.connection import (
    ClassificationConflictError,
    ClassificationValueError,
)

from app_main import dependencies
from app_main.api.app import create_app

DATABASETEKST = "Found 'extern' for field `classification` ... must conform to ... interne tekst"


@pytest.fixture
def app():
    application = create_app()
    yield application
    application.dependency_overrides.clear()


def _bron_dienst(fout: Exception) -> MagicMock:
    svc = MagicMock()
    svc.get = AsyncMock(return_value=SimpleNamespace(id="source:a"))
    svc.update = AsyncMock(side_effect=fout)
    svc.get_scoped = AsyncMock(side_effect=fout)
    return svc


@pytest.mark.parametrize(
    ("fout", "status", "melding"),
    [
        (ClassificationConflictError(DATABASETEKST), 409, "Classificatie niet toegestaan"),
        (ClassificationValueError(DATABASETEKST), 422, "Classificatie moet 'intern' of 'extern' zijn"),
    ],
    ids=["conflict", "ongeldig"],
)
def test_classificatiefout_achter_except_exception(app, fout, status, melding) -> None:
    app.dependency_overrides[dependencies.get_source_service] = lambda: _bron_dienst(fout)
    zoeken = MagicMock()
    zoeken.text_search = AsyncMock(side_effect=fout)
    app.dependency_overrides[dependencies.get_search_service] = lambda: zoeken
    client = TestClient(app, raise_server_exceptions=False)

    verzoeken = [
        ("PUT", "/api/sources/source:a", {"title": "Nieuw"}),  # update_source
        ("GET", "/api/sources/source:a", None),  # get_source
        ("POST", "/api/search", {"query": "x", "type": "text"}),  # search
    ]
    for methode, pad, body in verzoeken:
        r = client.request(methode, pad, json=body)
        assert r.status_code == status, (methode, pad, r.status_code, r.text)
        assert r.json() == {"detail": melding}
        assert "interne tekst" not in r.text


def test_andere_fout_blijft_500_zonder_inhoud(app) -> None:
    app.dependency_overrides[dependencies.get_source_service] = lambda: _bron_dienst(
        RuntimeError("interne tekst")
    )
    client = TestClient(app, raise_server_exceptions=False)
    r = client.put("/api/sources/source:a", json={"title": "Nieuw"})
    assert r.status_code == 500
    assert "interne tekst" not in r.text and "fout-id" in r.json()["detail"]
