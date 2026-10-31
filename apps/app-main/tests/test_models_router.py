"""Tests for the models router: het register, alleen lezen (F2-07, ADR 0007).

Dat de oude routes ``defaults``, ``status`` en ``providers`` 404 geven, toetst
``tests/test_geen_verwijzingen_trackj.py`` (F2-09).
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.models import router


def _make_app():
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


class TestListModels:
    """``GET /api/models`` toont het modelregister, alleen lezen."""

    def test_list_toont_het_register(self):
        resp = _make_app().get("/api/models")

        assert resp.status_code == 200
        namen = {m["naam"]: m for m in resp.json()}
        assert namen["werkmodel"]["aanroepwijze"] == "llama-swap"
        assert namen["werkmodel"]["toegestaan_intern"] == "ja"


class TestGeenSchrijfroutes:

    def test_aanmaken_en_verwijderen_bestaan_niet(self):
        client = _make_app()

        assert client.post("/api/models", json={"name": "m", "provider": "openai",
                                                "type": "language"}).status_code == 405
        assert client.delete("/api/models/model:1").status_code in (404, 405)
