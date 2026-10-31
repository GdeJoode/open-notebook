"""Tests for the models router."""

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.models import router
from app_main.dependencies import get_model_service
from app_main.services.model_service import ModelService
from tests.conftest import make_default_models


def _make_app(model_svc):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_model_service] = lambda: model_svc
    return TestClient(app)


class TestListModels:
    """``GET /api/models`` toont het modelregister, alleen lezen (F2-07, ADR 0007)."""

    def test_list_toont_het_register(self):
        svc = AsyncMock(spec=ModelService)
        client = _make_app(svc)

        resp = client.get("/api/models")

        assert resp.status_code == 200
        namen = {m["naam"]: m for m in resp.json()}
        assert namen["werkmodel"]["aanroepwijze"] == "llama-swap"
        assert namen["werkmodel"]["toegestaan_intern"] == "ja"
        svc.get_all.assert_not_called()


class TestGeenSchrijfroutes:

    def test_aanmaken_en_verwijderen_bestaan_niet(self):
        svc = AsyncMock(spec=ModelService)
        client = _make_app(svc)

        assert client.post("/api/models", json={"name": "m", "provider": "openai",
                                                "type": "language"}).status_code == 405
        assert client.delete("/api/models/model:1").status_code in (404, 405)
        svc.create.assert_not_called()
        svc.delete.assert_not_called()


class TestDefaultModels:

    def test_get_defaults(self):
        svc = AsyncMock(spec=ModelService)
        svc.get_defaults.return_value = make_default_models(
            default_chat_model="model:gpt4",
        )
        client = _make_app(svc)

        resp = client.get("/api/models/defaults")

        assert resp.status_code == 200
        assert resp.json()["default_chat_model"] == "model:gpt4"

    def test_update_defaults(self):
        svc = AsyncMock(spec=ModelService)
        svc.update_defaults.return_value = make_default_models(
            default_chat_model="model:new",
        )
        client = _make_app(svc)

        resp = client.put(
            "/api/models/defaults",
            json={"default_chat_model": "model:new"},
        )

        assert resp.status_code == 200
        assert resp.json()["default_chat_model"] == "model:new"
