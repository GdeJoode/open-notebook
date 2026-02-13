"""Tests for the models router."""

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.models import router
from app_main.dependencies import get_model_service
from app_main.services.model_service import ModelService
from tests.conftest import make_default_models, make_model


def _make_app(model_svc):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_model_service] = lambda: model_svc
    return TestClient(app)


class TestListModels:

    def test_list_all(self):
        svc = AsyncMock(spec=ModelService)
        svc.get_all.return_value = [make_model()]
        client = _make_app(svc)

        resp = client.get("/api/models")

        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_list_by_type(self):
        svc = AsyncMock(spec=ModelService)
        svc.get_by_type.return_value = [make_model(type="embedding")]
        client = _make_app(svc)

        resp = client.get("/api/models?type=embedding")

        assert resp.status_code == 200
        svc.get_by_type.assert_called_once_with("embedding")


class TestCreateModel:

    def test_create_model(self):
        svc = AsyncMock(spec=ModelService)
        m = make_model()
        svc.create.return_value = m
        client = _make_app(svc)

        resp = client.post(
            "/api/models",
            json={"name": "gpt-4o-mini", "provider": "openai", "type": "language"},
        )

        assert resp.status_code == 201
        assert resp.json()["name"] == "gpt-4o-mini"


class TestDeleteModel:

    def test_delete_model(self):
        svc = AsyncMock(spec=ModelService)
        svc.get.return_value = make_model()
        svc.delete.return_value = True
        client = _make_app(svc)

        resp = client.delete("/api/models/model:1")

        assert resp.status_code == 204

    def test_delete_model_not_found(self):
        svc = AsyncMock(spec=ModelService)
        svc.get.return_value = None
        client = _make_app(svc)

        resp = client.delete("/api/models/model:missing")

        assert resp.status_code == 404


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
