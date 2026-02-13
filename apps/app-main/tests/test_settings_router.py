"""Tests for the settings router."""

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.settings import router
from app_main.dependencies import get_settings_service
from app_main.services.settings_service import SettingsService
from tests.conftest import make_settings


def _make_app(settings_svc):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_settings_service] = lambda: settings_svc
    return TestClient(app)


class TestGetSettings:

    def test_get_settings(self):
        svc = AsyncMock(spec=SettingsService)
        svc.get.return_value = make_settings()
        client = _make_app(svc)

        resp = client.get("/api/settings")

        assert resp.status_code == 200
        data = resp.json()
        # Default values from ContentSettings
        assert data["default_content_processing_engine_doc"] == "docling"


class TestUpdateSettings:

    def test_update_settings(self):
        svc = AsyncMock(spec=SettingsService)
        svc.update.return_value = make_settings(default_embedding_option="always")
        client = _make_app(svc)

        resp = client.put(
            "/api/settings",
            json={"default_embedding_option": "always"},
        )

        assert resp.status_code == 200
        assert resp.json()["default_embedding_option"] == "always"

    def test_update_empty_body(self):
        svc = AsyncMock(spec=SettingsService)
        svc.update.return_value = make_settings()
        client = _make_app(svc)

        resp = client.put("/api/settings", json={})

        assert resp.status_code == 200
