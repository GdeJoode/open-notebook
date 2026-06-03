"""Tests for the settings router."""

from unittest.mock import AsyncMock

import pytest
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
        # Default values from ContentSettings (renamed in A.1b)
        assert data["parser_engine"] == "docling"


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


class TestParserEngineValidation:
    """Phase A.1b attempt-2: parser_engine must be one of the literal set.

    Prior to the fix, `SettingsResponse.parser_engine` and
    `SettingsUpdate.parser_engine` accepted arbitrary strings. A garbage
    value could be MERGE-upserted into SurrealDB before validation fired on
    response construction, leaving the singleton in a broken state. These
    tests pin the boundary check at the schema layer.
    """

    def test_put_rejects_invalid_parser_engine(self):
        svc = AsyncMock(spec=SettingsService)
        # If validation slips through, this fake update would echo back
        # the invalid value. The 422 from FastAPI must happen first.
        svc.update.return_value = make_settings()
        client = _make_app(svc)

        resp = client.put(
            "/api/settings",
            json={"parser_engine": "evil_value"},
        )

        assert resp.status_code == 422
        # The service must never see the invalid payload.
        svc.update.assert_not_awaited()

    @pytest.mark.parametrize(
        "engine",
        ["simple", "docling", "mineru", "auto"],
    )
    def test_put_accepts_each_valid_parser_engine(self, engine):
        svc = AsyncMock(spec=SettingsService)
        svc.update.return_value = make_settings(parser_engine=engine)
        client = _make_app(svc)

        resp = client.put(
            "/api/settings",
            json={"parser_engine": engine},
        )

        assert resp.status_code == 200
        assert resp.json()["parser_engine"] == engine
