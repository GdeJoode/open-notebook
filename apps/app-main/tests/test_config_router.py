"""Tests for the config router."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.config import router


def _make_app():
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


class TestGetConfig:

    def test_get_config(self):
        client = _make_app()

        with (
            patch("app_main.api.routers.config.get_version", return_value="1.0.0"),
            patch(
                "app_main.api.routers.config.get_latest_version_cached",
                return_value=("1.0.1", True),
            ),
            patch(
                "app_main.api.routers.config.check_database_health",
                new_callable=AsyncMock,
                return_value={"status": "online"},
            ),
        ):
            resp = client.get("/api/config")

        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1.0.0"
        assert data["hasUpdate"] is True
        assert data["dbStatus"] == "online"
