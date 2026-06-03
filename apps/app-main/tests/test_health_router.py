"""Tests for the health router.

Critical contract under test: ``GET /api/health/mineru`` must always
respond 200, even when the underlying probe raises. The UI chip relies
on this to render its offline state cleanly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.health import router
from app_main.services.mineru_http_client import MineruHealthResult


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


class TestMineruHealthEndpoint:
    def test_returns_healthy_when_service_responds_ok(self):
        client = _make_app()
        with patch(
            "app_main.api.routers.health.MineruHttpClient.health_check",
            new=AsyncMock(
                return_value=MineruHealthResult(healthy=True, version="2.0.0")
            ),
        ):
            resp = client.get("/api/health/mineru")

        assert resp.status_code == 200
        data = resp.json()
        assert data["healthy"] is True
        assert data["version"] == "2.0.0"
        assert data["error"] is None

    def test_returns_unhealthy_when_service_returns_unhealthy_result(self):
        """A non-healthy result still produces a 200 — the chip needs that."""
        client = _make_app()
        with patch(
            "app_main.api.routers.health.MineruHttpClient.health_check",
            new=AsyncMock(
                return_value=MineruHealthResult(
                    healthy=False, error="connection refused"
                )
            ),
        ):
            resp = client.get("/api/health/mineru")

        assert resp.status_code == 200
        data = resp.json()
        assert data["healthy"] is False
        assert data["error"] == "connection refused"
        assert data["version"] is None

    def test_endpoint_never_raises_even_when_client_raises(self):
        """If the client itself raises, the router must catch and still return 200."""
        client = _make_app()
        with patch(
            "app_main.api.routers.health.MineruHttpClient.health_check",
            new=AsyncMock(side_effect=RuntimeError("boom — unexpected")),
        ):
            resp = client.get("/api/health/mineru")

        # Crucially still 200 — the UI must never see a 5xx for this probe.
        assert resp.status_code == 200
        data = resp.json()
        assert data["healthy"] is False
        assert "boom" in (data.get("error") or "")

    def test_returns_healthy_without_version(self):
        client = _make_app()
        with patch(
            "app_main.api.routers.health.MineruHttpClient.health_check",
            new=AsyncMock(return_value=MineruHealthResult(healthy=True)),
        ):
            resp = client.get("/api/health/mineru")

        assert resp.status_code == 200
        data = resp.json()
        assert data["healthy"] is True
        assert data["version"] is None
