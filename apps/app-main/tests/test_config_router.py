"""Tests for the config router."""

import asyncio
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


class TestCheckDatabaseHealth:
    """Slow is not offline; offline needs a failed probe (WB.0)."""

    @pytest.fixture(autouse=True)
    def _fresh_probe_state(self, monkeypatch):
        from app_main.api.routers import config as config_mod

        monkeypatch.setattr(config_mod, "_probe", None)
        monkeypatch.setattr(config_mod, "_last_probe_error", None)
        monkeypatch.setattr(config_mod, "HEALTH_CHECK_TIMEOUT", 0.05)
        self.mod = config_mod

    def _db(self, monkeypatch, behaviour):
        calls = []

        async def fake_execute_query(query, params=None, config=None):
            calls.append(query)
            return await behaviour()

        monkeypatch.setattr(self.mod, "execute_query", fake_execute_query)
        return calls

    async def test_fast_answer_is_online(self, monkeypatch):
        async def ok():
            return [1]

        self._db(monkeypatch, ok)
        assert await self.mod.check_database_health() == {"status": "online"}

    async def test_error_is_offline(self, monkeypatch):
        async def refused():
            raise OSError("connection refused")

        self._db(monkeypatch, refused)
        health = await self.mod.check_database_health()
        assert health["status"] == "offline"
        assert "refused" in health["error"]

    async def test_slow_sign_in_is_online_and_finishes(self, monkeypatch):
        gate = asyncio.Event()

        async def slow():
            await gate.wait()
            return [1]

        calls = self._db(monkeypatch, slow)
        assert (await self.mod.check_database_health())["status"] == "online"
        # A second check while the first probe runs waits on that probe
        # instead of starting another sign-in.
        assert (await self.mod.check_database_health())["status"] == "online"
        assert len(calls) == 1
        # The timed-out probe was not cancelled: it completes.
        gate.set()
        assert await self.mod._probe == [1]

    async def test_slow_after_a_failed_probe_is_offline(self, monkeypatch):
        outcomes = [OSError("sign-in timed out")]
        gate = asyncio.Event()

        async def fail_then_hang():
            if outcomes:
                raise outcomes.pop()
            await gate.wait()
            return [1]

        self._db(monkeypatch, fail_then_hang)
        assert (await self.mod.check_database_health())["status"] == "offline"
        health = await self.mod.check_database_health()
        assert health["status"] == "offline"
        assert "timed out" in health["error"]
        # Still offline while that probe keeps hanging, until one succeeds.
        assert (await self.mod.check_database_health())["status"] == "offline"
        gate.set()
        await self.mod._probe
        # The last finished probe succeeded, so slow again is online again.
        gate.clear()
        assert (await self.mod.check_database_health())["status"] == "online"
        gate.set()
        await self.mod._probe

    async def test_cancelled_probe_is_offline_not_an_exception(self, monkeypatch):
        """The SDK cancels a pending query when the socket drops; the probe
        task then ends cancelled while this request was not cancelled."""
        gate = asyncio.Event()
        outcomes = ["cancel"]

        async def cancelled_then_hang():
            if outcomes:
                outcomes.pop()
                raise asyncio.CancelledError()
            await gate.wait()
            return [1]

        self._db(monkeypatch, cancelled_then_hang)
        health = await self.mod.check_database_health()
        assert health["status"] == "offline"
        # It also counts as the last failure for a following slow probe.
        assert (await self.mod.check_database_health())["status"] == "offline"
        gate.set()
        await self.mod._probe

    async def test_cancelling_the_request_still_cancels_it(self, monkeypatch):
        gate = asyncio.Event()

        async def slow():
            await gate.wait()
            return [1]

        self._db(monkeypatch, slow)
        self.mod.HEALTH_CHECK_TIMEOUT = 5.0
        check = asyncio.ensure_future(self.mod.check_database_health())
        await asyncio.sleep(0.01)
        check.cancel()
        with pytest.raises(asyncio.CancelledError):
            await check
        gate.set()
        await self.mod._probe
