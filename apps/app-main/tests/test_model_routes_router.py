"""Tests for the model-routes router (Track J.5).

Covers the three endpoints with mocked repos/service/registry:
* ``GET /api/model-routes`` returns the default chain for unconfigured tasks
  and the persisted chain otherwise.
* ``PUT /api/model-routes/{task}`` validates task + provider + model id and
  upserts.
* ``GET /api/model-routes/health`` reports configured + circuit state per
  provider and is visually distinct for unconfigured cloud providers.
"""

from unittest.mock import AsyncMock, MagicMock

from app_main.api.routers.model_routes import router
from app_main.dependencies import (
    get_circuit_breaker_registry,
    get_model_route_repo,
    get_model_service,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.models.llm import ModelRoute

from tests.conftest import make_model


def _make_app(*, route_repo=None, model_svc=None, registry=None):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    if route_repo is not None:
        app.dependency_overrides[get_model_route_repo] = lambda: route_repo
    if model_svc is not None:
        app.dependency_overrides[get_model_service] = lambda: model_svc
    if registry is not None:
        app.dependency_overrides[get_circuit_breaker_registry] = lambda: registry
    return TestClient(app)


class TestListModelRoutes:

    def test_empty_returns_default_chain_for_all_tasks(self):
        repo = AsyncMock()
        repo.list_all.return_value = []
        client = _make_app(route_repo=repo)

        resp = client.get("/api/model-routes")

        assert resp.status_code == 200
        body = resp.json()
        tasks = {r["task"] for r in body}
        assert tasks == {"entity_extraction", "summarization", "chat"}
        for route in body:
            assert route["is_default"] is True
            # Default cloud chain ends in the local fallback provider.
            assert route["provider_chain"][-1]["provider"] == "ollama"
            assert all(e["provider"] == "ollama" for e in route["private_chain"])

    def test_persisted_row_overrides_default(self):
        repo = AsyncMock()
        repo.list_all.return_value = [
            ModelRoute(
                task="chat",
                provider_chain=[
                    {"provider": "openai", "model_id": None, "enabled": True},
                    {"provider": "ollama", "model_id": None, "enabled": True},
                ],
                private_chain=[
                    {"provider": "ollama", "model_id": None, "enabled": True},
                ],
            )
        ]
        client = _make_app(route_repo=repo)

        resp = client.get("/api/model-routes")

        assert resp.status_code == 200
        chat = next(r for r in resp.json() if r["task"] == "chat")
        assert chat["is_default"] is False
        assert chat["provider_chain"][0]["provider"] == "openai"
        # Other tasks still fall back to the default chain.
        extraction = next(
            r for r in resp.json() if r["task"] == "entity_extraction"
        )
        assert extraction["is_default"] is True


class TestUpdateModelRoute:

    def test_reorder_upserts_and_reflects(self):
        repo = AsyncMock()
        repo.get_by_task.return_value = ModelRoute(
            task="chat",
            provider_chain=[
                {"provider": "ollama", "model_id": None, "enabled": True},
                {"provider": "openai", "model_id": None, "enabled": True},
            ],
            private_chain=[],
        )
        svc = AsyncMock()
        client = _make_app(route_repo=repo, model_svc=svc)

        resp = client.put(
            "/api/model-routes/chat",
            json={
                "provider_chain": [
                    {"provider": "ollama", "enabled": True},
                    {"provider": "openai", "enabled": True},
                ]
            },
        )

        assert resp.status_code == 200
        repo.upsert.assert_awaited_once()
        assert resp.json()["provider_chain"][0]["provider"] == "ollama"

    def test_unknown_task_is_400(self):
        client = _make_app(route_repo=AsyncMock(), model_svc=AsyncMock())
        resp = client.put(
            "/api/model-routes/embedding",
            json={"provider_chain": []},
        )
        assert resp.status_code == 400

    def test_unknown_provider_is_400(self):
        client = _make_app(route_repo=AsyncMock(), model_svc=AsyncMock())
        resp = client.put(
            "/api/model-routes/chat",
            json={"provider_chain": [{"provider": "made_up", "enabled": True}]},
        )
        assert resp.status_code == 400

    def test_unknown_model_id_is_400(self):
        svc = AsyncMock()
        svc.get.return_value = None
        client = _make_app(route_repo=AsyncMock(), model_svc=svc)
        resp = client.put(
            "/api/model-routes/chat",
            json={
                "provider_chain": [
                    {"provider": "openai", "model_id": "model:ghost", "enabled": True}
                ]
            },
        )
        assert resp.status_code == 400

    def test_valid_model_id_passes(self):
        repo = AsyncMock()
        repo.get_by_task.return_value = None
        svc = AsyncMock()
        svc.get.return_value = make_model()
        client = _make_app(route_repo=repo, model_svc=svc)
        resp = client.put(
            "/api/model-routes/chat",
            json={
                "provider_chain": [
                    {"provider": "openai", "model_id": "model:1", "enabled": True}
                ]
            },
        )
        assert resp.status_code == 200

    def test_cloud_provider_in_private_chain_is_400(self):
        """A cloud provider in the private chain violates the LOCAL-only contract."""
        client = _make_app(route_repo=AsyncMock(), model_svc=AsyncMock())
        resp = client.put(
            "/api/model-routes/chat",
            json={
                "provider_chain": [{"provider": "ollama", "enabled": True}],
                "private_chain": [{"provider": "openai", "enabled": True}],
            },
        )
        assert resp.status_code == 400
        assert "openai" in resp.json()["detail"]

    def test_empty_provider_chain_is_400(self):
        client = _make_app(route_repo=AsyncMock(), model_svc=AsyncMock())
        resp = client.put(
            "/api/model-routes/chat",
            json={"provider_chain": []},
        )
        assert resp.status_code == 400

    def test_duplicate_provider_in_chain_is_400(self):
        client = _make_app(route_repo=AsyncMock(), model_svc=AsyncMock())
        resp = client.put(
            "/api/model-routes/chat",
            json={
                "provider_chain": [
                    {"provider": "openai", "enabled": True},
                    {"provider": "openai", "enabled": False},
                ]
            },
        )
        assert resp.status_code == 400
        assert "openai" in resp.json()["detail"]

    def test_valid_local_private_chain_passes(self):
        """Valid cloud provider_chain + LOCAL-only private_chain is the happy path."""
        repo = AsyncMock()
        repo.get_by_task.return_value = None
        svc = AsyncMock()
        client = _make_app(route_repo=repo, model_svc=svc)
        resp = client.put(
            "/api/model-routes/chat",
            json={
                "provider_chain": [
                    {"provider": "openai", "enabled": True},
                    {"provider": "ollama", "enabled": True},
                ],
                "private_chain": [{"provider": "ollama", "enabled": True}],
            },
        )
        assert resp.status_code == 200
        repo.upsert.assert_awaited_once()


class TestRoutingSummary:
    """The per-notebook recent-routing summary (Track J.6 telemetry surface)."""

    def _patch_metrics(self, monkeypatch, rows, captured=None):
        """Patch the lazily-imported execute_query to return canned metric rows.

        ``captured`` (optional dict) records the params the query was called with
        so a test can assert the notebook filter was threaded.
        """

        async def _fake_execute_query(query, params=None):
            if captured is not None:
                captured["query"] = query
                captured["params"] = params
            return rows

        import surrealdb_service.connection as conn

        monkeypatch.setattr(conn, "execute_query", _fake_execute_query)

    def test_summary_classifies_cloud_vs_local_and_fallback(self, monkeypatch):
        rows = [
            {"payload": {"served_provider": "nvidia", "task": "summarization",
                         "was_failover": False}},
            {"payload": {"served_provider": "ollama", "task": "entity_extraction",
                         "was_failover": True, "fallback_from": ["nvidia"]}},
            {"payload": {"served_provider": "ollama", "task": "chat",
                         "was_failover": False}},
        ]
        self._patch_metrics(monkeypatch, rows)
        client = _make_app()

        resp = client.get("/api/model-routes/summary")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_events"] == 3
        assert body["cloud_count"] == 1  # nvidia
        assert body["local_count"] == 2  # two ollama
        assert body["fallback_count"] == 1
        assert body["by_provider"] == {"nvidia": 1, "ollama": 2}
        assert body["by_task"] == {
            "summarization": 1,
            "entity_extraction": 1,
            "chat": 1,
        }
        assert body["notebook_id"] is None

    def test_summary_scoped_to_notebook_threads_filter(self, monkeypatch):
        captured: dict = {}
        rows = [
            {"payload": {"served_provider": "ollama", "task": "chat",
                         "was_failover": False}},
        ]
        self._patch_metrics(monkeypatch, rows, captured=captured)
        client = _make_app()

        resp = client.get("/api/model-routes/summary?notebook_id=notebook:abc")

        assert resp.status_code == 200
        assert resp.json()["notebook_id"] == "notebook:abc"
        # The notebook filter must reach the query params.
        assert captured["params"]["notebook"] == "notebook:abc"
        assert "notebook = $notebook" in captured["query"]

    def test_summary_telemetry_failure_returns_zeroed(self, monkeypatch):
        async def _boom(query, params=None):
            raise RuntimeError("metrics table gone")

        import surrealdb_service.connection as conn

        monkeypatch.setattr(conn, "execute_query", _boom)
        client = _make_app()

        resp = client.get("/api/model-routes/summary")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_events"] == 0
        assert body["cloud_count"] == 0
        assert body["local_count"] == 0


class TestModelRouteHealth:

    def test_health_reports_configured_and_circuit_state(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPEN_NOTEBOOK_DISABLE_METRICS", "1")

        registry = MagicMock()
        registry.get_nowait.return_value = None  # no breaker => closed
        client = _make_app(registry=registry)

        resp = client.get("/api/model-routes/health")

        assert resp.status_code == 200
        by_provider = {p["provider"]: p for p in resp.json()["providers"]}
        # Cloud provider with key set is configured.
        assert by_provider["openai"]["configured"] is True
        assert by_provider["openai"]["is_local"] is False
        # Cloud provider without key is unconfigured (visually distinct chip).
        assert by_provider["anthropic"]["configured"] is False
        # Local provider always configured.
        assert by_provider["ollama"]["configured"] is True
        assert by_provider["ollama"]["is_local"] is True
        # Default circuit state is closed when no breaker exists yet.
        assert by_provider["openai"]["circuit_state"] == "closed"
