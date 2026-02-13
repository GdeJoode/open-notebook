"""Tests for API routers."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_manager.api.routers import models, providers, usage
from llm_manager.api.routers.usage import set_tracker
from llm_manager.config import LLMManagerConfig, set_config
from llm_manager.manager import ModelManager
from llm_manager.usage import UsageTracker


@pytest.fixture
def app(test_config):
    """Create test FastAPI app."""
    ModelManager.reset_instance()
    app = FastAPI()
    app.include_router(models.router, prefix="/models", tags=["models"])
    app.include_router(providers.router, prefix="/providers", tags=["providers"])
    app.include_router(usage.router, prefix="/usage", tags=["usage"])
    set_tracker(UsageTracker(enabled=True))
    yield app
    ModelManager.reset_instance()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestModelsRouter:
    """Tests for models router."""

    def test_get_model_types(self, client):
        """Test getting model types."""
        response = client.get("/models/types")
        assert response.status_code == 200
        types = response.json()
        assert "language" in types
        assert "embedding" in types
        assert "speech_to_text" in types
        assert "text_to_speech" in types

    def test_validate_model_config_valid(self, client):
        """Test validating valid model config."""
        response = client.post(
            "/models/validate",
            json={
                "name": "gpt-4",
                "provider": "openai",
                "type": "language",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert len(data["warnings"]) == 0

    def test_validate_model_config_unknown_provider(self, client):
        """Test validating with unknown provider."""
        response = client.post(
            "/models/validate",
            json={
                "name": "model",
                "provider": "unknown_provider",
                "type": "language",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["warnings"]) > 0
        assert any("Unknown provider" in w for w in data["warnings"])

    def test_get_cache_stats(self, client):
        """Test getting cache stats."""
        response = client.get("/models/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_entries" in data
        assert "valid_entries" in data
        assert "cache_enabled" in data

    def test_clear_cache(self, client):
        """Test clearing cache."""
        response = client.post("/models/cache/clear")
        assert response.status_code == 200
        assert "cleared" in response.json()["status"].lower()

    def test_invalidate_model_cache(self, client):
        """Test invalidating model cache."""
        response = client.post("/models/cache/invalidate/model:123")
        assert response.status_code == 200
        assert "invalidated" in response.json()["status"].lower()

    def test_get_defaults(self, client):
        """Test getting defaults."""
        response = client.get("/models/defaults")
        assert response.status_code == 200
        data = response.json()
        assert "default_chat_model" in data
        assert "default_embedding_model" in data

    def test_update_defaults(self, client):
        """Test updating defaults."""
        response = client.put(
            "/models/defaults",
            json={"default_chat_model": "model:123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["default_chat_model"] == "model:123"


class TestProvidersRouter:
    """Tests for providers router."""

    def test_get_providers(self, client):
        """Test getting all providers."""
        response = client.get("/providers")
        assert response.status_code == 200
        providers = response.json()
        assert len(providers) > 0
        provider_names = [p["name"] for p in providers]
        assert "openai" in provider_names
        assert "anthropic" in provider_names

    def test_get_configured_providers(self, client):
        """Test getting configured providers."""
        response = client.get("/providers/configured")
        assert response.status_code == 200
        # Local providers should be included
        providers = response.json()
        provider_names = [p["name"] for p in providers]
        assert "ollama" in provider_names  # Local provider

    def test_get_provider(self, client):
        """Test getting specific provider."""
        response = client.get("/providers/openai")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "openai"
        assert data["display_name"] == "OpenAI"
        assert "api_key_env_var" in data

    def test_get_provider_not_found(self, client):
        """Test getting non-existent provider."""
        response = client.get("/providers/nonexistent")
        assert response.status_code == 404

    def test_get_provider_capabilities(self, client):
        """Test getting provider capabilities."""
        response = client.get("/providers/openai/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "openai"
        assert "capabilities" in data
        assert data["capabilities"]["language_models"] is True


class TestUsageRouter:
    """Tests for usage router."""

    def test_get_usage_metrics(self, client):
        """Test getting usage metrics."""
        response = client.get("/usage/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "total_tokens" in data
        assert "by_model" in data

    def test_get_usage_history(self, client):
        """Test getting usage history."""
        response = client.get("/usage/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_usage_history_with_limit(self, client):
        """Test getting usage history with limit."""
        response = client.get("/usage/history?limit=10")
        assert response.status_code == 200

    def test_estimate_cost(self, client):
        """Test cost estimation."""
        response = client.post(
            "/usage/estimate",
            json={
                "input_tokens": 1000,
                "output_tokens": 500,
                "input_cost_per_1k": 0.01,
                "output_cost_per_1k": 0.03,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["input_tokens"] == 1000
        assert data["output_tokens"] == 500
        assert data["total_cost"] == 0.025

    def test_reset_usage(self, client):
        """Test resetting usage."""
        response = client.post("/usage/reset")
        assert response.status_code == 200
        assert "reset" in response.json()["status"].lower()

    def test_get_usage_by_model(self, client):
        """Test getting usage by model."""
        response = client.get("/usage/summary/by-model")
        assert response.status_code == 200
        data = response.json()
        assert "by_model" in data

    def test_get_usage_by_provider(self, client):
        """Test getting usage by provider."""
        response = client.get("/usage/summary/by-provider")
        assert response.status_code == 200
        data = response.json()
        assert "by_provider" in data

    def test_get_usage_by_operation(self, client):
        """Test getting usage by operation."""
        response = client.get("/usage/summary/by-operation")
        assert response.status_code == 200
        data = response.json()
        assert "by_operation" in data
