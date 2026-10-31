"""Tests for model manager."""

import time

import pytest

from llm_manager.config import LLMManagerConfig
from llm_manager.manager import ModelManager, get_model_manager
from shared.models import DefaultModels, Model


class TestModelManager:
    """Tests for ModelManager."""

    def test_singleton_pattern(self, test_config):
        """Test that ModelManager is a singleton."""
        ModelManager.reset_instance()
        manager1 = ModelManager(test_config)
        manager2 = ModelManager()
        assert manager1 is manager2
        ModelManager.reset_instance()

    def test_reset_instance(self, test_config):
        """Test singleton reset."""
        manager1 = ModelManager(test_config)
        ModelManager.reset_instance()
        manager2 = ModelManager(test_config)
        assert manager1 is not manager2
        ModelManager.reset_instance()

    def test_clear_cache(self, model_manager):
        """Test cache clearing."""
        model_manager._model_cache["key1"] = "model1"
        model_manager._model_cache["key2"] = "model2"
        model_manager._cache_timestamps["key1"] = time.time()
        model_manager._cache_timestamps["key2"] = time.time()

        model_manager.clear_cache()

        assert len(model_manager._model_cache) == 0
        assert len(model_manager._cache_timestamps) == 0

    def test_invalidate_model(self, model_manager):
        """Test invalidating specific model cache."""
        model_manager._model_cache["model1:{}"] = "instance1"
        model_manager._model_cache["model1:{'temp': 0.5}"] = "instance1b"
        model_manager._model_cache["model2:{}"] = "instance2"
        model_manager._cache_timestamps["model1:{}"] = time.time()
        model_manager._cache_timestamps["model1:{'temp': 0.5}"] = time.time()
        model_manager._cache_timestamps["model2:{}"] = time.time()

        model_manager.invalidate_model("model1")

        assert "model1:{}" not in model_manager._model_cache
        assert "model1:{'temp': 0.5}" not in model_manager._model_cache
        assert "model2:{}" in model_manager._model_cache

    def test_get_cache_stats(self, model_manager):
        """Test cache statistics."""
        model_manager._model_cache["key1"] = "model1"
        model_manager._model_cache["key2"] = "model2"
        model_manager._cache_timestamps["key1"] = time.time()
        model_manager._cache_timestamps["key2"] = time.time() - 7200  # Expired

        stats = model_manager.get_cache_stats()

        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 1
        assert stats["cache_enabled"] is True
        assert "ttl_seconds" in stats

    def test_set_and_get_defaults(self, model_manager):
        """Test setting and getting default models."""
        defaults = DefaultModels(
            default_chat_model="model:123",
            default_embedding_model="model:456",
        )

        model_manager.set_defaults(defaults)
        retrieved = model_manager.get_defaults()

        assert retrieved is not None
        assert retrieved.default_chat_model == "model:123"
        assert retrieved.default_embedding_model == "model:456"

    def test_get_defaults_none(self, model_manager):
        """Test getting defaults when not set."""
        retrieved = model_manager.get_defaults()
        assert retrieved is None


class TestGetModelManager:
    """Tests for get_model_manager function."""

    def test_get_model_manager(self, test_config):
        """Test getting global model manager."""
        ModelManager.reset_instance()
        manager = get_model_manager()
        assert isinstance(manager, ModelManager)
        ModelManager.reset_instance()

    def test_get_model_manager_singleton(self, test_config):
        """Test that get_model_manager returns singleton."""
        ModelManager.reset_instance()
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        assert manager1 is manager2
        ModelManager.reset_instance()
