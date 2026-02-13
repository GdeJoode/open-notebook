"""Pytest configuration for llm-manager tests."""

import pytest

from llm_manager.config import LLMManagerConfig, set_config
from llm_manager.manager import ModelManager
from llm_manager.usage import UsageTracker


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = LLMManagerConfig(
        cache_enabled=True,
        cache_ttl_seconds=60,
        track_usage=True,
        _env_file=None,
    )
    set_config(config)
    return config


@pytest.fixture
def model_manager(test_config):
    """Create a fresh model manager for testing."""
    ModelManager.reset_instance()
    manager = ModelManager(test_config)
    yield manager
    ModelManager.reset_instance()


@pytest.fixture
def usage_tracker():
    """Create a fresh usage tracker for testing."""
    return UsageTracker(enabled=True, max_history_size=100)
