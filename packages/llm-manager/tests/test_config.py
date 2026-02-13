"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from llm_manager.config import LLMManagerConfig, get_config, set_config


class TestLLMManagerConfig:
    """Tests for LLMManagerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        env_vars_to_clear = [k for k in os.environ if k.startswith("LLM_")]
        with patch.dict(os.environ, {k: "" for k in env_vars_to_clear}, clear=False):
            for k in env_vars_to_clear:
                os.environ.pop(k, None)
            config = LLMManagerConfig(_env_file=None)
            assert config.default_provider == "openai"
            assert config.cache_enabled is True
            assert config.cache_ttl_seconds == 3600
            assert config.track_usage is True
            assert config.max_retries == 3
            assert config.request_timeout_seconds == 120

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMManagerConfig(
            default_provider="anthropic",
            cache_enabled=False,
            cache_ttl_seconds=1800,
            track_usage=False,
            max_retries=5,
            _env_file=None,
        )
        assert config.default_provider == "anthropic"
        assert config.cache_enabled is False
        assert config.cache_ttl_seconds == 1800
        assert config.track_usage is False
        assert config.max_retries == 5

    def test_from_env(self):
        """Test loading config from environment variables."""
        with patch.dict(os.environ, {
            "LLM_DEFAULT_PROVIDER": "google",
            "LLM_CACHE_ENABLED": "false",
            "LLM_MAX_RETRIES": "10",
        }, clear=False):
            config = LLMManagerConfig(_env_file=None)
            assert config.default_provider == "google"
            assert config.cache_enabled is False
            assert config.max_retries == 10

    def test_local_model_settings(self):
        """Test local model configuration."""
        config = LLMManagerConfig(
            local_model_base_url="http://localhost:11434",
            local_model_provider="ollama",
            _env_file=None,
        )
        assert config.local_model_base_url == "http://localhost:11434"
        assert config.local_model_provider == "ollama"


class TestGlobalConfig:
    """Tests for global config management."""

    def test_get_and_set_config(self):
        """Test getting and setting global config."""
        custom_config = LLMManagerConfig(
            default_provider="anthropic",
            _env_file=None,
        )
        set_config(custom_config)
        retrieved = get_config()
        assert retrieved.default_provider == "anthropic"
