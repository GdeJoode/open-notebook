"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from surrealdb_service.config import SurrealDBConfig, get_config, set_config


class TestSurrealDBConfig:
    """Tests for SurrealDBConfig."""

    def test_default_values(self):
        """Test default configuration values when no env vars are set."""
        # Clear SURREAL_* env vars to test actual defaults
        env_vars_to_clear = [k for k in os.environ if k.startswith("SURREAL_")]
        with patch.dict(os.environ, {k: "" for k in env_vars_to_clear}, clear=False):
            # Remove the keys entirely
            for k in env_vars_to_clear:
                os.environ.pop(k, None)
            config = SurrealDBConfig(
                _env_file=None,  # Disable loading from .env file
            )
            assert config.url == "ws://localhost:8000/rpc"
            assert config.username == "root"
            assert config.password == "root"
            assert config.namespace == "open_notebook"
            assert config.database == "open_notebook"
            assert config.pool_size == 5
            assert config.connection_timeout == 30

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SurrealDBConfig(
            url="ws://custom:9000/rpc",
            username="testuser",
            password="testpass",
            namespace="test_ns",
            database="test_db",
        )
        assert config.url == "ws://custom:9000/rpc"
        assert config.username == "testuser"
        assert config.password == "testpass"
        assert config.namespace == "test_ns"
        assert config.database == "test_db"

    def test_from_env_with_url(self):
        """Test from_env with SURREAL_URL set."""
        with patch.dict(os.environ, {
            "SURREAL_URL": "ws://env-host:8000/rpc",
            "SURREAL_USER": "envuser",
            "SURREAL_PASSWORD": "envpass",
            "SURREAL_NAMESPACE": "env_ns",
            "SURREAL_DATABASE": "env_db",
        }, clear=False):
            config = SurrealDBConfig.from_env()
            assert config.url == "ws://env-host:8000/rpc"
            assert config.username == "envuser"
            assert config.password == "envpass"
            assert config.namespace == "env_ns"
            assert config.database == "env_db"

    def test_from_env_with_address_port(self):
        """Test from_env with SURREAL_ADDRESS and SURREAL_PORT."""
        with patch.dict(os.environ, {
            "SURREAL_ADDRESS": "db-host",
            "SURREAL_PORT": "9000",
            "SURREAL_USER": "user1",
            "SURREAL_PASS": "pass1",  # backward compat
        }, clear=True):
            config = SurrealDBConfig.from_env()
            assert config.url == "ws://db-host:9000/rpc"
            assert config.password == "pass1"


class TestGlobalConfig:
    """Tests for global config management."""

    def test_get_and_set_config(self):
        """Test getting and setting global config."""
        custom_config = SurrealDBConfig(
            url="ws://test:8000/rpc",
            username="test",
            password="test",
        )
        set_config(custom_config)
        retrieved = get_config()
        assert retrieved.url == "ws://test:8000/rpc"
        assert retrieved.username == "test"
