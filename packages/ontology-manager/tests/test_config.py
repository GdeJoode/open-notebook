"""Tests for ontology manager configuration."""

import os
from unittest.mock import patch

import pytest

from ontology_manager.config import OntologyManagerConfig, get_config, set_config


class TestOntologyManagerConfig:
    """Tests for OntologyManagerConfig defaults and custom values."""

    def test_default_values(self):
        """Test default configuration values."""
        env_vars_to_clear = [k for k in os.environ if k.startswith("ONTOLOGY_")]
        with patch.dict(os.environ, {}, clear=False):
            for k in env_vars_to_clear:
                os.environ.pop(k, None)
            config = OntologyManagerConfig(_env_file=None)
            assert config.ontologies_dir == "ontologies"
            assert config.default_ontology == "general"
            assert config.cache_enabled is True
            assert config.gap_tracking_enabled is True
            assert config.gap_auto_propose_threshold == 5

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = OntologyManagerConfig(
            ontologies_dir="/custom/path",
            default_ontology="scholarly",
            cache_enabled=False,
            gap_tracking_enabled=False,
            gap_auto_propose_threshold=10,
            _env_file=None,
        )
        assert config.ontologies_dir == "/custom/path"
        assert config.default_ontology == "scholarly"
        assert config.cache_enabled is False
        assert config.gap_tracking_enabled is False
        assert config.gap_auto_propose_threshold == 10

    def test_from_env_variables(self):
        """Test loading config from environment variables with ONTOLOGY_ prefix."""
        with patch.dict(os.environ, {
            "ONTOLOGY_ONTOLOGIES_DIR": "/env/ontologies",
            "ONTOLOGY_DEFAULT_ONTOLOGY": "policy",
            "ONTOLOGY_CACHE_ENABLED": "false",
            "ONTOLOGY_GAP_TRACKING_ENABLED": "false",
            "ONTOLOGY_GAP_AUTO_PROPOSE_THRESHOLD": "20",
        }, clear=False):
            config = OntologyManagerConfig(_env_file=None)
            assert config.ontologies_dir == "/env/ontologies"
            assert config.default_ontology == "policy"
            assert config.cache_enabled is False
            assert config.gap_tracking_enabled is False
            assert config.gap_auto_propose_threshold == 20

    def test_extra_fields_ignored(self):
        """Test that extra fields are silently ignored."""
        config = OntologyManagerConfig(
            ontologies_dir="ontologies",
            unknown_field="should be ignored",
            _env_file=None,
        )
        assert config.ontologies_dir == "ontologies"
        assert not hasattr(config, "unknown_field")


class TestGlobalConfig:
    """Tests for get_config() and set_config() module functions."""

    def setup_method(self):
        """Reset global config before each test."""
        import ontology_manager.config as config_module
        config_module._config = None

    def test_get_config_returns_default(self):
        """Test get_config() creates default config when none set."""
        config = get_config()
        assert isinstance(config, OntologyManagerConfig)
        assert config.default_ontology == "general"

    def test_set_config_and_get(self):
        """Test setting and retrieving global config."""
        custom = OntologyManagerConfig(
            default_ontology="scholarly",
            _env_file=None,
        )
        set_config(custom)
        retrieved = get_config()
        assert retrieved.default_ontology == "scholarly"
        assert retrieved is custom

    def test_get_config_caches(self):
        """Test that get_config returns the same instance on subsequent calls."""
        config_a = get_config()
        config_b = get_config()
        assert config_a is config_b

    def test_set_config_replaces(self):
        """Test that set_config replaces the existing config."""
        first = OntologyManagerConfig(default_ontology="first", _env_file=None)
        second = OntologyManagerConfig(default_ontology="second", _env_file=None)
        set_config(first)
        assert get_config().default_ontology == "first"
        set_config(second)
        assert get_config().default_ontology == "second"
