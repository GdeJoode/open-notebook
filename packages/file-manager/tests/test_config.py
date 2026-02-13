"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from file_manager.config import FileManagerConfig, get_config, set_config


class TestFileManagerConfig:
    """Tests for FileManagerConfig."""

    def test_default_values(self):
        """Test default configuration values when no env vars are set."""
        # Clear FILE_MANAGER_* env vars to test actual defaults
        env_vars_to_clear = [k for k in os.environ if k.startswith("FILE_MANAGER_")]
        with patch.dict(os.environ, {k: "" for k in env_vars_to_clear}, clear=False):
            for k in env_vars_to_clear:
                os.environ.pop(k, None)
            config = FileManagerConfig(_env_file=None)
            assert config.upload_dir == "./data/uploads"
            assert config.input_dir == "./data/input"
            assert config.output_dir == "./data/output"
            assert config.markdown_dir == "./data/markdown"
            assert config.temp_dir == "./data/temp"
            assert config.default_file_operation == "copy"
            assert config.default_naming_scheme == "date_prefix"
            assert config.max_file_size_mb == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FileManagerConfig(
            upload_dir="/custom/uploads",
            input_dir="/custom/input",
            output_dir="/custom/output",
            markdown_dir="/custom/markdown",
            temp_dir="/custom/temp",
            default_file_operation="move",
            default_naming_scheme="timestamp_prefix",
            max_file_size_mb=50,
            _env_file=None,
        )
        assert config.upload_dir == "/custom/uploads"
        assert config.default_file_operation == "move"
        assert config.default_naming_scheme == "timestamp_prefix"
        assert config.max_file_size_mb == 50

    def test_allowed_extensions_list(self):
        """Test allowed extensions parsing."""
        config = FileManagerConfig(
            allowed_extensions=".pdf,.docx,.txt",
            _env_file=None,
        )
        assert config.allowed_extensions_list == [".pdf", ".docx", ".txt"]

    def test_max_file_size_bytes(self):
        """Test max file size in bytes calculation."""
        config = FileManagerConfig(max_file_size_mb=10, _env_file=None)
        assert config.max_file_size_bytes == 10 * 1024 * 1024

    def test_from_env(self):
        """Test loading config from environment variables."""
        with patch.dict(os.environ, {
            "FILE_MANAGER_UPLOAD_DIR": "/env/uploads",
            "FILE_MANAGER_MAX_FILE_SIZE_MB": "200",
        }, clear=False):
            config = FileManagerConfig(_env_file=None)
            assert config.upload_dir == "/env/uploads"
            assert config.max_file_size_mb == 200


class TestGlobalConfig:
    """Tests for global config management."""

    def test_get_and_set_config(self):
        """Test getting and setting global config."""
        custom_config = FileManagerConfig(
            upload_dir="/test/uploads",
            _env_file=None,
        )
        set_config(custom_config)
        retrieved = get_config()
        assert retrieved.upload_dir == "/test/uploads"
