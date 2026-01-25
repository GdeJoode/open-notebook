"""Pytest configuration for file-manager tests."""

import tempfile
from pathlib import Path

import pytest

from file_manager.config import FileManagerConfig, set_config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration with temporary directories."""
    config = FileManagerConfig(
        upload_dir=str(temp_dir / "uploads"),
        input_dir=str(temp_dir / "input"),
        output_dir=str(temp_dir / "output"),
        markdown_dir=str(temp_dir / "markdown"),
        temp_dir=str(temp_dir / "temp"),
        _env_file=None,
    )
    set_config(config)
    return config


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("Hello, World!")
    return file_path


@pytest.fixture
def sample_pdf(temp_dir):
    """Create a sample PDF file path for testing."""
    file_path = temp_dir / "document.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake pdf content")
    return file_path
