"""Pytest configuration for file-manager tests."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from file_manager.config import FileManagerConfig, set_config
from shared.models.file_tracking import KnowledgeBase, Project, TrackedFile
from shared.types.enums import FileStatus, StorageLocation


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


# --- Mock repository fixtures ---


@pytest.fixture
def mock_project_repo():
    """Mock ProjectRepository with async methods."""
    return AsyncMock()


@pytest.fixture
def mock_kb_repo():
    """Mock KnowledgeBaseRepository."""
    return AsyncMock()


@pytest.fixture
def mock_file_repo():
    """Mock TrackedFileRepository."""
    return AsyncMock()


# --- Model factory helpers ---


def make_project(**overrides) -> Project:
    """Create a Project instance with sensible defaults."""
    defaults = {
        "id": "project:test_id",
        "name": "Test Project",
        "slug": "test-project",
        "description": "A test project",
        "base_path": "/tmp/projects/test-project",
        "input_path": "/tmp/projects/test-project/input",
        "output_path": "/tmp/projects/test-project/output",
        "exports_path": "/tmp/projects/test-project/exports",
        "file_count": 0,
        "total_size_bytes": 0,
        "is_processing": False,
        "created": datetime(2025, 1, 1),
        "updated": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    return Project(**defaults)


def make_kb(**overrides) -> KnowledgeBase:
    """Create a KnowledgeBase instance with sensible defaults."""
    defaults = {
        "id": "knowledge_base:test_id",
        "name": "Test KB",
        "slug": "test-kb",
        "description": "A test knowledge base",
        "base_path": "/tmp/knowledgebases/test-kb",
        "sources_path": "/tmp/knowledgebases/test-kb/sources",
        "exports_path": "/tmp/knowledgebases/test-kb/exports",
        "media_path": "/tmp/knowledgebases/test-kb/media",
        "file_count": 0,
        "total_size_bytes": 0,
        "created": datetime(2025, 1, 1),
        "updated": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    return KnowledgeBase(**defaults)


def make_tracked_file(**overrides) -> TrackedFile:
    """Create a TrackedFile instance with sensible defaults."""
    defaults = {
        "id": "tracked_file:test_id",
        "filename": "document.pdf",
        "file_path": "/tmp/test/document.pdf",
        "file_hash": "abc123hash",
        "file_size_bytes": 1024,
        "mime_type": "application/pdf",
        "status": FileStatus.PENDING.value,
        "storage_location": StorageLocation.TEMP.value,
        "created": datetime(2025, 1, 1),
        "updated": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    return TrackedFile(**defaults)
