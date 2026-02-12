"""Tests for PipelineCacheService."""

import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from file_manager.services.pipeline_cache import PipelineCacheService
from shared.models.file_tracking import PipelineCacheEntry, SourceFolder


def make_source_folder(**overrides) -> SourceFolder:
    defaults = {
        "id": "source_folder:test_id",
        "source_id": "Ab3kX9q2",
        "title": "Test Document",
        "slug": "Test_Document",
        "folder_path": "/tmp/sources/Test_Document_Ab3kX9q2",
        "source_type": "document",
        "total_cache_entries": 0,
        "total_folder_size_bytes": 0,
        "metadata": {},
        "created": datetime(2025, 1, 1),
        "updated": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    return SourceFolder(**defaults)


def make_cache_entry(**overrides) -> PipelineCacheEntry:
    defaults = {
        "id": "pipeline_cache:test_id",
        "source_folder_id": "source_folder:test_id",
        "source_id": "Ab3kX9q2",
        "pipeline_step": "docling_parse",
        "version": 1,
        "is_current": True,
        "file_path": "/tmp/sources/Test_Document_Ab3kX9q2/pipeline_cache/Ab3kX9q2_docling_parse.json",
        "file_format": "json",
        "file_size_bytes": 512,
        "metadata": {},
        "created": datetime(2025, 1, 1),
        "updated": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    return PipelineCacheEntry(**defaults)


class TestPipelineCacheService:
    @pytest.fixture
    def mock_cache_repo(self):
        return AsyncMock()

    @pytest.fixture
    def mock_folder_repo(self):
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_cache_repo, mock_folder_repo):
        return PipelineCacheService(
            cache_repo=mock_cache_repo,
            folder_repo=mock_folder_repo,
        )

    @pytest.mark.asyncio
    async def test_save_output_new(self, service, mock_cache_repo, mock_folder_repo, temp_dir):
        """Save a new output with no existing version."""
        folder_path = temp_dir / "sources" / "Test_Ab3kX9q2"
        (folder_path / "pipeline_cache").mkdir(parents=True)

        folder = make_source_folder(folder_path=str(folder_path))
        mock_folder_repo.get_or_raise.return_value = folder
        mock_cache_repo.get_current.return_value = None
        mock_cache_repo.get_all_for_source.return_value = []

        entry = make_cache_entry(
            file_path=str(folder_path / "pipeline_cache" / "Ab3kX9q2_docling_parse.json"),
        )
        mock_cache_repo.create.return_value = entry

        result = await service.save_output(
            source_folder_id="source_folder:test_id",
            pipeline_step="docling_parse",
            data={"parsed": True},
            file_format="json",
            processing_time=1.5,
        )

        # Verify file was written
        expected_path = folder_path / "pipeline_cache" / "Ab3kX9q2_docling_parse.json"
        assert expected_path.exists()
        content = json.loads(expected_path.read_text())
        assert content == {"parsed": True}

        # Verify DB create was called
        mock_cache_repo.create.assert_called_once()
        call_data = mock_cache_repo.create.call_args[0][0]
        assert call_data["version"] == 1
        assert call_data["is_current"] is True
        assert call_data["pipeline_step"] == "docling_parse"

    @pytest.mark.asyncio
    async def test_save_output_versioning(self, service, mock_cache_repo, mock_folder_repo, temp_dir):
        """Saving when existing version exists archives old file."""
        folder_path = temp_dir / "sources" / "Test_Ab3kX9q2"
        cache_dir = folder_path / "pipeline_cache"
        cache_dir.mkdir(parents=True)

        # Create existing file
        old_file = cache_dir / "Ab3kX9q2_docling_parse.json"
        old_file.write_text('{"old": true}')

        folder = make_source_folder(folder_path=str(folder_path))
        mock_folder_repo.get_or_raise.return_value = folder

        existing_entry = make_cache_entry(
            version=1,
            file_path=str(old_file),
        )
        mock_cache_repo.get_current.return_value = existing_entry
        mock_cache_repo.get_all_for_source.return_value = []

        new_entry = make_cache_entry(version=2)
        mock_cache_repo.create.return_value = new_entry

        await service.save_output(
            source_folder_id="source_folder:test_id",
            pipeline_step="docling_parse",
            data={"new": True},
        )

        # Old file should be archived (renamed with timestamp)
        archived_files = [f for f in cache_dir.iterdir() if "Ab3kX9q2_docling_parse.json." in f.name]
        assert len(archived_files) == 1

        # New file should exist with canonical name
        new_file = cache_dir / "Ab3kX9q2_docling_parse.json"
        assert new_file.exists()
        assert json.loads(new_file.read_text()) == {"new": True}

        # mark_not_current should have been called
        mock_cache_repo.mark_not_current.assert_called_once()

        # New entry version should be 2
        call_data = mock_cache_repo.create.call_args[0][0]
        assert call_data["version"] == 2

    @pytest.mark.asyncio
    async def test_save_output_string_data(self, service, mock_cache_repo, mock_folder_repo, temp_dir):
        """Save string data to cache."""
        folder_path = temp_dir / "sources" / "Test_Ab3kX9q2"
        (folder_path / "pipeline_cache").mkdir(parents=True)

        folder = make_source_folder(folder_path=str(folder_path))
        mock_folder_repo.get_or_raise.return_value = folder
        mock_cache_repo.get_current.return_value = None
        mock_cache_repo.get_all_for_source.return_value = []
        mock_cache_repo.create.return_value = make_cache_entry(file_format="md")

        await service.save_output(
            source_folder_id="source_folder:test_id",
            pipeline_step="summarization",
            data="# Summary\n\nThis is a summary.",
            file_format="md",
        )

        path = folder_path / "pipeline_cache" / "Ab3kX9q2_summarization.md"
        assert path.exists()
        assert path.read_text() == "# Summary\n\nThis is a summary."

    @pytest.mark.asyncio
    async def test_save_output_bytes_data(self, service, mock_cache_repo, mock_folder_repo, temp_dir):
        """Save bytes data to cache."""
        folder_path = temp_dir / "sources" / "Test_Ab3kX9q2"
        (folder_path / "pipeline_cache").mkdir(parents=True)

        folder = make_source_folder(folder_path=str(folder_path))
        mock_folder_repo.get_or_raise.return_value = folder
        mock_cache_repo.get_current.return_value = None
        mock_cache_repo.get_all_for_source.return_value = []
        mock_cache_repo.create.return_value = make_cache_entry(file_format="bin")

        await service.save_output(
            source_folder_id="source_folder:test_id",
            pipeline_step="embeddings",
            data=b"\x00\x01\x02\x03",
            file_format="bin",
        )

        path = folder_path / "pipeline_cache" / "Ab3kX9q2_embeddings.bin"
        assert path.exists()
        assert path.read_bytes() == b"\x00\x01\x02\x03"

    @pytest.mark.asyncio
    async def test_save_output_with_config_hash(self, service, mock_cache_repo, mock_folder_repo, temp_dir):
        """Config hash is computed when config_used is provided."""
        folder_path = temp_dir / "sources" / "Test_Ab3kX9q2"
        (folder_path / "pipeline_cache").mkdir(parents=True)

        folder = make_source_folder(folder_path=str(folder_path))
        mock_folder_repo.get_or_raise.return_value = folder
        mock_cache_repo.get_current.return_value = None
        mock_cache_repo.get_all_for_source.return_value = []
        mock_cache_repo.create.return_value = make_cache_entry()

        await service.save_output(
            source_folder_id="source_folder:test_id",
            pipeline_step="docling_parse",
            data={"test": True},
            config_used={"model": "granite"},
        )

        call_data = mock_cache_repo.create.call_args[0][0]
        assert call_data["config_hash"] is not None
        assert len(call_data["config_hash"]) == 12

    @pytest.mark.asyncio
    async def test_get_output(self, service, mock_cache_repo, temp_dir):
        """Get current cached output content."""
        cache_file = temp_dir / "cache.json"
        cache_file.write_text('{"result": "data"}')

        entry = make_cache_entry(file_path=str(cache_file))
        mock_cache_repo.get_current.return_value = entry

        content = await service.get_output("source_folder:test_id", "docling_parse")
        assert content == '{"result": "data"}'

    @pytest.mark.asyncio
    async def test_get_output_missing(self, service, mock_cache_repo):
        """Returns None when no cache entry exists."""
        mock_cache_repo.get_current.return_value = None
        content = await service.get_output("source_folder:test_id", "nonexistent_step")
        assert content is None

    @pytest.mark.asyncio
    async def test_get_output_file_missing(self, service, mock_cache_repo):
        """Returns None when cache entry exists but file is missing."""
        entry = make_cache_entry(file_path="/nonexistent/path.json")
        mock_cache_repo.get_current.return_value = entry

        content = await service.get_output("source_folder:test_id", "docling_parse")
        assert content is None

    @pytest.mark.asyncio
    async def test_get_output_specific_version(self, service, mock_cache_repo, temp_dir):
        """Get a specific version of cached output."""
        cache_file = temp_dir / "cache_v1.json"
        cache_file.write_text('{"version": 1}')

        v1 = make_cache_entry(version=1, file_path=str(cache_file))
        v2 = make_cache_entry(version=2, file_path="/other.json")
        mock_cache_repo.get_version_history.return_value = [v2, v1]

        content = await service.get_output("source_folder:test_id", "docling_parse", version=1)
        assert content == '{"version": 1}'

    @pytest.mark.asyncio
    async def test_has_output_true(self, service, mock_cache_repo, temp_dir):
        cache_file = temp_dir / "cache.json"
        cache_file.write_text("{}")
        entry = make_cache_entry(file_path=str(cache_file))
        mock_cache_repo.get_current.return_value = entry

        assert await service.has_output("source_folder:test_id", "docling_parse") is True

    @pytest.mark.asyncio
    async def test_has_output_false(self, service, mock_cache_repo):
        mock_cache_repo.get_current.return_value = None
        assert await service.has_output("source_folder:test_id", "docling_parse") is False

    @pytest.mark.asyncio
    async def test_get_cache_entry(self, service, mock_cache_repo):
        entry = make_cache_entry()
        mock_cache_repo.get_current.return_value = entry
        result = await service.get_cache_entry("source_folder:test_id", "docling_parse")
        assert result.pipeline_step == "docling_parse"

    @pytest.mark.asyncio
    async def test_list_cache_entries(self, service, mock_cache_repo):
        entries = [
            make_cache_entry(id=f"pipeline_cache:{i}", pipeline_step=f"step_{i}")
            for i in range(3)
        ]
        mock_cache_repo.get_all_for_source.return_value = entries
        result = await service.list_cache_entries("source_folder:test_id")
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_version_history(self, service, mock_cache_repo):
        entries = [
            make_cache_entry(version=2, is_current=True),
            make_cache_entry(version=1, is_current=False),
        ]
        mock_cache_repo.get_version_history.return_value = entries
        result = await service.get_version_history("source_folder:test_id", "docling_parse")
        assert len(result) == 2
        assert result[0].version == 2

    @pytest.mark.asyncio
    async def test_invalidate(self, service, mock_cache_repo, mock_folder_repo):
        entry = make_cache_entry()
        mock_cache_repo.get_current.return_value = entry
        mock_cache_repo.get_all_for_source.return_value = []

        result = await service.invalidate("source_folder:test_id", "docling_parse")
        assert result is True
        mock_cache_repo.mark_not_current.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent(self, service, mock_cache_repo):
        mock_cache_repo.get_current.return_value = None
        result = await service.invalidate("source_folder:test_id", "nonexistent")
        assert result is False
