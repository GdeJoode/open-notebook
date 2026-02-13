"""Tests for SourceFolderService."""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from file_manager.services.source_folder import (
    SourceFolderService,
    _generate_source_id,
    _sanitize_name,
)
from shared.models.file_tracking import SourceFolder


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


class TestSanitizeName:
    def test_basic(self):
        assert _sanitize_name("Hello World") == "Hello_World"

    def test_special_characters(self):
        result = _sanitize_name('test<>:"/\\|?*file')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result

    def test_length_limit(self):
        long_name = "a" * 200
        assert len(_sanitize_name(long_name)) <= 80

    def test_collapse_underscores(self):
        assert _sanitize_name("hello___world") == "hello_world"


class TestGenerateSourceId:
    def test_length(self):
        sid = _generate_source_id()
        assert len(sid) == 8

    def test_uniqueness(self):
        ids = {_generate_source_id() for _ in range(100)}
        assert len(ids) == 100


class TestSourceFolderService:
    @pytest.fixture
    def mock_repo(self):
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_repo, temp_dir):
        return SourceFolderService(
            repo=mock_repo,
            storage_root=str(temp_dir),
        )

    @pytest.mark.asyncio
    async def test_create_folder(self, service, mock_repo, temp_dir):
        """Test folder creation creates directories and DB record."""
        mock_repo.create.return_value = make_source_folder(
            folder_path=str(temp_dir / "sources" / "Test_Doc_Ab3kX9q2")
        )

        folder = await service.create_folder(
            title="Test Doc",
            source_type="document",
            original_filename="test.pdf",
            mime_type="application/pdf",
            file_size_bytes=1024,
        )

        # Verify DB create was called
        mock_repo.create.assert_called_once()
        call_data = mock_repo.create.call_args[0][0]
        assert call_data["title"] == "Test Doc"
        assert call_data["source_type"] == "document"
        assert call_data["original_filename"] == "test.pdf"
        assert len(call_data["source_id"]) == 8

        # Verify directory was created
        created_path = Path(call_data["folder_path"])
        assert created_path.exists()
        assert (created_path / "original").exists()
        assert (created_path / "ingestion").exists()
        assert (created_path / "ingestion" / "figures").exists()
        assert (created_path / "ingestion" / "tables").exists()
        assert (created_path / "ingestion" / "images").exists()
        assert (created_path / "pipeline_cache").exists()

    @pytest.mark.asyncio
    async def test_create_folder_audio(self, service, mock_repo, temp_dir):
        """Audio sources get a transcription/ subdirectory."""
        mock_repo.create.return_value = make_source_folder(source_type="audio")

        await service.create_folder(title="Podcast", source_type="audio")

        call_data = mock_repo.create.call_args[0][0]
        created_path = Path(call_data["folder_path"])
        assert (created_path / "transcription").exists()

    @pytest.mark.asyncio
    async def test_store_original(self, service, mock_repo, temp_dir):
        """Test storing an original file into the source folder."""
        folder_path = temp_dir / "sources" / "Test_Ab3kX9q2"
        (folder_path / "original").mkdir(parents=True)

        folder = make_source_folder(folder_path=str(folder_path))
        mock_repo.get_or_raise.return_value = folder
        mock_repo.update.return_value = folder

        # Create a source file
        src_file = temp_dir / "input" / "document.pdf"
        src_file.parent.mkdir(parents=True)
        src_file.write_bytes(b"%PDF-1.4 test content")

        result = await service.store_original(
            folder_id="source_folder:test_id",
            source_file_path=str(src_file),
            copy=True,
        )

        # Original file should still exist (copy mode)
        assert src_file.exists()

        # Verify update was called with file metadata
        mock_repo.update.assert_called_once()
        update_data = mock_repo.update.call_args[0][1]
        assert update_data["original_filename"] == "document.pdf"
        assert "Ab3kX9q2_document.pdf" in update_data["original_file_path"]

    @pytest.mark.asyncio
    async def test_store_original_move(self, service, mock_repo, temp_dir):
        """Test moving (not copying) original file."""
        folder_path = temp_dir / "sources" / "Test_Ab3kX9q2"
        (folder_path / "original").mkdir(parents=True)

        folder = make_source_folder(folder_path=str(folder_path))
        mock_repo.get_or_raise.return_value = folder
        mock_repo.update.return_value = folder

        src_file = temp_dir / "input" / "doc.pdf"
        src_file.parent.mkdir(parents=True)
        src_file.write_bytes(b"%PDF-1.4 test")

        await service.store_original(
            folder_id="source_folder:test_id",
            source_file_path=str(src_file),
            copy=False,
        )

        # Original file should be gone (move mode)
        assert not src_file.exists()

    @pytest.mark.asyncio
    async def test_store_original_file_not_found(self, service, mock_repo):
        """FileNotFoundError when source file doesn't exist."""
        mock_repo.get_or_raise.return_value = make_source_folder()
        with pytest.raises(FileNotFoundError):
            await service.store_original("source_folder:test_id", "/nonexistent/file.pdf")

    @pytest.mark.asyncio
    async def test_get_folder(self, service, mock_repo):
        folder = make_source_folder()
        mock_repo.get.return_value = folder
        result = await service.get_folder("source_folder:test_id")
        assert result.source_id == "Ab3kX9q2"

    @pytest.mark.asyncio
    async def test_get_folder_by_source_id(self, service, mock_repo):
        folder = make_source_folder()
        mock_repo.get_by_source_id.return_value = folder
        result = await service.get_folder_by_source_id("Ab3kX9q2")
        assert result.title == "Test Document"

    @pytest.mark.asyncio
    async def test_get_folder_for_source(self, service, mock_repo):
        folder = make_source_folder(source_record_id="source:abc")
        mock_repo.get_by_source_record.return_value = folder
        result = await service.get_folder_for_source("source:abc")
        assert result is not None

    @pytest.mark.asyncio
    async def test_link_to_source(self, service, mock_repo):
        updated = make_source_folder(source_record_id="source:linked")
        mock_repo.update.return_value = updated
        result = await service.link_to_source("source_folder:test_id", "source:linked")
        mock_repo.update.assert_called_once_with(
            "source_folder:test_id", {"source_record_id": "source:linked"}
        )

    @pytest.mark.asyncio
    async def test_list_folders(self, service, mock_repo):
        folders = [make_source_folder(id=f"source_folder:{i}") for i in range(3)]
        mock_repo.get_all.return_value = folders
        result = await service.list_folders(limit=10, offset=0)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_folder_contents(self, service, mock_repo, temp_dir):
        folder_path = temp_dir / "sources" / "Test_Ab3kX9q2"
        (folder_path / "original").mkdir(parents=True)
        (folder_path / "pipeline_cache").mkdir(parents=True)
        (folder_path / "original" / "Ab3kX9q2_doc.pdf").write_bytes(b"test")

        folder = make_source_folder(folder_path=str(folder_path))
        mock_repo.get_or_raise.return_value = folder

        contents = await service.get_folder_contents("source_folder:test_id")
        assert "original" in contents
        assert len(contents["original"]) == 1
        assert contents["original"][0]["name"] == "Ab3kX9q2_doc.pdf"

    @pytest.mark.asyncio
    async def test_delete_folder_physical(self, service, mock_repo, temp_dir):
        folder_path = temp_dir / "sources" / "Test_Ab3kX9q2"
        folder_path.mkdir(parents=True)
        (folder_path / "test.txt").write_text("hello")

        folder = make_source_folder(folder_path=str(folder_path))
        mock_repo.get.return_value = folder
        mock_repo.delete.return_value = True

        result = await service.delete_folder("source_folder:test_id", delete_physical=True)
        assert result is True
        assert not folder_path.exists()

    @pytest.mark.asyncio
    async def test_delete_folder_db_only(self, service, mock_repo, temp_dir):
        folder_path = temp_dir / "sources" / "Test_Ab3kX9q2"
        folder_path.mkdir(parents=True)

        folder = make_source_folder(folder_path=str(folder_path))
        mock_repo.get.return_value = folder
        mock_repo.delete.return_value = True

        result = await service.delete_folder("source_folder:test_id", delete_physical=False)
        assert result is True
        assert folder_path.exists()  # Directory still present

    @pytest.mark.asyncio
    async def test_delete_folder_not_found(self, service, mock_repo):
        mock_repo.get.return_value = None
        result = await service.delete_folder("source_folder:nonexistent")
        assert result is False
