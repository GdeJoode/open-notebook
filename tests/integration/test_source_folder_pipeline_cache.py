"""
Service integration: SourceFolderService + PipelineCacheService on real filesystem.

Tests the full chain: service logic → mock repos → real filesystem I/O.
"""

import json

import pytest

pytestmark = pytest.mark.integration


class TestSourceFolderWithCache:
    """SourceFolderService and PipelineCacheService working together."""

    async def test_create_folder_then_save_cache(
        self, source_folder_service, pipeline_cache_service
    ):
        """Create folder → save JSON cache → file exists on disk with correct content."""
        folder = await source_folder_service.create_folder(title="Test Paper")

        entry = await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="docling_parse",
            data={"pages": 5, "text": "Hello world"},
        )

        from pathlib import Path

        cache_path = Path(entry.file_path)
        assert cache_path.exists()
        content = json.loads(cache_path.read_text())
        assert content["pages"] == 5
        assert content["text"] == "Hello world"

    async def test_save_and_read_json_cache(
        self, source_folder_service, pipeline_cache_service
    ):
        """Save dict → get_output() → parseable JSON matches input."""
        folder = await source_folder_service.create_folder(title="JSON Test")
        test_data = {"entities": ["Alice", "Bob"], "count": 2}

        await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="entity_extraction",
            data=test_data,
        )

        content = await pipeline_cache_service.get_output(
            folder.id, "entity_extraction"
        )
        assert content is not None
        parsed = json.loads(content)
        assert parsed["entities"] == ["Alice", "Bob"]
        assert parsed["count"] == 2

    async def test_save_and_read_text_cache(
        self, source_folder_service, pipeline_cache_service
    ):
        """Save markdown string → read back → content matches."""
        folder = await source_folder_service.create_folder(title="Markdown Test")
        md_content = "# Summary\n\nThis is a test summary with **bold** text."

        await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="summarization",
            data=md_content,
            file_format="md",
        )

        content = await pipeline_cache_service.get_output(
            folder.id, "summarization"
        )
        assert content is not None
        assert "# Summary" in content
        assert "**bold**" in content

    async def test_store_original_file(
        self, source_folder_service, integration_temp_dir
    ):
        """Create folder → store_original() → file in original/ with {source_id}_ prefix."""
        folder = await source_folder_service.create_folder(
            title="Original Store Test"
        )

        # Create a source file to store
        source_file = integration_temp_dir / "test_document.pdf"
        source_file.write_text("fake pdf content")

        updated = await source_folder_service.store_original(
            folder_id=folder.id,
            source_file_path=str(source_file),
        )

        from pathlib import Path

        original_dir = Path(folder.folder_path) / "original"
        files = list(original_dir.iterdir())
        assert len(files) == 1
        assert files[0].name.startswith(folder.source_id)
        assert "test_document.pdf" in files[0].name
        assert files[0].read_text() == "fake pdf content"

    async def test_folder_contents_listing(
        self, source_folder_service, pipeline_cache_service, integration_temp_dir
    ):
        """Create folder + store original + save cache → get_folder_contents() returns structured listing."""
        folder = await source_folder_service.create_folder(title="Contents Test")

        # Store an original file
        source_file = integration_temp_dir / "contents_test.txt"
        source_file.write_text("test content")
        await source_folder_service.store_original(
            folder_id=folder.id,
            source_file_path=str(source_file),
        )

        # Save a cache entry
        await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="parse",
            data={"result": "parsed"},
        )

        contents = await source_folder_service.get_folder_contents(folder.id)

        assert "original" in contents
        assert "pipeline_cache" in contents
        assert len(contents["original"]) == 1
        assert len(contents["pipeline_cache"]) == 1

    async def test_delete_folder_removes_directory(
        self, source_folder_service
    ):
        """Create folder → delete_folder(delete_physical=True) → directory gone."""
        folder = await source_folder_service.create_folder(
            title="Delete Test"
        )

        from pathlib import Path

        folder_path = Path(folder.folder_path)
        assert folder_path.exists()

        deleted = await source_folder_service.delete_folder(
            folder.id, delete_physical=True
        )
        assert deleted is True
        assert not folder_path.exists()

    async def test_audio_folder_has_transcription_dir(
        self, source_folder_service
    ):
        """Create folder with source_type='audio' → transcription/ subdir exists."""
        folder = await source_folder_service.create_folder(
            title="Audio Test",
            source_type="audio",
        )

        from pathlib import Path

        folder_path = Path(folder.folder_path)
        assert (folder_path / "transcription").is_dir()
        assert (folder_path / "original").is_dir()
        assert (folder_path / "pipeline_cache").is_dir()

    async def test_cache_after_original_storage(
        self, source_folder_service, pipeline_cache_service, integration_temp_dir
    ):
        """Store original then save cache → both original/ and pipeline_cache/ populated."""
        folder = await source_folder_service.create_folder(title="Both Test")

        # Store original
        source_file = integration_temp_dir / "both_test.pdf"
        source_file.write_text("pdf content")
        await source_folder_service.store_original(
            folder_id=folder.id,
            source_file_path=str(source_file),
        )

        # Save cache
        await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="docling_parse",
            data={"pages": 3},
        )

        from pathlib import Path

        folder_path = Path(folder.folder_path)
        original_files = list((folder_path / "original").iterdir())
        cache_files = list((folder_path / "pipeline_cache").iterdir())

        assert len(original_files) == 1
        assert len(cache_files) == 1
