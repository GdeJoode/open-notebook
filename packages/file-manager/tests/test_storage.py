"""Tests for storage module."""

import pytest

from file_manager.storage import StorageManager


class TestStorageManager:
    """Tests for StorageManager."""

    @pytest.mark.asyncio
    async def test_setup(self, test_config, temp_dir):
        """Test storage setup creates directories."""
        manager = StorageManager(test_config)
        await manager.setup()

        assert (temp_dir / "uploads").exists()
        assert (temp_dir / "input").exists()
        assert (temp_dir / "output").exists()
        assert (temp_dir / "markdown").exists()
        assert (temp_dir / "temp").exists()

    @pytest.mark.asyncio
    async def test_store_upload(self, test_config, sample_file):
        """Test storing uploaded file."""
        manager = StorageManager(test_config)
        await manager.setup()

        dest_path = await manager.store_upload(str(sample_file))
        assert dest_path.exists()
        assert dest_path.name == sample_file.name

    @pytest.mark.asyncio
    async def test_store_upload_with_rename(self, test_config, sample_file):
        """Test storing uploaded file with custom name."""
        manager = StorageManager(test_config)
        await manager.setup()

        dest_path = await manager.store_upload(str(sample_file), "custom_name.txt")
        assert dest_path.exists()
        assert dest_path.name == "custom_name.txt"

    @pytest.mark.asyncio
    async def test_organize_file_copy(self, test_config, sample_pdf):
        """Test organizing file with copy operation."""
        manager = StorageManager(test_config)
        await manager.setup()

        input_path, output_path = await manager.organize_file(
            str(sample_pdf), file_operation="copy"
        )

        assert input_path is not None
        assert input_path.exists()
        assert output_path.exists()
        # Original should still exist after copy
        assert sample_pdf.exists()

    @pytest.mark.asyncio
    async def test_organize_file_move(self, test_config, sample_pdf):
        """Test organizing file with move operation."""
        manager = StorageManager(test_config)
        await manager.setup()

        input_path, output_path = await manager.organize_file(
            str(sample_pdf), file_operation="move"
        )

        assert input_path is not None
        assert input_path.exists()
        assert output_path.exists()
        # Original should not exist after move
        assert not sample_pdf.exists()

    @pytest.mark.asyncio
    async def test_organize_file_none_operation(self, test_config, sample_pdf):
        """Test organizing file with no input operation."""
        manager = StorageManager(test_config)
        await manager.setup()

        input_path, output_path = await manager.organize_file(
            str(sample_pdf), file_operation="none"
        )

        assert input_path is None
        assert output_path.exists()
        # Original should still exist
        assert sample_pdf.exists()

    @pytest.mark.asyncio
    async def test_organize_file_not_found(self, test_config):
        """Test organizing non-existent file."""
        manager = StorageManager(test_config)
        await manager.setup()

        with pytest.raises(FileNotFoundError):
            await manager.organize_file("/nonexistent/file.pdf")

    @pytest.mark.asyncio
    async def test_create_document_directory(self, test_config, temp_dir):
        """Test creating document directory structure."""
        manager = StorageManager(test_config)
        await manager.setup()

        paths = await manager.create_document_directory("test_doc", add_timestamp=False)

        assert paths["root"].exists()
        assert paths["images"].exists()
        assert paths["tables"].exists()
        assert paths["images"].parent == paths["root"]
        assert paths["tables"].parent == paths["root"]

    @pytest.mark.asyncio
    async def test_create_document_directory_with_timestamp(self, test_config, temp_dir):
        """Test creating document directory with timestamp."""
        manager = StorageManager(test_config)
        await manager.setup()

        paths = await manager.create_document_directory("test_doc", add_timestamp=True)

        # Name should include timestamp
        assert "test_doc_" in str(paths["root"])

    @pytest.mark.asyncio
    async def test_get_uploads(self, test_config, temp_dir):
        """Test getting uploads list."""
        manager = StorageManager(test_config)
        await manager.setup()

        # Create test files in uploads
        uploads_dir = temp_dir / "uploads"
        (uploads_dir / "file1.pdf").write_text("test")
        (uploads_dir / "file2.pdf").write_text("test")

        files = await manager.get_uploads()
        assert len(files) == 2

    @pytest.mark.asyncio
    async def test_get_outputs(self, test_config, temp_dir):
        """Test getting outputs list."""
        manager = StorageManager(test_config)
        await manager.setup()

        # Create test files in output
        output_dir = temp_dir / "output"
        (output_dir / "file1.pdf").write_text("test")

        files = await manager.get_outputs()
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_cleanup_temp(self, test_config, temp_dir):
        """Test cleaning up temp files."""
        manager = StorageManager(test_config)
        await manager.setup()

        # Create temp files
        temp_dir_path = temp_dir / "temp"
        (temp_dir_path / "temp1.txt").write_text("test")
        (temp_dir_path / "temp2.txt").write_text("test")

        count = await manager.cleanup_temp()
        assert count == 2

        # Verify files are deleted
        files = list(temp_dir_path.glob("*"))
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_get_storage_stats(self, test_config, temp_dir):
        """Test getting storage statistics."""
        manager = StorageManager(test_config)
        await manager.setup()

        # Create some test files
        (temp_dir / "uploads" / "file1.pdf").write_text("test content")
        (temp_dir / "output" / "file2.pdf").write_text("more content")

        stats = await manager.get_storage_stats()

        assert "uploads" in stats
        assert "input" in stats
        assert "output" in stats
        assert "markdown" in stats
        assert "temp" in stats

        assert stats["uploads"]["file_count"] == 1
        assert stats["output"]["file_count"] == 1
        assert stats["uploads"]["total_size_bytes"] > 0
