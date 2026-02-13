"""Tests for file operations module."""

import pytest

from file_manager.operations import (
    copy_file,
    delete_file,
    ensure_directory,
    generate_output_filename,
    generate_unique_name,
    get_file_size,
    list_files,
    move_file,
    read_file,
    sanitize_filename,
    write_file,
)


class TestFilenameGeneration:
    """Tests for filename generation functions."""

    def test_generate_output_filename_original(self):
        """Test original naming scheme."""
        result = generate_output_filename("document.pdf", "original")
        assert result == "document.pdf"

    def test_generate_output_filename_date_prefix(self):
        """Test date prefix naming scheme."""
        result = generate_output_filename("document.pdf", "date_prefix")
        # Should be YYYY-MM-DD_document.pdf format
        assert result.endswith("_document.pdf")
        assert len(result.split("_")[0]) == 10  # YYYY-MM-DD

    def test_generate_output_filename_timestamp_prefix(self):
        """Test timestamp prefix naming scheme."""
        result = generate_output_filename("document.pdf", "timestamp_prefix")
        # Should be YYYYMMDD_HHMMSS_document.pdf format
        parts = result.split("_")
        assert len(parts) >= 3
        assert parts[-1] == "document.pdf"

    def test_generate_output_filename_datetime_suffix(self):
        """Test datetime suffix naming scheme."""
        result = generate_output_filename("document.pdf", "datetime_suffix")
        # Should be document_YYYYMMDD_HHMMSS.pdf format
        assert result.startswith("document_")
        assert result.endswith(".pdf")

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        assert sanitize_filename("normal_file.txt") == "normal_file.txt"
        assert sanitize_filename("file-name.pdf") == "file-name.pdf"

    def test_sanitize_filename_special_chars(self):
        """Test sanitization of special characters."""
        result = sanitize_filename("file with spaces.txt")
        assert " " not in result

    def test_sanitize_filename_empty(self):
        """Test sanitization of empty filename."""
        assert sanitize_filename("") == "unnamed"

    def test_generate_unique_name_with_timestamp(self):
        """Test unique name generation with timestamp."""
        result = generate_unique_name("document", add_timestamp=True)
        assert result.startswith("document_")

    def test_generate_unique_name_without_timestamp(self):
        """Test unique name generation without timestamp."""
        result = generate_unique_name("document", add_timestamp=False)
        assert result == "document"


class TestDirectoryOperations:
    """Tests for directory operations."""

    @pytest.mark.asyncio
    async def test_ensure_directory(self, temp_dir):
        """Test directory creation."""
        new_dir = temp_dir / "new_directory"
        result = await ensure_directory(str(new_dir))
        assert result.exists()
        assert result.is_dir()

    @pytest.mark.asyncio
    async def test_ensure_directory_existing(self, temp_dir):
        """Test ensure_directory on existing directory."""
        result = await ensure_directory(str(temp_dir))
        assert result.exists()


class TestFileOperations:
    """Tests for file operations."""

    @pytest.mark.asyncio
    async def test_copy_file(self, sample_file, temp_dir):
        """Test file copying."""
        dest_dir = temp_dir / "dest"
        dest_dir.mkdir()
        dest_path, was_copied = await copy_file(str(sample_file), str(dest_dir))
        assert was_copied
        assert dest_path.exists()
        assert dest_path.read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_copy_file_with_rename(self, sample_file, temp_dir):
        """Test file copying with rename."""
        dest_dir = temp_dir / "dest"
        dest_dir.mkdir()
        dest_path, was_copied = await copy_file(
            str(sample_file), str(dest_dir), new_filename="renamed.txt"
        )
        assert was_copied
        assert dest_path.name == "renamed.txt"

    @pytest.mark.asyncio
    async def test_copy_file_no_overwrite(self, sample_file, temp_dir):
        """Test file copying without overwrite."""
        dest_dir = temp_dir / "dest"
        dest_dir.mkdir()
        # Copy once
        await copy_file(str(sample_file), str(dest_dir))
        # Try to copy again without overwrite
        dest_path, was_copied = await copy_file(
            str(sample_file), str(dest_dir), overwrite=False
        )
        assert not was_copied

    @pytest.mark.asyncio
    async def test_copy_file_not_found(self, temp_dir):
        """Test copy_file with non-existent source."""
        with pytest.raises(FileNotFoundError):
            await copy_file("/nonexistent/file.txt", str(temp_dir))

    @pytest.mark.asyncio
    async def test_move_file(self, sample_file, temp_dir):
        """Test file moving."""
        dest_dir = temp_dir / "dest"
        dest_dir.mkdir()
        original_path = str(sample_file)
        dest_path, was_moved = await move_file(original_path, str(dest_dir))
        assert was_moved
        assert dest_path.exists()
        assert not sample_file.exists()

    @pytest.mark.asyncio
    async def test_delete_file(self, sample_file):
        """Test file deletion."""
        assert sample_file.exists()
        result = await delete_file(str(sample_file))
        assert result
        assert not sample_file.exists()

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self):
        """Test delete_file with non-existent file."""
        result = await delete_file("/nonexistent/file.txt")
        assert not result

    @pytest.mark.asyncio
    async def test_get_file_size(self, sample_file):
        """Test getting file size."""
        size = await get_file_size(str(sample_file))
        assert size == len("Hello, World!")

    @pytest.mark.asyncio
    async def test_list_files(self, temp_dir):
        """Test listing files."""
        # Create some test files
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")
        (temp_dir / "file3.pdf").write_text("content3")

        files = await list_files(str(temp_dir))
        assert len(files) == 3

        txt_files = await list_files(str(temp_dir), pattern="*.txt")
        assert len(txt_files) == 2

    @pytest.mark.asyncio
    async def test_list_files_empty_dir(self, temp_dir):
        """Test listing files in empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        files = await list_files(str(empty_dir))
        assert files == []

    @pytest.mark.asyncio
    async def test_read_write_file(self, temp_dir):
        """Test reading and writing files."""
        file_path = str(temp_dir / "test.txt")
        content = "Test content"

        # Write
        result_path = await write_file(file_path, content)
        assert result_path.exists()

        # Read
        read_content = await read_file(file_path)
        assert read_content == content
