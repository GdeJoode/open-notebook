"""
Storage management - file organization and directory structure.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from loguru import logger

from file_manager.config import FileManagerConfig, get_config
from file_manager.operations import (
    copy_file,
    delete_file,
    ensure_directory,
    generate_output_filename,
    generate_unique_name,
    list_files,
    move_file,
)


class StorageManager:
    """
    Manages file storage and organization.

    Handles uploading, organizing, and retrieving files with
    configurable directory structure and naming schemes.
    """

    def __init__(self, config: Optional[FileManagerConfig] = None):
        """
        Initialize storage manager.

        Args:
            config: Optional configuration. Uses global config if not provided.
        """
        self.config = config or get_config()

    async def setup(self) -> None:
        """Set up storage directories."""
        await ensure_directory(self.config.upload_dir)
        await ensure_directory(self.config.input_dir)
        await ensure_directory(self.config.output_dir)
        await ensure_directory(self.config.markdown_dir)
        await ensure_directory(self.config.temp_dir)
        logger.info("Storage directories initialized")

    async def store_upload(
        self,
        source_path: str,
        original_filename: Optional[str] = None,
    ) -> Path:
        """
        Store an uploaded file in the uploads directory.

        Args:
            source_path: Path to the uploaded file.
            original_filename: Original filename (uses source name if None).

        Returns:
            Path to stored file.
        """
        filename = original_filename or Path(source_path).name
        destination, _ = await copy_file(
            source_path,
            self.config.upload_dir,
            filename,
            overwrite=True,
        )
        return destination

    async def organize_file(
        self,
        file_path: str,
        file_operation: Optional[Literal["copy", "move", "none"]] = None,
        naming_scheme: Optional[Literal[
            "timestamp_prefix", "date_prefix", "datetime_suffix", "original"
        ]] = None,
    ) -> Tuple[Optional[Path], Path]:
        """
        Organize a file according to file management settings.

        Workflow:
        1. Copy/Move to INPUT directory (if operation is not "none")
        2. Copy to OUTPUT directory with naming scheme

        Args:
            file_path: Path to the file to organize.
            file_operation: Operation to perform (defaults to config).
            naming_scheme: Naming scheme for output (defaults to config).

        Returns:
            Tuple of (input_path, output_path).
            input_path is None if file_operation is "none".
        """
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        operation = file_operation or self.config.default_file_operation
        scheme = naming_scheme or self.config.default_naming_scheme

        input_path = None
        source_for_output = source  # Track which file to use for output copy

        # Step 1: Organize into INPUT directory
        if operation == "copy":
            input_path, _ = await copy_file(str(source), self.config.input_dir)
            logger.info(f"Copied to INPUT: {input_path}")
        elif operation == "move":
            input_path, _ = await move_file(str(source), self.config.input_dir)
            source_for_output = input_path  # Use moved location for output
            logger.info(f"Moved to INPUT: {input_path}")
        else:
            logger.info(f"Keeping file in original location: {source}")

        # Step 2: Copy to OUTPUT directory with naming scheme
        output_filename = generate_output_filename(source.name, scheme)
        output_path, _ = await copy_file(
            str(source_for_output), self.config.output_dir, new_filename=output_filename
        )
        logger.info(f"Saved to OUTPUT: {output_path}")

        return input_path, output_path

    async def create_document_directory(
        self,
        document_name: str,
        add_timestamp: bool = True,
    ) -> Dict[str, Path]:
        """
        Create subdirectory structure for a document's markdown and assets.

        Creates:
            markdown_dir/document_name/
            markdown_dir/document_name/images/
            markdown_dir/document_name/tables/

        Args:
            document_name: Name of the document.
            add_timestamp: Whether to add timestamp for uniqueness.

        Returns:
            Dictionary with paths: {root, images, tables}.
        """
        unique_name = generate_unique_name(document_name, add_timestamp)
        root = Path(self.config.markdown_dir) / unique_name

        root_path = await ensure_directory(str(root))
        images_path = await ensure_directory(str(root / "images"))
        tables_path = await ensure_directory(str(root / "tables"))

        logger.info(f"Created document directory structure: {root_path}")

        return {
            "root": root_path,
            "images": images_path,
            "tables": tables_path,
        }

    async def get_uploads(self, pattern: str = "*") -> list[Path]:
        """
        Get list of files in uploads directory.

        Args:
            pattern: Glob pattern for filtering files.

        Returns:
            List of file paths.
        """
        return await list_files(self.config.upload_dir, pattern)

    async def get_outputs(self, pattern: str = "*") -> list[Path]:
        """
        Get list of files in output directory.

        Args:
            pattern: Glob pattern for filtering files.

        Returns:
            List of file paths.
        """
        return await list_files(self.config.output_dir, pattern)

    async def cleanup_temp(self) -> int:
        """
        Clean up temporary files.

        Returns:
            Number of files deleted.
        """
        temp_files = await list_files(self.config.temp_dir)
        count = 0
        for file_path in temp_files:
            try:
                await delete_file(str(file_path))
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")
        logger.info(f"Cleaned up {count} temporary files")
        return count

    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage stats for each directory.
        """
        stats = {}
        for name, dir_path in [
            ("uploads", self.config.upload_dir),
            ("input", self.config.input_dir),
            ("output", self.config.output_dir),
            ("markdown", self.config.markdown_dir),
            ("temp", self.config.temp_dir),
        ]:
            path = Path(dir_path)
            if path.exists():
                files = await list_files(dir_path, recursive=True)
                total_size = sum(f.stat().st_size for f in files)
                stats[name] = {
                    "path": str(path.absolute()),
                    "file_count": len(files),
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                }
            else:
                stats[name] = {
                    "path": str(path.absolute()),
                    "file_count": 0,
                    "total_size_bytes": 0,
                    "total_size_mb": 0,
                }
        return stats
