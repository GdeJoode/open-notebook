"""
Source folder service.

Manages per-source directory creation, original file storage,
and folder lifecycle operations.
"""

import os
import re
import secrets
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from file_manager.mime import get_mime_type
from shared.models.file_tracking import SourceFolder
from surrealdb_service.repositories.file_tracking import SourceFolderRepository


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a directory name."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Collapse multiple underscores/spaces
    sanitized = re.sub(r'[\s_]+', '_', sanitized).strip('_')
    if len(sanitized) > 80:
        sanitized = sanitized[:80].rstrip('_')
    return sanitized


def _generate_source_id() -> str:
    """Generate an 8-character URL-safe random ID."""
    return secrets.token_urlsafe(6)


class SourceFolderService:
    """
    Manages source folders: filesystem + DB operations.

    Each source document gets its own folder with subdirectories
    for original files, ingestion artifacts, and pipeline cache.
    """

    def __init__(
        self,
        repo: SourceFolderRepository,
        storage_root: str,
    ):
        self.repo = repo
        self.storage_root = storage_root
        self.sources_dir = os.path.join(storage_root, "sources")

    def _get_subdirs(self, source_type: str) -> List[str]:
        """Get subdirectory list based on source type."""
        dirs = [
            "original",
            "ingestion",
            "ingestion/figures",
            "ingestion/tables",
            "ingestion/images",
            "pipeline_cache",
        ]
        if source_type in ("audio", "video"):
            dirs.append("transcription")
        return dirs

    async def create_folder(
        self,
        title: str,
        source_type: str = "document",
        original_filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SourceFolder:
        """
        Create a new source folder with directory structure + DB record.

        Args:
            title: Human-readable title for the source.
            source_type: Type of source (document, audio, video, text, url).
            original_filename: Original filename if available.
            mime_type: MIME type of the original file.
            file_size_bytes: Size of the original file in bytes.
            metadata: Additional metadata.

        Returns:
            Created SourceFolder record.
        """
        source_id = _generate_source_id()
        slug = _sanitize_name(title)
        folder_name = f"{slug}_{source_id}"
        folder_path = os.path.join(self.sources_dir, folder_name)

        # Create directory structure
        for subdir in self._get_subdirs(source_type):
            os.makedirs(os.path.join(folder_path, subdir), exist_ok=True)

        logger.info(f"Created source folder: {folder_path}")

        # Create DB record
        data = {
            "source_id": source_id,
            "title": title,
            "slug": slug,
            "folder_path": folder_path,
            "original_filename": original_filename,
            "file_size_bytes": file_size_bytes,
            "mime_type": mime_type,
            "source_type": source_type,
            "total_cache_entries": 0,
            "total_folder_size_bytes": 0,
            "metadata": metadata or {},
        }

        return await self.repo.create(data)

    async def store_original(
        self,
        folder_id: str,
        source_file_path: str,
        copy: bool = True,
    ) -> SourceFolder:
        """
        Store the original file into the source folder's original/ directory.

        The file is renamed to {source_id}_{original_filename} for global uniqueness.

        Args:
            folder_id: Source folder DB record ID.
            source_file_path: Path to the file to store.
            copy: If True, copy the file; if False, move it.

        Returns:
            Updated SourceFolder record.
        """
        folder = await self.repo.get_or_raise(folder_id)
        source_path = Path(source_file_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_file_path}")

        original_filename = source_path.name
        dest_filename = f"{folder.source_id}_{original_filename}"
        dest_dir = Path(folder.folder_path) / "original"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / dest_filename

        if copy:
            shutil.copy2(source_path, dest_path)
        else:
            shutil.move(str(source_path), str(dest_path))

        # Detect MIME type
        mime_type = get_mime_type(str(dest_path))
        file_size = dest_path.stat().st_size

        # Update DB record
        update_data = {
            "original_filename": original_filename,
            "original_file_path": str(dest_path),
            "file_size_bytes": file_size,
            "mime_type": mime_type,
        }

        return await self.repo.update(folder_id, update_data)

    async def get_folder(self, folder_id: str) -> Optional[SourceFolder]:
        """Get a source folder by its DB ID."""
        return await self.repo.get(folder_id)

    async def get_folder_by_source_id(self, source_id: str) -> Optional[SourceFolder]:
        """Get a source folder by its short random ID."""
        return await self.repo.get_by_source_id(source_id)

    async def get_folder_for_source(self, source_record_id: str) -> Optional[SourceFolder]:
        """Get a source folder by its linked domain Source record."""
        return await self.repo.get_by_source_record(source_record_id)

    async def link_to_source(self, folder_id: str, source_record_id: str) -> SourceFolder:
        """Link a source folder to a domain Source record."""
        return await self.repo.update(folder_id, {"source_record_id": source_record_id})

    async def list_folders(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[SourceFolder]:
        """List source folders with pagination."""
        return await self.repo.get_all(
            order_by="created DESC",
            limit=limit,
            offset=offset,
        )

    async def get_folder_contents(self, folder_id: str) -> Dict[str, Any]:
        """
        Get filesystem listing of a source folder.

        Returns:
            Dictionary with subdirectory names as keys and file lists as values.
        """
        folder = await self.repo.get_or_raise(folder_id)
        folder_path = Path(folder.folder_path)

        if not folder_path.exists():
            return {"error": "Folder does not exist on disk", "path": str(folder_path)}

        contents: Dict[str, Any] = {}
        for item in sorted(folder_path.iterdir()):
            if item.is_dir():
                files = []
                for f in sorted(item.rglob("*")):
                    if f.is_file():
                        files.append({
                            "name": f.name,
                            "path": str(f.relative_to(folder_path)),
                            "size_bytes": f.stat().st_size,
                        })
                contents[item.name] = files
            elif item.is_file():
                contents.setdefault("_root_files", []).append({
                    "name": item.name,
                    "size_bytes": item.stat().st_size,
                })

        return contents

    async def delete_folder(
        self,
        folder_id: str,
        delete_physical: bool = True,
    ) -> bool:
        """
        Delete a source folder (DB record and optionally filesystem).

        Args:
            folder_id: Source folder DB record ID.
            delete_physical: Whether to also delete the physical directory.

        Returns:
            True if deleted successfully.
        """
        folder = await self.repo.get(folder_id)
        if not folder:
            return False

        if delete_physical:
            folder_path = Path(folder.folder_path)
            if folder_path.exists():
                shutil.rmtree(folder_path)
                logger.info(f"Deleted source folder directory: {folder_path}")

        await self.repo.delete(folder_id)
        return True
