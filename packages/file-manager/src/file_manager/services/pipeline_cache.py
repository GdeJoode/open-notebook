"""
Pipeline cache service.

Manages versioned caching of pipeline step outputs within source folders.
Old outputs are preserved with timestamps, never deleted.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from shared.models.file_tracking import PipelineCacheEntry, SourceFolder
from surrealdb_service.repositories.file_tracking import (
    PipelineCacheRepository,
    SourceFolderRepository,
)


class PipelineCacheService:
    """
    Manages pipeline cache: read/write with automatic versioning.

    When saving a new output for a step that already has a cached version:
    1. The old file gets a timestamp suffix (e.g. `.20260212_143000`)
    2. The old DB entry is marked `is_current=False`
    3. The new file is written with the canonical name
    4. A new DB entry is created with `is_current=True`
    """

    def __init__(
        self,
        cache_repo: PipelineCacheRepository,
        folder_repo: SourceFolderRepository,
    ):
        self.cache_repo = cache_repo
        self.folder_repo = folder_repo

    async def save_output(
        self,
        source_folder_id: str,
        pipeline_step: str,
        data: Union[str, bytes, Dict, List],
        file_format: str = "json",
        processing_time: Optional[float] = None,
        config_used: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineCacheEntry:
        """
        Save a pipeline step output, versioning any existing output.

        Args:
            source_folder_id: DB ID of the source folder.
            pipeline_step: Pipeline step name (e.g. "docling_parse").
            data: Output data (dict/list for JSON, str for text, bytes for binary).
            file_format: File format extension (json, md, txt, etc.).
            processing_time: Seconds taken to produce this output.
            config_used: Configuration dict (hashed for change detection).
            metadata: Additional metadata.

        Returns:
            Created PipelineCacheEntry record.
        """
        folder = await self.folder_repo.get_or_raise(source_folder_id)
        cache_dir = Path(folder.folder_path) / "pipeline_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        canonical_name = f"{folder.source_id}_{pipeline_step}.{file_format}"
        canonical_path = cache_dir / canonical_name

        # Check for existing current version
        existing = await self.cache_repo.get_current(source_folder_id, pipeline_step)
        new_version = 1

        if existing:
            new_version = existing.version + 1

            # Rename old file with timestamp suffix
            old_path = Path(existing.file_path)
            if old_path.exists():
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                archived_path = old_path.parent / f"{old_path.name}.{timestamp}"
                old_path.rename(archived_path)
                logger.debug(f"Archived old cache: {old_path.name} -> {archived_path.name}")

            # Mark old entry as not current
            await self.cache_repo.mark_not_current(source_folder_id, pipeline_step)

        # Write new file
        if isinstance(data, (dict, list)):
            content = json.dumps(data, indent=2, default=str)
            canonical_path.write_text(content, encoding="utf-8")
        elif isinstance(data, bytes):
            canonical_path.write_bytes(data)
        else:
            canonical_path.write_text(str(data), encoding="utf-8")

        file_size = canonical_path.stat().st_size

        # Compute config hash if config provided
        config_hash = None
        if config_used:
            import hashlib
            config_str = json.dumps(config_used, sort_keys=True, default=str)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

        # Create new DB entry
        entry_data = {
            "source_folder_id": source_folder_id,
            "source_id": folder.source_id,
            "pipeline_step": pipeline_step,
            "version": new_version,
            "is_current": True,
            "file_path": str(canonical_path),
            "file_format": file_format,
            "file_size_bytes": file_size,
            "processing_time_seconds": processing_time,
            "config_hash": config_hash,
            "metadata": metadata or {},
        }

        entry = await self.cache_repo.create(entry_data)

        # Update folder stats
        all_current = await self.cache_repo.get_all_for_source(source_folder_id)
        await self.folder_repo.update(source_folder_id, {
            "total_cache_entries": len(all_current),
        })

        logger.info(
            f"Cached {pipeline_step} v{new_version} for {folder.source_id} "
            f"({file_size} bytes)"
        )

        return entry

    async def get_output(
        self,
        source_folder_id: str,
        pipeline_step: str,
        version: Optional[int] = None,
    ) -> Optional[str]:
        """
        Read a cached pipeline output.

        Args:
            source_folder_id: DB ID of the source folder.
            pipeline_step: Pipeline step name.
            version: Specific version to retrieve (None = current).

        Returns:
            File contents as string, or None if not found.
        """
        if version is not None:
            entries = await self.cache_repo.get_version_history(
                source_folder_id, pipeline_step
            )
            entry = next((e for e in entries if e.version == version), None)
        else:
            entry = await self.cache_repo.get_current(source_folder_id, pipeline_step)

        if not entry:
            return None

        file_path = Path(entry.file_path)
        if not file_path.exists():
            logger.warning(f"Cache file missing: {file_path}")
            return None

        return file_path.read_text(encoding="utf-8")

    async def has_output(
        self,
        source_folder_id: str,
        pipeline_step: str,
    ) -> bool:
        """Check if a current cached output exists for a pipeline step."""
        entry = await self.cache_repo.get_current(source_folder_id, pipeline_step)
        if not entry:
            return False
        return Path(entry.file_path).exists()

    async def get_cache_entry(
        self,
        source_folder_id: str,
        pipeline_step: str,
    ) -> Optional[PipelineCacheEntry]:
        """Get current cache entry metadata (without reading file)."""
        return await self.cache_repo.get_current(source_folder_id, pipeline_step)

    async def list_cache_entries(
        self,
        source_folder_id: str,
    ) -> List[PipelineCacheEntry]:
        """List all current cache entries for a source folder."""
        return await self.cache_repo.get_all_for_source(source_folder_id)

    async def get_version_history(
        self,
        source_folder_id: str,
        pipeline_step: str,
    ) -> List[PipelineCacheEntry]:
        """Get all versions of a pipeline step's cache."""
        return await self.cache_repo.get_version_history(
            source_folder_id, pipeline_step
        )

    async def invalidate(
        self,
        source_folder_id: str,
        pipeline_step: str,
    ) -> bool:
        """
        Mark a pipeline step's cache as not current without deleting files.

        Returns:
            True if an entry was invalidated.
        """
        entry = await self.cache_repo.get_current(source_folder_id, pipeline_step)
        if not entry:
            return False

        await self.cache_repo.mark_not_current(source_folder_id, pipeline_step)

        # Update folder stats
        all_current = await self.cache_repo.get_all_for_source(source_folder_id)
        await self.folder_repo.update(source_folder_id, {
            "total_cache_entries": len(all_current),
        })

        return True
