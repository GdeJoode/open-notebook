"""
File tracking repositories for managing tracked files, projects, and knowledge bases.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from shared.models import KnowledgeBase, PipelineCacheEntry, Project, SourceFolder, TrackedFile
from shared.types.enums import FileStatus, StorageLocation
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.base import BaseRepository


class TrackedFileRepository(BaseRepository[TrackedFile]):
    """Repository for tracked file operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(TrackedFile, config)

    async def get_by_path(self, file_path: str) -> Optional[TrackedFile]:
        """Get a tracked file by its path."""
        results = await self.query(
            "file_path = $file_path",
            {"file_path": file_path},
            limit=1,
        )
        return results[0] if results else None

    async def get_by_hash(self, file_hash: str) -> Optional[TrackedFile]:
        """Get a tracked file by its hash (for deduplication)."""
        results = await self.query(
            "file_hash = $file_hash",
            {"file_hash": file_hash},
            limit=1,
        )
        return results[0] if results else None

    async def get_by_status(
        self,
        status: FileStatus,
        limit: Optional[int] = None,
    ) -> List[TrackedFile]:
        """Get all files with a specific status."""
        return await self.query(
            "status = $status",
            {"status": status.value},
            order_by="created DESC",
            limit=limit,
        )

    async def get_by_project(
        self,
        project_id: str,
        status: Optional[FileStatus] = None,
    ) -> List[TrackedFile]:
        """Get all files in a project, optionally filtered by status."""
        where = "project_id = $project_id"
        params: Dict = {"project_id": project_id}
        if status:
            where += " AND status = $status"
            params["status"] = status.value
        return await self.query(where, params, order_by="created DESC")

    async def get_by_knowledge_base(
        self,
        kb_id: str,
        status: Optional[FileStatus] = None,
    ) -> List[TrackedFile]:
        """Get all files in a knowledge base, optionally filtered by status."""
        where = "knowledge_base_id = $kb_id"
        params: Dict = {"kb_id": kb_id}
        if status:
            where += " AND status = $status"
            params["status"] = status.value
        return await self.query(where, params, order_by="created DESC")

    async def update_status(
        self,
        file_id: str,
        status: FileStatus,
        message: Optional[str] = None,
    ) -> TrackedFile:
        """Update the status of a tracked file."""
        data = {
            "status": status.value,
            "last_status_change": datetime.now(timezone.utc),
        }
        if message:
            data["status_message"] = message
        return await self.update(file_id, data)

    async def link_to_source(self, file_id: str, source_id: str) -> TrackedFile:
        """Link a tracked file to its created Source record."""
        return await self.update(file_id, {"source_id": source_id})

    async def link_to_transcription(self, file_id: str, transcription_id: str) -> TrackedFile:
        """Link a tracked file to its transcription record."""
        return await self.update(file_id, {"transcription_id": transcription_id})

    async def move_to_project(self, file_id: str, project_id: str) -> TrackedFile:
        """Move a file to a project."""
        return await self.update(file_id, {
            "project_id": project_id,
            "knowledge_base_id": None,
            "storage_location": StorageLocation.PROJECT.value,
        })

    async def move_to_knowledge_base(self, file_id: str, kb_id: str) -> TrackedFile:
        """Move a file to a knowledge base."""
        return await self.update(file_id, {
            "knowledge_base_id": kb_id,
            "project_id": None,
            "storage_location": StorageLocation.KNOWLEDGE_BASE.value,
        })

    async def get_pending_files(self, limit: int = 100) -> List[TrackedFile]:
        """Get files pending processing."""
        return await self.get_by_status(FileStatus.PENDING, limit=limit)

    async def get_stats(self) -> Dict:
        """Get statistics about tracked files."""
        query = """
        SELECT
            status,
            count() as count,
            math::sum(file_size_bytes) as total_size
        FROM tracked_file
        GROUP BY status
        """
        try:
            results = await execute_query(query, config=self.config)
            return {
                "by_status": {r["status"]: {"count": r["count"], "size": r["total_size"]} for r in results},
                "total_count": sum(r["count"] for r in results),
                "total_size": sum(r.get("total_size", 0) or 0 for r in results),
            }
        except Exception as e:
            logger.error(f"Failed to get file stats: {e}")
            return {"by_status": {}, "total_count": 0, "total_size": 0}


class ProjectRepository(BaseRepository[Project]):
    """Repository for project operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Project, config)

    async def get_by_slug(self, slug: str) -> Optional[Project]:
        """Get a project by its slug."""
        results = await self.query(
            "slug = $slug",
            {"slug": slug},
            limit=1,
        )
        return results[0] if results else None

    async def get_by_path(self, base_path: str) -> Optional[Project]:
        """Get a project by its base path."""
        results = await self.query(
            "base_path = $base_path",
            {"base_path": base_path},
            limit=1,
        )
        return results[0] if results else None

    async def get_active_projects(self) -> List[Project]:
        """Get all non-empty projects ordered by last activity."""
        return await self.query(
            "file_count > 0",
            order_by="updated DESC",
        )

    async def increment_file_count(self, project_id: str, size_bytes: int = 0) -> Project:
        """Increment file count and total size for a project."""
        query = f"""
        UPDATE {project_id} SET
            file_count = file_count + 1,
            total_size_bytes = total_size_bytes + $size
        """
        results = await execute_query(query, {"size": size_bytes}, self.config)
        if results:
            return Project(**results[0])
        return await self.get_or_raise(project_id)

    async def decrement_file_count(self, project_id: str, size_bytes: int = 0) -> Project:
        """Decrement file count and total size for a project."""
        query = f"""
        UPDATE {project_id} SET
            file_count = file_count - 1,
            total_size_bytes = total_size_bytes - $size
        """
        results = await execute_query(query, {"size": size_bytes}, self.config)
        if results:
            return Project(**results[0])
        return await self.get_or_raise(project_id)


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    """Repository for knowledge base operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(KnowledgeBase, config)

    async def get_by_slug(self, slug: str) -> Optional[KnowledgeBase]:
        """Get a knowledge base by its slug."""
        results = await self.query(
            "slug = $slug",
            {"slug": slug},
            limit=1,
        )
        return results[0] if results else None

    async def get_by_path(self, base_path: str) -> Optional[KnowledgeBase]:
        """Get a knowledge base by its base path."""
        results = await self.query(
            "base_path = $base_path",
            {"base_path": base_path},
            limit=1,
        )
        return results[0] if results else None

    async def get_default(self) -> Optional[KnowledgeBase]:
        """Get the default knowledge base (_default)."""
        return await self.get_by_slug("_default")

    async def get_or_create_default(self, storage_root: str) -> KnowledgeBase:
        """Get or create the default knowledge base."""
        existing = await self.get_default()
        if existing:
            return existing

        import os
        base_path = os.path.join(storage_root, "knowledgebases", "_default")
        return await self.create({
            "name": "Default",
            "slug": "_default",
            "description": "Default knowledge base for unassigned files",
            "base_path": base_path,
            "sources_path": os.path.join(base_path, "sources"),
            "exports_path": os.path.join(base_path, "exports"),
            "media_path": os.path.join(base_path, "media"),
        })

    async def increment_file_count(self, kb_id: str, size_bytes: int = 0) -> KnowledgeBase:
        """Increment file count and total size for a KB."""
        query = f"""
        UPDATE {kb_id} SET
            file_count = file_count + 1,
            total_size_bytes = total_size_bytes + $size
        """
        results = await execute_query(query, {"size": size_bytes}, self.config)
        if results:
            return KnowledgeBase(**results[0])
        return await self.get_or_raise(kb_id)

    async def decrement_file_count(self, kb_id: str, size_bytes: int = 0) -> KnowledgeBase:
        """Decrement file count and total size for a KB."""
        query = f"""
        UPDATE {kb_id} SET
            file_count = file_count - 1,
            total_size_bytes = total_size_bytes - $size
        """
        results = await execute_query(query, {"size": size_bytes}, self.config)
        if results:
            return KnowledgeBase(**results[0])
        return await self.get_or_raise(kb_id)


class SourceFolderRepository(BaseRepository[SourceFolder]):
    """Repository for source folder operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(SourceFolder, config)

    async def get_by_source_id(self, source_id: str) -> Optional[SourceFolder]:
        """Lookup a source folder by its short random ID."""
        results = await self.query(
            "source_id = $source_id",
            {"source_id": source_id},
            limit=1,
        )
        return results[0] if results else None

    async def get_by_source_record(self, source_record_id: str) -> Optional[SourceFolder]:
        """Lookup a source folder by its linked Source record."""
        results = await self.query(
            "source_record_id = $source_record_id",
            {"source_record_id": source_record_id},
            limit=1,
        )
        return results[0] if results else None

    async def get_by_path(self, folder_path: str) -> Optional[SourceFolder]:
        """Lookup a source folder by its filesystem path."""
        results = await self.query(
            "folder_path = $folder_path",
            {"folder_path": folder_path},
            limit=1,
        )
        return results[0] if results else None

    async def search_by_filename(
        self,
        filename: str,
        mime_type: Optional[str] = None,
    ) -> List[SourceFolder]:
        """Search source folders by original filename for duplicate detection."""
        where = "original_filename CONTAINS $filename"
        params: Dict = {"filename": filename}
        if mime_type:
            where += " AND mime_type = $mime_type"
            params["mime_type"] = mime_type
        return await self.query(where, params, order_by="created DESC")


class PipelineCacheRepository(BaseRepository[PipelineCacheEntry]):
    """Repository for pipeline cache operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(PipelineCacheEntry, config)

    async def get_current(
        self,
        source_folder_id: str,
        pipeline_step: str,
    ) -> Optional[PipelineCacheEntry]:
        """Get the active cached output for a pipeline step."""
        results = await self.query(
            "source_folder_id = $sf_id AND pipeline_step = $step AND is_current = true",
            {"sf_id": source_folder_id, "step": pipeline_step},
            limit=1,
        )
        return results[0] if results else None

    async def get_all_for_source(self, source_folder_id: str) -> List[PipelineCacheEntry]:
        """Get all current cache entries for a source folder."""
        return await self.query(
            "source_folder_id = $sf_id AND is_current = true",
            {"sf_id": source_folder_id},
            order_by="pipeline_step ASC",
        )

    async def get_version_history(
        self,
        source_folder_id: str,
        pipeline_step: str,
    ) -> List[PipelineCacheEntry]:
        """Get all versions of a pipeline step's cache."""
        return await self.query(
            "source_folder_id = $sf_id AND pipeline_step = $step",
            {"sf_id": source_folder_id, "step": pipeline_step},
            order_by="version DESC",
        )

    async def mark_not_current(
        self,
        source_folder_id: str,
        pipeline_step: str,
    ) -> None:
        """Mark all existing entries for a step as not current."""
        query = """
        UPDATE pipeline_cache SET is_current = false
        WHERE source_folder_id = $sf_id AND pipeline_step = $step AND is_current = true
        """
        try:
            await execute_query(
                query,
                {"sf_id": source_folder_id, "step": pipeline_step},
                self.config,
            )
        except Exception as e:
            logger.error(f"Failed to mark cache entries as not current: {e}")
