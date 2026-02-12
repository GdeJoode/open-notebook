"""
Repository layer for SurrealDB operations.
"""

from surrealdb_service.repositories.base import BaseRepository, RecordRepository
from surrealdb_service.repositories.entity import EntityRepository
from surrealdb_service.repositories.file_tracking import (
    KnowledgeBaseRepository,
    PipelineCacheRepository,
    ProjectRepository,
    SourceFolderRepository,
    TrackedFileRepository,
)
from surrealdb_service.repositories.notebook import (
    ChatMessageRepository,
    ChatSessionRepository,
    NoteRepository,
    NotebookRepository,
)
from surrealdb_service.repositories.search import SearchRepository
from surrealdb_service.repositories.settings import ContentSettingsRepository
from surrealdb_service.repositories.source import (
    ChunkRepository,
    SourceEmbeddingRepository,
    SourceInsightRepository,
    SourceRepository,
)

__all__ = [
    # Base
    "BaseRepository",
    "RecordRepository",
    # Entity resolution
    "EntityRepository",
    # File tracking
    "TrackedFileRepository",
    "ProjectRepository",
    "KnowledgeBaseRepository",
    "SourceFolderRepository",
    "PipelineCacheRepository",
    # Notebook
    "NotebookRepository",
    "NoteRepository",
    "ChatSessionRepository",
    "ChatMessageRepository",
    # Source
    "SourceRepository",
    "ChunkRepository",
    "SourceInsightRepository",
    "SourceEmbeddingRepository",
    # Settings
    "ContentSettingsRepository",
    # Search
    "SearchRepository",
]
