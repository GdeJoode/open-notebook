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
from surrealdb_service.repositories.model import (
    DefaultModelsRepository,
    ModelRepository,
)
from surrealdb_service.repositories.notebook import (
    ChatMessageRepository,
    ChatSessionRepository,
    NoteRepository,
    NotebookRepository,
)
from surrealdb_service.repositories.notebook_schema import (
    NotebookSchemaRepository,
    Pass1ResultRepository,
)
from surrealdb_service.repositories.podcast import (
    EpisodeProfileRepository,
    PodcastEpisodeRepository,
    SpeakerProfileRepository,
)
from surrealdb_service.repositories.search import SearchRepository
from surrealdb_service.repositories.summary import SummaryRepository
from surrealdb_service.repositories.settings import ContentSettingsRepository
from surrealdb_service.repositories.source import (
    ChunkRepository,
    SourceEmbeddingRepository,
    SourceInsightRepository,
    SourceRepository,
)
from surrealdb_service.repositories.transformation import (
    DefaultPromptsRepository,
    TransformationRepository,
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
    # Model
    "ModelRepository",
    "DefaultModelsRepository",
    # Notebook
    "NotebookRepository",
    "NoteRepository",
    "ChatSessionRepository",
    "ChatMessageRepository",
    # Notebook schema (Phase B.1b)
    "NotebookSchemaRepository",
    "Pass1ResultRepository",
    # Podcast
    "SpeakerProfileRepository",
    "EpisodeProfileRepository",
    "PodcastEpisodeRepository",
    # Source
    "SourceRepository",
    "ChunkRepository",
    "SourceInsightRepository",
    "SourceEmbeddingRepository",
    # Settings
    "ContentSettingsRepository",
    # Search
    "SearchRepository",
    # Summary
    "SummaryRepository",
    # Transformation
    "TransformationRepository",
    "DefaultPromptsRepository",
]
