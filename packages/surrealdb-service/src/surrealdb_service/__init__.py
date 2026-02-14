"""
SurrealDB Service - Database layer for open-notebook.

This package provides:
- Connection management for SurrealDB
- Repository pattern for all domain models
- Search capabilities (text and vector)
- REST API for database operations

Usage:
    from surrealdb_service import SourceRepository, NotebookRepository

    # Create repositories
    source_repo = SourceRepository()
    notebook_repo = NotebookRepository()

    # Use repositories
    source = await source_repo.get("source:123")
    notebooks = await notebook_repo.get_all()
"""

from surrealdb_service.config import SurrealDBConfig, get_config, set_config
from surrealdb_service.connection import (
    ConnectionPool,
    db_connection,
    ensure_record_id,
    execute_query,
    get_pool,
    parse_record_ids,
)
from surrealdb_service.migrations import AsyncMigrationManager
from surrealdb_service.repositories import (
    BaseRepository,
    ChatMessageRepository,
    ChatSessionRepository,
    ChunkRepository,
    ContentSettingsRepository,
    NoteRepository,
    NotebookRepository,
    RecordRepository,
    SearchRepository,
    SourceEmbeddingRepository,
    SourceInsightRepository,
    SourceRepository,
)

__version__ = "0.1.0"

__all__ = [
    # Migrations
    "AsyncMigrationManager",
    # Config
    "SurrealDBConfig",
    "get_config",
    "set_config",
    # Connection
    "db_connection",
    "execute_query",
    "parse_record_ids",
    "ensure_record_id",
    "ConnectionPool",
    "get_pool",
    # Base repositories
    "BaseRepository",
    "RecordRepository",
    # Domain repositories
    "NotebookRepository",
    "NoteRepository",
    "ChatSessionRepository",
    "ChatMessageRepository",
    "SourceRepository",
    "ChunkRepository",
    "SourceInsightRepository",
    "SourceEmbeddingRepository",
    "ContentSettingsRepository",
    "SearchRepository",
]
