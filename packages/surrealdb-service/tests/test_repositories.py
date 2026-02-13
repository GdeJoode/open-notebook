"""Tests for repository classes (unit tests without database)."""

import pytest

from shared.models import Notebook, Note, Source, Chunk, ContentSettings
from surrealdb_service.repositories import (
    BaseRepository,
    ChunkRepository,
    ContentSettingsRepository,
    NoteRepository,
    NotebookRepository,
    RecordRepository,
    SourceRepository,
)


class TestBaseRepository:
    """Tests for BaseRepository initialization."""

    def test_notebook_repository_table_name(self):
        """Test NotebookRepository has correct table_name."""
        repo = NotebookRepository()
        assert repo.table_name == "notebook"
        assert repo.model_class == Notebook

    def test_source_repository_table_name(self):
        """Test SourceRepository has correct table_name."""
        repo = SourceRepository()
        assert repo.table_name == "source"
        assert repo.model_class == Source

    def test_note_repository_table_name(self):
        """Test NoteRepository has correct table_name."""
        repo = NoteRepository()
        assert repo.table_name == "note"
        assert repo.model_class == Note

    def test_chunk_repository_table_name(self):
        """Test ChunkRepository has correct table_name."""
        repo = ChunkRepository()
        assert repo.table_name == "chunk"
        assert repo.model_class == Chunk


class TestRecordRepository:
    """Tests for RecordRepository initialization."""

    def test_content_settings_repository(self):
        """Test ContentSettingsRepository has correct record_id."""
        repo = ContentSettingsRepository()
        assert repo.record_id == "open_notebook:content_settings"
        assert repo.model_class == ContentSettings


class TestRepositoryMethods:
    """Test repository method signatures and structure."""

    def test_base_repository_methods_exist(self):
        """Test that BaseRepository has expected methods."""
        repo = NotebookRepository()
        assert hasattr(repo, "create")
        assert hasattr(repo, "get")
        assert hasattr(repo, "get_or_raise")
        assert hasattr(repo, "get_all")
        assert hasattr(repo, "update")
        assert hasattr(repo, "delete")
        assert hasattr(repo, "upsert")
        assert hasattr(repo, "query")
        assert hasattr(repo, "count")
        assert hasattr(repo, "exists")
        assert hasattr(repo, "relate")

    def test_notebook_repository_methods_exist(self):
        """Test that NotebookRepository has specialized methods."""
        repo = NotebookRepository()
        assert hasattr(repo, "get_sources")
        assert hasattr(repo, "get_notes")
        assert hasattr(repo, "get_chat_sessions")

    def test_source_repository_methods_exist(self):
        """Test that SourceRepository has specialized methods."""
        repo = SourceRepository()
        assert hasattr(repo, "add_to_notebook")
        assert hasattr(repo, "get_insights")
        assert hasattr(repo, "get_chunks")
        assert hasattr(repo, "get_embeddings")
        assert hasattr(repo, "get_embedding_count")
        assert hasattr(repo, "delete_embeddings")
        assert hasattr(repo, "add_insight")
        assert hasattr(repo, "add_embedding")

    def test_note_repository_methods_exist(self):
        """Test that NoteRepository has specialized methods."""
        repo = NoteRepository()
        assert hasattr(repo, "add_to_notebook")
        assert hasattr(repo, "create_with_embedding")

    def test_chunk_repository_methods_exist(self):
        """Test that ChunkRepository has specialized methods."""
        repo = ChunkRepository()
        assert hasattr(repo, "get_by_source")
        assert hasattr(repo, "delete_by_source")
        assert hasattr(repo, "bulk_create")

    def test_record_repository_methods_exist(self):
        """Test that RecordRepository has expected methods."""
        repo = ContentSettingsRepository()
        assert hasattr(repo, "get")
        assert hasattr(repo, "update")
        assert hasattr(repo, "patch")
