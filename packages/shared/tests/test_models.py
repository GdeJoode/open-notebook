"""Tests for shared domain models."""

from datetime import datetime

import pytest

from shared.models import (
    Asset,
    ChatMessage,
    ChatSession,
    Chunk,
    ContentSettings,
    Note,
    Notebook,
    ObjectModel,
    RecordModel,
    Source,
    SourceEmbedding,
    SourceInsight,
)


class TestObjectModel:
    """Tests for the ObjectModel base class."""

    def test_object_model_creation(self):
        """Test basic ObjectModel can be instantiated."""
        model = ObjectModel()
        assert model.id is None
        assert model.created is None
        assert model.updated is None

    def test_object_model_with_id(self):
        """Test ObjectModel with id."""
        model = ObjectModel(id="test:123")
        assert model.id == "test:123"

    def test_datetime_parsing_from_string(self):
        """Test datetime fields are parsed from ISO strings."""
        model = ObjectModel(
            created="2024-01-15T10:30:00",
            updated="2024-01-15T11:00:00Z",
        )
        assert isinstance(model.created, datetime)
        assert isinstance(model.updated, datetime)

    def test_to_dict(self):
        """Test to_dict conversion."""
        model = ObjectModel(id="test:123")
        data = model.to_dict()
        assert data["id"] == "test:123"
        assert "created" not in data  # None values excluded by default

    def test_to_dict_include_none(self):
        """Test to_dict with include_none=False."""
        model = ObjectModel(id="test:123")
        data = model.to_dict(exclude_none=False)
        assert data["id"] == "test:123"
        assert data["created"] is None


class TestNotebook:
    """Tests for the Notebook model."""

    def test_notebook_creation(self):
        """Test Notebook creation with required fields."""
        notebook = Notebook(name="My Notebook", description="A test notebook")
        assert notebook.name == "My Notebook"
        assert notebook.description == "A test notebook"
        assert notebook.archived is False

    def test_notebook_table_name(self):
        """Test Notebook has correct table_name."""
        assert Notebook.table_name == "notebook"

    def test_notebook_empty_name_fails(self):
        """Test that empty notebook name raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Notebook(name="", description="test")

    def test_notebook_whitespace_name_fails(self):
        """Test that whitespace-only notebook name raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Notebook(name="   ", description="test")

    def test_notebook_name_stripped(self):
        """Test that notebook name is stripped."""
        notebook = Notebook(name="  My Notebook  ", description="test")
        assert notebook.name == "My Notebook"


class TestNote:
    """Tests for the Note model."""

    def test_note_creation(self):
        """Test Note creation."""
        note = Note(title="Test Note", content="Some content", note_type="human")
        assert note.title == "Test Note"
        assert note.content == "Some content"
        assert note.note_type == "human"

    def test_note_table_name(self):
        """Test Note has correct table_name."""
        assert Note.table_name == "note"

    def test_note_empty_content_fails(self):
        """Test that empty content raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Note(title="Test", content="")

    def test_note_whitespace_content_fails(self):
        """Test that whitespace-only content raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Note(title="Test", content="   ")

    def test_note_none_content_allowed(self):
        """Test that None content is allowed."""
        note = Note(title="Test")
        assert note.content is None

    def test_note_get_context_short(self):
        """Test get_context with short context."""
        note = Note(id="note:1", title="Test", content="A" * 200)
        context = note.get_context("short")
        assert context["id"] == "note:1"
        assert len(context["content"]) == 100

    def test_note_get_context_long(self):
        """Test get_context with long context."""
        note = Note(id="note:1", title="Test", content="A" * 200)
        context = note.get_context("long")
        assert len(context["content"]) == 200

    def test_note_needs_embedding(self):
        """Test needs_embedding returns True."""
        note = Note(title="Test", content="Some content")
        assert note.needs_embedding() is True

    def test_note_get_embedding_content(self):
        """Test get_embedding_content returns content."""
        note = Note(title="Test", content="Some content")
        assert note.get_embedding_content() == "Some content"


class TestSource:
    """Tests for the Source model."""

    def test_source_creation(self):
        """Test Source creation."""
        source = Source(title="Test Source")
        assert source.title == "Test Source"
        assert source.topics == []
        assert source.full_text is None

    def test_source_table_name(self):
        """Test Source has correct table_name."""
        assert Source.table_name == "source"

    def test_source_with_asset(self):
        """Test Source with asset."""
        asset = Asset(file_path="/path/to/file.pdf")
        source = Source(title="Test", asset=asset)
        assert source.asset.file_path == "/path/to/file.pdf"

    def test_source_topics_from_none(self):
        """Test topics defaults to empty list."""
        source = Source(title="Test", topics=None)
        assert source.topics == []

    def test_source_topics_from_string(self):
        """Test topics can be created from string."""
        source = Source(title="Test", topics="single topic")
        assert source.topics == ["single topic"]


class TestChunk:
    """Tests for the Chunk model."""

    def test_chunk_creation(self):
        """Test Chunk creation."""
        chunk = Chunk(
            text="Some text content",
            order=0,
            physical_page=0,
            element_type="paragraph",
        )
        assert chunk.text == "Some text content"
        assert chunk.order == 0
        assert chunk.physical_page == 0
        assert chunk.element_type == "paragraph"

    def test_chunk_table_name(self):
        """Test Chunk has correct table_name."""
        assert Chunk.table_name == "chunk"

    def test_chunk_with_positions(self):
        """Test Chunk with bounding box positions."""
        chunk = Chunk(
            text="Text",
            order=0,
            physical_page=0,
            element_type="paragraph",
            positions=[[0, 10.0, 100.0, 20.0, 50.0]],
        )
        assert len(chunk.positions) == 1
        assert chunk.positions[0] == [0, 10.0, 100.0, 20.0, 50.0]


class TestSourceInsight:
    """Tests for the SourceInsight model."""

    def test_source_insight_creation(self):
        """Test SourceInsight creation."""
        insight = SourceInsight(
            insight_type="summary",
            content="This is a summary of the document.",
        )
        assert insight.insight_type == "summary"
        assert insight.content == "This is a summary of the document."

    def test_source_insight_table_name(self):
        """Test SourceInsight has correct table_name."""
        assert SourceInsight.table_name == "source_insight"


class TestSourceEmbedding:
    """Tests for the SourceEmbedding model."""

    def test_source_embedding_creation(self):
        """Test SourceEmbedding creation."""
        embedding = SourceEmbedding(content="Embedded text", order=0)
        assert embedding.content == "Embedded text"
        assert embedding.order == 0
        assert embedding.embedding is None

    def test_source_embedding_table_name(self):
        """Test SourceEmbedding has correct table_name."""
        assert SourceEmbedding.table_name == "source_embedding"


class TestChatSession:
    """Tests for the ChatSession model."""

    def test_chat_session_creation(self):
        """Test ChatSession creation."""
        session = ChatSession(title="Test Session")
        assert session.title == "Test Session"
        assert session.model_override is None

    def test_chat_session_table_name(self):
        """Test ChatSession has correct table_name."""
        assert ChatSession.table_name == "chat_session"


class TestChatMessage:
    """Tests for the ChatMessage model."""

    def test_chat_message_creation(self):
        """Test ChatMessage creation."""
        message = ChatMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"

    def test_chat_message_table_name(self):
        """Test ChatMessage has correct table_name."""
        assert ChatMessage.table_name == "chat_message"


class TestAsset:
    """Tests for the Asset model."""

    def test_asset_with_file(self):
        """Test Asset with file path."""
        asset = Asset(file_path="/path/to/file.pdf")
        assert asset.file_path == "/path/to/file.pdf"
        assert asset.url is None

    def test_asset_with_url(self):
        """Test Asset with URL."""
        asset = Asset(url="https://example.com/document")
        assert asset.url == "https://example.com/document"
        assert asset.file_path is None


class TestContentSettings:
    """Tests for the ContentSettings model."""

    def test_content_settings_defaults(self):
        """Test ContentSettings with default values."""
        settings = ContentSettings()
        assert settings.parser_engine == "docling"
        assert ".pdf" in (settings.mineru_supported_extensions or [])
        assert settings.docling_gpu_enabled is True
        assert settings.docling_ocr_engine == "easyocr"

    def test_content_settings_record_id(self):
        """Test ContentSettings has correct record_id."""
        assert ContentSettings.record_id == "open_notebook:content_settings"

    def test_content_settings_custom_values(self):
        """Test ContentSettings with custom values."""
        settings = ContentSettings(
            docling_gpu_enabled=False,
            docling_ocr_engine="tesseract",
            docling_chunking_max_tokens=1024,
        )
        assert settings.docling_gpu_enabled is False
        assert settings.docling_ocr_engine == "tesseract"
        assert settings.docling_chunking_max_tokens == 1024
