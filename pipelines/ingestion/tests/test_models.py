"""Tests for ingestion data models."""

from datetime import datetime
from pathlib import Path

import pytest

from ingestion.models import (
    BoundingBox,
    ElementType,
    ExtractedDocument,
    ExtractedElement,
    ExtractedImage,
    ExtractedTable,
    IngestionResult,
    PageContent,
    SourceMetadata,
    SourceType,
    SpeakerInfo,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


class TestBoundingBox:
    """Tests for BoundingBox."""

    def test_creation(self):
        """Test bounding box creation."""
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4, page=1)

        assert bbox.x == 0.1
        assert bbox.y == 0.2
        assert bbox.width == 0.3
        assert bbox.height == 0.4
        assert bbox.page == 1

    def test_computed_properties(self):
        """Test computed properties."""
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4, page=1)

        assert bbox.x2 == pytest.approx(0.4)
        assert bbox.y2 == pytest.approx(0.6)
        assert bbox.center == pytest.approx((0.25, 0.4))
        assert bbox.area == pytest.approx(0.12)

    def test_to_dict(self):
        """Test dictionary conversion."""
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4, page=1)
        data = bbox.to_dict()

        assert data["x"] == 0.1
        assert data["page"] == 1


class TestExtractedElement:
    """Tests for ExtractedElement."""

    def test_creation(self):
        """Test element creation."""
        element = ExtractedElement(
            content="Hello world",
            element_type=ElementType.TEXT,
            page=1,
        )

        assert element.content == "Hello world"
        assert element.element_type == ElementType.TEXT
        assert element.page == 1

    def test_with_bbox(self):
        """Test element with bounding box."""
        bbox = BoundingBox(x=0, y=0, width=1, height=1, page=1)
        element = ExtractedElement(
            content="Test",
            element_type=ElementType.PARAGRAPH,
            page=1,
            bbox=bbox,
        )

        assert element.bbox is not None
        assert element.bbox.page == 1

    def test_to_dict(self):
        """Test dictionary conversion."""
        element = ExtractedElement(
            content="Test",
            element_type=ElementType.TEXT,
            page=1,
        )
        data = element.to_dict()

        assert data["content"] == "Test"
        assert data["type"] == "text"
        assert data["page"] == 1


class TestExtractedTable:
    """Tests for ExtractedTable."""

    def test_creation_with_data(self):
        """Test table creation with structured data."""
        table = ExtractedTable(
            content="| A | B |",
            element_type=ElementType.TABLE,
            page=1,
            headers=["A", "B"],
            rows=[["1", "2"], ["3", "4"]],
        )

        assert table.num_rows == 2
        assert table.num_cols == 2
        assert table.headers == ["A", "B"]

    def test_to_csv_string(self):
        """Test CSV generation."""
        table = ExtractedTable(
            content="",
            element_type=ElementType.TABLE,
            page=1,
            headers=["Name", "Value"],
            rows=[["a", "1"], ["b", "2"]],
        )

        csv = table.to_csv_string()
        assert "Name,Value" in csv
        assert "a,1" in csv

    def test_to_markdown_string(self):
        """Test Markdown generation."""
        table = ExtractedTable(
            content="",
            element_type=ElementType.TABLE,
            page=1,
            headers=["Name", "Value"],
            rows=[["a", "1"]],
        )

        md = table.to_markdown_string()
        assert "| Name" in md
        assert "| a" in md
        assert "---" in md  # Separator line


class TestExtractedImage:
    """Tests for ExtractedImage."""

    def test_creation(self):
        """Test image creation."""
        image = ExtractedImage(
            content="A chart showing data",
            element_type=ElementType.IMAGE,
            page=1,
            classification="chart",
            description="This chart shows quarterly revenue.",
        )

        assert image.classification == "chart"
        assert image.description == "This chart shows quarterly revenue."
        assert image.element_type == ElementType.IMAGE

    def test_to_dict(self):
        """Test dictionary conversion."""
        image = ExtractedImage(
            content="Image",
            element_type=ElementType.IMAGE,
            page=1,
            width=800,
            height=600,
            format="png",
        )
        data = image.to_dict()

        assert data["width"] == 800
        assert data["height"] == 600
        assert data["format"] == "png"


class TestPageContent:
    """Tests for PageContent."""

    def test_creation(self):
        """Test page content creation."""
        page = PageContent(page_number=1)
        assert page.page_number == 1
        assert page.element_count == 0

    def test_element_count(self):
        """Test element counting."""
        page = PageContent(page_number=1)
        page.elements = [
            ExtractedElement("a", ElementType.TEXT, 1),
            ExtractedElement("b", ElementType.TEXT, 1),
        ]
        page.tables = [
            ExtractedTable("", ElementType.TABLE, 1),
        ]

        assert page.element_count == 3


class TestExtractedDocument:
    """Tests for ExtractedDocument."""

    def test_creation(self):
        """Test document creation."""
        doc = ExtractedDocument(
            source_path=Path("/test/doc.pdf"),
            title="Test Document",
            page_count=5,
        )

        assert doc.title == "Test Document"
        assert doc.page_count == 5
        assert doc.table_count == 0
        assert doc.image_count == 0

    def test_get_page(self):
        """Test page retrieval."""
        doc = ExtractedDocument(source_path=Path("/test.pdf"))
        doc.pages = [
            PageContent(page_number=1),
            PageContent(page_number=2),
        ]

        page = doc.get_page(1)
        assert page is not None
        assert page.page_number == 1

        assert doc.get_page(99) is None


class TestTranscriptionWord:
    """Tests for TranscriptionWord."""

    def test_creation(self):
        """Test word creation."""
        word = TranscriptionWord(
            word="hello",
            start=0.0,
            end=0.5,
            confidence=0.95,
        )

        assert word.word == "hello"
        assert word.duration == 0.5


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment."""

    def test_creation(self):
        """Test segment creation."""
        segment = TranscriptionSegment(
            text="Hello world",
            start=0.0,
            end=2.0,
            speaker="SPEAKER_00",
        )

        assert segment.text == "Hello world"
        assert segment.duration == 2.0
        assert segment.speaker == "SPEAKER_00"

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        segment = TranscriptionSegment(text="", start=3661.5, end=3662.0)
        # 3661.5 seconds = 1 hour, 1 minute, 1.5 seconds
        assert segment.start_formatted == "01:01:01.500"

    def test_to_srt_format(self):
        """Test SRT format generation."""
        segment = TranscriptionSegment(
            text="Hello",
            start=1.0,
            end=2.0,
            speaker="SPEAKER_00",
        )
        srt = segment.to_srt_format(1)

        assert "1\n" in srt
        assert "[SPEAKER_00] Hello" in srt


class TestTranscriptionResult:
    """Tests for TranscriptionResult."""

    def test_creation(self):
        """Test transcription result creation."""
        result = TranscriptionResult(
            source_path=Path("/test/audio.mp3"),
            language="en",
            duration_seconds=60.0,
        )

        assert result.language == "en"
        assert result.duration_seconds == 60.0

    def test_speaker_stats(self):
        """Test speaker statistics."""
        result = TranscriptionResult(source_path=Path("/test.mp3"))
        result.speakers = [
            SpeakerInfo("SPEAKER_00", total_time=30.0, segment_count=5),
            SpeakerInfo("SPEAKER_01", total_time=20.0, segment_count=3),
        ]

        assert result.speaker_count == 2

    def test_to_srt(self):
        """Test SRT export."""
        result = TranscriptionResult(source_path=Path("/test.mp3"))
        result.segments = [
            TranscriptionSegment("Hello", 0.0, 1.0),
            TranscriptionSegment("World", 1.0, 2.0),
        ]

        srt = result.to_srt()
        assert "1\n" in srt
        assert "2\n" in srt
        assert "Hello" in srt


class TestSourceMetadata:
    """Tests for SourceMetadata."""

    def test_creation(self):
        """Test metadata creation."""
        metadata = SourceMetadata(
            filename="test.pdf",
            source_type=SourceType.DOCUMENT,
            file_extension=".pdf",
            file_size_bytes=1024000,
        )

        assert metadata.filename == "test.pdf"
        assert metadata.source_type == SourceType.DOCUMENT

    def test_format_size(self):
        """Test file size formatting."""
        metadata = SourceMetadata(
            filename="test.pdf",
            source_type=SourceType.DOCUMENT,
            file_extension=".pdf",
            file_size_bytes=1024 * 1024,  # 1 MB
        )

        assert metadata.format_size() == "1.0 MB"

    def test_format_duration(self):
        """Test duration formatting."""
        metadata = SourceMetadata(
            filename="test.mp3",
            source_type=SourceType.AUDIO,
            file_extension=".mp3",
            duration_seconds=3661,  # 1h 1m 1s
        )

        assert metadata.format_duration() == "1h 1m"


class TestIngestionResult:
    """Tests for IngestionResult."""

    def test_success_result(self):
        """Test successful result creation."""
        metadata = SourceMetadata("test.pdf", SourceType.DOCUMENT, ".pdf")
        result = IngestionResult(source_metadata=metadata)
        result.complete(success=True)

        assert result.success is True
        assert result.is_document is True
        assert result.completed_at is not None

    def test_error_result(self):
        """Test error result creation."""
        result = IngestionResult.create_error(
            Path("/test/missing.pdf"),
            "File not found",
        )

        assert result.success is False
        assert result.error_message == "File not found"
