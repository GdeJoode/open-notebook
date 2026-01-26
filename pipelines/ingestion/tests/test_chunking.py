"""Tests for chunk extraction."""

from pathlib import Path

import pytest

from ingestion.chunking import Chunk, ChunkExtractor, ChunkPosition
from ingestion.models import (
    BoundingBox,
    ElementType,
    ExtractedDocument,
    ExtractedElement,
    ExtractedTable,
    PageContent,
)


class TestChunkPosition:
    """Tests for ChunkPosition."""

    def test_creation(self):
        """Test position creation."""
        pos = ChunkPosition(page=1, x1=0.1, x2=0.9, y1=0.2, y2=0.8)

        assert pos.page == 1
        assert pos.x1 == 0.1
        assert pos.x2 == 0.9

    def test_to_list(self):
        """Test list conversion."""
        pos = ChunkPosition(page=1, x1=0.1, x2=0.9, y1=0.2, y2=0.8)
        lst = pos.to_list()

        assert lst == [1, 0.1, 0.9, 0.2, 0.8]

    def test_from_bbox(self):
        """Test creation from BoundingBox."""
        bbox = BoundingBox(x=0.1, y=0.2, width=0.8, height=0.6, page=1)
        pos = ChunkPosition.from_bbox(bbox)

        assert pos.page == 1
        assert pos.x1 == 0.1
        assert pos.x2 == pytest.approx(0.9)
        assert pos.y1 == 0.2
        assert pos.y2 == pytest.approx(0.8)


class TestChunk:
    """Tests for Chunk."""

    def test_creation(self):
        """Test chunk creation."""
        chunk = Chunk(
            text="This is a test paragraph.",
            order=0,
            element_type="text",
            physical_page=1,
        )

        assert chunk.text == "This is a test paragraph."
        assert chunk.order == 0
        assert chunk.element_type == "text"
        assert chunk.physical_page == 1

    def test_with_context(self):
        """Test chunk with chapter/section context."""
        chunk = Chunk(
            text="Content text",
            order=5,
            element_type="paragraph",
            physical_page=2,
            chapter="Introduction",
            section="Background",
            paragraph_number=3,
        )

        assert chunk.chapter == "Introduction"
        assert chunk.section == "Background"
        assert chunk.paragraph_number == 3

    def test_with_positions(self):
        """Test chunk with position data."""
        positions = [
            ChunkPosition(page=1, x1=0.1, x2=0.9, y1=0.1, y2=0.2),
            ChunkPosition(page=1, x1=0.1, x2=0.9, y1=0.2, y2=0.3),
        ]
        chunk = Chunk(
            text="Multi-line text",
            order=0,
            element_type="text",
            physical_page=1,
            positions=positions,
        )

        assert len(chunk.positions) == 2

    def test_to_dict(self):
        """Test dictionary conversion."""
        chunk = Chunk(
            text="Test",
            order=0,
            element_type="text",
            physical_page=1,
            chapter="Chapter 1",
        )
        data = chunk.to_dict()

        assert data["text"] == "Test"
        assert data["order"] == 0
        assert data["chapter"] == "Chapter 1"
        assert data["positions"] == []


class TestChunkExtractor:
    """Tests for ChunkExtractor."""

    def test_default_settings(self):
        """Test default extractor settings."""
        extractor = ChunkExtractor()

        assert extractor.min_chunk_size == 50
        assert extractor.max_chunk_size == 2000
        assert extractor.include_tables is True
        assert extractor.include_images is True

    def test_custom_settings(self):
        """Test custom extractor settings."""
        extractor = ChunkExtractor(
            min_chunk_size=100,
            max_chunk_size=1000,
            include_tables=False,
        )

        assert extractor.min_chunk_size == 100
        assert extractor.max_chunk_size == 1000
        assert extractor.include_tables is False

    def test_extract_from_document(self):
        """Test chunk extraction from ExtractedDocument."""
        # Create a test document
        doc = ExtractedDocument(source_path=Path("/test.pdf"), page_count=1)

        # Add page with elements
        page = PageContent(page_number=1)
        page.elements = [
            ExtractedElement(
                content="This is a long enough paragraph to be extracted as a chunk. " * 3,
                element_type=ElementType.PARAGRAPH,
                page=1,
            ),
            ExtractedElement(
                content="Another paragraph with sufficient content for extraction. " * 3,
                element_type=ElementType.PARAGRAPH,
                page=1,
            ),
        ]
        doc.pages = [page]

        # Extract chunks
        extractor = ChunkExtractor(min_chunk_size=50)
        chunks = extractor.extract(doc)

        assert len(chunks) == 2
        assert chunks[0].element_type == "paragraph"
        assert chunks[0].physical_page == 1
        assert chunks[0].order == 0
        assert chunks[1].order == 1

    def test_skip_short_content(self):
        """Test that short content is skipped."""
        doc = ExtractedDocument(source_path=Path("/test.pdf"), page_count=1)

        page = PageContent(page_number=1)
        page.elements = [
            ExtractedElement(
                content="Short",  # Less than min_chunk_size
                element_type=ElementType.TEXT,
                page=1,
            ),
            ExtractedElement(
                content="This is a sufficiently long paragraph for extraction. " * 2,
                element_type=ElementType.PARAGRAPH,
                page=1,
            ),
        ]
        doc.pages = [page]

        extractor = ChunkExtractor(min_chunk_size=50)
        chunks = extractor.extract(doc)

        assert len(chunks) == 1
        assert "sufficiently long" in chunks[0].text

    def test_chapter_tracking(self):
        """Test chapter/section context tracking."""
        doc = ExtractedDocument(source_path=Path("/test.pdf"), page_count=1)

        page = PageContent(page_number=1)
        page.elements = [
            ExtractedElement(
                content="Introduction",
                element_type=ElementType.TITLE,
                page=1,
            ),
            ExtractedElement(
                content="This is the introduction paragraph with enough content. " * 3,
                element_type=ElementType.PARAGRAPH,
                page=1,
            ),
            ExtractedElement(
                content="Background Section",
                element_type=ElementType.HEADING,
                page=1,
            ),
            ExtractedElement(
                content="This is the background paragraph with sufficient length. " * 3,
                element_type=ElementType.PARAGRAPH,
                page=1,
            ),
        ]
        doc.pages = [page]

        extractor = ChunkExtractor(min_chunk_size=50)
        chunks = extractor.extract(doc)

        # Check context is preserved
        intro_chunk = next(c for c in chunks if "introduction paragraph" in c.text.lower())
        assert intro_chunk.chapter == "Introduction"

        background_chunk = next(c for c in chunks if "background paragraph" in c.text.lower())
        assert background_chunk.chapter == "Introduction"
        assert background_chunk.section == "Background Section"

    def test_table_extraction(self):
        """Test table extraction as chunks."""
        doc = ExtractedDocument(source_path=Path("/test.pdf"), page_count=1)

        page = PageContent(page_number=1)
        page.tables = [
            ExtractedTable(
                content="| A | B |",
                element_type=ElementType.TABLE,
                page=1,
                headers=["Column A", "Column B"],
                rows=[["Value 1", "Value 2"], ["Value 3", "Value 4"]],
            ),
        ]
        doc.pages = [page]

        extractor = ChunkExtractor(min_chunk_size=10, include_tables=True)
        chunks = extractor.extract(doc)

        assert len(chunks) == 1
        assert chunks[0].element_type == "table"
        assert "Column A" in chunks[0].text

    def test_table_exclusion(self):
        """Test table exclusion option."""
        doc = ExtractedDocument(source_path=Path("/test.pdf"), page_count=1)

        page = PageContent(page_number=1)
        page.tables = [
            ExtractedTable(
                content="| A | B |",
                element_type=ElementType.TABLE,
                page=1,
                headers=["A", "B"],
                rows=[["1", "2"]],
            ),
        ]
        doc.pages = [page]

        extractor = ChunkExtractor(include_tables=False)
        chunks = extractor.extract(doc)

        assert len(chunks) == 0

    def test_multi_page_extraction(self):
        """Test extraction across multiple pages."""
        doc = ExtractedDocument(source_path=Path("/test.pdf"), page_count=2)

        page1 = PageContent(page_number=1)
        page1.elements = [
            ExtractedElement(
                content="Page one content with enough text for extraction. " * 3,
                element_type=ElementType.PARAGRAPH,
                page=1,
            ),
        ]

        page2 = PageContent(page_number=2)
        page2.elements = [
            ExtractedElement(
                content="Page two content with sufficient length for a chunk. " * 3,
                element_type=ElementType.PARAGRAPH,
                page=2,
            ),
        ]

        doc.pages = [page1, page2]

        extractor = ChunkExtractor(min_chunk_size=50)
        chunks = extractor.extract(doc)

        assert len(chunks) == 2
        assert chunks[0].physical_page == 1
        assert chunks[1].physical_page == 2
