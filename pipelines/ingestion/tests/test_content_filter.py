"""Tests for content classification (is_content flag) and downstream filtering.

Classification happens in ChunkExtractor._classify_is_content().
Downstream consumers (regrouper, etc.) filter on chunk.is_content.
"""

import pytest

from ingestion.chunking.extractor import Chunk, ChunkExtractor
from ingestion.chunking.regrouper import ChunkRegrouper
from ingestion.config import ContentFilterConfig, RegroupingConfig, RegroupingStrategy


def _make_chunk(
    text: str,
    order: int = 0,
    element_type: str = "paragraph",
    physical_page: int = 1,
    section_path: list[str] | None = None,
    metadata: dict | None = None,
    is_content: bool = True,
) -> Chunk:
    """Helper to create test chunks."""
    return Chunk(
        text=text,
        order=order,
        element_type=element_type,
        physical_page=physical_page,
        section_path=section_path or [],
        metadata=metadata or {},
        is_content=is_content,
    )


def _make_extractor(
    content_filter_config: ContentFilterConfig | None = None,
) -> ChunkExtractor:
    """Create extractor with given filter config."""
    return ChunkExtractor(
        min_chunk_size=1,  # Low so test chunks aren't skipped by size
        content_filter_config=content_filter_config or ContentFilterConfig(),
    )


# =============================================================================
# Test classification logic (extractor level)
# =============================================================================


class TestClassifyElementTypes:
    """Test that noise element types are classified as is_content=False."""

    def test_page_header_is_noise(self):
        ext = _make_extractor()
        chunk = _make_chunk("Journal of Testing Vol 1", element_type="page_header")
        assert ext._classify_is_content(chunk) is False

    def test_page_footer_is_noise(self):
        ext = _make_extractor()
        chunk = _make_chunk("Page 42 - Copyright 2024", element_type="page_footer")
        assert ext._classify_is_content(chunk) is False

    def test_footnote_is_noise(self):
        ext = _make_extractor()
        chunk = _make_chunk("1. See reference in appendix B for details.", element_type="footnote")
        assert ext._classify_is_content(chunk) is False

    def test_paragraph_is_content(self):
        ext = _make_extractor()
        chunk = _make_chunk("This is actual document content.", element_type="paragraph")
        assert ext._classify_is_content(chunk) is True

    def test_text_is_content(self):
        ext = _make_extractor()
        chunk = _make_chunk("Regular text element content.", element_type="text")
        assert ext._classify_is_content(chunk) is True

    def test_heading_is_content(self):
        ext = _make_extractor()
        chunk = _make_chunk("Chapter 1: Introduction Here", element_type="heading")
        assert ext._classify_is_content(chunk) is True

    def test_table_is_content(self):
        ext = _make_extractor()
        chunk = _make_chunk("| Col A | Col B |\n| 1 | 2 |", element_type="table")
        assert ext._classify_is_content(chunk) is True


class TestClassifyImageClassifications:
    """Test that noise image classifications are classified as is_content=False."""

    def test_logo_is_noise(self):
        ext = _make_extractor()
        chunk = _make_chunk("[Picture: Logo]", element_type="picture", metadata={"classification": "logo"})
        assert ext._classify_is_content(chunk) is False

    def test_stamp_is_noise(self):
        ext = _make_extractor()
        chunk = _make_chunk("[Picture: Stamp]", element_type="picture", metadata={"classification": "stamp"})
        assert ext._classify_is_content(chunk) is False

    def test_qr_code_is_noise(self):
        ext = _make_extractor()
        chunk = _make_chunk("[Picture: QR]", element_type="picture", metadata={"classification": "qr_code"})
        assert ext._classify_is_content(chunk) is False

    def test_icon_is_noise(self):
        ext = _make_extractor()
        chunk = _make_chunk("[Icon element here]", element_type="picture", metadata={"classification": "icon"})
        assert ext._classify_is_content(chunk) is False

    def test_signature_is_noise(self):
        ext = _make_extractor()
        chunk = _make_chunk("[Signature element]", element_type="picture", metadata={"classification": "signature"})
        assert ext._classify_is_content(chunk) is False

    def test_bar_code_is_noise(self):
        ext = _make_extractor()
        chunk = _make_chunk("[Barcode element here]", element_type="picture", metadata={"classification": "bar_code"})
        assert ext._classify_is_content(chunk) is False

    def test_chart_is_content(self):
        ext = _make_extractor()
        chunk = _make_chunk(
            "[Image: Chart] Revenue growth chart showing quarterly trends.",
            element_type="picture", metadata={"classification": "chart"},
        )
        assert ext._classify_is_content(chunk) is True

    def test_photograph_is_content(self):
        ext = _make_extractor()
        chunk = _make_chunk(
            "[Image: Photo] Team at headquarters.",
            element_type="picture", metadata={"classification": "photograph"},
        )
        assert ext._classify_is_content(chunk) is True

    def test_no_classification_is_content(self):
        ext = _make_extractor()
        chunk = _make_chunk(
            "[Image: Unknown] An image without classification.",
            element_type="picture", metadata={},
        )
        assert ext._classify_is_content(chunk) is True


class TestClassifyMinTextLength:
    """Test minimum text length classification."""

    def test_short_text_is_noise(self):
        ext = _make_extractor()
        chunk = _make_chunk("Short", element_type="paragraph")
        assert ext._classify_is_content(chunk) is False

    def test_text_at_threshold_is_content(self):
        ext = _make_extractor()
        chunk = _make_chunk("Exactly twenty chars", element_type="paragraph")  # 20 chars
        assert ext._classify_is_content(chunk) is True

    def test_custom_min_length(self):
        ext = _make_extractor(ContentFilterConfig(min_text_length=40))
        short = _make_chunk("This is 30 chars of text abcd.", element_type="paragraph")
        long = _make_chunk("This is a sufficiently long paragraph of text for testing.", element_type="paragraph")
        assert ext._classify_is_content(short) is False
        assert ext._classify_is_content(long) is True

    def test_whitespace_only_is_noise(self):
        ext = _make_extractor()
        chunk = _make_chunk("                    ", element_type="paragraph")
        assert ext._classify_is_content(chunk) is False


class TestClassifyDisabled:
    """Test that disabling classification marks everything as content."""

    def test_disabled_all_content(self):
        ext = _make_extractor(ContentFilterConfig(enabled=False))
        header = _make_chunk("Header", element_type="page_header")
        short = _make_chunk("X", element_type="paragraph")
        logo = _make_chunk("[Logo]", element_type="picture", metadata={"classification": "logo"})
        assert ext._classify_is_content(header) is True
        assert ext._classify_is_content(short) is True
        assert ext._classify_is_content(logo) is True


class TestClassifyChunksBatch:
    """Test _classify_chunks sets is_content on a list of chunks."""

    def test_batch_classification(self):
        ext = _make_extractor()
        chunks = [
            _make_chunk("Running Header Text Here", element_type="page_header"),
            _make_chunk("This is actual content paragraph.", order=1, element_type="paragraph"),
            _make_chunk("Page 1 of 10 footer text", order=2, element_type="page_footer"),
            _make_chunk("Another real content paragraph.", order=3, element_type="paragraph"),
        ]
        ext._classify_chunks(chunks)
        assert chunks[0].is_content is False  # page_header
        assert chunks[1].is_content is True   # paragraph
        assert chunks[2].is_content is False  # page_footer
        assert chunks[3].is_content is True   # paragraph


# =============================================================================
# Test downstream filtering (regrouper level)
# =============================================================================


class TestRegroupFiltersByIsContent:
    """Test that regrouper filters on is_content flag."""

    def test_noise_chunks_excluded_from_regroup(self):
        """Pre-classified noise chunks are excluded from regrouping."""
        chunks = [
            _make_chunk("Journal Header Text Here", element_type="page_header", is_content=False),
            _make_chunk("This is the actual content of the document paragraph.", order=1, is_content=True),
        ]
        config = RegroupingConfig(strategy=RegroupingStrategy.NONE)
        result = ChunkRegrouper(config).regroup(chunks)
        assert len(result) == 1
        assert "actual content" in result[0].text

    def test_all_noise_returns_empty(self):
        chunks = [
            _make_chunk("Header noise text", element_type="page_header", is_content=False),
            _make_chunk("Footer noise text", order=1, element_type="page_footer", is_content=False),
        ]
        config = RegroupingConfig(strategy=RegroupingStrategy.NONE)
        result = ChunkRegrouper(config).regroup(chunks)
        assert result == []

    def test_noise_excluded_from_structural_merge(self):
        """Noise chunks don't participate in structural merge."""
        chunks = [
            _make_chunk("Running Header Text Here", element_type="page_header",
                        section_path=["Ch1"], is_content=False),
            _make_chunk(
                "First paragraph with actual content that should be merged together.",
                order=1, element_type="paragraph", section_path=["Ch1"], is_content=True,
            ),
            _make_chunk("Page 1 of 10", order=2, element_type="page_footer",
                        section_path=["Ch1"], is_content=False),
            _make_chunk(
                "Second paragraph that continues the same section's discussion topic.",
                order=3, element_type="paragraph", section_path=["Ch1"], is_content=True,
            ),
        ]
        config = RegroupingConfig(strategy=RegroupingStrategy.STRUCTURAL)
        result = ChunkRegrouper(config).regroup(chunks)
        assert len(result) == 1
        assert "First paragraph" in result[0].text
        assert "Second paragraph" in result[0].text
        assert "Running Header" not in result[0].text

    def test_empty_input(self):
        config = RegroupingConfig(strategy=RegroupingStrategy.NONE)
        result = ChunkRegrouper(config).regroup([])
        assert result == []


# =============================================================================
# Test to_dict includes is_content
# =============================================================================


class TestChunkToDict:
    """Test that is_content is included in serialization."""

    def test_to_dict_includes_is_content(self):
        chunk = _make_chunk("Some content here...", is_content=True)
        d = chunk.to_dict()
        assert "is_content" in d
        assert d["is_content"] is True

    def test_to_dict_noise_chunk(self):
        chunk = _make_chunk("Header", element_type="page_header", is_content=False)
        d = chunk.to_dict()
        assert d["is_content"] is False
