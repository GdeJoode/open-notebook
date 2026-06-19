"""
Tests for the source-processing pipeline: SourceExtractor / SourceProcessor /
SourceEmbeddingOrchestrator / SourceSummarizationOrchestrator.

All external dependencies (repos, ingestion pipeline, embedding service,
transformation graph) are mocked — no DB or GPU required.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.models import Asset, Source
from shared.models.settings import ContentSettings

from app_main.services.chunking import chunk_builder
from app_main.services.ingestion.config_builder import build_ingestion_config
from app_main.services.source_embedding_orchestrator import (
    SourceEmbeddingOrchestrator,
)
from app_main.services.source_extractor import (
    ExtractionResult,
    SourceExtractor,
    _extract_text_from_html,
)
from app_main.services.source_processor import SourceProcessor
from app_main.services.source_summarization_orchestrator import (
    SourceSummarizationOrchestrator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 6, 15, 12, 0, 0)


def _make_source(**overrides: Any) -> Source:
    defaults = {
        "id": "source:test1",
        "title": "Test Source",
        "full_text": "Some content.",
        "topics": [],
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return Source(**defaults)


def _make_settings(**overrides: Any) -> ContentSettings:
    return ContentSettings(**overrides)


@pytest.fixture
def source_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=_make_source())
    repo.update = AsyncMock(side_effect=lambda id, data: _make_source(**data))
    repo.get_embedding_count = AsyncMock(return_value=5)
    repo.get_insights = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def chunk_repo():
    repo = AsyncMock()
    repo.delete_by_source = AsyncMock(return_value=0)
    repo.bulk_create = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def settings_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=_make_settings())
    return repo


@pytest.fixture
def transformation_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    repo.get_defaults = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def extractor():
    return SourceExtractor()


@pytest.fixture
def processor(source_repo, chunk_repo, settings_repo, extractor):
    return SourceProcessor(
        source_repo=source_repo,
        chunk_repo=chunk_repo,
        settings_repo=settings_repo,
        extractor=extractor,
    )


@pytest.fixture
def embedding_orchestrator(source_repo):
    return SourceEmbeddingOrchestrator(source_repo=source_repo)


@pytest.fixture
def summarization_orchestrator(source_repo, transformation_repo):
    return SourceSummarizationOrchestrator(
        source_repo=source_repo,
        transformation_repo=transformation_repo,
    )


# ---------------------------------------------------------------------------
# Helpers: fake ingestion models
# ---------------------------------------------------------------------------

@dataclass
class FakeBBox:
    x: float = 0.1
    y: float = 0.2
    width: float = 0.8
    height: float = 0.3
    page: int = 1

    @property
    def x2(self):
        return self.x + self.width

    @property
    def y2(self):
        return self.y + self.height


@dataclass
class FakeElement:
    content: str = "Hello world"
    element_type: MagicMock = field(default_factory=lambda: MagicMock(value="paragraph"))
    page: int = 1
    bbox: Optional[FakeBBox] = field(default_factory=FakeBBox)
    section_path: list = field(default_factory=list)
    section_level: int = 0


@dataclass
class FakeTable:
    content: str = "| A | B |"
    markdown: str = "| A | B |"
    element_type: MagicMock = field(default_factory=lambda: MagicMock(value="table"))
    page: int = 1
    bbox: Optional[FakeBBox] = field(default_factory=FakeBBox)
    num_rows: int = 2
    num_cols: int = 2


@dataclass
class FakePage:
    page_number: int = 1
    elements: list = field(default_factory=lambda: [FakeElement()])
    tables: list = field(default_factory=list)
    images: list = field(default_factory=list)


@dataclass
class FakeDocument:
    title: str = "Test Document"
    full_text: str = "Full document text."
    pages: list = field(default_factory=lambda: [FakePage()])
    source_path: Path = Path("/tmp/test.pdf")

    def to_markdown(self):
        return self.full_text


@dataclass
class FakeSegment:
    text: str = "Hello, this is a test."
    start: float = 0.0
    end: float = 5.0
    speaker: str = "Speaker 1"


@dataclass
class FakeTranscription:
    segments: list = field(default_factory=lambda: [FakeSegment()])

    def to_markdown(self):
        return "\n".join(s.text for s in self.segments)


@dataclass
class FakeSourceType:
    value: str = "document"


@dataclass
class FakeIngestionResult:
    success: bool = True
    error_message: Optional[str] = None
    document: Optional[FakeDocument] = field(default_factory=FakeDocument)
    transcription: Any = None
    source_type: FakeSourceType = field(default_factory=FakeSourceType)
    output_directory: Optional[Path] = None
    markdown_path: Optional[Path] = None
    processing_time_seconds: float = 1.5


# ===========================================================================
# Test: _process_text
# ===========================================================================

class TestProcessText:
    def test_wraps_content_as_single_chunk(self, extractor):
        result = extractor._process_text("This is raw text.")

        assert isinstance(result, ExtractionResult)
        assert result.full_text == "This is raw text."
        assert result.title == "This is raw text."
        assert len(result.chunks) == 1
        assert result.chunks[0]["text"] == "This is raw text."
        assert result.chunks[0]["element_type"] == "text"

    def test_empty_content(self, extractor):
        result = extractor._process_text("")
        assert result.title == "Untitled"
        assert result.full_text == ""

    def test_multiline_title_uses_first_line(self, extractor):
        result = extractor._process_text("First line\nSecond line")
        assert result.title == "First line"


# ===========================================================================
# Test: _process_file
# ===========================================================================

class TestProcessFile:
    @pytest.mark.asyncio
    async def test_processes_document(self, extractor, settings_repo, tmp_path):
        fake_result = FakeIngestionResult()
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake")

        with patch(
            "ingestion.workflow.IngestionWorkflow"
        ) as MockWorkflow:
            instance = MockWorkflow.return_value
            instance.process = MagicMock(return_value=fake_result)

            result = await extractor._process_file(
                file_path=str(test_file),
                content_settings=_make_settings(),
            )

        assert result.title == "Test Document"
        assert result.full_text == "Full document text."
        assert result.file_path == str(test_file)
        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_processes_transcription(self, extractor, tmp_path):
        fake_result = FakeIngestionResult(
            document=None,
            transcription=FakeTranscription(),
            source_type=FakeSourceType(value="audio"),
        )
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio")

        with patch(
            "ingestion.workflow.IngestionWorkflow"
        ) as MockWorkflow:
            instance = MockWorkflow.return_value
            instance.process = MagicMock(return_value=fake_result)

            result = await extractor._process_file(
                file_path=str(test_file),
                content_settings=_make_settings(),
            )

        assert result.full_text == "Hello, this is a test."
        assert len(result.chunks) == 1
        assert result.chunks[0]["element_type"] == "transcription_segment"
        assert result.chunks[0]["metadata"]["speaker"] == "Speaker 1"

    @pytest.mark.asyncio
    async def test_raises_on_missing_file(self, extractor):
        with pytest.raises(FileNotFoundError):
            await extractor._process_file(
                file_path="/nonexistent/file.pdf",
                content_settings=_make_settings(),
            )

    @pytest.mark.asyncio
    async def test_raises_on_ingestion_failure(self, extractor, tmp_path):
        fake_result = FakeIngestionResult(
            success=False,
            error_message="Parse error",
            document=None,
        )
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake")

        with patch(
            "ingestion.workflow.IngestionWorkflow"
        ) as MockWorkflow:
            instance = MockWorkflow.return_value
            instance.process = MagicMock(return_value=fake_result)

            with pytest.raises(RuntimeError, match="Ingestion failed"):
                await extractor._process_file(
                    file_path=str(test_file),
                    content_settings=_make_settings(),
                )

    @pytest.mark.asyncio
    async def test_records_parser_engine_used_in_metadata(self, extractor, tmp_path):
        """Default settings (parser_engine='docling') -> metadata records docling."""
        fake_result = FakeIngestionResult()
        test_file = tmp_path / "doc.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake")

        with patch("ingestion.workflow.IngestionWorkflow") as MockWorkflow:
            MockWorkflow.return_value.process = MagicMock(return_value=fake_result)

            result = await extractor._process_file(
                file_path=str(test_file),
                content_settings=_make_settings(),
            )

        assert result.metadata is not None
        assert result.metadata["parser_engine_used"] == "docling"

    @pytest.mark.asyncio
    async def test_routes_to_mineru_when_parser_engine_mineru(
        self, extractor, tmp_path
    ):
        """parser_engine='mineru' + supported ext -> MineruHttpClient.process called once."""
        fake_result = FakeIngestionResult()
        test_file = tmp_path / "paper.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake")

        # Mock the lazily-imported MineruHttpClient inside source_extractor.
        mineru_mock_client = MagicMock()
        mineru_mock_client.process = AsyncMock(return_value=fake_result)
        mineru_module = MagicMock()
        mineru_module.MineruHttpClient = MagicMock(return_value=mineru_mock_client)

        with patch.dict(
            "sys.modules",
            {"app_main.services.mineru_http_client": mineru_module},
        ):
            result = await extractor._process_file(
                file_path=str(test_file),
                content_settings=_make_settings(parser_engine="mineru"),
            )

        mineru_mock_client.process.assert_awaited_once()
        assert result.metadata["parser_engine_used"] == "mineru"

    @pytest.mark.asyncio
    async def test_falls_back_to_docling_for_unsupported_mineru_ext(
        self, extractor, tmp_path, caplog
    ):
        """parser_engine='mineru' + unsupported ext -> docling, INFO log emitted."""
        import logging

        fake_result = FakeIngestionResult()
        test_file = tmp_path / "page.html"
        test_file.write_text("<html><body>x</body></html>")

        mineru_module = MagicMock()
        mineru_module.MineruHttpClient = MagicMock()

        with patch.dict(
            "sys.modules",
            {"app_main.services.mineru_http_client": mineru_module},
        ):
            with patch("ingestion.workflow.IngestionWorkflow") as MockWorkflow:
                MockWorkflow.return_value.process = MagicMock(return_value=fake_result)
                with caplog.at_level(logging.INFO):
                    result = await extractor._process_file(
                        file_path=str(test_file),
                        content_settings=_make_settings(parser_engine="mineru"),
                    )

        # MinerU client must not have been instantiated.
        mineru_module.MineruHttpClient.assert_not_called()
        assert result.metadata["parser_engine_used"] == "docling"

    @pytest.mark.asyncio
    async def test_auto_high_confidence_keeps_docling(
        self, extractor, tmp_path
    ):
        """Phase A.1c: parser_engine='auto' + high-confidence docling
        result ⇒ MinerU is not called and metadata says docling.
        """
        # High-quality fake document: many pages, dense text, headings.
        pages: list[FakePage] = []
        for idx in range(1, 6):
            elements = [
                FakeElement(
                    content="Heading",
                    element_type=MagicMock(value="heading"),
                    page=idx,
                ),
                FakeElement(
                    content="Title",
                    element_type=MagicMock(value="title"),
                    page=idx,
                ),
            ]
            for _ in range(5):
                elements.append(
                    FakeElement(
                        content="paragraph",
                        element_type=MagicMock(value="paragraph"),
                        page=idx,
                    )
                )
            pages.append(FakePage(page_number=idx, elements=elements))

        good_doc = FakeDocument(
            title="Good doc",
            full_text="x" * (2500 * 5),
            pages=pages,
        )
        # Attach attributes the confidence scorer reads.
        good_doc.page_count = 5
        good_doc.all_tables = []
        good_doc.all_images = []

        fake_result = FakeIngestionResult(document=good_doc)
        test_file = tmp_path / "paper.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake")

        mineru_mock_client = MagicMock()
        mineru_mock_client.process = AsyncMock(return_value=fake_result)
        mineru_module = MagicMock()
        mineru_module.MineruHttpClient = MagicMock(return_value=mineru_mock_client)

        with patch.dict(
            "sys.modules",
            {"app_main.services.mineru_http_client": mineru_module},
        ):
            with patch("ingestion.workflow.IngestionWorkflow") as MockWorkflow:
                MockWorkflow.return_value.process = MagicMock(return_value=fake_result)

                result = await extractor._process_file(
                    file_path=str(test_file),
                    content_settings=_make_settings(parser_engine="auto"),
                )

        # MinerU must not have been called for a high-confidence doc.
        mineru_mock_client.process.assert_not_called()
        assert result.metadata["parser_engine_used"] == "docling"
        assert result.metadata["extraction_fallback_triggered"] is False
        assert 0.0 <= result.metadata["extraction_confidence"] <= 1.0
        assert "extraction_confidence_signals" in result.metadata

    @pytest.mark.asyncio
    async def test_auto_low_confidence_falls_back_to_mineru(
        self, extractor, tmp_path
    ):
        """Phase A.1c: parser_engine='auto' + low-confidence docling ⇒
        MinerU is called and metadata reports the fallback.
        """
        # Low-quality document: minimal text, no structure.
        low_doc = FakeDocument(
            title="Scan",
            full_text="x" * 50,
            pages=[FakePage(
                page_number=1,
                elements=[FakeElement(
                    content="ocr",
                    element_type=MagicMock(value="text"),
                    page=1,
                    bbox=None,
                    section_level=0,
                )],
            )],
        )
        low_doc.page_count = 1
        low_doc.all_tables = []
        low_doc.all_images = []
        # Mark the element as OCR-sourced with low confidence so the
        # OCR signal pulls the overall score under the threshold.
        low_doc.pages[0].elements[0].source = "ocr"
        low_doc.pages[0].elements[0].confidence = 0.4

        docling_result = FakeIngestionResult(document=low_doc)
        mineru_result = FakeIngestionResult()  # default fake doc
        test_file = tmp_path / "scan.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake")

        mineru_mock_client = MagicMock()
        mineru_mock_client.process = AsyncMock(return_value=mineru_result)
        mineru_module = MagicMock()
        mineru_module.MineruHttpClient = MagicMock(return_value=mineru_mock_client)

        with patch.dict(
            "sys.modules",
            {"app_main.services.mineru_http_client": mineru_module},
        ):
            with patch("ingestion.workflow.IngestionWorkflow") as MockWorkflow:
                MockWorkflow.return_value.process = MagicMock(return_value=docling_result)

                result = await extractor._process_file(
                    file_path=str(test_file),
                    content_settings=_make_settings(parser_engine="auto"),
                )

        mineru_mock_client.process.assert_awaited_once()
        assert result.metadata["parser_engine_used"] == "mineru"
        assert result.metadata["extraction_fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_auto_respects_docling_min_confidence_override(
        self, extractor, tmp_path
    ):
        """Phase A.1c acceptance criterion #7: a higher
        ``docling_min_confidence`` override on reprocess raises the bar
        for that one extraction.
        """
        # Build a "good but not perfect" doc — let image-text ratio dent
        # the score so a 0.99 bar can flip it to fallback.
        pages: list[FakePage] = []
        for idx in range(1, 4):
            elements = [
                FakeElement(
                    content="Heading",
                    element_type=MagicMock(value="heading"),
                    page=idx,
                ),
            ]
            for _ in range(2):
                elements.append(FakeElement(
                    content="para",
                    element_type=MagicMock(value="paragraph"),
                    page=idx,
                ))
            pages.append(FakePage(
                page_number=idx,
                elements=elements,
                images=[FakeElement(
                    content="img",
                    element_type=MagicMock(value="image"),
                    page=idx,
                ) for _ in range(3)],
            ))
        mid_doc = FakeDocument(
            title="Mid",
            full_text="x" * (2500 * 3),
            pages=pages,
        )
        mid_doc.page_count = 3
        mid_doc.all_tables = []
        mid_doc.all_images = [FakeElement() for _ in range(3 * 3)]

        docling_result = FakeIngestionResult(document=mid_doc)
        mineru_result = FakeIngestionResult()
        test_file = tmp_path / "paper.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake")

        mineru_mock_client = MagicMock()
        mineru_mock_client.process = AsyncMock(return_value=mineru_result)
        mineru_module = MagicMock()
        mineru_module.MineruHttpClient = MagicMock(return_value=mineru_mock_client)

        with patch.dict(
            "sys.modules",
            {"app_main.services.mineru_http_client": mineru_module},
        ):
            with patch("ingestion.workflow.IngestionWorkflow") as MockWorkflow:
                MockWorkflow.return_value.process = MagicMock(return_value=docling_result)

                result = await extractor._process_file(
                    file_path=str(test_file),
                    content_settings=_make_settings(
                        parser_engine="auto",
                        docling_min_confidence=0.99,
                    ),
                )

        # At 0.99 bar the mid-quality doc should fall back.
        mineru_mock_client.process.assert_awaited_once()
        assert result.metadata["parser_engine_used"] == "mineru"
        assert result.metadata["extraction_fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_auto_respects_threshold_zero_never_falls_back(
        self, extractor, tmp_path
    ):
        """Regression for A.1c attempt-1 Blocker #2: ``docling_min_confidence=0.0``
        means "always accept docling, never fall back". The previous code
        used ``getattr(...) or DEFAULT_THRESHOLD`` which silently demoted
        0.0 to 0.95 because Python treats 0.0 as falsy. With the fix this
        test verifies the threshold actually reaches the auto-fallback
        orchestrator and MinerU is *not* called even on a low-quality doc
        that would normally trigger fallback at the 0.95 default.
        """
        # Build the same low-quality fixture that
        # test_auto_low_confidence_falls_back_to_mineru uses — confirming
        # the *only* thing changing behaviour is the threshold override.
        low_doc = FakeDocument(
            title="Scan",
            full_text="x" * 50,
            pages=[FakePage(
                page_number=1,
                elements=[FakeElement(
                    content="ocr",
                    element_type=MagicMock(value="text"),
                    page=1,
                    bbox=None,
                    section_level=0,
                )],
            )],
        )
        low_doc.page_count = 1
        low_doc.all_tables = []
        low_doc.all_images = []
        low_doc.pages[0].elements[0].source = "ocr"
        low_doc.pages[0].elements[0].confidence = 0.4

        docling_result = FakeIngestionResult(document=low_doc)
        test_file = tmp_path / "scan.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake")

        mineru_mock_client = MagicMock()
        mineru_mock_client.process = AsyncMock(return_value=FakeIngestionResult())
        mineru_module = MagicMock()
        mineru_module.MineruHttpClient = MagicMock(return_value=mineru_mock_client)

        with patch.dict(
            "sys.modules",
            {"app_main.services.mineru_http_client": mineru_module},
        ):
            with patch("ingestion.workflow.IngestionWorkflow") as MockWorkflow:
                MockWorkflow.return_value.process = MagicMock(return_value=docling_result)

                result = await extractor._process_file(
                    file_path=str(test_file),
                    content_settings=_make_settings(
                        parser_engine="auto",
                        docling_min_confidence=0.0,
                    ),
                )

        # The bug-of-record assertion: a low-conf doc + threshold=0.0 must
        # stay with docling. MinerU must not have been called.
        mineru_mock_client.process.assert_not_called()
        assert result.metadata["parser_engine_used"] == "docling"
        assert result.metadata["extraction_fallback_triggered"] is False
        # The score was computed and threshold was honoured.
        assert "extraction_confidence" in result.metadata
        assert "extraction_confidence_signals" in result.metadata

    @pytest.mark.asyncio
    async def test_simple_setting_records_docling_engine_used(
        self, extractor, tmp_path
    ):
        """parser_engine='simple' falls through to Docling, so the engine
        that *actually ran* is "docling" — not "simple". This is the A.1b
        attempt-2 fix: the field reflects the executed engine, not the
        user setting, so A.1c's confidence-fallback can trust it.
        """
        fake_result = FakeIngestionResult()
        test_file = tmp_path / "doc.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake")

        with patch("ingestion.workflow.IngestionWorkflow") as MockWorkflow:
            MockWorkflow.return_value.process = MagicMock(return_value=fake_result)

            result = await extractor._process_file(
                file_path=str(test_file),
                content_settings=_make_settings(parser_engine="simple"),
            )

        assert result.metadata["parser_engine_used"] == "docling"

    @pytest.mark.asyncio
    async def test_audio_records_whisperx_engine_used(self, extractor, tmp_path):
        """Audio files route through IngestionWorkflow → WhisperX regardless
        of the parser_engine setting. The metadata must say "whisperx"
        (engine that ran), not the user setting. A.1b attempt-2 fix.
        """
        fake_result = FakeIngestionResult(
            document=None,
            transcription=FakeTranscription(),
            source_type=FakeSourceType(value="audio"),
        )
        test_file = tmp_path / "talk.mp3"
        test_file.write_bytes(b"fake audio")

        with patch("ingestion.workflow.IngestionWorkflow") as MockWorkflow:
            MockWorkflow.return_value.process = MagicMock(return_value=fake_result)

            result = await extractor._process_file(
                file_path=str(test_file),
                # Even with parser_engine="mineru", the audio extension
                # bypasses the dispatcher and IngestionWorkflow picks
                # WhisperX. The metadata must reflect what ran.
                content_settings=_make_settings(parser_engine="mineru"),
            )

        assert result.metadata["parser_engine_used"] == "whisperx"

    @pytest.mark.asyncio
    async def test_delete_source_flag(self, extractor, tmp_path):
        test_file = tmp_path / "deleteme.txt"
        test_file.write_text("content")
        assert test_file.exists()

        fake_result = FakeIngestionResult(
            document=FakeDocument(source_path=test_file),
        )

        with patch(
            "ingestion.workflow.IngestionWorkflow"
        ) as MockWorkflow:
            instance = MockWorkflow.return_value
            instance.process = MagicMock(return_value=fake_result)

            await extractor._process_file(
                file_path=str(test_file),
                content_settings=_make_settings(),
                delete_source=True,
            )

        assert not test_file.exists()


# ===========================================================================
# Test: _process_url
# ===========================================================================

class TestProcessUrl:
    @pytest.mark.asyncio
    async def test_extracts_url_content(self, extractor):
        with patch.object(
            SourceExtractor,
            "_fetch_url_content",
            new_callable=AsyncMock,
            return_value=("Page Title", "Extracted text content."),
        ):
            result = await extractor._process_url(
                url="https://example.com",
                content_settings=_make_settings(),
            )

        assert result.title == "Page Title"
        assert result.full_text == "Extracted text content."
        assert result.url == "https://example.com"
        assert len(result.chunks) == 1
        assert result.chunks[0]["element_type"] == "web_content"

    @pytest.mark.asyncio
    async def test_raises_on_empty_content(self, extractor):
        with patch.object(
            SourceExtractor,
            "_fetch_url_content",
            new_callable=AsyncMock,
            return_value=(None, ""),
        ):
            with pytest.raises(RuntimeError, match="No content extracted"):
                await extractor._process_url(
                    url="https://empty.com",
                    content_settings=_make_settings(),
                )


# ===========================================================================
# Test: _extract_text_from_html
# ===========================================================================

class TestHtmlExtraction:
    def test_extracts_main_content(self):
        html = """
        <html>
        <body>
            <nav>Skip me</nav>
            <main><p>Important content</p></main>
            <footer>Also skip</footer>
        </body>
        </html>
        """
        text = _extract_text_from_html(html)
        assert "Important content" in text
        assert "Skip me" not in text
        assert "Also skip" not in text

    def test_removes_scripts_and_styles(self):
        html = """
        <html>
        <body>
            <script>alert('bad')</script>
            <style>.hidden { display: none }</style>
            <p>Good content</p>
        </body>
        </html>
        """
        text = _extract_text_from_html(html)
        assert "Good content" in text
        assert "alert" not in text
        assert ".hidden" not in text

    def test_falls_back_to_body(self):
        html = "<html><body><p>Body content</p></body></html>"
        text = _extract_text_from_html(html)
        assert "Body content" in text


# ===========================================================================
# Test: _document_to_chunks
# ===========================================================================

class TestDocumentToChunks:
    def test_converts_elements(self):
        doc = FakeDocument()
        chunks = chunk_builder.from_document(doc)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello world"
        assert chunks[0]["physical_page"] == 0  # page_number(1) - 1
        assert chunks[0]["printed_page"] == 1
        assert len(chunks[0]["positions"]) == 1

    def test_converts_tables(self):
        page = FakePage(
            elements=[],
            tables=[FakeTable()],
        )
        doc = FakeDocument(pages=[page])
        chunks = chunk_builder.from_document(doc)

        assert len(chunks) == 1
        assert chunks[0]["element_type"] == "table"
        assert chunks[0]["metadata"]["num_rows"] == 2

    def test_preserves_order(self):
        page = FakePage(
            elements=[
                FakeElement(content="First"),
                FakeElement(content="Second"),
            ],
            tables=[FakeTable()],
        )
        doc = FakeDocument(pages=[page])
        chunks = chunk_builder.from_document(doc)

        assert len(chunks) == 3
        assert chunks[0]["order"] == 0
        assert chunks[1]["order"] == 1
        assert chunks[2]["order"] == 2

    def test_element_without_bbox(self):
        elem = FakeElement(bbox=None)
        page = FakePage(elements=[elem])
        doc = FakeDocument(pages=[page])
        chunks = chunk_builder.from_document(doc)

        assert chunks[0]["positions"] == []
        assert chunks[0]["metadata"]["has_spatial_data"] is False


# ===========================================================================
# Test: _transcription_to_chunks
# ===========================================================================

class TestTranscriptionToChunks:
    def test_converts_segments(self):
        transcription = FakeTranscription(
            segments=[
                FakeSegment(text="Hello", start=0.0, end=2.5, speaker="A"),
                FakeSegment(text="World", start=2.5, end=5.0, speaker="B"),
            ]
        )
        chunks = chunk_builder.from_transcription(transcription)

        assert len(chunks) == 2
        assert chunks[0]["text"] == "Hello"
        assert chunks[0]["metadata"]["speaker"] == "A"
        assert chunks[1]["order"] == 1


# ===========================================================================
# Test: _prepare_chunks_for_db
# ===========================================================================

class TestPrepareChunksForDb:
    def test_adds_source_id(self):
        chunks = [
            {
                "text": "chunk text",
                "order": 0,
                "physical_page": 0,
                "element_type": "paragraph",
            }
        ]
        prepared = chunk_builder.prepare_for_db(
            chunks, "source:abc"
        )

        assert len(prepared) == 1
        assert prepared[0]["source"] == "source:abc"
        assert prepared[0]["text"] == "chunk text"
        assert prepared[0]["positions"] == []
        assert prepared[0]["metadata"] == {}


# ===========================================================================
# Test: _build_ingestion_config
# ===========================================================================

class TestBuildIngestionConfig:
    def test_default_settings(self):
        settings = _make_settings()
        config = build_ingestion_config(settings)

        # Should not raise and should return a valid config
        assert config.docling is not None

    def test_gpu_disabled(self):
        settings = _make_settings(docling_gpu_enabled=False)
        config = build_ingestion_config(settings)

        from ingestion.config import AcceleratorDevice
        assert config.docling.device == AcceleratorDevice.CPU

    def test_ocr_engine_mapping(self):
        settings = _make_settings(docling_ocr_engine="tesseract")
        config = build_ingestion_config(settings)

        from ingestion.config import OcrEngine
        assert config.docling.ocr_engine == OcrEngine.TESSERACT

    # ---- Phase I.D-2: enrichment / page-image / classification fields ----

    def test_id2_defaults_preserve_current_behaviour(self):
        """Defaults must reproduce pre-I.D-2 effective values (AC3).

        With a VLM pipeline (the ContentSettings default), classification
        follows the VLM toggle, enrichment stays on, page images stay off,
        and the image scale stays at 2.0.
        """
        config = build_ingestion_config(_make_settings()).docling

        assert config.do_code_enrichment is True
        assert config.do_formula_extraction is True
        assert config.generate_page_images is False
        assert config.images_scale == 2.0
        # VLM is the default pipeline -> classification on, matching the
        # old `do_picture_classification=use_vlm` coupling.
        assert config.do_picture_classification is True

    def test_id2_classification_follows_vlm_when_unset(self):
        """Unset classification falls back to the VLM toggle, both ways."""
        vlm = build_ingestion_config(
            _make_settings(docling_pipeline="vlm")
        ).docling
        standard = build_ingestion_config(
            _make_settings(docling_pipeline="standard")
        ).docling

        assert vlm.do_picture_classification is True
        assert standard.do_picture_classification is False

    def test_id2_classification_decoupled_from_vlm(self):
        """Explicit classification overrides the VLM coupling in both
        directions (the core I.D-2 separation)."""
        on_without_vlm = build_ingestion_config(
            _make_settings(
                docling_pipeline="standard",
                docling_do_picture_classification=True,
            )
        ).docling
        off_with_vlm = build_ingestion_config(
            _make_settings(
                docling_pipeline="vlm",
                docling_do_picture_classification=False,
            )
        ).docling

        assert on_without_vlm.do_picture_classification is True
        assert off_with_vlm.do_picture_classification is False

    def test_id2_override_fields_map_through(self):
        """All five I.D-2 override fields reach DoclingConfig (AC1)."""
        config = build_ingestion_config(
            _make_settings(
                docling_do_code_enrichment=False,
                docling_do_formula_enrichment=False,
                docling_do_picture_classification=True,
                docling_generate_page_images=True,
                docling_image_scale=4.0,
            )
        ).docling

        assert config.do_code_enrichment is False
        assert config.do_formula_extraction is False
        assert config.do_picture_classification is True
        assert config.generate_page_images is True
        assert config.images_scale == 4.0

    def test_id2_override_payload_round_trip(self):
        """Simulate the per-run override merge (source_processor) then the
        config build, asserting the five fields survive the trip (AC1)."""
        base = _make_settings()
        overrides = {
            "docling_do_code_enrichment": False,
            "docling_do_formula_enrichment": False,
            "docling_do_picture_classification": True,
            "docling_generate_page_images": True,
            "docling_image_scale": 3.0,
        }
        merged_data = base.model_dump()
        merged_data.update(
            {k: v for k, v in overrides.items() if v is not None}
        )
        merged = ContentSettings(**merged_data)

        config = build_ingestion_config(merged).docling
        assert config.do_code_enrichment is False
        assert config.do_formula_extraction is False
        assert config.do_picture_classification is True
        assert config.generate_page_images is True
        assert config.images_scale == 3.0

    def test_id2_forwarded_to_docling_options(self):
        """to_docling_options() must forward the I.D-2 toggles to the docling
        PdfPipelineOptions (AC2). Skips when docling is not installed."""
        pytest.importorskip("docling")

        config = build_ingestion_config(
            _make_settings(
                docling_do_code_enrichment=False,
                docling_do_formula_enrichment=False,
                docling_do_picture_classification=True,
                docling_generate_page_images=True,
                docling_image_scale=3.0,
            )
        ).docling
        opts = config.to_docling_options()

        assert opts.do_code_enrichment is False
        assert opts.do_formula_enrichment is False
        assert opts.do_picture_classification is True
        assert opts.generate_page_images is True
        assert opts.images_scale == 3.0


# ===========================================================================
# Test: process_source (full orchestration)
# ===========================================================================

class TestProcessSource:
    @pytest.mark.asyncio
    async def test_full_text_processing(self, processor, source_repo, chunk_repo):
        result = await processor.process_source(
            source_id="source:test1",
            content_state={"content": "Hello world"},
        )

        assert result["source_id"] is not None
        chunk_repo.delete_by_source.assert_awaited_once_with("source:test1")
        chunk_repo.bulk_create.assert_awaited_once()
        source_repo.update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_source_raises(self, processor, source_repo):
        source_repo.get = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await processor.process_source(
                source_id="source:missing",
                content_state={"content": "test"},
            )

    @pytest.mark.asyncio
    async def test_invalid_content_state_raises(self, processor):
        with pytest.raises(ValueError, match="must contain"):
            await processor.process_source(
                source_id="source:test1",
                content_state={"invalid_key": "value"},
            )


# ===========================================================================
# Test: _update_source
# ===========================================================================

class TestUpdateSource:
    @pytest.mark.asyncio
    async def test_updates_with_title(self, processor, source_repo):
        extracted = ExtractionResult(
            title="New Title",
            full_text="New content",
            file_path="/tmp/test.pdf",
        )
        source = _make_source()

        await processor._update_source(source, extracted)

        source_repo.update.assert_awaited_once()
        call_args = source_repo.update.call_args
        data = call_args[0][1]
        assert data["title"] == "New Title"
        assert data["full_text"] == "New content"
        assert data["asset"]["file_path"] == "/tmp/test.pdf"

    @pytest.mark.asyncio
    async def test_skips_title_when_none(self, processor, source_repo):
        extracted = ExtractionResult(
            title=None,
            full_text="Content",
        )
        source = _make_source()

        await processor._update_source(source, extracted)

        call_args = source_repo.update.call_args
        data = call_args[0][1]
        assert "title" not in data

    @pytest.mark.asyncio
    async def test_lifts_extraction_metadata_to_source_metadata(
        self, processor, source_repo
    ):
        """Phase A.1c: provenance keys must flow to Source.metadata so
        the UI badge can render them (Q-A-3).
        """
        extracted = ExtractionResult(
            title="Doc",
            full_text="content",
            file_path="/tmp/doc.pdf",
            metadata={
                "source_type": "document",
                "parser_engine_used": "mineru",
                "extraction_confidence": 0.62,
                "extraction_confidence_signals": {
                    "ocr_confidence": 0.4,
                    "text_density": 0.5,
                },
                "extraction_fallback_triggered": True,
                # Non-provenance noise that must NOT be lifted.
                "processing_time": 12.3,
                "markdown_path": "/tmp/doc.md",
            },
        )
        source = _make_source()

        await processor._update_source(source, extracted)

        data = source_repo.update.call_args[0][1]
        assert "metadata" in data
        assert data["metadata"]["parser_engine_used"] == "mineru"
        assert data["metadata"]["extraction_confidence"] == 0.62
        assert data["metadata"]["extraction_confidence_signals"] == {
            "ocr_confidence": 0.4,
            "text_density": 0.5,
        }
        assert data["metadata"]["extraction_fallback_triggered"] is True
        # processing_time / markdown_path are not part of the Source.metadata
        # provenance contract — they stay on the ExtractionResult only.
        assert "processing_time" not in data["metadata"]
        assert "markdown_path" not in data["metadata"]

    @pytest.mark.asyncio
    async def test_omits_metadata_when_no_provenance_keys(
        self, processor, source_repo
    ):
        """If the ExtractionResult has no provenance keys, don't touch
        Source.metadata — keeps the DB row minimal for legacy paths.
        """
        extracted = ExtractionResult(
            title="Doc",
            full_text="content",
            metadata={"source_type": "document", "processing_time": 5.0},
        )
        source = _make_source()

        await processor._update_source(source, extracted)

        data = source_repo.update.call_args[0][1]
        assert "metadata" not in data


# ===========================================================================
# Characterization: embed_source (public API)
# ===========================================================================

class TestEmbedSource:
    @pytest.mark.asyncio
    async def test_calls_embedding_pipeline(self, embedding_orchestrator):
        embed_service = MagicMock()
        embed_service.embed_source = AsyncMock()
        with patch(
            "app_main.dependencies.get_embedding_service",
            AsyncMock(return_value=embed_service),
        ):
            await embedding_orchestrator.embed_source("source:test1")

        embed_service.embed_source.assert_awaited_once_with("source:test1")

    @pytest.mark.asyncio
    async def test_returns_source_id_and_chunk_count(self, embedding_orchestrator, source_repo):
        source_repo.get_embedding_count = AsyncMock(return_value=7)
        embed_service = MagicMock()
        embed_service.embed_source = AsyncMock()
        with patch(
            "app_main.dependencies.get_embedding_service",
            AsyncMock(return_value=embed_service),
        ):
            result = await embedding_orchestrator.embed_source("source:test1")

        assert result == {"source_id": "source:test1", "embedded_chunks": 7}

    @pytest.mark.asyncio
    async def test_raises_on_missing_source(self, embedding_orchestrator, source_repo):
        source_repo.get = AsyncMock(return_value=None)
        with pytest.raises(ValueError, match="not found"):
            await embedding_orchestrator.embed_source("source:missing")


# ===========================================================================
# Characterization: run_summaries (public API)
# ===========================================================================

class TestRunSummaries:
    @pytest.mark.asyncio
    async def test_raises_on_missing_source(self, summarization_orchestrator, source_repo):
        source_repo.get = AsyncMock(return_value=None)
        with pytest.raises(ValueError, match="not found"):
            await summarization_orchestrator.run_summaries("source:missing")

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_default_transformations(
        self, summarization_orchestrator, transformation_repo
    ):
        import sys

        transformation_repo.get_defaults = AsyncMock(return_value=[])

        # _run_transformations imports app_main.graphs.transformation at the top
        # of its body — patch sys.modules even though we expect an early return.
        fake_module = MagicMock()
        fake_module.graph = MagicMock()
        with patch.dict(
            sys.modules, {"app_main.graphs.transformation": fake_module}
        ):
            result = await summarization_orchestrator.run_summaries("source:test1")

        assert result["transformations_run"] == 0
        assert result["source_id"] == "source:test1"

    @pytest.mark.asyncio
    async def test_runs_specific_transformation_ids(
        self, summarization_orchestrator, transformation_repo, source_repo
    ):
        import sys

        from shared.models.transformation import Transformation

        t = Transformation(
            id="transformation:t1",
            name="summarize",
            title="Summary",
            description="Summarize",
            prompt="Summarize: {text}",
            created=_NOW,
            updated=_NOW,
        )
        transformation_repo.get = AsyncMock(return_value=t)
        source_repo.get_insights = AsyncMock(return_value=[{"id": "i1"}, {"id": "i2"}])

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock()
        fake_module = MagicMock()
        fake_module.graph = mock_graph

        with patch.dict(
            sys.modules, {"app_main.graphs.transformation": fake_module}
        ):
            result = await summarization_orchestrator.run_summaries(
                "source:test1", transformation_ids=["transformation:t1"]
            )

        mock_graph.ainvoke.assert_awaited_once()
        assert result["transformations_run"] == 1
        assert result["insights_created"] == 2

    @pytest.mark.asyncio
    async def test_raises_on_unknown_transformation_id(
        self, summarization_orchestrator, transformation_repo
    ):
        import sys

        transformation_repo.get = AsyncMock(return_value=None)

        fake_module = MagicMock()
        fake_module.graph = MagicMock()
        with patch.dict(
            sys.modules, {"app_main.graphs.transformation": fake_module}
        ):
            with pytest.raises(
                ValueError, match="Transformation 'transformation:bogus' not found"
            ):
                await summarization_orchestrator.run_summaries(
                    "source:test1", transformation_ids=["transformation:bogus"]
                )

    @pytest.mark.asyncio
    async def test_skips_when_source_has_no_content(
        self, summarization_orchestrator, source_repo, transformation_repo
    ):
        import sys

        from shared.models.transformation import Transformation

        source_repo.get = AsyncMock(return_value=_make_source(full_text=None))
        t = Transformation(
            id="transformation:t1",
            name="summarize",
            title="Summary",
            description="Summarize",
            prompt="Summarize: {text}",
            created=_NOW,
            updated=_NOW,
        )
        transformation_repo.get = AsyncMock(return_value=t)

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock()
        fake_module = MagicMock()
        fake_module.graph = mock_graph

        with patch.dict(
            sys.modules, {"app_main.graphs.transformation": fake_module}
        ):
            result = await summarization_orchestrator.run_summaries(
                "source:test1", transformation_ids=["transformation:t1"]
            )

        mock_graph.ainvoke.assert_not_awaited()
        assert result["transformations_run"] == 0
