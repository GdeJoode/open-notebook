"""
Source processing service — orchestrates content extraction, persistence,
embedding, and transformation for a single source document.

Replaces the monolith LangGraph source_graph with a plain async service.
Uses the ingestion pipeline (Docling + WhisperX) for file-based extraction.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


_DOCLING_PARSEABLE_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".pptx", ".ppt", ".html", ".htm", ".txt", ".md",
}


def _use_docling_service() -> bool:
    """True when USE_DOCLING_SERVICE env var is truthy."""
    return os.environ.get("USE_DOCLING_SERVICE", "").lower() in {
        "1", "true", "yes", "on"
    }

from loguru import logger

from app_main.services.chunking import chunk_builder
from app_main.services.ingestion.config_builder import build_ingestion_config
from app_main.services.log_stream import get_log_stream
from shared.models import Asset, Source
from shared.models.settings import ContentSettings
from shared.utils.text import strip_null_bytes
from surrealdb_service.repositories import (
    ChunkRepository,
    ContentSettingsRepository,
    SourceRepository,
    TransformationRepository,
)


@dataclass
class ExtractionResult:
    """Intermediate result from content extraction."""

    title: Optional[str] = None
    full_text: str = ""
    file_path: Optional[str] = None
    url: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


def _is_docling_parseable_extension(path: Path) -> bool:
    return path.suffix.lower() in _DOCLING_PARSEABLE_EXTENSIONS


class SourceProcessingService:
    """Processes a source end-to-end: extract -> persist -> embed -> transform."""

    def __init__(
        self,
        source_repo: SourceRepository,
        chunk_repo: ChunkRepository,
        settings_repo: ContentSettingsRepository,
        transformation_repo: TransformationRepository,
    ) -> None:
        self.source_repo = source_repo
        self.chunk_repo = chunk_repo
        self.settings_repo = settings_repo
        self.transformation_repo = transformation_repo

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def process_source(
        self,
        source_id: str,
        content_state: Dict[str, Any],
        *,
        notebook_ids: Optional[List[str]] = None,
        processing_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract content from a source and persist chunks.

        This method only handles extraction. Embedding (see ``embed_source``)
        and summarization (see ``run_summaries``) are triggered separately.

        Args:
            source_id: ID of the source record to process.
            content_state: Dict describing what to extract. Supports:
                - ``{"file_path": "/path/to/file"}`` — process via IngestionWorkflow
                - ``{"url": "https://..."}`` — extract web page content
                - ``{"content": "raw text"}`` — wrap as single chunk
            notebook_ids: Optional notebook IDs (associations created by API layer).
            processing_overrides: Per-submission overrides merged on top of
                the global ContentSettings.

        Returns:
            Dict with ``source_id`` and ``chunk_count``.
        """
        log_stream = get_log_stream()

        # 1. Load source from DB
        log_stream.emit(source_id, "Loading source record...")
        source = await self.source_repo.get(source_id)
        if source is None:
            log_stream.emit(source_id, f"Source '{source_id}' not found", "ERROR")
            log_stream.close(source_id)
            raise ValueError(f"Source '{source_id}' not found")

        # 2. Load settings and merge per-submission overrides
        log_stream.emit(source_id, "Loading processing settings...")
        content_settings = await self.settings_repo.get()
        if processing_overrides:
            settings_data = content_settings.model_dump()
            settings_data.update(
                {k: v for k, v in processing_overrides.items() if v is not None}
            )
            content_settings = ContentSettings(**settings_data)

        # 3. Route to extraction method
        method = "file" if "file_path" in content_state else (
            "url" if "url" in content_state else "text"
        )
        log_stream.emit(source_id, f"Starting extraction (method: {method})...")
        try:
            extracted = await self._extract(content_state, content_settings, source_id=source_id)
        except Exception as e:
            log_stream.emit(source_id, f"Extraction failed: {e}", "ERROR")
            log_stream.close(source_id)
            raise

        log_stream.emit(source_id, "Extraction completed successfully")

        # 4. Update source record
        log_stream.emit(source_id, "Updating source record...")
        source = await self._update_source(source, extracted)

        # 5. Replace chunks
        chunk_count = 0
        await self.chunk_repo.delete_by_source(source_id)
        if extracted.chunks:
            prepared = chunk_builder.prepare_for_db(extracted.chunks, source_id)
            log_stream.emit(source_id, f"Saving {len(prepared)} chunks...")
            await self.chunk_repo.bulk_create(prepared)
            chunk_count = len(prepared)
            logger.info(f"Saved {chunk_count} chunks for source {source_id}")

        log_stream.emit(
            source_id,
            f"Processing complete: {chunk_count} chunks created",
        )
        log_stream.close(source_id)

        return {
            "source_id": str(source.id),
            "chunk_count": chunk_count,
        }

    async def embed_source(self, source_id: str) -> Dict[str, Any]:
        """Generate embeddings for a source's chunks."""
        source = await self.source_repo.get(source_id)
        if source is None:
            raise ValueError(f"Source '{source_id}' not found")

        await self._embed(source_id)
        embedded_chunks = await self.source_repo.get_embedding_count(source_id)
        return {
            "source_id": str(source.id),
            "embedded_chunks": embedded_chunks,
        }

    async def run_summaries(
        self,
        source_id: str,
        transformation_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run summarization transformations on a source."""
        source = await self.source_repo.get(source_id)
        if source is None:
            raise ValueError(f"Source '{source_id}' not found")

        count = await self._run_transformations(
            source, transformation_ids=transformation_ids,
        )
        insights_list = await self.source_repo.get_insights(source_id)
        return {
            "source_id": str(source.id),
            "insights_created": len(insights_list),
            "transformations_run": count,
        }

    # ------------------------------------------------------------------
    # Extraction routing
    # ------------------------------------------------------------------

    async def _extract(
        self,
        content_state: Dict[str, Any],
        content_settings: ContentSettings,
        source_id: str | None = None,
    ) -> ExtractionResult:
        """Route extraction based on content_state keys."""
        if "file_path" in content_state:
            return await self._process_file(
                file_path=content_state["file_path"],
                content_settings=content_settings,
                delete_source=content_state.get("delete_source", False),
                source_id=source_id,
            )
        elif "url" in content_state:
            return await self._process_url(
                url=content_state["url"],
                content_settings=content_settings,
            )
        elif "content" in content_state:
            return self._process_text(content=content_state["content"])
        else:
            raise ValueError(
                "content_state must contain 'file_path', 'url', or 'content'"
            )

    # ------------------------------------------------------------------
    # File processing via IngestionWorkflow (Docling + WhisperX)
    # ------------------------------------------------------------------

    async def _process_file(
        self,
        file_path: str,
        content_settings: ContentSettings,
        delete_source: bool = False,
        source_id: str | None = None,
    ) -> ExtractionResult:
        """
        Process a local file through the ingestion pipeline.

        Uses Docling for documents and WhisperX for audio/video.
        """
        from ingestion.config import (
            AcceleratorDevice,
            DoclingConfig,
            IngestionConfig,
            OcrEngine,
            TableMode,
        )
        from ingestion.workflow import IngestionWorkflow

        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        # Build IngestionConfig from ContentSettings
        config = build_ingestion_config(content_settings)

        log_stream = get_log_stream()
        emit_key = source_id or str(source_path)

        logger.info(
            f"Processing file via ingestion pipeline: {source_path.name}"
        )
        log_stream.emit(emit_key, f"Processing file: {source_path.name}")

        # IngestionWorkflow.process() is synchronous — run in thread pool.
        # Bridge loguru logs from the ingestion pipeline into LogStreamService
        # so the frontend can display verbose processing output.
        use_service = _use_docling_service() and _is_docling_parseable_extension(
            source_path
        )

        if use_service:
            # Route to the GPU-accelerated docling service; skip in-process
            # IngestionWorkflow entirely. The service writes the same output
            # files (document.json, markdown, images) to /data/output.
            from app_main.services.docling_http_client import DoclingHttpClient

            logger.info(
                f"Routing {source_path.name} to docling service (GPU)"
            )
            log_stream.emit(
                emit_key,
                f"Routing to docling service: {source_path.name}",
            )
            client = DoclingHttpClient()
            result = await client.process(source_path)
        else:
            workflow = IngestionWorkflow(config)
            loop = asyncio.get_event_loop()

            def _loguru_to_stream(message: object) -> None:
                """Forward ingestion/docling loguru logs to LogStreamService."""
                record = message.record  # type: ignore[union-attr]
                name = record.get("name", "")
                # Only forward logs from the ingestion pipeline and docling
                if not (
                    name.startswith("ingestion")
                    or "docling" in name.lower()
                ):
                    return
                level = record["level"].name
                msg = str(record["message"]).strip()
                if not msg:
                    return
                try:
                    loop.call_soon_threadsafe(
                        log_stream.emit, emit_key, msg, level
                    )
                except RuntimeError:
                    pass  # Loop closed — ignore

            sink_id = logger.add(
                _loguru_to_stream,
                level="INFO",
                format="{message}",
            )
            try:
                result = await loop.run_in_executor(
                    None, workflow.process, source_path
                )
            finally:
                logger.remove(sink_id)

        if not result.success:
            raise RuntimeError(
                f"Ingestion failed for {source_path.name}: "
                f"{result.error_message}"
            )

        # Convert IngestionResult to ExtractionResult
        chunks = None
        title = None
        full_text = ""

        if result.document:
            title = result.document.title or source_path.stem
            full_text = result.document.full_text or result.document.to_markdown()
            chunks = chunk_builder.from_document(result.document)
        elif result.transcription:
            title = source_path.stem
            full_text = result.transcription.to_markdown()
            chunks = chunk_builder.from_transcription(result.transcription)

        # Optionally delete source file
        if delete_source and source_path.exists():
            try:
                source_path.unlink()
                logger.info(f"Deleted source file: {source_path}")
            except OSError as e:
                logger.warning(f"Failed to delete source file: {e}")

        return ExtractionResult(
            title=title,
            full_text=full_text,
            file_path=file_path,
            chunks=chunks,
            metadata={
                "source_type": result.source_type.value,
                "output_directory": (
                    str(result.output_directory) if result.output_directory else None
                ),
                "markdown_path": (
                    str(result.markdown_path) if result.markdown_path else None
                ),
                "processing_time": result.processing_time_seconds,
            },
        )

    # ------------------------------------------------------------------
    # URL processing
    # ------------------------------------------------------------------

    async def _process_url(
        self,
        url: str,
        content_settings: ContentSettings,
    ) -> ExtractionResult:
        """
        Extract content from a URL using BeautifulSoup.

        Falls back to basic HTTP fetch with readability extraction.
        For JS-heavy pages, uses Playwright when available.
        """
        title, text = await self._fetch_url_content(url)

        if not text:
            raise RuntimeError(f"No content extracted from URL: {url}")

        # Wrap as a single chunk
        chunks = [
            {
                "text": text,
                "order": 0,
                "physical_page": 0,
                "printed_page": None,
                "chapter": None,
                "paragraph_number": None,
                "element_type": "web_content",
                "positions": [],
                "metadata": {"url": url},
            }
        ]

        return ExtractionResult(
            title=title,
            full_text=text,
            url=url,
            chunks=chunks,
        )

    @staticmethod
    async def _fetch_url_content(url: str) -> tuple[Optional[str], str]:
        """
        Fetch and extract readable text from a URL.

        Strategy:
        1. Try Playwright for JS-rendered content (if available)
        2. Fall back to httpx + BeautifulSoup
        """
        # Try Playwright first for JS-heavy pages
        try:
            title, text = await _fetch_with_playwright(url)
            if text and len(text.strip()) > 100:
                return title, text
        except ImportError:
            pass  # Playwright not installed
        except Exception as e:
            logger.debug(f"Playwright extraction failed, falling back: {e}")

        # Fallback: httpx + BeautifulSoup
        return await _fetch_with_httpx(url)

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    @staticmethod
    def _process_text(content: str) -> ExtractionResult:
        """Wrap raw text content into a single-chunk ExtractionResult."""
        title = content[:80].strip().split("\n")[0] if content else "Untitled"

        chunks = [
            {
                "text": content,
                "order": 0,
                "physical_page": 0,
                "printed_page": None,
                "chapter": None,
                "paragraph_number": None,
                "element_type": "text",
                "positions": [],
                "metadata": {},
            }
        ]

        return ExtractionResult(
            title=title,
            full_text=content,
            chunks=chunks,
        )

    # ------------------------------------------------------------------
    # Source record persistence
    # ------------------------------------------------------------------

    async def _update_source(
        self,
        source: Source,
        extracted: ExtractionResult,
    ) -> Source:
        """Persist extracted content back to the source record."""
        update_data: Dict[str, Any] = {
            "asset": Asset(
                url=extracted.url,
                file_path=extracted.file_path,
            ).model_dump(),
            "full_text": strip_null_bytes(extracted.full_text),
        }
        if extracted.title:
            update_data["title"] = strip_null_bytes(extracted.title)

        # Persist output directory so we can serve images later
        output_dir = (extracted.metadata or {}).get("output_directory")
        if output_dir:
            update_data["output_directory"] = output_dir

        return await self.source_repo.update(source.id, update_data)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _embed(self, source_id: str) -> None:
        """Vectorise the source via the embeddings pipeline."""
        logger.debug("Embedding content for vector search")
        from app_main.dependencies import get_embedding_service

        service = await get_embedding_service()
        await service.embed_source(source_id)

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    async def _run_transformations(
        self,
        source: Source,
        *,
        transformation_ids: Optional[List[str]] = None,
    ) -> int:
        """Run transformation graph for each requested transformation."""
        from app_main.graphs.transformation import graph as transform_graph

        transformations = []
        if transformation_ids:
            for tid in transformation_ids:
                t = await self.transformation_repo.get(tid)
                if t is None:
                    raise ValueError(f"Transformation '{tid}' not found")
                transformations.append(t)
        else:
            transformations = await self.transformation_repo.get_defaults()

        if not transformations:
            return 0

        content = source.full_text
        if not content:
            logger.warning(
                f"Source {source.id} has no content for transformations"
            )
            return 0

        count = 0
        for transformation in transformations:
            logger.debug(f"Applying transformation {transformation.name}")
            try:
                await transform_graph.ainvoke(
                    dict(
                        input_text=content,
                        source=source,
                        transformation=transformation,
                    )
                )
                count += 1
            except Exception as e:
                logger.error(
                    f"Transformation '{transformation.name}' failed: {e}"
                )

        return count

# ======================================================================
# URL extraction helpers (module-level, used by _fetch_url_content)
# ======================================================================


async def _fetch_with_playwright(url: str) -> tuple[Optional[str], str]:
    """Render a page with Playwright and extract text via BS4."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30_000)
            html = await page.content()
            title = await page.title()
        finally:
            await browser.close()

    return title, _extract_text_from_html(html)


async def _fetch_with_httpx(url: str) -> tuple[Optional[str], str]:
    """Fetch a page with httpx and extract text via BS4."""
    import httpx

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; OpenNotebook/1.0; "
                "+https://github.com/open-notebook)"
            ),
        },
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    return None, _extract_text_from_html(html)


def _extract_text_from_html(html: str) -> str:
    """
    Extract readable text from HTML using BeautifulSoup.

    Removes scripts, styles, navigation, and other non-content elements.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "noscript"]
    ):
        tag.decompose()

    # Try to find main content area
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(role="main")
        or soup.find("div", class_="content")
        or soup.body
    )

    if main is None:
        main = soup

    # Extract text with reasonable whitespace
    text = main.get_text(separator="\n", strip=True)

    # Clean up excessive blank lines
    import re

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
