"""
SourceExtractor — content extraction from files, URLs, and raw text.

Routes a ``content_state`` dict to the appropriate extractor:
- ``{"file_path": "..."}`` → ingestion pipeline (Docling/WhisperX) or docling service
- ``{"url": "..."}`` → Playwright (preferred) or httpx + BeautifulSoup
- ``{"content": "..."}`` → wrap as single chunk

Pulled out of SourceProcessingService in Phase 3 of the refactor so the
orchestrator (SourceProcessor) only owns persistence concerns, not the
heavy lifting of turning input into chunks.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app_main.services.chunking import chunk_builder
from app_main.services.ingestion.config_builder import build_ingestion_config
from app_main.services.log_stream import get_log_stream
from shared.models.settings import ContentSettings


_DOCLING_PARSEABLE_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".pptx", ".ppt", ".html", ".htm", ".txt", ".md",
}


def _use_docling_service() -> bool:
    """True when ``USE_DOCLING_SERVICE`` env var is truthy."""
    return os.environ.get("USE_DOCLING_SERVICE", "").lower() in {
        "1", "true", "yes", "on"
    }


def _is_docling_parseable_extension(path: Path) -> bool:
    return path.suffix.lower() in _DOCLING_PARSEABLE_EXTENSIONS


@dataclass
class ExtractionResult:
    """Intermediate result from content extraction.

    Produced by SourceExtractor, consumed by SourceProcessor for source
    record update + chunk persistence.
    """

    title: Optional[str] = None
    full_text: str = ""
    file_path: Optional[str] = None
    url: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class SourceExtractor:
    """Content extraction over files, URLs, and raw text.

    Stateless — no constructor dependencies. Each ``extract`` call takes
    the input + settings and returns a populated ExtractionResult.
    """

    async def extract(
        self,
        content_state: Dict[str, Any],
        content_settings: ContentSettings,
        *,
        source_id: str | None = None,
    ) -> ExtractionResult:
        """Route extraction based on ``content_state`` keys."""
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
        """Process a local file through the ingestion pipeline.

        Uses Docling for documents and WhisperX for audio/video. When
        ``USE_DOCLING_SERVICE=1`` parseable files route to the GPU
        docling HTTP service instead of in-process Docling.
        """
        from ingestion.workflow import IngestionWorkflow

        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        config = build_ingestion_config(content_settings)

        log_stream = get_log_stream()
        emit_key = source_id or str(source_path)

        logger.info(
            f"Processing file via ingestion pipeline: {source_path.name}"
        )
        log_stream.emit(emit_key, f"Processing file: {source_path.name}")

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
        """Extract content from a URL using BeautifulSoup, with Playwright
        fallback for JS-heavy pages.
        """
        title, text = await self._fetch_url_content(url)

        if not text:
            raise RuntimeError(f"No content extracted from URL: {url}")

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
        """Fetch and extract readable text from a URL.

        Strategy:
        1. Try Playwright for JS-rendered content (if available)
        2. Fall back to httpx + BeautifulSoup
        """
        try:
            title, text = await _fetch_with_playwright(url)
            if text and len(text.strip()) > 100:
                return title, text
        except ImportError:
            pass  # Playwright not installed
        except Exception as e:
            logger.debug(f"Playwright extraction failed, falling back: {e}")

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


# ======================================================================
# URL extraction helpers (module-level, used by SourceExtractor._fetch_url_content)
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
    """Extract readable text from HTML using BeautifulSoup.

    Removes scripts, styles, navigation, and other non-content elements.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "noscript"]
    ):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(role="main")
        or soup.find("div", class_="content")
        or soup.body
    )

    if main is None:
        main = soup

    text = main.get_text(separator="\n", strip=True)

    import re

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
