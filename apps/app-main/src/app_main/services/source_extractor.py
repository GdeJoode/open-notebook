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
from app_main.services.parsing import (
    DEFAULT_THRESHOLD,
    DOCLING_PARSEABLE_EXTENSIONS,
    extract_with_auto_fallback,
    resolve_parser_route,
)
from shared.models.settings import ContentSettings


def _use_docling_service() -> bool:
    """True when ``USE_DOCLING_SERVICE`` env var is truthy."""
    return os.environ.get("USE_DOCLING_SERVICE", "").lower() in {
        "1", "true", "yes", "on"
    }


def _is_docling_parseable_extension(path: Path) -> bool:
    return path.suffix.lower() in DOCLING_PARSEABLE_EXTENSIONS


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

        Routing (Phase A.1c):
        * ``content_settings.parser_engine == "mineru"`` → MinerU HTTP
          service when the extension is in
          ``content_settings.mineru_supported_extensions``; otherwise
          fall back to Docling and log at INFO.
        * ``"docling"`` / ``"simple"`` → Docling path (current behaviour).
          ``"simple"`` is treated as Docling for now because there is no
          separate simple-extraction implementation in the codebase; the
          enum is preserved so user-facing copy is unchanged.
        * ``"auto"`` (A.1c, document extensions only) → confidence-driven
          fallback: Docling first, score the result, then re-run via
          MinerU when ``confidence < content_settings.docling_min_confidence``.
          Auto persists ``extraction_confidence``,
          ``extraction_confidence_signals``, and
          ``extraction_fallback_triggered`` on metadata so the UI badge
          can surface provenance to the user (Q-A-3).

        Audio / video continues to go through ``IngestionWorkflow``
        (WhisperX) regardless of ``parser_engine`` because the dispatcher
        only fires on document-style files.

        ``parser_engine_used`` reflects the engine that *executed*, not
        the user setting — audio routed through WhisperX records
        ``"whisperx"`` and ``parser_engine="simple"`` falling through to
        Docling records ``"docling"``.
        """
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        log_stream = get_log_stream()
        emit_key = source_id or str(source_path)

        logger.info(
            f"Processing file via ingestion pipeline: {source_path.name}"
        )
        log_stream.emit(emit_key, f"Processing file: {source_path.name}")

        # Decide how to parse this file in ONE place. resolve_parser_route is
        # the single source of truth: it folds the concrete-engine choice and
        # the A.1c auto-fallback decision together so they can't drift. Settings
        # default to "docling"; only routable extensions get the chance to pick
        # MinerU, and only an explicit "auto" on a docling-parseable file arms
        # the confidence fallback.
        parser_engine_setting = getattr(content_settings, "parser_engine", "docling") or "docling"
        route = resolve_parser_route(
            parser_engine_setting,
            source_path,
            mineru_supported_extensions=getattr(
                content_settings, "mineru_supported_extensions", None
            ),
        )
        use_auto_fallback = route.use_auto_fallback
        use_mineru = route.engine == "mineru"

        engine_used: str
        result: Any
        confidence_metadata: Dict[str, Any] = {}

        if use_auto_fallback:
            engine_used, result, confidence_metadata = await self._run_auto_fallback(
                source_path=source_path,
                content_settings=content_settings,
                emit_key=emit_key,
                log_stream=log_stream,
                source_id=source_id,
            )
        elif use_mineru:
            from app_main.services.mineru_http_client import MineruHttpClient

            logger.info(f"Routing {source_path.name} to MinerU service")
            log_stream.emit(
                emit_key,
                f"Routing to MinerU service: {source_path.name}",
            )
            client = MineruHttpClient()
            result = await client.process(source_path)
            engine_used = "mineru"
        else:
            result, engine_used = await self._run_docling(
                source_path=source_path,
                content_settings=content_settings,
                emit_key=emit_key,
                log_stream=log_stream,
            )

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

        metadata: Dict[str, Any] = {
            "source_type": result.source_type.value,
            "output_directory": (
                str(result.output_directory) if result.output_directory else None
            ),
            "markdown_path": (
                str(result.markdown_path) if result.markdown_path else None
            ),
            "processing_time": result.processing_time_seconds,
            "parser_engine_used": engine_used,
        }
        metadata.update(confidence_metadata)

        return ExtractionResult(
            title=title,
            full_text=full_text,
            file_path=file_path,
            chunks=chunks,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Engine dispatch helpers
    # ------------------------------------------------------------------

    async def _run_docling(
        self,
        *,
        source_path: Path,
        content_settings: ContentSettings,
        emit_key: str,
        log_stream: Any,
    ) -> tuple[Any, str]:
        """Run the docling extraction (HTTP service or in-process workflow).

        Returns ``(IngestionResult, engine_used)`` — engine_used is
        ``"docling"`` for documents or ``"whisperx"`` when the in-process
        workflow routed audio/video through transcription instead.
        """
        from ingestion.workflow import IngestionWorkflow

        is_doc_ext = _is_docling_parseable_extension(source_path)
        if _use_docling_service() and is_doc_ext:
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
            return result, "docling"

        config = build_ingestion_config(content_settings)
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

        # IngestionWorkflow dispatches internally to either Docling
        # (documents) or WhisperX (audio/video).
        if result.transcription is not None and result.document is None:
            return result, "whisperx"
        return result, "docling"

    async def _run_auto_fallback(
        self,
        *,
        source_path: Path,
        content_settings: ContentSettings,
        emit_key: str,
        log_stream: Any,
        source_id: str | None = None,
    ) -> tuple[str, Any, Dict[str, Any]]:
        """Run docling+MinerU through the auto-fallback orchestrator.

        Returns ``(engine_used, IngestionResult, confidence_metadata)``.
        ``confidence_metadata`` contains ``extraction_confidence``,
        ``extraction_confidence_signals`` and
        ``extraction_fallback_triggered`` for the UI badge.
        """
        from app_main.services.mineru_http_client import MineruHttpClient

        log_stream.emit(
            emit_key,
            f"Auto-mode: trying Docling first for {source_path.name}",
        )

        # Explicit None check — not truthy — because 0.0 is a legal value
        # meaning "always use docling, never fall back" (the API schema
        # permits ge=0.0). `value or DEFAULT_THRESHOLD` would silently
        # demote 0.0 to 0.95, the opposite of what the caller asked for.
        raw_threshold = getattr(content_settings, "docling_min_confidence", None)
        threshold = float(
            raw_threshold if raw_threshold is not None else DEFAULT_THRESHOLD
        )

        # Adapter so the orchestrator can call ``client.process(path)``
        # without caring whether docling went via the HTTP service or the
        # in-process workflow.
        class _DoclingAdapter:
            def __init__(self, extractor: SourceExtractor) -> None:
                self._extractor = extractor
                # Cache the engine label inferred from _run_docling so
                # auto-fallback's "engine_used" reflects whisperx for
                # audio-via-workflow. In practice auto-fallback only
                # runs on docling-parseable extensions, but we preserve
                # the field for forward-compat.
                self.last_engine_label: str = "docling"

            async def process(self, path: Path):
                result, engine = await self._extractor._run_docling(
                    source_path=path,
                    content_settings=content_settings,
                    emit_key=emit_key,
                    log_stream=log_stream,
                )
                self.last_engine_label = engine
                return result

        docling_adapter = _DoclingAdapter(self)
        mineru_client = MineruHttpClient()

        chosen_result, engine_used, score = await extract_with_auto_fallback(
            source_path,
            docling_client=docling_adapter,
            mineru_client=mineru_client,
            threshold=threshold,
            source_id=source_id,
        )

        # When docling actually ran (engine_used="docling") and the
        # adapter saw whisperx (audio file slipped through), prefer the
        # adapter's label so the metadata reflects what executed.
        if engine_used == "docling" and docling_adapter.last_engine_label != "docling":
            engine_used = docling_adapter.last_engine_label

        log_stream.emit(
            emit_key,
            f"Auto-mode picked {engine_used} (confidence {score.overall:.3f}, "
            f"threshold {score.threshold:.3f})",
        )

        confidence_metadata: Dict[str, Any] = {
            "extraction_confidence": score.overall,
            "extraction_confidence_signals": dict(score.signals),
            "extraction_fallback_triggered": score.decision == "fallback",
        }

        return engine_used, chosen_result, confidence_metadata

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
