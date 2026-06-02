"""
SourceProcessor — orchestrates extraction → persistence for one source.

Owns just the orchestration: loading the source row, merging settings,
delegating extraction to SourceExtractor, updating the source record,
and replacing chunks. Embedding and summarization are separate services
(SourceEmbeddingOrchestrator / SourceSummarizationOrchestrator).

Replaces SourceProcessingService after the Phase 3 god-split.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from app_main.services.chunking import chunk_builder
from app_main.services.log_stream import get_log_stream
from app_main.services.source_extractor import ExtractionResult, SourceExtractor
from shared.models import Asset, Source
from shared.models.settings import ContentSettings
from shared.utils.text import strip_null_bytes
from surrealdb_service.repositories import (
    ChunkRepository,
    ContentSettingsRepository,
    SourceRepository,
)


class SourceProcessor:
    """Orchestrates per-source extraction and chunk persistence."""

    def __init__(
        self,
        source_repo: SourceRepository,
        chunk_repo: ChunkRepository,
        settings_repo: ContentSettingsRepository,
        extractor: SourceExtractor,
    ) -> None:
        self.source_repo = source_repo
        self.chunk_repo = chunk_repo
        self.settings_repo = settings_repo
        self.extractor = extractor

    async def process_source(
        self,
        source_id: str,
        content_state: Dict[str, Any],
        *,
        notebook_ids: Optional[List[str]] = None,
        processing_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract content from a source and persist chunks.

        Args:
            source_id: ID of the source record to process.
            content_state: Dict describing what to extract. Supports:
                - ``{"file_path": "/path/to/file"}`` → IngestionWorkflow
                - ``{"url": "https://..."}`` → web extraction
                - ``{"content": "raw text"}`` → single chunk wrap
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
            extracted = await self.extractor.extract(
                content_state, content_settings, source_id=source_id,
            )
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

        output_dir = (extracted.metadata or {}).get("output_directory")
        if output_dir:
            update_data["output_directory"] = output_dir

        return await self.source_repo.update(source.id, update_data)
