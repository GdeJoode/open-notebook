"""
SourceProcessor — orchestrates extraction → persistence for one source.

Owns just the orchestration: loading the source row, merging settings,
delegating extraction to SourceExtractor, updating the source record,
and writing chunks. Embedding and summarization are separate services
(SourceEmbeddingOrchestrator / SourceSummarizationOrchestrator).

F2-04 (ADR 0003): elke verwerking is een run (``processing_run``) op een
bronversie (``source_version``, uniek op de inhoudshash). Fragmenten en de
documentstructuur krijgen die versie en run; de fragmenten van eerdere runs
blijven staan. De bronversie wijst daarna met ``current_run`` naar deze run.

Replaces SourceProcessingService after the Phase 3 god-split.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from shared.models import Asset, Source
from shared.models.settings import ContentSettings
from shared.utils.text import strip_null_bytes
from surrealdb_service.repositories import (
    ChunkRepository,
    ContentSettingsRepository,
    ProcessingRunRepository,
    SourceRepository,
    SourceVersionRepository,
)

from app_main.services.chunking import chunk_builder
from app_main.services.flow_register import fork_commit
from app_main.services.graph.doc_graph_builder import DocGraphBuilder
from app_main.services.log_stream import get_log_stream
from app_main.services.source_extractor import ExtractionResult, SourceExtractor

#: De pipeline van een run die de fork zelf start (F2-04). Geen flow uit Git,
#: dus geen controle van de flow; die geldt voor runs van Langflow.
PIPELINE_ID = "source_processor"


def _inhoudshash(content_state: Dict[str, Any], extracted: ExtractionResult) -> str:
    """SHA-256 van de ruwe inhoud: het bestand, anders de tekst.

    Bij een URL is de ruwe inhoud hier niet beschikbaar; dan telt de
    geëxtraheerde tekst.
    """
    pad = content_state.get("file_path") or extracted.file_path
    h = hashlib.sha256()
    if pad and Path(pad).is_file():
        with open(pad, "rb") as f:
            for blok in iter(lambda: f.read(1 << 20), b""):
                h.update(blok)
        return h.hexdigest()
    tekst = extracted.full_text or content_state.get("content") or ""
    h.update(tekst.encode("utf-8"))
    return h.hexdigest()


class SourceProcessor:
    """Orchestrates per-source extraction and chunk persistence."""

    def __init__(
        self,
        source_repo: SourceRepository,
        chunk_repo: ChunkRepository,
        settings_repo: ContentSettingsRepository,
        extractor: SourceExtractor,
        graph_builder: Optional[DocGraphBuilder] = None,
        version_repo: Optional[SourceVersionRepository] = None,
        run_repo: Optional[ProcessingRunRepository] = None,
    ) -> None:
        self.source_repo = source_repo
        self.chunk_repo = chunk_repo
        self.settings_repo = settings_repo
        self.extractor = extractor
        self.version_repo = version_repo or SourceVersionRepository()
        self.run_repo = run_repo or ProcessingRunRepository()
        # Structure-graph builder (Phase I.F). Optional + best-effort: a build
        # failure must never fail the ingestion (see process_source).
        self.graph_builder = graph_builder or DocGraphBuilder()

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

        # 3. Run (F2-04, ADR 0003): elke verwerking is een run, ook een die
        # mislukt. De eerste verwerking heet `inname`; `handmatig` is alleen
        # voor correcties door de beheerder (ADR 0005). De pipeline is code van
        # de fork zelf, geen flow; die staat in Git (`flow_vastgelegd`), met
        # de commit van de fork als `pipeline_version`.
        run = await self.run_repo.create(
            trigger="herverwerking" if source.current_version else "inname",
            pipeline_id=PIPELINE_ID,
            pipeline_version={"git_commit": fork_commit(), "flow_hash": None},
            input_classification=source.classification or "onbekend",
            flow_vastgelegd=True,
        )
        run_id = str(run.id)

        # 4. Route to extraction method
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
            await self.run_repo.close(run_id, status="mislukt", error=type(e).__name__)
            raise

        log_stream.emit(source_id, "Extraction completed successfully")

        try:
            # 5. Update source record
            log_stream.emit(source_id, "Updating source record...")
            source = await self._update_source(source, extracted)

            # 6. Bronversie: een nieuwe run met nieuwe fragmenten; de
            # fragmenten van eerdere runs blijven staan.
            versie, nieuwe_versie = await self.version_repo.register(
                source_id,
                _inhoudshash(content_state, extracted),
                run_id,
                original=extracted.file_path or extracted.url,
            )
            versie_id = str(versie.id)

            # 7. Write chunks for this version and run.
            chunk_count = 0
            created_chunks: List[Any] = []
            if extracted.chunks:
                extractor = (extracted.metadata or {}).get("parser_engine_used") or method
                prepared = [
                    {**c, "source_version": versie_id, "run": run_id,
                     "extractor": str(extractor)}
                    for c in chunk_builder.prepare_for_db(extracted.chunks, source_id)
                ]
                log_stream.emit(source_id, f"Saving {len(prepared)} chunks...")
                created_chunks = await self.chunk_repo.bulk_create(prepared)
                chunk_count = len(prepared)
                logger.info(f"Saved {chunk_count} chunks for source {source_id}")
            await self.version_repo.set_current_run(versie_id, run_id)
        except Exception as e:
            await self.run_repo.close(run_id, status="mislukt", error=type(e).__name__)
            raise

        # 8. Build the document structure graph (Phase I.F). Best-effort: a
        # graph-build failure must NOT fail the whole ingestion — the chunks are
        # already persisted and every other downstream feature works without the
        # structure graph. Log it in the run and continue.
        regels: List[Dict[str, Any]] = []
        if created_chunks:
            fout = await self._build_structure_graph(
                source_id, created_chunks, source_version=versie_id, run=run_id
            )
            if fout:
                regels.append({"at": datetime.now(timezone.utc), "step": "structuur",
                               "status": "mislukt", "message": fout})
        await self.run_repo.close(
            run_id,
            status="geslaagd" if (chunk_count or nieuwe_versie) else "niets_nieuws",
            log=regels,
        )

        log_stream.emit(
            source_id,
            f"Processing complete: {chunk_count} chunks created",
        )
        log_stream.close(source_id)

        return {
            "source_id": str(source.id),
            "chunk_count": chunk_count,
        }

    async def _build_structure_graph(
        self,
        source_id: str,
        created_chunks: List[Any],
        *,
        source_version: str,
        run: str,
    ) -> Optional[str]:
        """Build the document structure graph for a source (Phase I.F).

        Best-effort hook: converts the freshly-persisted ``Chunk`` rows into the
        dict shape ``DocGraphBuilder`` expects and delegates. Any failure is
        logged and swallowed so the structure graph can never break ingestion —
        the chunks are already saved and every other feature is graph-agnostic.
        """
        try:
            chunk_dicts = [
                {
                    "id": c.id,
                    "order": c.order,
                    "element_type": c.element_type,
                    "text": c.text,
                    "physical_page": c.physical_page,
                    "positions": c.positions,
                    "metadata": c.metadata,
                }
                for c in created_chunks
            ]
            graph = await self.graph_builder.build(
                source_id, chunks=chunk_dicts, source_version=source_version, run=run
            )
            logger.info(
                f"Structure graph built for {source_id}: "
                f"{len(graph.nodes)} nodes"
            )
        except Exception as e:  # noqa: BLE001 — best-effort, never fatal
            logger.warning(
                f"Structure-graph build failed for {source_id} "
                f"(ingestion continues): {e}"
            )
            # Voor het runlogboek: alleen het soort fout, geen inhoud.
            return f"Structuur niet opgebouwd: {type(e).__name__}"
        return None

    async def _update_source(
        self,
        source: Source,
        extracted: ExtractionResult,
    ) -> Source:
        """Persist extracted content back to the source record.

        Lifts a curated subset of ``ExtractionResult.metadata`` onto the
        Source-level ``metadata`` dict (Phase A.1c, Q-A-3) so the UI can
        render a "parsed by MinerU (auto-fallback)" badge and tooltip.
        Only the parser-engine-provenance keys are persisted at the
        Source level — large per-element bags stay on the
        ExtractionResult so the DB row doesn't bloat.
        """
        update_data: Dict[str, Any] = {
            "asset": Asset(
                url=extracted.url,
                file_path=extracted.file_path,
            ).model_dump(),
            "full_text": strip_null_bytes(extracted.full_text),
        }
        if extracted.title:
            update_data["title"] = strip_null_bytes(extracted.title)

        extraction_metadata = extracted.metadata or {}
        output_dir = extraction_metadata.get("output_directory")
        if output_dir:
            update_data["output_directory"] = output_dir

        # Merge parser-engine provenance onto Source.metadata. Keep keys
        # consistent with the UI badge contract in Phase A.2:
        #   parser_engine_used, extraction_confidence,
        #   extraction_confidence_signals, extraction_fallback_triggered.
        # Anything else (output_directory, markdown_path, processing_time)
        # is already lifted onto top-level Source fields elsewhere.
        existing_metadata: Dict[str, Any] = dict(source.metadata or {})
        provenance_keys = (
            "parser_engine_used",
            "extraction_confidence",
            "extraction_confidence_signals",
            "extraction_fallback_triggered",
        )
        for key in provenance_keys:
            if key in extraction_metadata:
                existing_metadata[key] = extraction_metadata[key]
        if existing_metadata != (source.metadata or {}):
            update_data["metadata"] = existing_metadata

        return await self.source_repo.update(source.id, update_data)
