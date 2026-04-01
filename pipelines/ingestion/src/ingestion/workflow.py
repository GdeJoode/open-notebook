"""
Main ingestion workflow.

Orchestrates the full ingestion pipeline:
1. Detect source type (document, audio, video)
2. Process with appropriate parser/transcriber
3. Export to standard directory structure
4. Handle destination (project vs knowledge base)
"""

import asyncio
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ingestion.config import (
    DestinationType,
    IngestionConfig,
    TableFormat,
)
from ingestion.models import (
    IngestionResult,
    SourceMetadata,
    SourceType,
)
from ingestion.parsers import DoclingParser
from ingestion.audio import WhisperXTranscriber
from ingestion.exporters import DocumentExporter, TranscriptionExporter


class IngestionWorkflow:
    """
    Main ingestion workflow orchestrator.

    Handles:
    - Source type detection
    - Document parsing with Docling
    - Audio/video transcription with WhisperX
    - Content export to standard directory structure
    - Destination routing (project vs knowledge base)
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        """
        Initialize the ingestion workflow.

        Args:
            config: IngestionConfig instance. If None, uses defaults.
        """
        self.config = config or IngestionConfig()

        # Lazy-initialized processors
        self._docling_parser: Optional[DoclingParser] = None
        self._whisperx_transcriber: Optional[WhisperXTranscriber] = None
        self._document_exporter: Optional[DocumentExporter] = None
        self._transcription_exporter: Optional[TranscriptionExporter] = None

    @property
    def docling_parser(self) -> DoclingParser:
        """Get or create Docling parser."""
        if self._docling_parser is None:
            self._docling_parser = DoclingParser(self.config.docling)
        return self._docling_parser

    @property
    def whisperx_transcriber(self) -> WhisperXTranscriber:
        """Get or create WhisperX transcriber."""
        if self._whisperx_transcriber is None:
            self._whisperx_transcriber = WhisperXTranscriber(self.config.whisperx)
        return self._whisperx_transcriber

    @property
    def document_exporter(self) -> DocumentExporter:
        """Get or create document exporter."""
        if self._document_exporter is None:
            self._document_exporter = DocumentExporter(self.config.output)
        return self._document_exporter

    @property
    def transcription_exporter(self) -> TranscriptionExporter:
        """Get or create transcription exporter."""
        if self._transcription_exporter is None:
            self._transcription_exporter = TranscriptionExporter(self.config.output)
        return self._transcription_exporter

    def process(
        self,
        source_path: Path,
        destination: Optional[DestinationType] = None,
        project_id: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> IngestionResult:
        """
        Process a single source file.

        Args:
            source_path: Path to the source file
            destination: Override destination type
            project_id: Project ID if destination is PROJECT
            source_name: Custom name for output directory

        Returns:
            IngestionResult with all extracted content and file paths
        """
        source_path = Path(source_path)

        if not source_path.exists():
            return IngestionResult.create_error(
                source_path, f"Source file not found: {source_path}"
            )

        if not self.config.is_supported(source_path):
            return IngestionResult.create_error(
                source_path, f"Unsupported file type: {source_path.suffix}"
            )

        # Determine source type
        if self.config.is_document(source_path):
            return self._process_document(source_path, source_name)
        elif self.config.is_av(source_path):
            return self._process_audio_video(source_path, source_name)
        else:
            return IngestionResult.create_error(
                source_path, f"Unknown source type: {source_path.suffix}"
            )

    def _process_document(
        self,
        source_path: Path,
        source_name: Optional[str] = None,
    ) -> IngestionResult:
        """Process a document file."""
        logger.info(f"Processing document: {source_path}")

        # Create initial result
        metadata = SourceMetadata.from_path(source_path, SourceType.DOCUMENT)
        result = IngestionResult(source_metadata=metadata)

        try:
            # Parse document
            document = self.docling_parser.parse(source_path)
            result.document = document
            metadata.page_count = document.page_count

            # Export to directory structure
            exported = self.document_exporter.export(
                document=document,
                source_name=source_name,
                copy_original=True,
                table_format=TableFormat.BOTH,
            )

            result.output_directory = exported.get("root")
            result.markdown_path = exported.get("markdown")
            result.json_path = exported.get("json")
            result.original_path = exported.get("original")
            result.table_paths = exported.get("tables", [])
            result.image_paths = exported.get("images", [])

            return result.complete(success=True)

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            # Clean up partial output directory on failure
            if result.output_directory and Path(result.output_directory).exists():
                import shutil

                try:
                    shutil.rmtree(result.output_directory)
                    logger.info(f"Cleaned up partial output: {result.output_directory}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to clean up partial output: {cleanup_err}")
            return result.complete(success=False, error=str(e))

    def _process_audio_video(
        self,
        source_path: Path,
        source_name: Optional[str] = None,
    ) -> IngestionResult:
        """Process an audio or video file."""
        logger.info(f"Processing audio/video: {source_path}")

        # Determine if audio or video
        source_type = (
            SourceType.VIDEO if self.config.is_video(source_path) else SourceType.AUDIO
        )

        # Create initial result with media metadata
        metadata = SourceMetadata.from_media_file(source_path, source_type)
        result = IngestionResult(source_metadata=metadata)

        try:
            # Transcribe
            transcription = self.whisperx_transcriber.transcribe(source_path)
            result.transcription = transcription

            # Export to directory structure (no original file stored)
            exported = self.transcription_exporter.export(
                transcription=transcription,
                source_name=source_name,
                export_subtitles=True,
            )

            result.output_directory = exported.get("root")
            result.markdown_path = exported.get("markdown")
            result.json_path = exported.get("json")
            # original_path is None for audio/video (not stored)

            return result.complete(success=True)

        except Exception as e:
            logger.error(f"Audio/video processing failed: {e}")
            # Clean up partial output directory on failure
            if result.output_directory and Path(result.output_directory).exists():
                import shutil

                try:
                    shutil.rmtree(result.output_directory)
                    logger.info(f"Cleaned up partial output: {result.output_directory}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to clean up partial output: {cleanup_err}")
            return result.complete(success=False, error=str(e))

    async def process_batch(
        self,
        source_paths: list[Path],
        on_progress: Optional[Callable[[int, int, IngestionResult], None]] = None,
    ) -> list[IngestionResult]:
        """
        Process multiple source files.

        Args:
            source_paths: List of source file paths
            on_progress: Optional callback(current, total, result) for progress updates

        Returns:
            List of IngestionResult for each source
        """
        results = []
        total = len(source_paths)

        for idx, source_path in enumerate(source_paths, 1):
            result = self.process(source_path)
            results.append(result)

            if on_progress:
                on_progress(idx, total, result)

            logger.info(f"Processed {idx}/{total}: {source_path.name}")

        return results

    async def process_batch_concurrent(
        self,
        source_paths: list[Path],
        max_concurrent: Optional[int] = None,
    ) -> list[IngestionResult]:
        """
        Process multiple source files concurrently.

        Args:
            source_paths: List of source file paths
            max_concurrent: Maximum concurrent operations (default from config)

        Returns:
            List of IngestionResult for each source
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_files
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(path: Path) -> IngestionResult:
            async with semaphore:
                # Run sync process in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.process, path)

        tasks = [process_with_semaphore(path) for path in source_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        final_results = []
        for path, result in zip(source_paths, results):
            if isinstance(result, Exception):
                final_results.append(
                    IngestionResult.create_error(path, str(result))
                )
            else:
                final_results.append(result)

        return final_results


def ingest(
    source_path: Path,
    config: Optional[IngestionConfig] = None,
    destination: Optional[DestinationType] = None,
) -> IngestionResult:
    """
    Convenience function to ingest a single source.

    Args:
        source_path: Path to the source file
        config: Optional IngestionConfig
        destination: Optional destination type override

    Returns:
        IngestionResult with all extracted content
    """
    workflow = IngestionWorkflow(config)
    return workflow.process(source_path, destination=destination)


async def ingest_batch(
    source_paths: list[Path],
    config: Optional[IngestionConfig] = None,
    concurrent: bool = True,
) -> list[IngestionResult]:
    """
    Convenience function to ingest multiple sources.

    Args:
        source_paths: List of source file paths
        config: Optional IngestionConfig
        concurrent: Whether to process concurrently

    Returns:
        List of IngestionResult for each source
    """
    workflow = IngestionWorkflow(config)

    if concurrent:
        return await workflow.process_batch_concurrent(source_paths)
    else:
        return await workflow.process_batch(source_paths)
