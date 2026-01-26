"""
Ingestion Pipeline

Document ingestion pipeline for parsing, chunking, and source creation.

Features:
- Full Docling support for PDF, DOCX, XLSX, PPTX, HTML
- WhisperX transcription for audio/video with speaker diarization
- Table extraction with MD and CSV export
- Image extraction with VLM descriptions
- Per-page content organization
- Fixed directory structure for outputs

Usage:
    from ingestion import ingest, IngestionConfig

    # Simple usage
    result = ingest(Path("document.pdf"))

    # With custom config
    config = IngestionConfig.for_cli()
    result = ingest(Path("document.pdf"), config=config)

    # Batch processing
    results = await ingest_batch([Path("a.pdf"), Path("b.mp3")])
"""

from ingestion.config import (
    AcceleratorDevice,
    DestinationType,
    DoclingConfig,
    ImageRefMode,
    IngestionConfig,
    OcrEngine,
    OutputConfig,
    TableFormat,
    TableMode,
    WhisperModel,
    WhisperXConfig,
)
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
from ingestion.workflow import (
    IngestionWorkflow,
    ingest,
    ingest_batch,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "AcceleratorDevice",
    "DestinationType",
    "DoclingConfig",
    "ImageRefMode",
    "IngestionConfig",
    "OcrEngine",
    "OutputConfig",
    "TableFormat",
    "TableMode",
    "WhisperModel",
    "WhisperXConfig",
    # Models
    "BoundingBox",
    "ElementType",
    "ExtractedDocument",
    "ExtractedElement",
    "ExtractedImage",
    "ExtractedTable",
    "IngestionResult",
    "PageContent",
    "SourceMetadata",
    "SourceType",
    "SpeakerInfo",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionWord",
    # Workflow
    "IngestionWorkflow",
    "ingest",
    "ingest_batch",
]
