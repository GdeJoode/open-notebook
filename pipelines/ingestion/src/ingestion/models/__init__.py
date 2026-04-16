"""
Ingestion pipeline data models.

Provides structured data models for:
- Document extraction results
- Table and image extraction
- Audio/video transcription
- Source metadata
"""

from .document import (
    AnnotationInfo,
    ElementType,
    BoundingBox,
    ExtractedElement,
    ExtractedTable,
    ExtractedImage,
    PageContent,
    ExtractedDocument,
)
from .transcription import (
    TranscriptionWord,
    TranscriptionSegment,
    SpeakerInfo,
    TranscriptionResult,
)
from .source import (
    SourceType,
    SourceMetadata,
    IngestionResult,
)

__all__ = [
    # Document models
    "AnnotationInfo",
    "ElementType",
    "BoundingBox",
    "ExtractedElement",
    "ExtractedTable",
    "ExtractedImage",
    "PageContent",
    "ExtractedDocument",
    # Transcription models
    "TranscriptionWord",
    "TranscriptionSegment",
    "SpeakerInfo",
    "TranscriptionResult",
    # Source models
    "SourceType",
    "SourceMetadata",
    "IngestionResult",
]
