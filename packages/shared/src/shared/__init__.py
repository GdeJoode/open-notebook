"""
Shared package - Pure Pydantic models and utilities.

This package provides:
- Domain models (Notebook, Source, Chunk, Note, etc.)
- Type definitions and enumerations
- Text processing utilities

All models are pure Pydantic without database operations.
Database persistence is handled by the surrealdb-service package.
"""

from shared.models import (
    Asset,
    ChatMessage,
    ChatSession,
    Chunk,
    ContentSettings,
    Note,
    Notebook,
    ObjectModel,
    RecordModel,
    Source,
    SourceEmbedding,
    SourceInsight,
)
from shared.types import (
    ChunkingMethod,
    ContentProcessingEngine,
    DoclingPipeline,
    ElementType,
    EmbeddingOption,
    FileOperation,
    GpuDevice,
    InsightType,
    NamingScheme,
    NoteType,
    OcrEngine,
    PipelineConfig,
    PipelineTask,
    TableMode,
    TaskPriority,
    TaskResult,
    TaskStatus,
    UrlProcessingEngine,
    VlmFramework,
    VlmModel,
)
from shared.utils import (
    clean_text,
    count_tokens_estimate,
    extract_title,
    normalize_whitespace,
    split_into_sentences,
    split_text,
    truncate_text,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "ObjectModel",
    "RecordModel",
    "Notebook",
    "Note",
    "ChatSession",
    "ChatMessage",
    "Source",
    "Chunk",
    "SourceInsight",
    "SourceEmbedding",
    "Asset",
    "ContentSettings",
    # Enums
    "ChunkingMethod",
    "ContentProcessingEngine",
    "DoclingPipeline",
    "ElementType",
    "EmbeddingOption",
    "FileOperation",
    "GpuDevice",
    "InsightType",
    "NamingScheme",
    "NoteType",
    "OcrEngine",
    "TableMode",
    "UrlProcessingEngine",
    "VlmFramework",
    "VlmModel",
    # Pipeline types
    "PipelineConfig",
    "PipelineTask",
    "TaskPriority",
    "TaskResult",
    "TaskStatus",
    # Utilities
    "clean_text",
    "count_tokens_estimate",
    "extract_title",
    "normalize_whitespace",
    "split_into_sentences",
    "split_text",
    "truncate_text",
]
