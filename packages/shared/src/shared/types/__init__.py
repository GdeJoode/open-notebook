"""
Shared types and enumerations.
"""

from shared.types.enums import (
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
    TableMode,
    UrlProcessingEngine,
    VlmFramework,
    VlmModel,
)
from shared.types.pipeline import (
    PipelineConfig,
    PipelineTask,
    TaskPriority,
    TaskResult,
    TaskStatus,
)

__all__ = [
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
    # Pipeline
    "PipelineConfig",
    "PipelineTask",
    "TaskPriority",
    "TaskResult",
    "TaskStatus",
]
