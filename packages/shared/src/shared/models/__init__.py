"""
Shared domain models.

Pure Pydantic models without database operations.
"""

from shared.models.base import ObjectModel, RecordModel
from shared.models.entity import Entity, Relation
from shared.models.export import (
    ExportFilter,
    ExportReport,
    JsonlExportRequest,
    NetworkxExportRequest,
    NetworkxFormat,
    ObsidianExportRequest,
)
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    FilteredResult,
)
from shared.models.file_tracking import (
    KnowledgeBase,
    PipelineCacheEntry,
    Project,
    SourceFolder,
    TrackedFile,
)
from shared.models.jobs import (
    Job,
    JobConfig,
    JobListResponse,
    JobProgress,
    JobResult,
    JobStatsResponse,
    JobSubmitRequest,
)
from shared.models.llm import (
    DefaultModels,
    Model,
    ModelRoute,
    ModelTypeStr,
    ModelUsageRecord,
)
from shared.models.notebook import ChatMessage, ChatSession, Note, Notebook
from shared.models.notebook_schema import (
    DEFAULT_BASE_ONTOLOGY,
    NotebookEvent,
    NotebookSchema,
    Pass1Result,
)
from shared.models.podcast import EpisodeProfile, PodcastEpisode, SpeakerProfile
from shared.models.settings import ContentSettings
from shared.models.source import Asset, Chunk, Source, SourceEmbedding, SourceInsight
from shared.models.transformation import DefaultPrompts, Transformation

__all__ = [
    # Base models
    "ObjectModel",
    "RecordModel",
    # LLM models
    "Model",
    "DefaultModels",
    "ModelRoute",
    "ModelUsageRecord",
    "ModelTypeStr",
    # Notebook models
    "Notebook",
    "Note",
    "ChatSession",
    "ChatMessage",
    # Notebook schema models (Phase B.1b)
    "DEFAULT_BASE_ONTOLOGY",
    "NotebookSchema",
    "Pass1Result",
    # Notebook event stream (Phase B.3b)
    "NotebookEvent",
    # Source models
    "Source",
    "Chunk",
    "SourceInsight",
    "SourceEmbedding",
    "Asset",
    # Transformation models
    "Transformation",
    "DefaultPrompts",
    # Podcast models
    "SpeakerProfile",
    "EpisodeProfile",
    "PodcastEpisode",
    # File tracking models
    "TrackedFile",
    "Project",
    "KnowledgeBase",
    "SourceFolder",
    "PipelineCacheEntry",
    # Job models
    "Job",
    "JobConfig",
    "JobProgress",
    "JobResult",
    "JobSubmitRequest",
    "JobListResponse",
    "JobStatsResponse",
    # Settings
    "ContentSettings",
    # Extraction models
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionResult",
    "FilteredResult",
    # Knowledge-graph entity/relation models (Phase B.1a)
    "Entity",
    "Relation",
    # Export contracts (Phase D.0) — shared by D.1 / D.2 / D.3
    "ExportFilter",
    "ExportReport",
    "JsonlExportRequest",
    "NetworkxExportRequest",
    "NetworkxFormat",
    "ObsidianExportRequest",
]
