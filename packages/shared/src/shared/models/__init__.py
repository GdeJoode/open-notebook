"""
Shared domain models.

Pure Pydantic models without database operations.
"""

from shared.models.base import ObjectModel, RecordModel
from shared.models.llm import DefaultModels, Model, ModelTypeStr, ModelUsageRecord
from shared.models.notebook import ChatMessage, ChatSession, Note, Notebook
from shared.models.settings import ContentSettings
from shared.models.source import Asset, Chunk, Source, SourceEmbedding, SourceInsight

__all__ = [
    # Base models
    "ObjectModel",
    "RecordModel",
    # LLM models
    "Model",
    "DefaultModels",
    "ModelUsageRecord",
    "ModelTypeStr",
    # Notebook models
    "Notebook",
    "Note",
    "ChatSession",
    "ChatMessage",
    # Source models
    "Source",
    "Chunk",
    "SourceInsight",
    "SourceEmbedding",
    "Asset",
    # Settings
    "ContentSettings",
]
