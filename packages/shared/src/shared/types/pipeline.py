"""
Pipeline task definitions for the document processing system.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineTask(str, Enum):
    """
    All available pipeline tasks in the document processing system.

    Tasks are grouped by pipeline:
    - INGEST_*: Document ingestion and parsing
    - EXTRACT_*: Entity and ontology extraction
    - SUMMARIZE_*: Document summarization
    - EMBED_*: Vector embedding generation
    - ENRICH_*: Knowledge enrichment
    - RETRIEVE_*: Information retrieval
    """
    # Ingestion Pipeline
    INGEST_PARSE = "ingest_parse"
    INGEST_CHUNK = "ingest_chunk"
    INGEST_CREATE_SOURCE = "ingest_create_source"
    INGEST_SCRAPE_URL = "ingest_scrape_url"
    INGEST_TRANSCRIBE = "ingest_transcribe"

    # Ontology Extraction Pipeline
    EXTRACT_ENTITIES = "extract_entities"
    EXTRACT_RELATIONS = "extract_relations"
    EXTRACT_ONTOLOGY = "extract_ontology"
    EXTRACT_TOPICS = "extract_topics"

    # Summarization Pipeline
    SUMMARIZE_CHUNK = "summarize_chunk"
    SUMMARIZE_SECTION = "summarize_section"
    SUMMARIZE_DOCUMENT = "summarize_document"
    SUMMARIZE_RAPTOR = "summarize_raptor"

    # Embedding Pipeline
    EMBED_CHUNKS = "embed_chunks"
    EMBED_SUMMARIES = "embed_summaries"
    EMBED_ENTITIES = "embed_entities"
    EMBED_INSIGHTS = "embed_insights"

    # Enrichment Pipeline
    ENRICH_ENTITY_LINKING = "enrich_entity_linking"
    ENRICH_FACT_VERIFICATION = "enrich_fact_verification"
    ENRICH_METADATA = "enrich_metadata"
    ENRICH_CROSS_REFERENCE = "enrich_cross_reference"

    # Retrieval Pipeline
    RETRIEVE_VECTOR_SEARCH = "retrieve_vector_search"
    RETRIEVE_TEXT_SEARCH = "retrieve_text_search"
    RETRIEVE_GRAPH_SEARCH = "retrieve_graph_search"
    RETRIEVE_HYBRID_SEARCH = "retrieve_hybrid_search"


class TaskStatus(str, Enum):
    """Status of a pipeline task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class TaskPriority(str, Enum):
    """Priority levels for pipeline tasks."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskResult(BaseModel):
    """Result of a completed pipeline task."""
    task: PipelineTask
    status: TaskStatus
    source_id: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Configuration for a pipeline execution."""
    tasks: List[PipelineTask]
    source_id: Optional[str] = None
    notebook_id: Optional[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    options: Dict[str, Any] = Field(default_factory=dict)

    # Task-specific overrides
    skip_on_error: bool = False
    retry_count: int = 0
    timeout_seconds: Optional[int] = None
