"""
File tracking domain models.

Models for tracking files through the ingestion pipeline and organizing
them into projects or knowledge bases.
"""

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from shared.models.base import ObjectModel
from shared.types.enums import FileStatus, SourceType, StorageLocation


class TrackedFile(ObjectModel):
    """
    A file being tracked through the ingestion pipeline.

    Tracks the processing status and location of files as they move
    through the pipeline stages: pending -> ingested -> entities_extracted
    -> summarized -> saved_in_kb/saved_as_project.
    """
    table_name: ClassVar[str] = "tracked_file"

    # File identification
    filename: str = Field(description="Original filename")
    file_path: str = Field(description="Full path to the file")
    file_hash: Optional[str] = Field(None, description="SHA256 hash for deduplication")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")

    # Processing status
    status: str = Field(
        default=FileStatus.PENDING.value,
        description="Current processing status"
    )
    status_message: Optional[str] = Field(None, description="Status details or error message")
    last_status_change: Optional[datetime] = Field(None, description="When status last changed")

    # Location information
    storage_location: str = Field(
        default=StorageLocation.TEMP.value,
        description="Type of storage location"
    )
    project_id: Optional[str] = Field(None, description="Reference to project (if in project)")
    knowledge_base_id: Optional[str] = Field(None, description="Reference to KB (if in KB)")

    # Processing results references
    source_id: Optional[str] = Field(None, description="Reference to created Source record")
    transcription_id: Optional[str] = Field(None, description="Reference to transcription (audio/video)")
    source_folder_id: Optional[str] = Field(None, description="Reference to source_folder record")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional file metadata"
    )


class Project(ObjectModel):
    """
    A standalone project for organizing related files.

    Projects are used for batch processing and keeping related documents
    together without linking them to the knowledge graph.
    """
    table_name: ClassVar[str] = "project"

    # Project identification
    name: str = Field(description="Project name")
    slug: str = Field(description="URL-safe project identifier")
    description: Optional[str] = Field(None, description="Project description")

    # Directory paths
    base_path: str = Field(description="Base directory path for the project")
    input_path: Optional[str] = Field(None, description="Input files directory")
    output_path: Optional[str] = Field(None, description="Output files directory")
    exports_path: Optional[str] = Field(None, description="Exports directory")

    # Statistics
    file_count: int = Field(default=0, description="Number of files in project")
    total_size_bytes: int = Field(default=0, description="Total size of all files")

    # Processing status
    is_processing: bool = Field(default=False, description="Whether batch processing is active")
    last_processed: Optional[datetime] = Field(None, description="When project was last processed")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional project metadata"
    )


class KnowledgeBase(ObjectModel):
    """
    A knowledge base for storing and linking related content.

    Files in a knowledge base are tracked in the knowledge graph and
    can be queried and connected to other sources.
    """
    table_name: ClassVar[str] = "knowledge_base"

    # KB identification
    name: str = Field(description="Knowledge base name")
    slug: str = Field(description="URL-safe KB identifier")
    description: Optional[str] = Field(None, description="KB description")

    # Directory paths
    base_path: str = Field(description="Base directory path for the KB")
    sources_path: Optional[str] = Field(None, description="Source files directory")
    exports_path: Optional[str] = Field(None, description="Exports directory")
    media_path: Optional[str] = Field(None, description="Media files directory")

    # Statistics
    file_count: int = Field(default=0, description="Number of files in KB")
    total_size_bytes: int = Field(default=0, description="Total size of all files")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional KB metadata"
    )


class SourceFolder(ObjectModel):
    """
    A per-source directory record.

    Each source document gets its own folder with a short random ID.
    Pipeline outputs are cached per-step in this folder.
    """
    table_name: ClassVar[str] = "source_folder"

    # Identification
    source_id: str = Field(description="8-char random ID for global uniqueness")
    source_record_id: Optional[str] = Field(None, description="Link to domain source table")
    title: str = Field(description="Human-readable title")
    slug: str = Field(description="Filesystem-safe version of title")
    folder_path: str = Field(description="Absolute path to the folder")

    # Original file info
    original_filename: Optional[str] = Field(None, description="Original filename")
    original_file_path: Optional[str] = Field(None, description="Path to original file in folder")
    file_size_bytes: Optional[int] = Field(None, description="Original file size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type of original file")

    # Classification
    source_type: str = Field(
        default=SourceType.DOCUMENT.value,
        description="Type of source (document, audio, video, text, url)"
    )

    # Statistics
    total_cache_entries: int = Field(default=0, description="Number of cached pipeline outputs")
    total_folder_size_bytes: int = Field(default=0, description="Total size of all files in folder")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source folder metadata"
    )


class PipelineCacheEntry(ObjectModel):
    """
    A versioned cache record for a pipeline step output.

    Each pipeline step's output is cached per source folder.
    Old outputs are timestamped and preserved, never deleted.
    """
    table_name: ClassVar[str] = "pipeline_cache"

    # References
    source_folder_id: str = Field(description="Reference to source_folder record")
    source_id: str = Field(description="Short source ID (for filename construction)")

    # Pipeline step info
    pipeline_step: str = Field(description="Pipeline step name (e.g. 'docling_parse')")
    version: int = Field(default=1, description="Incrementing version number")
    is_current: bool = Field(default=True, description="Whether this is the active version")

    # File info
    file_path: str = Field(description="Path to cached output file")
    file_format: str = Field(default="json", description="File format (json, md, txt, etc.)")
    file_size_bytes: Optional[int] = Field(None, description="Size of cached file")

    # Processing metadata
    processing_time_seconds: Optional[float] = Field(None, description="Time taken to produce this output")
    config_hash: Optional[str] = Field(None, description="Hash of config used, to detect changes")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional cache entry metadata"
    )
