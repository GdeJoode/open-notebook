"""
Source-related domain models.

Pure Pydantic models for documents, chunks, insights, and embeddings.
Database operations are handled by the surrealdb-service package.
"""

from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from shared.models.base import ObjectModel
from shared.types.enums import ElementType, InsightType


class Asset(BaseModel):
    """Reference to an external asset (file or URL)."""
    file_path: Optional[str] = None
    url: Optional[str] = None


class Chunk(ObjectModel):
    """
    Represents a document chunk with spatial information.

    Each chunk corresponds to a document element (paragraph, table, title, etc.)
    extracted during document processing with bounding box positions.
    """
    table_name: ClassVar[str] = "chunk"

    source: Optional[str] = Field(None, description="Reference to the source document")
    text: str = Field(description="Text content of the chunk")
    order: int = Field(description="Order of chunk in the document (0-indexed)")

    # Page information
    physical_page: int = Field(description="Physical page number in PDF (0-indexed)")
    printed_page: Optional[int] = Field(None, description="Printed page number if available")

    # Structural metadata
    chapter: Optional[str] = Field(None, description="Chapter/section heading")
    paragraph_number: Optional[int] = Field(None, description="Paragraph index within section")
    element_type: str = Field(
        default=ElementType.PARAGRAPH.value,
        description="Type of element (paragraph, title, table, list, etc.)"
    )

    # Bounding box positions: [[page, x1, x2, y1, y2], ...]
    positions: List[List[float]] = Field(
        default_factory=list,
        description="Bounding box positions [[page, x1, x2, y1, y2], ...]"
    )

    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chunk metadata"
    )

    # Section hierarchy from document structure
    section_path: List[str] = Field(
        default_factory=list,
        description="Heading breadcrumb path e.g. ['Chapter 1', 'Section 1.2']"
    )
    section_level: int = Field(default=0, description="Nesting depth in document")
    chunk_type: str = Field(default="original", description="original|merged|table|overlap")
    source_element_count: int = Field(default=1, description="Elements merged into this chunk")


class SourceInsight(ObjectModel):
    """
    An insight extracted from a source document.

    Insights are derived information like summaries, key points, entities, etc.
    """
    table_name: ClassVar[str] = "source_insight"

    source: Optional[str] = Field(None, description="Reference to the source document")
    insight_type: str = Field(
        default=InsightType.SUMMARY.value,
        description="Type of insight"
    )
    content: str = Field(description="The insight content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceEmbedding(ObjectModel):
    """
    A vector embedding for source content.

    Used for semantic search and retrieval.
    """
    table_name: ClassVar[str] = "source_embedding"

    source: Optional[str] = Field(None, description="Reference to the source document")
    content: str = Field(description="The text content that was embedded")
    order: int = Field(default=0, description="Order in the document")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    chunk: Optional[str] = Field(None, description="Reference to the chunk record")


class Source(ObjectModel):
    """
    A source document in the system.

    Represents any imported content: PDF, URL, video, etc.
    """
    table_name: ClassVar[str] = "source"

    asset: Optional[Asset] = None
    title: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    full_text: Optional[str] = None
    command: Optional[str] = Field(
        None, description="Link to processing job/command"
    )

    @field_validator("topics", mode="before")
    @classmethod
    def ensure_topics_list(cls, v: Any) -> List[str]:
        """Ensure topics is always a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)
