"""
Source-related domain models.

Pure Pydantic models for documents, chunks, insights, and embeddings.
Database operations are handled by the surrealdb-service package.
"""

from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from shared.models.base import ObjectModel
from shared.types.enums import ElementType, InsightType

#: Classificatie intern/extern (F2-03, ADR 0001). ``None`` bij schrijven laat de
#: database de standaard kiezen: 'intern' voor een bron, de waarde van de bron
#: voor een fragment, embedding of inzicht (``migrations/89.surrealql``).
Classification = Literal["intern", "extern"]


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

    # Content relevance flag — set at extraction time.
    # False for noise elements (page headers/footers, logos, stamps, etc.)
    # Downstream consumers (regrouper, summarizer, embeddings) filter on this.
    is_content: bool = Field(default=True, description="Whether this chunk is relevant content vs noise")
    classification: Optional[Classification] = Field(
        None, description="intern of extern; standaard die van de bron (F2-03)"
    )
    # Herkomst (F2-04, ADR 0003): verplicht in de database (migratie 90).
    source_version: Optional[str] = Field(None, description="De bronversie van het fragment")
    run: Optional[str] = Field(None, description="De verwerkingsrun die het fragment maakte")
    extractor: Optional[str] = Field(None, description="De extractor, met versie")


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
    classification: Optional[Classification] = Field(
        None, description="intern of extern; standaard die van de bron (F2-03)"
    )


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
    classification: Optional[Classification] = Field(
        None, description="intern of extern; standaard die van de bron (F2-03)"
    )
    run: Optional[str] = Field(None, description="De verwerkingsrun (F2-04)")
    embedding_model: Optional[str] = Field(None, description="Embeddingmodel met versie (F2-04)")
    afgekapt: Optional[bool] = Field(
        None, description="Tekst vóór de embedding afgekapt (ADR 0008, F2-15)"
    )


class Source(ObjectModel):
    """
    A source document in the system.

    Represents any imported content: PDF, URL, video, etc.
    """
    table_name: ClassVar[str] = "source"

    asset: Optional[Asset] = None
    title: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    command: Optional[str] = Field(
        None, description="Link to processing job/command"
    )

    # Source-level aggregate embedding (Track R.0). One vector per source,
    # produced by mean-pooling the source's per-chunk vectors from the
    # ``source_embedding`` table — NOT a fresh LLM call. Enables source<->source
    # cosine kNN in R.1. Declared on the SCHEMAFULL ``source`` table by
    # ``migrations/63.surrealql`` as ``option<array<float>>``; ``None`` means
    # "not yet computed / no chunk vectors" (distinct from an empty list). Legacy
    # rows read back as None. Dimension matches the configured embedding model
    # (mxbai-embed-large, 1024-dim) — never schema-pinned.
    embedding: Optional[List[float]] = Field(
        None,
        description="Source-level aggregate vector (mean-pool of chunk embeddings)",
    )

    # Per-source extraction provenance bag (Phase A.1c, Q-A-3).
    # Keys currently written by the auto-fallback path:
    #   parser_engine_used              "docling" | "mineru" | "whisperx"
    #   extraction_confidence           float in [0, 1] — docling score
    #   extraction_confidence_signals   per-signal breakdown for the UI tooltip
    #   extraction_fallback_triggered   bool — true when auto-mode re-routed
    # The `source` table is SCHEMAFULL (migration 1), so this field is
    # declared in `migrations/43.surrealql` as
    # `FLEXIBLE TYPE option<object>` to accept arbitrary additive keys
    # without further schema bumps. Legacy rows written before #43 read
    # back as NONE/missing and deserialise via the Pydantic default
    # (`Field(default_factory=dict)`) plus the `ensure_metadata_dict`
    # validator below.
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-source metadata bag (extraction confidence, parser engine used, etc.)",
    )

    # Per-source pipeline status (Track PL.2). Records HOW FAR the auto-chain
    # (parse -> embed -> EXTRACT -> GRAPH -> INSIGHTS) has carried this source,
    # so the pipeline is resumable and the UI can show per-document progress.
    # Written at each transition by the job handlers:
    #   ingested                -> after parse (chunks persisted)
    #   embedded                -> after the per-chunk + aggregate embed
    #   extracted               -> after a successful entity extraction
    #   awaiting_schema_review  -> the schema-review gate parked extraction
    #   graphed / complete      -> PL.3/PL.4 (mentions refresh / full chain done)
    #   failed                  -> a hard failure in a stage
    # Backed by ``migrations/71.surrealql`` as a strict ``string DEFAULT
    # "ingested"`` on the SCHEMAFULL ``source`` table; legacy rows are backfilled
    # to ``"ingested"`` (drift-only, S.4-safe). There is no honest "no stage"
    # value, so it is a plain (non-optional) string with a default rather than
    # ``option<>``.
    processing_stage: str = Field(
        default="ingested",
        description="Pipeline stage: ingested|embedded|extracted|awaiting_schema_review|graphed|complete|failed",
    )

    # Classificatie intern/extern (F2-03, ADR 0001); standaard 'intern'. Van
    # 'intern' naar 'extern' kan alleen via de ClassificationService.
    classification: Optional[Classification] = Field(
        None, description="intern of extern; standaard intern (F2-03)"
    )
    current_version: Optional[str] = Field(None, description="De nieuwste bronversie (F2-04)")
    # Binding van de bron (F2-14, migratie 94): de sleutel waarmee een import
    # de bron terugvindt, en de bronkaartregel waarmee ze binnenkwam.
    external_id: Optional[str] = Field(None, description="Extern id van de bron (F2-14)")
    source_map_rule: Optional[str] = Field(None, description="Bronkaartregel van de bron (F2-14)")

    @field_validator("topics", mode="before")
    @classmethod
    def ensure_topics_list(cls, v: Any) -> List[str]:
        """Ensure topics is always a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @field_validator("metadata", mode="before")
    @classmethod
    def ensure_metadata_dict(cls, v: Any) -> Dict[str, Any]:
        """Defensively coerce legacy/null metadata into an empty dict.

        Surreal records written before migration #43 (the migration
        that declared this field) read back as NONE; they deserialise
        with the default. Should the column ever land in the DB with
        an explicit ``NULL`` or non-dict value, Pydantic would refuse
        to parse. Coerce to ``{}`` for resilience.
        """
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {}
