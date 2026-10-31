"""
API request/response schemas for app-main.

Pure Pydantic models for API serialization (DTOs).
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Classificatie in een antwoord (F2-05, ADR 0001). Elk response-schema van een
#: record met een classificatie geeft haar terug.
ClassificationField = Optional[Literal["intern", "extern"]]


# Chunk schemas
class ChunkUpdate(BaseModel):
    """Partial update for a chunk."""
    text: Optional[str] = None
    element_type: Optional[str] = None
    positions: Optional[List[List[float]]] = None
    is_content: Optional[bool] = None
    chapter: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChunkMergeRequest(BaseModel):
    """Merge a chunk with an adjacent one (Track I.D-3).

    When ``target_chunk_id`` is omitted the chunk is merged with its
    immediate successor by ``order``; otherwise the named chunk is used and
    must be adjacent.
    """
    target_chunk_id: Optional[str] = Field(
        None, description="Adjacent chunk to merge with; defaults to the next chunk"
    )


class ChunkSplitRequest(BaseModel):
    """Split a chunk at a character offset into its text (Track I.D-3)."""
    cursorOffset: int = Field(
        ..., ge=0, description="Character offset into the chunk text to split at"
    )


class ChunkCreate(BaseModel):
    """Create a new manual chunk."""
    text: str = Field(..., description="Chunk text content")
    element_type: str = Field("paragraph", description="Element type")
    physical_page: int = Field(..., description="Physical page number")
    positions: List[List[float]] = Field(default_factory=list, description="Bounding box positions")
    is_content: bool = Field(True, description="Whether this is content (vs noise)")
    chapter: Optional[str] = Field(None, description="Chapter/section heading")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Notebook schemas
class NotebookCreate(BaseModel):
    name: str = Field(..., description="Name of the notebook")
    description: str = Field(default="", description="Description of the notebook")


class NotebookUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Name of the notebook")
    description: Optional[str] = Field(None, description="Description of the notebook")
    archived: Optional[bool] = Field(
        None, description="Whether the notebook is archived"
    )
    # Track J.3: per-notebook privacy layer. 'cloud' | 'private' set it; the
    # value persists through notebook_service.update. (The 'inherit'/None reset
    # affordance is a J.5 UI concern — PUT here uses exclude_none.)
    privacy_mode: Optional[str] = Field(
        None, description="Per-notebook privacy mode: 'cloud' | 'private'"
    )
    # Track PL.3: per-notebook auto-INSIGHTS toggle. True keeps the LLM-cost
    # INSIGHTS stage auto-chaining off ingest (the default); False opts the
    # notebook out (it still gets the free document graph). Persists through
    # notebook_service.update (PUT uses exclude_none, so omitting it leaves the
    # stored value untouched).
    auto_insights: Optional[bool] = Field(
        None, description="Auto-run INSIGHTS extraction after embed for this notebook"
    )
    librarian_enabled: Optional[bool] = Field(
        None, description="Opt-in: periodically re-run the quality audit (F.5)"
    )


class NotebookResponse(BaseModel):
    id: str
    name: str
    description: str
    archived: bool
    created: str
    updated: str
    source_count: int
    note_count: int
    # Track J.3: None = inherit global default; 'cloud' | 'private' otherwise.
    privacy_mode: Optional[str] = None
    # Track PL.3: per-notebook auto-INSIGHTS toggle (default True).
    auto_insights: bool = True
    # Track F.5: per-notebook librarian opt-in (default False).
    librarian_enabled: bool = False
    classification: ClassificationField = None


# Search schemas
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    type: Literal["text", "vector", "hybrid"] = Field(
        "text", description="Search type"
    )
    limit: int = Field(100, description="Maximum number of results", ge=1, le=1000)
    # F2-05 (ADR 0004): zoeken leest alleen fragmenten. `search_sources` en
    # `search_notes` vervallen; een client die ze stuurt, krijgt geen fout
    # (Pydantic negeert onbekende velden), maar ze hebben geen effect meer.
    history: bool = Field(
        False,
        description=(
            "Ook fragmenten van oudere bronversies en runs; standaard alleen "
            "de huidige versie (ADR 0003)"
        ),
    )
    minimum_score: float = Field(
        0.2, description="Minimum score for vector/hybrid search", ge=0, le=1
    )
    text_weight: float = Field(
        0.5,
        description=(
            "Weight for the BM25 (text) component in hybrid search; the "
            "vector component gets (1 - text_weight). Only used when "
            "type='hybrid'."
        ),
        ge=0,
        le=1,
    )
    rerank: bool = Field(
        False,
        description=(
            "When true (hybrid search only), re-score the top fused results "
            "with the local cross-encoder reranker service and reorder by its "
            "score. Falls back to the heuristic reranker if the service is "
            "unavailable. Default false keeps the path identical to plain "
            "hybrid search."
        ),
    )
    rerank_top_n: int = Field(
        50,
        description=(
            "How many of the top fused hybrid results to send to the reranker "
            "when rerank=true. Only the top-N are re-scored; the remainder keep "
            "their fused order beneath them."
        ),
        ge=1,
        le=200,
    )


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results")
    search_type: str = Field(..., description="Type of search performed")


class AskRequest(BaseModel):
    question: str = Field(..., description="Question to ask the knowledge base")
    # F2-08: de modellen komen uit het register; deze velden tellen niet meer
    # en vervallen met F2-09.
    strategy_model: Optional[str] = Field(None, description="Vervalt; het register beslist")
    answer_model: Optional[str] = Field(None, description="Vervalt; het register beslist")
    final_answer_model: Optional[str] = Field(None, description="Vervalt; het register beslist")


class AskResponse(BaseModel):
    answer: str = Field(..., description="Final answer from the knowledge base")
    question: str = Field(..., description="Original question")
    citations: list[dict] = Field(
        default_factory=list,
        description=(
            "Structured source citations derived from the retrieval context "
            "(Track X.2): [{source, page, chunk_id, section}]. Additive — may "
            "be empty when no provenance is available."
        ),
    )


# Models API schemas
class ModelCreate(BaseModel):
    name: str = Field(..., description="Model name (e.g., gpt-5-mini, claude, gemini)")
    provider: str = Field(
        ..., description="Provider name (e.g., openai, anthropic, gemini)"
    )
    type: str = Field(
        ...,
        description="Model type (language, embedding, text_to_speech, speech_to_text)",
    )


class ModelResponse(BaseModel):
    id: str
    name: str
    provider: str
    type: str
    created: str
    updated: str


class DefaultModelsResponse(BaseModel):
    default_chat_model: Optional[str] = None
    default_transformation_model: Optional[str] = None
    large_context_model: Optional[str] = None
    default_text_to_speech_model: Optional[str] = None
    default_speech_to_text_model: Optional[str] = None
    default_embedding_model: Optional[str] = None
    default_tools_model: Optional[str] = None
    default_extraction_model: Optional[str] = None


class ProviderAvailabilityResponse(BaseModel):
    available: List[str] = Field(..., description="List of available providers")
    unavailable: List[str] = Field(..., description="List of unavailable providers")
    supported_types: Dict[str, List[str]] = Field(
        ..., description="Provider to supported model types mapping"
    )


class ModelStatusIssue(BaseModel):
    type: Literal["ollama_unreachable", "model_unavailable", "provider_unavailable", "no_defaults"] = Field(
        ..., description="Issue type"
    )
    message: str = Field(..., description="Human-readable issue description")
    model_name: Optional[str] = Field(None, description="Affected model name")
    provider: Optional[str] = Field(None, description="Affected provider")
    default_slot: Optional[str] = Field(None, description="Affected default slot name")


class ModelStatusResponse(BaseModel):
    valid: bool = Field(..., description="Whether model configuration is valid")
    issues: List[ModelStatusIssue] = Field(
        default_factory=list, description="List of configuration issues"
    )


# Transformations API schemas
class TransformationCreate(BaseModel):
    name: str = Field(..., description="Transformation name")
    title: str = Field(..., description="Display title for the transformation")
    description: str = Field(
        ..., description="Description of what this transformation does"
    )
    prompt: str = Field(..., description="The transformation prompt")
    apply_default: bool = Field(
        False, description="Whether to apply this transformation by default"
    )


class TransformationUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Transformation name")
    title: Optional[str] = Field(
        None, description="Display title for the transformation"
    )
    description: Optional[str] = Field(
        None, description="Description of what this transformation does"
    )
    prompt: Optional[str] = Field(None, description="The transformation prompt")
    apply_default: Optional[bool] = Field(
        None, description="Whether to apply this transformation by default"
    )


class TransformationResponse(BaseModel):
    id: str
    name: str
    title: str
    description: str
    prompt: str
    apply_default: bool
    created: str
    updated: str


class TransformationExecuteRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    transformation_id: str = Field(
        ..., description="ID of the transformation to execute"
    )
    input_text: str = Field(..., description="Text to transform")
    model_id: str = Field(..., description="Model ID to use for the transformation")


class TransformationExecuteResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    output: str = Field(..., description="Transformed text")
    transformation_id: str = Field(..., description="ID of the transformation used")
    model_id: str = Field(..., description="Model ID used")


# Default Prompt schemas
class DefaultPromptResponse(BaseModel):
    transformation_instructions: str = Field(
        ..., description="Default transformation instructions"
    )


class DefaultPromptUpdate(BaseModel):
    transformation_instructions: str = Field(
        ..., description="Default transformation instructions"
    )


# Notes API schemas
class NoteCreate(BaseModel):
    title: Optional[str] = Field(None, description="Note title")
    content: str = Field(..., description="Note content")
    note_type: Optional[str] = Field("human", description="Type of note (human, ai)")
    notebook_id: Optional[str] = Field(
        None, description="Notebook ID to add the note to"
    )


class NoteUpdate(BaseModel):
    title: Optional[str] = Field(None, description="Note title")
    content: Optional[str] = Field(None, description="Note content")
    note_type: Optional[str] = Field(None, description="Type of note (human, ai)")


class NoteResponse(BaseModel):
    id: str
    title: Optional[str]
    content: Optional[str]
    note_type: Optional[str]
    created: str
    updated: str
    classification: ClassificationField = None
    is_derived: bool = True


class NoteAutoLinkResponse(BaseModel):
    """Summary of an auto-link run (Track Y.2 + NS.2).

    One call links a note to both its related NOTES (``related_note`` edges, the
    bare counters) and the SOURCES it's about (``note_about`` edges, the
    ``source_*`` counters).
    """

    note_id: str = Field(..., description="The note that was linked from")
    status: str = Field(
        ...,
        description=(
            "linked | needs_embedding | not_found — needs_embedding means the "
            "note had no content to embed; not_found means no such note. "
            "(The MCP auto_link_note tool can additionally return invalid_id; "
            "the HTTP route validates the id at the boundary and returns 422 "
            "first, so this response never carries invalid_id — Y.2 follow-up.)"
        ),
    )
    created: int = Field(0, description="related_note edges (re)written")
    skipped_existing: int = Field(
        0, description="Note candidates whose relate was refused/failed"
    )
    below_threshold: int = Field(
        0, description="Ranked note candidates dropped below min_similarity"
    )
    candidates_considered: int = Field(
        0, description="Ranked note candidates examined (top-k by cosine)"
    )
    source_links_created: int = Field(
        0, description="note_about edges (re)written (NS.2)"
    )
    source_skipped_existing: int = Field(
        0, description="Source candidates whose relate was refused/failed"
    )
    source_below_threshold: int = Field(
        0, description="Ranked source candidates dropped below min_similarity"
    )
    source_candidates_considered: int = Field(
        0, description="Ranked source candidates examined (top-k by cosine)"
    )
    embedded: bool = Field(
        False, description="Whether this run embedded the note first"
    )
    min_similarity: float = Field(
        ..., description="Effective min cosine applied"
    )
    k: int = Field(..., description="Effective top-k applied")
    linked_note_ids: list[str] = Field(
        default_factory=list, description="Note ids linked this run"
    )
    linked_source_ids: list[str] = Field(
        default_factory=list, description="Source ids linked this run (NS.2)"
    )


class SourceJudgeContradictionsResponse(BaseModel):
    """Summary of an on-demand contradiction-judge run over a source (Track Z.3).

    Mirrors :class:`app_main.services.contradiction_judge_service.JudgeRunResult`.
    Precision-first: ``edges_written`` counts ONLY confident
    contradicts/reinforces verdicts; ``neutral`` and ``below_threshold`` are
    judged but written nowhere.
    """

    source_id: str = Field(..., description="The source that was judged")
    status: str = Field(
        ...,
        description="judged | not_found — not_found means no such source.",
    )
    judged: int = Field(0, description="Related pairs judged this run")
    contradicts: int = Field(0, description="Pairs judged contradicts")
    reinforces: int = Field(0, description="Pairs judged reinforces")
    neutral: int = Field(0, description="Pairs judged neutral (no edge)")
    below_threshold: int = Field(
        0,
        description=(
            "Contradicts/reinforces verdicts below min_confidence (no edge)"
        ),
    )
    edges_written: int = Field(
        0, description="source_verdict edges (re)written (confident only)"
    )
    candidates_considered: int = Field(
        0, description="Related pairs considered (top-k from Track R substrate)"
    )
    min_confidence: float = Field(
        ..., description="Effective precision gate applied"
    )
    k: int = Field(..., description="Effective top-k related fan-out applied")
    verdict_pairs: list[dict] = Field(
        default_factory=list,
        description="Per-pair {from, to, verdict, confidence, edge_written}",
    )


# Embedding API schemas
class EmbedRequest(BaseModel):
    item_id: str = Field(..., description="ID of the item to embed")
    item_type: str = Field(..., description="Type of item (source, note)")
    async_processing: bool = Field(
        False, description="Process asynchronously in background"
    )


class EmbedResponse(BaseModel):
    success: bool = Field(..., description="Whether embedding was successful")
    message: str = Field(..., description="Result message")
    item_id: str = Field(..., description="ID of the item that was embedded")
    item_type: str = Field(..., description="Type of item that was embedded")
    command_id: Optional[str] = Field(
        None, description="Command ID for async processing"
    )


# Rebuild schemas
class RebuildRequest(BaseModel):
    mode: Literal["existing", "all"] = Field(
        ...,
        description="Rebuild mode: 'existing' only re-embeds items with embeddings, 'all' embeds everything",
    )
    include_sources: bool = Field(True, description="Include sources in rebuild")
    include_notes: bool = Field(True, description="Include notes in rebuild")
    include_insights: bool = Field(True, description="Include insights in rebuild")


class RebuildResponse(BaseModel):
    command_id: str = Field(..., description="Command ID to track progress")
    total_items: int = Field(..., description="Estimated number of items to process")
    message: str = Field(..., description="Status message")


class RebuildProgress(BaseModel):
    processed: int = Field(..., description="Number of items processed")
    total: int = Field(..., description="Total items to process")
    percentage: float = Field(..., description="Progress percentage")


class RebuildStats(BaseModel):
    sources: int = Field(0, description="Sources processed")
    notes: int = Field(0, description="Notes processed")
    insights: int = Field(0, description="Insights processed")
    failed: int = Field(0, description="Failed items")


class RebuildStatusResponse(BaseModel):
    command_id: str = Field(..., description="Command ID")
    status: str = Field(..., description="Status: queued, running, completed, failed")
    progress: Optional[RebuildProgress] = None
    stats: Optional[RebuildStats] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


# Settings API schemas
class SettingsResponse(BaseModel):
    # Constrained to the same literal set as ContentSettings.parser_engine so
    # invalid values are rejected at the API boundary instead of being
    # MERGE-upserted into SurrealDB and only failing on response hydration.
    parser_engine: Optional[Literal["simple", "docling", "mineru", "markitdown", "auto"]] = None
    mineru_supported_extensions: Optional[List[str]] = None
    # Phase A.1c: auto-mode confidence threshold ([0, 1]).
    docling_min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    default_content_processing_engine_url: Optional[str] = None
    default_embedding_option: Optional[str] = None
    auto_delete_files: Optional[str] = None
    youtube_preferred_languages: Optional[List[str]] = None
    docling_gpu_enabled: Optional[bool] = None
    docling_gpu_device: Optional[str] = None
    docling_pipeline: Optional[str] = None
    docling_vlm_model: Optional[str] = None
    docling_vlm_framework: Optional[str] = None
    docling_ocr_engine: Optional[str] = None
    docling_ocr_languages: Optional[List[str]] = None
    docling_ocr_use_gpu: Optional[bool] = None
    docling_table_mode: Optional[str] = None
    docling_auto_export_images: Optional[bool] = None
    docling_image_scale: Optional[float] = None
    docling_generate_page_images: Optional[bool] = None
    docling_do_code_enrichment: Optional[bool] = None
    docling_do_formula_enrichment: Optional[bool] = None
    docling_do_picture_classification: Optional[bool] = None
    docling_chunking_enabled: Optional[bool] = None
    docling_chunking_method: Optional[str] = None
    docling_chunking_max_tokens: Optional[int] = None
    classification_max_chars: Optional[int] = None
    input_directory_path: Optional[str] = None
    markdown_directory_path: Optional[str] = None
    output_directory_path: Optional[str] = None
    file_operation: Optional[str] = None
    output_naming_scheme: Optional[str] = None
    # Track J.3: global default privacy mode for LLM routing.
    default_privacy_mode: Optional[Literal["cloud", "private"]] = None


class SettingsUpdate(BaseModel):
    # Constrained to the same literal set as ContentSettings.parser_engine so
    # invalid values are rejected at the API boundary instead of being
    # MERGE-upserted into SurrealDB and only failing on response hydration.
    parser_engine: Optional[Literal["simple", "docling", "mineru", "markitdown", "auto"]] = None
    mineru_supported_extensions: Optional[List[str]] = None
    # Phase A.1c: auto-mode confidence threshold ([0, 1]).
    docling_min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    default_content_processing_engine_url: Optional[str] = None
    default_embedding_option: Optional[str] = None
    auto_delete_files: Optional[str] = None
    youtube_preferred_languages: Optional[List[str]] = None
    docling_gpu_enabled: Optional[bool] = None
    docling_gpu_device: Optional[str] = None
    docling_pipeline: Optional[str] = None
    docling_vlm_model: Optional[str] = None
    docling_vlm_framework: Optional[str] = None
    docling_ocr_engine: Optional[str] = None
    docling_ocr_languages: Optional[List[str]] = None
    docling_ocr_use_gpu: Optional[bool] = None
    docling_table_mode: Optional[str] = None
    docling_auto_export_images: Optional[bool] = None
    docling_image_scale: Optional[float] = None
    docling_generate_page_images: Optional[bool] = None
    docling_do_code_enrichment: Optional[bool] = None
    docling_do_formula_enrichment: Optional[bool] = None
    docling_do_picture_classification: Optional[bool] = None
    docling_chunking_enabled: Optional[bool] = None
    docling_chunking_method: Optional[str] = None
    docling_chunking_max_tokens: Optional[int] = None
    classification_max_chars: Optional[int] = None
    input_directory_path: Optional[str] = None
    markdown_directory_path: Optional[str] = None
    output_directory_path: Optional[str] = None
    file_operation: Optional[str] = None
    output_naming_scheme: Optional[str] = None
    # Track J.3: global default privacy mode for LLM routing ('cloud' | 'private').
    default_privacy_mode: Optional[Literal["cloud", "private"]] = None


# Sources API schemas
class AssetModel(BaseModel):
    file_path: Optional[str] = None
    url: Optional[str] = None


class SourceCreate(BaseModel):
    notebook_id: Optional[str] = Field(
        None, description="Notebook ID to add the source to (deprecated, use notebooks)"
    )
    notebooks: Optional[List[str]] = Field(
        None, description="List of notebook IDs to add the source to"
    )
    type: str = Field(..., description="Source type: link, upload, or text")
    url: Optional[str] = Field(None, description="URL for link type")
    file_path: Optional[str] = Field(None, description="File path for upload type")
    content: Optional[str] = Field(None, description="Text content for text type")
    title: Optional[str] = Field(None, description="Source title")
    transformations: Optional[List[str]] = Field(
        default_factory=list, description="Transformation IDs to apply"
    )
    embed: bool = Field(False, description="Whether to embed content for vector search")
    delete_source: bool = Field(
        False, description="Whether to delete uploaded file after processing"
    )
    async_processing: bool = Field(
        False, description="Whether to process source asynchronously"
    )
    processing_overrides: Optional[Dict[str, Any]] = Field(
        None, description="Per-submission overrides for processing settings"
    )
    private: bool = Field(
        False,
        description=(
            "Force this source's LLM stages local (Track J.3 sticky privacy "
            "override). When True the document is never routed to a cloud provider."
        ),
    )

    @model_validator(mode="after")
    def validate_notebook_fields(self):
        if self.notebook_id is not None and self.notebooks is not None:
            raise ValueError(
                "Cannot specify both 'notebook_id' and 'notebooks'. Use 'notebooks' for multi-notebook support."
            )
        if self.notebook_id is not None:
            self.notebooks = [self.notebook_id]
        if self.notebooks is None:
            self.notebooks = []
        return self


class SourceUpdate(BaseModel):
    title: Optional[str] = Field(None, description="Source title")
    topics: Optional[List[str]] = Field(None, description="Source topics")


class ClassificationRequest(BaseModel):
    """Herclassificatie van een bron (F2-03, ADR 0001): alleen ``beheerder``."""

    classification: Literal["intern", "extern"] = Field(
        ..., description="De nieuwe classificatie"
    )
    reason: str = Field(..., description="Verplichte reden, komt in het logboek")

    @field_validator("reason")
    @classmethod
    def _reden_niet_leeg(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Een reden is verplicht")
        return value.strip()


class ClassificationResponse(BaseModel):
    source_id: str
    from_classification: Literal["intern", "extern"]
    to_classification: Literal["intern", "extern"]
    changed: bool
    log_id: Optional[str] = None


class PipelineVersionModel(BaseModel):
    """Git-commit en SHA-256 van de flow-JSON (F2-04, ADR 0022 E)."""

    git_commit: Optional[str] = None
    flow_hash: Optional[str] = None


class RunCreateRequest(BaseModel):
    """``POST /api/v1/runs`` (F2-04, plan §5.4)."""

    trigger: Literal["schema", "mapcontrole", "feed", "handmatig", "herverwerking"]
    pipeline_id: str = Field(..., min_length=1)
    pipeline_version: PipelineVersionModel = Field(default_factory=PipelineVersionModel)
    source_map_rule_id: Optional[str] = None
    label_schema: Optional[Dict[str, Any]] = None
    #: Binding aan een batch (bulkupload) of wachtrij-item (F2-14).
    intake_batch_id: Optional[str] = None
    intake_item_id: Optional[str] = None


class RunLogEntryModel(BaseModel):
    at: Optional[datetime] = None
    step: str
    status: str
    message: str = ""


class RunPatchRequest(BaseModel):
    """``PATCH /api/v1/runs/{id}`` (F2-04, plan §5.4)."""

    status: Optional[Literal["bezig", "geslaagd", "mislukt", "niets_nieuws"]] = None
    ended_at: Optional[datetime] = None
    log: List[RunLogEntryModel] = Field(default_factory=list)
    models: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class RunCreateResponse(BaseModel):
    run_id: str
    status: str


class RunResponse(BaseModel):
    """Een verwerkingsrun (F2-04, ADR 0003)."""

    id: str
    status: str
    trigger: str
    pipeline_id: str
    pipeline_version: Optional[PipelineVersionModel] = None
    input_classification: str
    flow_vastgelegd: bool = False
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    models: List[Dict[str, Any]] = Field(default_factory=list)
    log: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    attempts: int = 0
    source_map_rule_id: Optional[str] = None
    source_map_rule_rev: Optional[int] = None
    actor: Optional[str] = None
    intake_batch: Optional[str] = None
    intake_item: Optional[str] = None


class SourceResponse(BaseModel):
    id: str
    title: Optional[str]
    topics: Optional[List[str]]
    asset: Optional[AssetModel]
    full_text: Optional[str]
    embedded: bool
    embedded_chunks: int
    insights_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    file_available: Optional[bool] = None
    created: str
    updated: str
    command_id: Optional[str] = None
    status: Optional[str] = None
    processing_info: Optional[Dict] = None
    notebooks: Optional[List[str]] = None
    # Track J.3/J.5: per-document privacy override. True forces the document's
    # LLM stages local (sticky). Surfaced as a read-only "Private" badge.
    private: bool = False
    # Track PL.4: the per-source pipeline stage (ingested | embedded | extracted
    # | awaiting_schema_review | graphed | complete | failed) so the UI can show
    # per-document progress. Defaults "ingested" (migration 71 backfill).
    processing_stage: str = "ingested"
    classification: ClassificationField = None


class SourceListResponse(BaseModel):
    id: str
    title: Optional[str]
    topics: Optional[List[str]]
    asset: Optional[AssetModel]
    embedded: bool
    embedded_chunks: int
    insights_count: int
    created: str
    updated: str
    file_available: Optional[bool] = None
    command_id: Optional[str] = None
    status: Optional[str] = None
    processing_info: Optional[Dict[str, Any]] = None
    # Track UX.1: surface the per-source pipeline stage on the list endpoint too
    # (mirrors SourceResponse.processing_stage) so list-derived cards read the
    # spine directly without an N-per-card fan-out. Defaults "ingested".
    processing_stage: str = "ingested"
    # WB.3: stopped short of complete (services/stopped_sources.py), so the
    # Sources board can mark it; the same predicate as ``?stalled=true``.
    stalled: bool = False
    # WB.3: failed, awaiting schema review or stopped (NEEDS_YOU_STAGES plus
    # ``stalled``): the rows ``?sort_by=attention`` lists first.
    needs_you: bool = False
    classification: ClassificationField = None


class SourceCountsResponse(BaseModel):
    """``GET /sources/counts``: the Sources board's filter chips (WB.3).

    Each count is the length of the list the chip's filter returns:
    ``stages[s]`` for ``?stage=s``, ``stalled`` for ``?stalled=true``, both
    within ``notebook_id`` when given. ``needs_you`` is what
    ``?sort_by=attention`` puts first: failed, awaiting schema review and
    stopped sources.
    """

    total: int
    stages: Dict[str, int]
    stalled: int
    needs_you: int


class RelatedSourceResponse(BaseModel):
    """One related source returned by ``GET /sources/{id}/related`` (R.1).

    ``score`` is the cosine similarity (0..1 for the non-negative mxbai vectors)
    between the queried source's aggregate embedding and this source's.
    """

    id: str
    title: Optional[str] = None
    score: float


class KGSharedEntity(BaseModel):
    """One shared entity driving a KG-related-source score (Track R.2).

    The per-pair "why matched" lineage: ``weight`` is this entity's contribution
    (``type_salience × rarity``), ``document_frequency`` how many sources it
    appears in, ``via_relation`` whether it was reached by a 1-hop relation path
    (vs directly shared).
    """

    entity_id: str
    name: str
    type: str
    document_frequency: int
    weight: float
    via_relation: bool = False


class RelatedSourceKGResponse(BaseModel):
    """One related source ranked by KG proximity (``/sources/{id}/related-kg``).

    Parallel to :class:`RelatedSourceResponse` (R.1 dense) but the ``score`` is
    the KG-proximity score (sum of shared-entity weights), and ``explanation``
    carries the shared entities that drove it.
    """

    id: str
    title: Optional[str] = None
    score: float
    explanation: List[KGSharedEntity] = Field(default_factory=list)


class SignalProvenanceResponse(BaseModel):
    """One signal's contribution to a fused hybrid result (Track R.3).

    The per-signal "why": whether the source was present in this signal, its
    raw score (cosine for dense; KG weighted-sum for KG), its 1-based rank
    within the signal, and the signal's exact additive RRF contribution to the
    fused score. ``score``/``rank`` are ``None`` when the source was absent from
    the signal (it still ranks on its other signal — never dropped).
    """

    present: bool
    score: Optional[float] = None
    rank: Optional[int] = None
    contribution: float


class RelatedSourceHybridResponse(BaseModel):
    """One fused related source (``GET /sources/{id}/related-hybrid``, R.3).

    Combines the dense (R.1) and KG (R.2) signals via Reciprocal Rank Fusion,
    KG-prominent by default. ``fused_score`` is the weighted RRF total;
    ``dense``/``kg`` carry per-signal provenance (score, rank, contribution);
    ``kg_entities`` is the R.2 driving-entities lineage (the "why matched").
    """

    id: str
    title: Optional[str] = None
    fused_score: float
    dense: SignalProvenanceResponse
    kg: SignalProvenanceResponse
    kg_entities: List[KGSharedEntity] = Field(default_factory=list)


# Context API schemas
class ContextConfig(BaseModel):
    sources: Dict[str, str] = Field(
        default_factory=dict, description="Source inclusion config {source_id: level}"
    )
    notes: Dict[str, str] = Field(
        default_factory=dict, description="Note inclusion config {note_id: level}"
    )


class ContextRequest(BaseModel):
    notebook_id: str = Field(..., description="Notebook ID to get context for")
    context_config: Optional[ContextConfig] = Field(
        None, description="Context configuration"
    )


class ContextResponse(BaseModel):
    notebook_id: str
    sources: List[Dict[str, Any]] = Field(..., description="Source context data")
    notes: List[Dict[str, Any]] = Field(..., description="Note context data")
    total_tokens: Optional[int] = Field(None, description="Estimated token count")


# Insights API schemas
class SourceInsightResponse(BaseModel):
    id: str
    source_id: str
    insight_type: str
    content: str
    created: str
    updated: str
    classification: ClassificationField = None
    is_derived: bool = True


class SaveAsNoteRequest(BaseModel):
    notebook_id: Optional[str] = Field(None, description="Notebook ID to add note to")


class CreateSourceInsightRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    transformation_id: str = Field(..., description="ID of transformation to apply")
    model_id: Optional[str] = Field(
        None, description="Model ID (uses default if not provided)"
    )


# Source status response
class SourceStatusResponse(BaseModel):
    status: Optional[str] = Field(None, description="Processing status")
    message: str = Field(..., description="Descriptive message about the status")
    processing_info: Optional[Dict[str, Any]] = Field(
        None, description="Detailed processing information"
    )
    command_id: Optional[str] = Field(None, description="Command ID if available")


# Error response
class ErrorResponse(BaseModel):
    error: str
    message: str


# ---------------------------------------------------------------------------
# Bronkaart, innamebatch en innamewachtrij (F2-13, ADR 0020, plan §5.2)
# ---------------------------------------------------------------------------

_Classificatie = Literal["intern", "extern"]
_RouteType = Literal["api", "mcp", "feed", "website", "bulkupload", "watchfolder", "upload"]
_SourceType = Literal["pdf", "map", "website", "api", "video", "podcast", "mcp"]
_FetchContent = Literal["metadata", "transcript", "volledige_media", "volledig_document"]
#: Een flow-id in Git (``route``): letters, cijfers, ``_``, ``-`` en ``.``.
_ROUTE = r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$"
#: SHA-256 van de ruwe inhoud, zoals in het importverzoek (plan §5.1).
_SHA256 = r"^sha256:[0-9a-f]{64}$"


class ScheduleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cron: str = Field(..., min_length=1)
    timezone: str = "Europe/Amsterdam"
    catch_up: Literal["coalesce", "skip"] = "coalesce"


class LabelSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_name: str = Field(..., alias="schema", min_length=1)
    version: str = Field(..., min_length=1)

    def opslag(self) -> Dict[str, str]:
        return {"schema": self.schema_name, "version": self.version}


class SourceMapRuleCreate(BaseModel):
    """``POST /api/v1/source-map-rules`` (beheerder). Zonder ``classification``
    geldt de standaard van de route (ADR 0019); ``confirm`` legt ``extern``
    meteen vast met de persoon van de sleutel."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    route_type: _RouteType
    source_type: _SourceType
    location: str = Field(..., min_length=1)
    schedule: Optional[ScheduleModel] = None
    classification: Optional[_Classificatie] = None
    document_type: Optional[List[str]] = None
    route: str = Field(..., pattern=_ROUTE)
    selection: Optional[List[str]] = None
    label_schema: Optional[LabelSchemaModel] = None
    owner: Optional[str] = None
    fetch_content: Optional[_FetchContent] = None
    active: Optional[bool] = None
    confirm: bool = False
    reason: Optional[str] = None


class SourceMapRulePatch(BaseModel):
    """``PATCH /api/v1/source-map-rules/{id}`` (beheerder), met ``rev``.
    Alleen meegestuurde velden veranderen; ``null`` wist een optioneel veld."""

    model_config = ConfigDict(extra="forbid")

    rev: int
    name: Optional[str] = Field(None, min_length=1)
    route_type: Optional[_RouteType] = None
    source_type: Optional[_SourceType] = None
    location: Optional[str] = Field(None, min_length=1)
    schedule: Optional[ScheduleModel] = None
    classification: Optional[_Classificatie] = None
    document_type: Optional[List[str]] = None
    route: Optional[str] = Field(None, pattern=_ROUTE)
    selection: Optional[List[str]] = None
    label_schema: Optional[LabelSchemaModel] = None
    owner: Optional[str] = Field(None, min_length=1)
    fetch_content: Optional[_FetchContent] = None
    active: Optional[bool] = None
    confirm: bool = False
    reason: Optional[str] = None


class ConfirmRequest(BaseModel):
    """Bevestiging van ``extern`` door de beheerder, met ``rev``."""

    model_config = ConfigDict(extra="forbid")

    rev: int
    reason: Optional[str] = None


class SourceMapRuleResponse(BaseModel):
    id: str
    name: str
    route_type: str
    source_type: str
    location: str
    schedule: Optional[Dict[str, Any]] = None
    classification: str
    classification_confirmed_by: Optional[str] = None
    classification_confirmed_at: Optional[datetime] = None
    effective_classification: str
    pipeline_start: str
    document_type: Optional[List[str]] = None
    route: str
    selection: Optional[List[str]] = None
    label_schema: Optional[Dict[str, Any]] = None
    owner: str
    fetch_content: str
    active: bool
    rev: int
    modified_by: Optional[str] = None
    modified_at: Optional[datetime] = None


class IntakeBatchCreate(BaseModel):
    """``POST /api/v1/intake/batches`` (beheerder); zonder keuze ``intern``."""

    model_config = ConfigDict(extra="forbid")

    route: str = Field(..., pattern=_ROUTE)
    files: List[str] = Field(default_factory=list)
    classification: Optional[_Classificatie] = None
    source_map_rule_id: Optional[str] = None
    confirm: bool = False
    reason: Optional[str] = None


class IntakeBatchPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rev: int
    route: Optional[str] = Field(None, pattern=_ROUTE)
    files: Optional[List[str]] = None
    classification: Optional[_Classificatie] = None
    confirm: bool = False
    reason: Optional[str] = None


class IntakeBatchResponse(BaseModel):
    id: str
    classification: str
    classification_confirmed_by: Optional[str] = None
    classification_confirmed_at: Optional[datetime] = None
    effective_classification: str
    route: str
    source_map_rule: Optional[str] = None
    files: List[str] = Field(default_factory=list)
    file_hashes: List[Dict[str, Any]] = Field(default_factory=list)
    created_by: str
    rev: int
    modified_by: Optional[str] = None
    modified_at: Optional[datetime] = None


class GatekeeperModel(BaseModel):
    """Het voorstel van de poortwachter: kenmerken, geen tekst uit het document."""

    model_config = ConfigDict(extra="forbid")

    decision: _Classificatie
    internal_markers: Optional[List[str]] = None
    method: Optional[str] = None
    text_source: Optional[str] = None
    probability: Optional[float] = None
    model: Optional[str] = None


class IntakeItemCreate(BaseModel):
    """``POST /api/v1/intake/items`` (pijplijn)."""

    model_config = ConfigDict(extra="forbid")

    origin: Literal["watchfolder", "upload", "zonder_regel"]
    path: str = Field(..., min_length=1)
    content_hash: str = Field(..., pattern=_SHA256)
    gatekeeper: Optional[GatekeeperModel] = None


class IntakeItemPatch(BaseModel):
    """``PATCH /api/v1/intake/items/{id}`` (pijplijn): alleen ``verwerkt``."""

    model_config = ConfigDict(extra="forbid")

    rev: int
    status: Literal["verwerkt"]
    run_id: Optional[str] = None


class IntakeDecisionRequest(BaseModel):
    """``POST /api/v1/intake/items/{id}/decision`` (beheerder)."""

    model_config = ConfigDict(extra="forbid")

    rev: int
    route: str = Field(..., pattern=_ROUTE)
    classification: _Classificatie
    reason: Optional[str] = None


class IntakeItemResponse(BaseModel):
    id: str
    origin: str
    path: str
    content_hash: str
    proposed_classification: Optional[str] = None
    gatekeeper: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    effective_classification: str
    status: str
    run: Optional[str] = None
    rev: int
    modified_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Importeindpunt (F2-14, ADR 0010, plan §5.1)
# ---------------------------------------------------------------------------


class ImportGatekeeper(BaseModel):
    """De beslissing van de poortwachter: kenmerken, geen inhoud."""

    decision: Optional[Literal["intern", "extern"]] = None
    internal_markers: List[str] = Field(default_factory=list)
    method: Optional[str] = None
    text_source: Optional[str] = None
    probability: Optional[float] = None
    model: Optional[str] = None


class ImportClassification(BaseModel):
    """Wat de aanroeper opgeeft. Die kan alleen strenger maken: de
    classificatie komt uit de regel, de batch of het besluit over het item."""

    value: Optional[Literal["intern", "extern"]] = None
    basis: Optional[str] = None
    gatekeeper: Optional[ImportGatekeeper] = None


class ImportSource(BaseModel):
    external_id: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)
    title: Optional[str] = None


class ImportSourceVersion(BaseModel):
    #: SHA-256 van de ruwe inhoud (het origineel in het bronarchief).
    content_hash: str = Field(..., pattern=r"^sha256:[0-9a-f]{64}$")
    fetched_at: Optional[datetime] = None
    #: Het origineel, relatief in het bronarchief.
    original: str = Field(..., min_length=1)
    feed_item_id: Optional[str] = None


class ImportExtraction(BaseModel):
    extractor: str = Field(..., min_length=1)
    version: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class ImportStructure(BaseModel):
    format: str = "docling_document"
    content: Dict[str, Any] = Field(default_factory=dict)


class ImportPosition(BaseModel):
    page: int = Field(..., ge=0)
    bbox: List[float] = Field(..., min_length=4, max_length=4)


class ImportChunk(BaseModel):
    seq: int = Field(..., ge=0)
    text: str
    section_path: List[str] = Field(default_factory=list)
    page: int = Field(0, ge=0)
    printed_page: Optional[str] = None
    element_type: str = "paragraph"
    positions: List[ImportPosition] = Field(default_factory=list)
    doc_ref: Optional[str] = None
    labels: List[Dict[str, Any]] = Field(default_factory=list)


class ImportRequest(BaseModel):
    """``POST /api/v1/import`` (plan §5.1)."""

    run_id: str = Field(..., min_length=1)
    source_map_rule_id: Optional[str] = None
    batch_id: Optional[str] = None
    #: Het wachtrij-item (watchfolder, losse upload); F2-14.
    intake_item_id: Optional[str] = None
    source: ImportSource
    source_version: ImportSourceVersion
    classification: ImportClassification = Field(default_factory=ImportClassification)
    extraction: ImportExtraction
    structure: Optional[ImportStructure] = None
    chunks: List[ImportChunk] = Field(default_factory=list)

    @field_validator("chunks")
    @classmethod
    def unieke_volgorde(cls, chunks: List[ImportChunk]) -> List[ImportChunk]:
        if len({c.seq for c in chunks}) != len(chunks):
            raise ValueError("seq moet uniek zijn")
        return chunks


class ImportResponse(BaseModel):
    status: Literal["nieuw", "ongewijzigd"]
    source_id: str
    source_version_id: str
    version_no: int
    run_id: str
    classification: Literal["intern", "extern"]
    chunk_ids: List[str] = Field(default_factory=list)
