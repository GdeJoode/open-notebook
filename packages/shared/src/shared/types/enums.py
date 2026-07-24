"""
Common enumerations used across the application.
"""

from enum import Enum


class NoteType(str, Enum):
    """Type of note - human created or AI generated."""
    HUMAN = "human"
    AI = "ai"


class ContentProcessingEngine(str, Enum):
    """Document processing engine options."""
    AUTO = "auto"
    DOCLING = "docling"
    SIMPLE = "simple"


class UrlProcessingEngine(str, Enum):
    """URL content extraction engine options."""
    AUTO = "auto"
    FIRECRAWL = "firecrawl"
    JINA = "jina"
    SIMPLE = "simple"


class GpuDevice(str, Enum):
    """GPU device selection options."""
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class DoclingPipeline(str, Enum):
    """Docling processing pipeline options."""
    AUTO = "auto"
    STANDARD = "standard"
    VLM = "vlm"


class VlmModel(str, Enum):
    """Vision-Language Model options."""
    GRANITE_DOCLING = "granite-docling-258m"
    SMOLDOCLING = "smoldocling-256m"


class VlmFramework(str, Enum):
    """VLM framework options."""
    AUTO = "auto"
    TRANSFORMERS = "transformers"
    MLX = "mlx"


class OcrEngine(str, Enum):
    """OCR engine options."""
    AUTO = "auto"
    EASYOCR = "easyocr"
    RAPIDOCR = "rapidocr"
    TESSERACT = "tesseract"


class TableMode(str, Enum):
    """Table structure recognition mode."""
    ACCURATE = "accurate"
    FAST = "fast"


class ChunkingMethod(str, Enum):
    """Document chunking method options."""
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"


class EmbeddingOption(str, Enum):
    """Default embedding behavior options."""
    ASK = "ask"
    ALWAYS = "always"
    NEVER = "never"


class FileOperation(str, Enum):
    """File operation after processing."""
    COPY = "copy"
    MOVE = "move"
    NONE = "none"


class NamingScheme(str, Enum):
    """Output file naming scheme options."""
    TIMESTAMP_PREFIX = "timestamp_prefix"
    DATE_PREFIX = "date_prefix"
    DATETIME_SUFFIX = "datetime_suffix"
    ORIGINAL = "original"


class InsightType(str, Enum):
    """Types of insights that can be extracted from sources."""
    SUMMARY = "summary"
    KEY_POINTS = "key_points"
    ENTITIES = "entities"
    TOPICS = "topics"
    QUESTIONS = "questions"
    CUSTOM = "custom"


class ElementType(str, Enum):
    """Document element types for chunks."""
    PARAGRAPH = "paragraph"
    TITLE = "title"
    HEADING = "heading"
    TABLE = "table"
    LIST = "list"
    LIST_ITEM = "list_item"
    FIGURE = "figure"
    CAPTION = "caption"
    CODE = "code"
    QUOTE = "quote"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    UNKNOWN = "unknown"


class FileStatus(str, Enum):
    """Processing status for tracked files in the ingestion pipeline."""
    PENDING = "pending"  # File uploaded but not yet processed
    INGESTED = "ingested"  # File has been parsed/transcribed
    ENTITIES_EXTRACTED = "entities_extracted"  # Named entities extracted
    SUMMARIZED = "summarized"  # Summary generated
    SAVED_IN_KB = "saved_in_kb"  # Saved to a knowledge base
    SAVED_AS_PROJECT = "saved_as_project"  # Saved as standalone project
    FAILED = "failed"  # Processing failed


class StorageLocation(str, Enum):
    """Type of storage location for files."""
    KNOWLEDGE_BASE = "knowledge_base"
    PROJECT = "project"
    TEMP = "temp"


class JobType(str, Enum):
    """Types of background processing jobs."""
    DOCUMENT_PARSE = "document_parse"  # Docling document processing
    AUDIO_TRANSCRIBE = "audio_transcribe"  # WhisperX transcription
    BATCH_PROCESS = "batch_process"  # Multiple files
    CHUNK_EXTRACT = "chunk_extract"  # Semantic chunking
    EMBEDDING_GENERATE = "embedding_generate"  # Vector embeddings
    INSIGHT_EXTRACT = "insight_extract"  # LLM-based insights
    ENTITY_EXTRACT = "entity_extract"  # Ontology-guided entity extraction
    # Track D Phase D.0: async Obsidian vault export. JSONL (D.2) and
    # NetworkX (D.3) stay sync-only in V1 (Q-D-2), so this is the sole
    # export-side JobType. Track E may add ``RESEARCH`` next to this --
    # whoever lands first wins, the second track rebases.
    EXPORT_OBSIDIAN = "export_obsidian"
    # Track Y.3: background auto-link. Enqueued (best-effort) after a note is
    # successfully embedded, this links the note to its most-related notes by
    # embedding similarity (the ``related_note`` graph edges). A separate job
    # from EMBEDDING_GENERATE so a linking failure stays isolated from the
    # (already-persisted) note + embedding.
    NOTE_AUTO_LINK = "note_auto_link"
    # Track V.5: reference-extraction post-ingest pass. Enqueued (best-effort,
    # config-gated OFF by default) after a source is ingested, this extracts the
    # corpus' references (V.1-V.3) and feeds U.3's whole-corpus ``cites``
    # materialization. A separate job from DOCUMENT_PARSE so a reference/matching
    # failure stays isolated from the (already-persisted) source + chunks.
    REFERENCE_EXTRACT = "reference_extract"


class JobStatus(str, Enum):
    """Status of a background job."""
    QUEUED = "queued"  # Job submitted, waiting for worker
    PROCESSING = "processing"  # Worker is executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Error during processing
    CANCELLED = "cancelled"  # Manually cancelled
    RETRYING = "retrying"  # Failed, attempting retry
    # B.1f: extraction blocked pending notebook-schema review (multi-schema
    # path). The UI surfaces a "review queue" entry; once the user accepts
    # or rejects the proposed extensions the job can be re-queued.
    PAUSED_FOR_REVIEW = "paused_for_review"


class JobPriority(str, Enum):
    """Priority levels for job queue."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class PipelineStep(str, Enum):
    """Known pipeline processing steps for cache identification."""
    DOCLING_PARSE = "docling_parse"
    TRANSCRIPTION = "transcription"
    RAPTOR_SUMMARIES = "raptor_summaries"
    TREEKG_STRUCTURE = "treekg_structure"
    OPENIE_EXTRACTION = "openie_extraction"
    ENTITY_DEDUP = "entity_dedup"
    EMBEDDINGS = "embeddings"
    ENRICHMENT = "enrichment"
    SUMMARIZATION = "summarization"


class SourceType(str, Enum):
    """Type of source being stored."""
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    URL = "url"


class ProcessingStage(str, Enum):
    """Per-source pipeline stage (Track PL.2).

    Records how far the auto-chain (parse -> embed -> EXTRACT -> GRAPH ->
    INSIGHTS) has carried a source, so the pipeline is resumable and the UI can
    surface per-document progress. Persisted on ``source.processing_stage``
    (migration 71). PL.2 writes through ``EXTRACTED`` plus the gate/failure
    states; ``GRAPHED``/``COMPLETE`` are reached in PL.3/PL.4.
    """

    INGESTED = "ingested"
    EMBEDDED = "embedded"
    EXTRACTED = "extracted"
    AWAITING_SCHEMA_REVIEW = "awaiting_schema_review"
    GRAPHED = "graphed"
    COMPLETE = "complete"
    FAILED = "failed"
