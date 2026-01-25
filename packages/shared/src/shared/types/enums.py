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
