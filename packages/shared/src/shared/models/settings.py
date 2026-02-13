"""
Settings models for application configuration.

Pure Pydantic models for settings. Database persistence handled by surrealdb-service.
"""

from typing import ClassVar, List, Literal, Optional

from pydantic import Field

from shared.models.base import RecordModel


class ContentSettings(RecordModel):
    """
    Content processing settings for the application.

    Singleton record identified by record_id.
    """
    record_id: ClassVar[str] = "open_notebook:content_settings"

    # Document Processing Engine
    default_content_processing_engine_doc: Optional[
        Literal["auto", "docling", "simple"]
    ] = Field("docling", description="Default Content Processing Engine for Documents")
    default_content_processing_engine_url: Optional[
        Literal["auto", "firecrawl", "jina", "simple"]
    ] = Field("auto", description="Default Content Processing Engine for URLs")

    # GPU Acceleration Settings
    docling_gpu_enabled: Optional[bool] = Field(
        True, description="Enable GPU acceleration for Docling"
    )
    docling_gpu_device: Optional[Literal["auto", "cuda", "cpu"]] = Field(
        "auto", description="GPU device selection"
    )

    # Pipeline Settings
    docling_pipeline: Optional[Literal["auto", "standard", "vlm"]] = Field(
        "standard", description="Docling processing pipeline"
    )

    # VLM Settings (used when pipeline=vlm)
    docling_vlm_model: Optional[
        Literal["granite-docling-258m", "smoldocling-256m"]
    ] = Field("granite-docling-258m", description="VLM model for document processing")
    docling_vlm_framework: Optional[Literal["auto", "transformers", "mlx"]] = Field(
        "auto", description="VLM framework selection"
    )

    # OCR Settings (used when pipeline=standard)
    docling_ocr_engine: Optional[
        Literal["auto", "easyocr", "rapidocr", "tesseract"]
    ] = Field("easyocr", description="OCR engine for text recognition")
    docling_ocr_languages: Optional[List[str]] = Field(
        default_factory=lambda: ["en"], description="OCR languages"
    )
    docling_ocr_use_gpu: Optional[bool] = Field(
        True, description="Use GPU for OCR acceleration"
    )

    # Table Processing Settings
    docling_table_mode: Optional[Literal["accurate", "fast"]] = Field(
        "accurate", description="Table structure recognition mode"
    )

    # Image Export Settings
    docling_auto_export_images: Optional[bool] = Field(
        False, description="Automatically export images during extraction"
    )
    docling_image_scale: Optional[float] = Field(
        2.0, description="Image extraction scale (1.0-4.0)"
    )

    # Chunking Settings
    docling_chunking_enabled: Optional[bool] = Field(
        False, description="Enable automatic chunking"
    )
    docling_chunking_method: Optional[Literal["hybrid", "hierarchical"]] = Field(
        "hybrid", description="Chunking method"
    )
    docling_chunking_max_tokens: Optional[int] = Field(
        512, description="Maximum tokens per chunk"
    )

    # Embedding Settings
    default_embedding_option: Optional[Literal["ask", "always", "never"]] = Field(
        "ask", description="Default Embedding Option for Vector Search"
    )

    # File Management Settings
    auto_delete_files: Optional[Literal["yes", "no"]] = Field(
        "yes", description="Auto Delete Uploaded Files"
    )
    youtube_preferred_languages: Optional[List[str]] = Field(
        default_factory=lambda: ["en", "pt", "es", "de", "nl", "en-GB", "fr", "hi", "ja"],
        description="Preferred languages for YouTube transcripts",
    )

    # Directory Paths
    input_directory_path: Optional[str] = Field(
        "./data/input", description="Directory path for organized input files"
    )
    markdown_directory_path: Optional[str] = Field(
        "./data/markdown", description="Directory path for markdown output with assets"
    )
    output_directory_path: Optional[str] = Field(
        "./data/output", description="Directory path for final processed files"
    )

    # File Operations
    file_operation: Optional[Literal["copy", "move", "none"]] = Field(
        "copy", description="File operation: copy, move, or none"
    )
    output_naming_scheme: Optional[
        Literal["timestamp_prefix", "date_prefix", "datetime_suffix", "original"]
    ] = Field("date_prefix", description="Naming scheme for output files")
