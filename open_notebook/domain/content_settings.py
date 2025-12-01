from typing import ClassVar, List, Literal, Optional

from pydantic import Field

from open_notebook.domain.base import RecordModel


class ContentSettings(RecordModel):
    record_id: ClassVar[str] = "open_notebook:content_settings"

    # Document Engine (simplified - GPU/VLM controlled via content-core settings)
    default_content_processing_engine_doc: Optional[
        Literal["auto", "docling", "simple"]
    ] = Field("docling", description="Default Content Processing Engine for Documents")
    default_content_processing_engine_url: Optional[
        Literal["auto", "firecrawl", "jina", "simple"]
    ] = Field("auto", description="Default Content Processing Engine for URLs")

    # GPU Acceleration Settings (content-core)
    docling_gpu_enabled: Optional[bool] = Field(
        True, description="Enable GPU acceleration for Docling"
    )
    docling_gpu_device: Optional[Literal["auto", "cuda", "cpu"]] = Field(
        "auto", description="GPU device selection"
    )

    # Pipeline Settings (content-core)
    docling_pipeline: Optional[Literal["auto", "standard", "vlm"]] = Field(
        "standard", description="Docling processing pipeline (standard=GPU OCR, vlm=Vision-Language Model)"
    )

    # VLM Settings (content-core) - used when pipeline=vlm
    docling_vlm_model: Optional[
        Literal["granite-docling-258m", "smoldocling-256m"]
    ] = Field("granite-docling-258m", description="VLM model for document processing")
    docling_vlm_framework: Optional[Literal["auto", "transformers", "mlx"]] = Field(
        "auto", description="VLM framework selection"
    )

    # OCR Settings (content-core) - used when pipeline=standard
    docling_ocr_engine: Optional[
        Literal["auto", "easyocr", "rapidocr", "tesseract"]
    ] = Field("easyocr", description="OCR engine for text recognition")
    docling_ocr_languages: Optional[List[str]] = Field(
        ["en"], description="OCR languages"
    )
    docling_ocr_use_gpu: Optional[bool] = Field(
        True, description="Use GPU for OCR acceleration"
    )

    # Table Processing Settings (content-core)
    docling_table_mode: Optional[Literal["accurate", "fast"]] = Field(
        "accurate", description="Table structure recognition mode"
    )

    # Image Export Settings (content-core) - not yet functional
    docling_auto_export_images: Optional[bool] = Field(
        False, description="Automatically export images during extraction"
    )
    docling_image_scale: Optional[float] = Field(
        2.0, description="Image extraction scale (1.0-4.0)"
    )

    # Chunking Settings (content-core)
    docling_chunking_enabled: Optional[bool] = Field(
        False, description="Enable automatic chunking"
    )
    docling_chunking_method: Optional[Literal["hybrid", "hierarchical"]] = Field(
        "hybrid", description="Chunking method"
    )
    docling_chunking_max_tokens: Optional[int] = Field(
        512, description="Maximum tokens per chunk"
    )

    # Existing settings
    default_embedding_option: Optional[Literal["ask", "always", "never"]] = Field(
        "ask", description="Default Embedding Option for Vector Search"
    )
    auto_delete_files: Optional[Literal["yes", "no"]] = Field(
        "yes", description="Auto Delete Uploaded Files"
    )
    youtube_preferred_languages: Optional[List[str]] = Field(
        ["en", "pt", "es", "de", "nl", "en-GB", "fr", "de", "hi", "ja"],
        description="Preferred languages for YouTube transcripts",
    )

    # File Management Settings
    input_directory_path: Optional[str] = Field(
        "./data/input", description="Directory path for organized input files"
    )
    markdown_directory_path: Optional[str] = Field(
        "./data/markdown", description="Directory path for markdown output with assets"
    )
    output_directory_path: Optional[str] = Field(
        "./data/output", description="Directory path for final processed files"
    )
    file_operation: Optional[Literal["copy", "move", "none"]] = Field(
        "copy", description="File operation: copy, move, or none (keep in uploads)"
    )
    output_naming_scheme: Optional[
        Literal["timestamp_prefix", "date_prefix", "datetime_suffix", "original"]
    ] = Field(
        "date_prefix", description="Naming scheme for output files"
    )
