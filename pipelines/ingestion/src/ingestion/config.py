"""
Ingestion pipeline configuration.

Defines configuration dataclasses for:
- DoclingConfig: Full Docling options for document parsing
- WhisperXConfig: Audio/video transcription options
- OutputConfig: Directory structure for extracted content
- IngestionConfig: Main configuration combining all options
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class AcceleratorDevice(str, Enum):
    """Device for GPU acceleration."""
    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class OcrEngine(str, Enum):
    """OCR engine selection."""
    EASYOCR = "easyocr"
    RAPIDOCR = "rapidocr"
    TESSERACT = "tesseract"
    NONE = "none"


class TableMode(str, Enum):
    """Table extraction mode."""
    FAST = "fast"
    ACCURATE = "accurate"


class ImageRefMode(str, Enum):
    """How to reference images in output."""
    PLACEHOLDER = "placeholder"  # <!-- image --> in markdown
    EMBEDDED = "embedded"  # Base64 inline
    REFERENCED = "referenced"  # Separate files with links


class TableFormat(str, Enum):
    """Table output format."""
    MARKDOWN = "markdown"
    CSV = "csv"
    BOTH = "both"


class WhisperModel(str, Enum):
    """WhisperX model size."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large-v3"


class RegroupingStrategy(str, Enum):
    """Chunk regrouping strategy."""
    STRUCTURAL = "structural"      # Merge by section headings
    SLIDING_WINDOW = "sliding_window"  # Fixed-size with overlap
    NONE = "none"                  # Keep 1:1 (legacy)


class DestinationType(str, Enum):
    """Where to store the processed content."""
    PROJECT = "project"
    KNOWLEDGE_BASE = "knowledge_base"
    ASK = "ask"  # Ask user at end (CLI mode)


@dataclass
class DoclingConfig:
    """
    Configuration for Docling document parser.

    Supports full Docling feature set:
    - GPU acceleration (CUDA, MPS, CPU)
    - OCR with multiple engines
    - Table extraction with structure recognition
    - Image extraction with VLM descriptions
    - LaTeX and math formula extraction
    - Per-page organization option
    """
    # GPU Acceleration
    device: AcceleratorDevice = AcceleratorDevice.AUTO
    num_threads: int = 4

    # OCR Settings
    do_ocr: bool = True
    ocr_engine: OcrEngine = OcrEngine.EASYOCR
    ocr_lang: list[str] = field(default_factory=lambda: ["nl", "en"])

    # Table Extraction
    do_table_structure: bool = True
    table_mode: TableMode = TableMode.ACCURATE
    table_format: TableFormat = TableFormat.BOTH  # Both MD and CSV

    # Image Extraction
    generate_page_images: bool = False
    generate_picture_images: bool = True
    images_scale: float = 2.0
    image_ref_mode: ImageRefMode = ImageRefMode.REFERENCED

    # Image Classification & Description
    do_picture_classification: bool = True
    do_picture_description: bool = True
    vlm_model: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    vlm_prompt: str = "Describe this image in 2-3 sentences. Focus on key information."

    # Formula Extraction
    do_formula_extraction: bool = True

    # Code Block Handling
    do_code_enrichment: bool = True

    # Output Organization
    organize_by_page: bool = True  # Organize elements by page in output

    def to_docling_options(self):
        """Convert to Docling PdfPipelineOptions."""
        try:
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice as DoclingAccelDevice,
                AcceleratorOptions,
                PdfPipelineOptions,
                TableStructureOptions,
                EasyOcrOptions,
                TesseractOcrOptions,
                TesseractCliOcrOptions,
                RapidOcrOptions,
                PictureDescriptionVlmOptions,
            )
        except ImportError:
            raise ImportError("docling not installed. Install with: pip install docling>=2.58.0")

        # Map device
        device_map = {
            AcceleratorDevice.AUTO: DoclingAccelDevice.AUTO,
            AcceleratorDevice.CUDA: DoclingAccelDevice.CUDA,
            AcceleratorDevice.MPS: DoclingAccelDevice.MPS,
            AcceleratorDevice.CPU: DoclingAccelDevice.CPU,
        }

        accel_options = AcceleratorOptions(
            device=device_map[self.device],
            num_threads=self.num_threads,
        )

        # Table options
        table_options = TableStructureOptions(
            mode=self.table_mode.value,
        )

        # OCR options - use specific class based on engine
        ocr_options = None
        if self.do_ocr and self.ocr_engine != OcrEngine.NONE:
            if self.ocr_engine == OcrEngine.EASYOCR:
                ocr_options = EasyOcrOptions(lang=self.ocr_lang)
            elif self.ocr_engine == OcrEngine.TESSERACT:
                # Tesseract uses different language codes (3-letter)
                ocr_options = TesseractCliOcrOptions(lang=self.ocr_lang)
            elif self.ocr_engine == OcrEngine.RAPIDOCR:
                ocr_options = RapidOcrOptions(lang=self.ocr_lang)

        # Picture description options
        picture_desc_options = None
        if self.do_picture_description:
            picture_desc_options = PictureDescriptionVlmOptions(
                repo_id=self.vlm_model,
                prompt=self.vlm_prompt,
            )

        kwargs: dict = dict(
            accelerator_options=accel_options,
            do_ocr=self.do_ocr,
            ocr_options=ocr_options,
            table_structure_options=table_options,
            generate_page_images=self.generate_page_images,
            generate_picture_images=self.generate_picture_images,
            images_scale=self.images_scale,
            do_picture_classification=self.do_picture_classification,
            do_picture_description=self.do_picture_description,
            do_formula_enrichment=self.do_formula_extraction,
            do_code_enrichment=self.do_code_enrichment,
        )

        # Only pass picture_description_options when not None —
        # docling requires a valid PictureDescriptionBaseOptions instance, not None.
        if picture_desc_options is not None:
            kwargs["picture_description_options"] = picture_desc_options

        return PdfPipelineOptions(**kwargs)


@dataclass
class WhisperXConfig:
    """
    Configuration for WhisperX audio/video transcription.

    Features:
    - Multiple model sizes (tiny to large-v3)
    - Speaker diarization
    - Word-level timestamps
    - Language detection or specification
    """
    # Model Settings
    model_size: WhisperModel = WhisperModel.LARGE  # large-v3 for best quality
    language: Optional[str] = None  # None for auto-detect

    # Device
    device: AcceleratorDevice = AcceleratorDevice.AUTO
    compute_type: str = "float16"  # float16 for GPU, int8 for lower memory
    batch_size: int = 8  # Audio chunks processed at once (lower = less VRAM)

    # Diarization
    enable_diarization: bool = True
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    # Output
    include_word_timestamps: bool = True

    # HuggingFace token for diarization (pyannote)
    hf_token: Optional[str] = None

    def get_device(self) -> str:
        """Get the compute device string."""
        if self.device == AcceleratorDevice.AUTO:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return self.device.value


@dataclass
class OutputConfig:
    """
    Configuration for output directory structure.

    Structure:
    ./[base_dir]/[source_name]/
        original_file/          # Original document (non-audio/video only)
        output/
            extracted_info/     # Main extracted content
                document.md     # Full markdown
                document.json   # Structured JSON
                images/         # Extracted images
                tables/         # Extracted tables (MD and CSV)
                metadata.json   # Extraction metadata
    """
    base_dir: Path = field(default_factory=lambda: Path("./ingestion_output"))

    # Subdirectory names
    original_subdir: str = "original_file"
    output_subdir: str = "output"
    extracted_subdir: str = "extracted_info"
    images_subdir: str = "images"
    tables_subdir: str = "tables"

    def get_source_dir(self, source_name: str) -> Path:
        """Get the root directory for a source."""
        return self.base_dir / self._sanitize_name(source_name)

    def get_original_dir(self, source_name: str) -> Path:
        """Get directory for original file storage."""
        return self.get_source_dir(source_name) / self.original_subdir

    def get_output_dir(self, source_name: str) -> Path:
        """Get directory for extracted output."""
        return self.get_source_dir(source_name) / self.output_subdir / self.extracted_subdir

    def get_images_dir(self, source_name: str) -> Path:
        """Get directory for extracted images."""
        return self.get_output_dir(source_name) / self.images_subdir

    def get_tables_dir(self, source_name: str) -> Path:
        """Get directory for extracted tables."""
        return self.get_output_dir(source_name) / self.tables_subdir

    def create_directories(self, source_name: str, include_original: bool = True) -> dict[str, Path]:
        """
        Create all directories for a source.

        Args:
            source_name: Name of the source (used for directory naming)
            include_original: Whether to create original_file dir (False for audio/video)

        Returns:
            Dictionary of directory paths created
        """
        dirs = {
            "root": self.get_source_dir(source_name),
            "output": self.get_output_dir(source_name),
            "images": self.get_images_dir(source_name),
            "tables": self.get_tables_dir(source_name),
        }

        if include_original:
            dirs["original"] = self.get_original_dir(source_name)

        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        return dirs

    def create_directories_in_source_folder(
        self,
        source_folder_path: Path,
        source_type: str = "document",
    ) -> dict[str, Path]:
        """
        Create ingestion subdirectory structure within a pre-existing source folder.

        Used when a source folder already exists (created by file-manager) and
        the ingestion pipeline needs to write outputs into it.

        Args:
            source_folder_path: Path to the existing source folder.
            source_type: Type of source (affects which subdirs are created).

        Returns:
            Dictionary of directory paths created, matching create_directories() format.
        """
        dirs = {
            "root": source_folder_path,
            "output": source_folder_path / "ingestion",
            "images": source_folder_path / "ingestion" / self.images_subdir,
            "tables": source_folder_path / "ingestion" / self.tables_subdir,
        }

        if source_type in ("audio", "video"):
            dirs["transcription"] = source_folder_path / "transcription"

        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        return dirs

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize a name for use as directory name."""
        import re
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Limit length
        if len(sanitized) > 80:
            sanitized = sanitized[:80]
        return sanitized


@dataclass
class ContentFilterConfig:
    """Configuration for classifying chunks as content vs noise at extraction time.

    Used by ChunkExtractor to set is_content=False on noise elements.
    Downstream consumers (regrouper, summarizer, embeddings, entity extraction)
    filter on chunk.is_content rather than reimplementing filter logic.
    """
    enabled: bool = True
    # Element types classified as noise (matched against element_type)
    noise_element_types: list[str] = field(default_factory=lambda: [
        "page_header", "page_footer", "footnote",
    ])
    # Image classifications classified as noise (matched against metadata.classification)
    noise_image_classifications: list[str] = field(default_factory=lambda: [
        "logo", "icon", "signature", "stamp", "qr_code", "bar_code",
    ])
    # Text shorter than this is classified as noise
    min_text_length: int = 20


@dataclass
class RegroupingConfig:
    """Configuration for chunk regrouping."""
    strategy: RegroupingStrategy = RegroupingStrategy.STRUCTURAL
    target_tokens: int = 800          # ~3200 chars
    max_tokens: int = 1200            # Hard ceiling
    overlap_tokens: int = 100         # For sliding window
    chars_per_token: float = 4.0      # Approximation
    tables_as_standalone: bool = True  # Tables get own chunk
    prepend_section_context: bool = True  # Prefix "Section: ..." on chunks


@dataclass
class IngestionConfig:
    """
    Main configuration for the ingestion pipeline.

    Combines:
    - DoclingConfig: Document parsing options
    - WhisperXConfig: Audio/video transcription options
    - OutputConfig: Directory structure
    - RegroupingConfig: Chunk regrouping options
    - Destination handling
    """
    # Component configs
    docling: DoclingConfig = field(default_factory=DoclingConfig)
    whisperx: WhisperXConfig = field(default_factory=WhisperXConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    regrouping: RegroupingConfig = field(default_factory=RegroupingConfig)
    content_filter: ContentFilterConfig = field(default_factory=ContentFilterConfig)

    # Destination
    destination: DestinationType = DestinationType.ASK
    project_id: Optional[str] = None  # If destination is PROJECT

    # Processing options
    store_original_for_av: bool = False  # Don't store original audio/video files
    concurrent_processing: bool = True
    max_concurrent_files: int = 3

    # Supported file types
    supported_documents: list[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".doc", ".xlsx", ".xls",
        ".pptx", ".ppt", ".html", ".htm", ".txt", ".md"
    ])
    supported_audio: list[str] = field(default_factory=lambda: [
        ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma"
    ])
    supported_video: list[str] = field(default_factory=lambda: [
        ".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv"
    ])

    def is_document(self, path: Path) -> bool:
        """Check if path is a supported document."""
        return path.suffix.lower() in self.supported_documents

    def is_audio(self, path: Path) -> bool:
        """Check if path is a supported audio file."""
        return path.suffix.lower() in self.supported_audio

    def is_video(self, path: Path) -> bool:
        """Check if path is a supported video file."""
        return path.suffix.lower() in self.supported_video

    def is_av(self, path: Path) -> bool:
        """Check if path is audio or video."""
        return self.is_audio(path) or self.is_video(path)

    def is_supported(self, path: Path) -> bool:
        """Check if path is any supported file type."""
        return self.is_document(path) or self.is_av(path)

    @classmethod
    def from_env(cls) -> "IngestionConfig":
        """
        Create config from environment variables.

        Environment variables:
        - INGESTION_OUTPUT_DIR: Base output directory
        - INGESTION_DEVICE: GPU device (auto, cuda, mps, cpu)
        - INGESTION_OCR_LANG: OCR languages (comma-separated)
        - WHISPERX_MODEL: WhisperX model size
        - WHISPERX_DEVICE: WhisperX device
        - HF_TOKEN: HuggingFace token for diarization
        """
        import os

        config = cls()

        # Output directory
        if output_dir := os.getenv("INGESTION_OUTPUT_DIR"):
            config.output.base_dir = Path(output_dir)

        # Device settings
        if device := os.getenv("INGESTION_DEVICE"):
            config.docling.device = AcceleratorDevice(device)
            config.whisperx.device = AcceleratorDevice(device)

        # OCR languages
        if ocr_lang := os.getenv("INGESTION_OCR_LANG"):
            config.docling.ocr_lang = [lang.strip() for lang in ocr_lang.split(",")]

        # WhisperX settings
        if whisper_model := os.getenv("WHISPERX_MODEL"):
            config.whisperx.model_size = WhisperModel(whisper_model)

        if whisper_device := os.getenv("WHISPERX_DEVICE"):
            config.whisperx.device = AcceleratorDevice(whisper_device)

        # HuggingFace token
        if hf_token := os.getenv("HF_TOKEN"):
            config.whisperx.hf_token = hf_token

        return config

    @classmethod
    def for_cli(cls, output_dir: Optional[Path] = None) -> "IngestionConfig":
        """Create config for CLI usage with user prompts."""
        config = cls.from_env()
        config.destination = DestinationType.ASK
        if output_dir:
            config.output.base_dir = output_dir
        return config

    @classmethod
    def for_api(
        cls,
        destination: DestinationType,
        project_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> "IngestionConfig":
        """Create config for API usage with explicit destination."""
        config = cls.from_env()
        config.destination = destination
        config.project_id = project_id
        if output_dir:
            config.output.base_dir = output_dir
        return config
