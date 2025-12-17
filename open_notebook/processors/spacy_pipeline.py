"""
spaCy-Layout Document Processing Pipeline.

Replaces content-core with spacy-layout for document parsing.
spacy-layout integrates Docling internally for PDF/DOCX/PPTX/etc parsing.

Features:
- Multi-format support (PDF, DOCX, PPTX, XLSX, HTML, MD, Images)
- Layout-aware chunking with bounding boxes
- NER with EntityRuler (fed by existing KG entities)
- GPU acceleration (CUDA/MPS/CPU)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from open_notebook.processors.gpu_detection import (
    GPUConfig,
    get_gpu_config,
    get_optimal_config,
    setup_spacy_gpu,
)


@dataclass
class ProcessingInput:
    """Input for document processing (replaces ProcessSourceInput)."""
    file_path: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None
    title: Optional[str] = None


@dataclass
class ProcessingOutput:
    """Output from document processing (replaces ProcessSourceOutput)."""
    content: str
    file_path: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkData:
    """Structured chunk data with spatial information."""
    text: str
    chunk_order: int
    physical_page: int
    printed_page: Optional[int] = None
    chapter: Optional[str] = None
    paragraph_number: Optional[int] = None
    element_type: str = "text"
    positions: List[List[float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for storage."""
        return {
            "text": self.text,
            "chunk_order": self.chunk_order,
            "physical_page": self.physical_page,
            "printed_page": self.printed_page,
            "chapter": self.chapter,
            "paragraph_number": self.paragraph_number,
            "element_type": self.element_type,
            "positions": self.positions,
            "metadata": self.metadata,
        }


class SpacyLayoutPipeline:
    """
    Document processing pipeline using spacy-layout.

    spacy-layout integrates Docling internally, so we get:
    - Document parsing (PDF, DOCX, PPTX, etc.)
    - Layout analysis
    - Text extraction with bounding boxes
    - Table extraction
    - Image extraction

    We add on top:
    - NER with EntityRuler (fed by KG entities)
    - EntityLinker for resolution
    - GPU acceleration
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_trf",
        gpu_enabled: bool = True,
        gpu_device: Optional[str] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            spacy_model: spaCy model to use for NER
            gpu_enabled: Enable GPU acceleration
            gpu_device: Specific GPU device ('cuda', 'mps', 'cpu', 'auto')
        """
        self.spacy_model_name = spacy_model
        self.gpu_config = get_optimal_config(gpu_device, gpu_enabled)
        self._nlp = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of spaCy pipeline."""
        if self._initialized:
            return

        # Setup GPU for spaCy
        setup_spacy_gpu(self.gpu_config)

        # Load spaCy model
        try:
            import spacy
            self._nlp = spacy.load(self.spacy_model_name)
            logger.info(f"✅ Loaded spaCy model: {self.spacy_model_name}")
        except OSError:
            logger.warning(f"Model {self.spacy_model_name} not found, downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", self.spacy_model_name], check=True)
            import spacy
            self._nlp = spacy.load(self.spacy_model_name)

        self._initialized = True

    async def process(
        self,
        input_data: ProcessingInput
    ) -> tuple[ProcessingOutput, List[Dict[str, Any]]]:
        """
        Process a document and extract content with chunks.

        Args:
            input_data: Input containing file_path, url, or content

        Returns:
            Tuple of (ProcessingOutput, list of chunk dicts)
        """
        self._ensure_initialized()

        if input_data.file_path:
            return await self._process_file(input_data)
        elif input_data.url:
            return await self._process_url(input_data)
        elif input_data.content:
            return await self._process_text(input_data)
        else:
            raise ValueError("No file_path, url, or content provided")

    async def _process_file(
        self,
        input_data: ProcessingInput
    ) -> tuple[ProcessingOutput, List[Dict[str, Any]]]:
        """Process a local file using spacy-layout."""
        file_path = Path(input_data.file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        logger.info(f"📄 Processing file: {file_path.name} (type: {suffix})")

        # Use spacy-layout for document processing
        try:
            import spacy_layout
            from spacy_layout import spaCyLayout

            # Create layout parser with our spaCy model
            layout = spaCyLayout(self._nlp)

            # Process the document - spacy-layout uses Docling internally
            doc = layout(str(file_path))

            # Extract content as markdown
            content = self._doc_to_markdown(doc)

            # Extract chunks with bounding boxes
            chunks = self._extract_chunks_from_layout(doc)

            # Build output
            output = ProcessingOutput(
                content=content,
                file_path=str(file_path),
                title=input_data.title or file_path.stem,
                metadata={
                    "extraction_engine": "spacy-layout",
                    "file_type": suffix,
                    "num_chunks": len(chunks),
                    "gpu_device": self.gpu_config.torch_device,
                }
            )

            logger.info(f"✅ Extracted {len(chunks)} chunks from {file_path.name}")
            return output, [c.to_dict() for c in chunks]

        except ImportError:
            logger.warning("spacy-layout not available, falling back to basic extraction")
            return await self._fallback_extraction(input_data)

    async def _process_url(
        self,
        input_data: ProcessingInput
    ) -> tuple[ProcessingOutput, List[Dict[str, Any]]]:
        """Process a URL - download and process."""
        import httpx
        from tempfile import NamedTemporaryFile

        logger.info(f"🌐 Fetching URL: {input_data.url}")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(input_data.url)
            response.raise_for_status()

            # Determine file extension from content-type or URL
            content_type = response.headers.get("content-type", "")
            suffix = self._get_suffix_from_content_type(content_type, input_data.url)

            # Save to temp file
            with NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(response.content)
                temp_path = f.name

        try:
            # Process the downloaded file
            file_input = ProcessingInput(
                file_path=temp_path,
                url=input_data.url,
                title=input_data.title,
            )
            output, chunks = await self._process_file(file_input)
            output.url = input_data.url
            return output, chunks
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    async def _process_text(
        self,
        input_data: ProcessingInput
    ) -> tuple[ProcessingOutput, List[Dict[str, Any]]]:
        """Process raw text content."""
        self._ensure_initialized()

        content = input_data.content or ""

        # Process with spaCy for NER
        doc = self._nlp(content)

        # Create single chunk for text input
        chunks = [
            ChunkData(
                text=content,
                chunk_order=0,
                physical_page=0,
                element_type="text",
                metadata={"source": "raw_text"}
            )
        ]

        output = ProcessingOutput(
            content=content,
            title=input_data.title,
            metadata={
                "extraction_engine": "spacy-text",
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
            }
        )

        return output, [c.to_dict() for c in chunks]

    def _doc_to_markdown(self, doc) -> str:
        """Convert spacy-layout document to markdown."""
        # spacy-layout docs have layout spans we can iterate
        parts = []

        for span in doc.spans.get("layout", []):
            text = span.text.strip()
            if not text:
                continue

            # Get layout label if available
            label = getattr(span, "label_", "text")

            if label in ["title", "heading", "section_header"]:
                parts.append(f"## {text}\n")
            elif label == "list_item":
                parts.append(f"- {text}")
            elif label == "table":
                parts.append(f"\n{text}\n")
            else:
                parts.append(text)

        return "\n\n".join(parts) if parts else doc.text

    def _extract_chunks_from_layout(self, doc) -> List[ChunkData]:
        """Extract chunks with bounding boxes from spacy-layout document."""
        chunks = []
        current_chapter = None

        # Get layout spans from spacy-layout
        layout_spans = doc.spans.get("layout", [])

        for idx, span in enumerate(layout_spans):
            text = span.text.strip()
            if not text:
                continue

            # Get span properties
            label = getattr(span, "label_", "text")

            # Track chapters
            if label in ["title", "heading", "section_header"]:
                current_chapter = text

            # Extract bounding box if available
            positions = []
            physical_page = 0

            # spacy-layout stores layout info in span._.layout
            if hasattr(span._, "layout") and span._.layout:
                layout_info = span._.layout
                if hasattr(layout_info, "page_no"):
                    physical_page = layout_info.page_no

                # Extract bounding box
                if hasattr(layout_info, "bbox"):
                    bbox = layout_info.bbox
                    # Normalize to 0-1 range (assuming page size in layout_info)
                    page_width = getattr(layout_info, "page_width", 595.0)
                    page_height = getattr(layout_info, "page_height", 842.0)

                    x_left = bbox.x / page_width if hasattr(bbox, "x") else 0
                    x_right = (bbox.x + bbox.width) / page_width if hasattr(bbox, "width") else 1
                    y_top = bbox.y / page_height if hasattr(bbox, "y") else 0
                    y_bottom = (bbox.y + bbox.height) / page_height if hasattr(bbox, "height") else 1

                    positions.append([
                        physical_page,
                        max(0, min(1, x_left)),
                        max(0, min(1, x_right)),
                        max(0, min(1, y_top)),
                        max(0, min(1, y_bottom)),
                    ])

            chunk = ChunkData(
                text=text,
                chunk_order=idx,
                physical_page=physical_page,
                printed_page=physical_page,
                chapter=current_chapter,
                element_type=label,
                positions=positions,
                metadata={
                    "has_spatial_data": len(positions) > 0,
                    "num_locations": len(positions),
                }
            )
            chunks.append(chunk)

        return chunks

    async def _fallback_extraction(
        self,
        input_data: ProcessingInput
    ) -> tuple[ProcessingOutput, List[Dict[str, Any]]]:
        """Fallback extraction using direct Docling (without spacy-layout)."""
        from open_notebook.processors.chunk_extractor import extract_chunks_from_docling

        logger.info("Using fallback Docling extraction")

        content, chunks, _ = extract_chunks_from_docling(
            input_data.file_path,
            output_format="markdown"
        )

        output = ProcessingOutput(
            content=content,
            file_path=input_data.file_path,
            title=input_data.title or Path(input_data.file_path).stem,
            metadata={
                "extraction_engine": "docling-fallback",
                "num_chunks": len(chunks),
            }
        )

        return output, chunks

    def _get_suffix_from_content_type(self, content_type: str, url: str) -> str:
        """Determine file suffix from content-type or URL."""
        type_map = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "text/html": ".html",
            "text/markdown": ".md",
        }

        for mime, suffix in type_map.items():
            if mime in content_type:
                return suffix

        # Try URL extension
        path = Path(url.split("?")[0])
        if path.suffix:
            return path.suffix

        return ".pdf"  # Default


# Convenience function for backward compatibility
async def process_document(
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    content: Optional[str] = None,
    title: Optional[str] = None,
    gpu_enabled: bool = True,
    gpu_device: Optional[str] = None,
    spacy_model: str = "en_core_web_trf",
) -> tuple[ProcessingOutput, List[Dict[str, Any]]]:
    """
    Process a document using the spaCy-Layout pipeline.

    This is the main entry point replacing content-core's extract_content().

    Args:
        file_path: Path to local file
        url: URL to download and process
        content: Raw text content
        title: Document title
        gpu_enabled: Enable GPU acceleration
        gpu_device: GPU device preference
        spacy_model: spaCy model for NER

    Returns:
        Tuple of (ProcessingOutput, list of chunk dicts)
    """
    pipeline = SpacyLayoutPipeline(
        spacy_model=spacy_model,
        gpu_enabled=gpu_enabled,
        gpu_device=gpu_device,
    )

    input_data = ProcessingInput(
        file_path=file_path,
        url=url,
        content=content,
        title=title,
    )

    return await pipeline.process(input_data)
