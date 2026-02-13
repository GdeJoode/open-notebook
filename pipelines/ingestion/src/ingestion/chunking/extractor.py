"""
Chunk extractor for semantic search.

Extracts text chunks with:
- Bounding box positions for PDF highlighting
- Chapter/section context
- Element type classification
- Paragraph ordering
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from ingestion.config import ContentFilterConfig, RegroupingConfig, RegroupingStrategy
from ingestion.models import ExtractedDocument, BoundingBox


@dataclass
class ChunkPosition:
    """
    Position of a chunk on a page.

    Coordinates are normalized (0.0 to 1.0) for cross-resolution compatibility.
    """
    page: int  # 1-indexed page number
    x1: float  # Left edge (0.0 to 1.0)
    x2: float  # Right edge (0.0 to 1.0)
    y1: float  # Top edge (0.0 to 1.0)
    y2: float  # Bottom edge (0.0 to 1.0)

    def to_list(self) -> list[float]:
        """Convert to list format [page, x1, x2, y1, y2]."""
        return [self.page, self.x1, self.x2, self.y1, self.y2]

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "page": self.page,
            "x1": self.x1,
            "x2": self.x2,
            "y1": self.y1,
            "y2": self.y2,
        }

    @classmethod
    def from_bbox(cls, bbox: BoundingBox) -> "ChunkPosition":
        """Create from BoundingBox."""
        return cls(
            page=bbox.page,
            x1=bbox.x,
            x2=bbox.x2,
            y1=bbox.y,
            y2=bbox.y2,
        )


@dataclass
class Chunk:
    """
    Semantic chunk for vector embedding and search.

    Contains:
    - Text content
    - Position information for PDF highlighting
    - Contextual metadata (chapter, section)
    - Element classification
    """
    text: str
    order: int  # Position in document reading order
    element_type: str  # text, table, picture, heading, etc.

    # Page information
    physical_page: int  # 1-indexed physical page number
    printed_page: Optional[int] = None  # Printed page number if different

    # Positions for highlighting (can span multiple pages)
    positions: list[ChunkPosition] = field(default_factory=list)

    # Context
    chapter: Optional[str] = None
    section: Optional[str] = None
    paragraph_number: Optional[int] = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Section hierarchy from document structure
    section_path: list[str] = field(default_factory=list)  # e.g. ["Chapter 1", "Section 1.2"]
    section_level: int = 0  # Nesting depth in document
    chunk_type: str = "original"       # "original" | "merged" | "table" | "overlap"
    source_element_count: int = 1      # How many elements were merged

    # Content relevance flag — set at extraction time.
    # False for noise elements (page headers/footers, logos, stamps, etc.)
    is_content: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "order": self.order,
            "element_type": self.element_type,
            "physical_page": self.physical_page,
            "printed_page": self.printed_page,
            "positions": [p.to_list() for p in self.positions],
            "chapter": self.chapter,
            "section": self.section,
            "paragraph_number": self.paragraph_number,
            "metadata": self.metadata,
            "section_path": self.section_path,
            "section_level": self.section_level,
            "chunk_type": self.chunk_type,
            "source_element_count": self.source_element_count,
            "is_content": self.is_content,
        }


class ChunkExtractor:
    """
    Extract semantic chunks from documents.

    Features:
    - Bounding box positions for PDF highlighting
    - Chapter/section context tracking
    - Reading order preservation
    - Element type classification
    """

    def __init__(
        self,
        min_chunk_size: int = 50,
        max_chunk_size: int = 2000,
        include_tables: bool = True,
        include_images: bool = True,
        regrouping_config: Optional[RegroupingConfig] = None,
        content_filter_config: Optional[ContentFilterConfig] = None,
    ):
        """
        Initialize the chunk extractor.

        Args:
            min_chunk_size: Minimum characters for a chunk
            max_chunk_size: Maximum characters for a chunk (will split larger)
            include_tables: Include tables as chunks
            include_images: Include image captions as chunks
            regrouping_config: Optional config for chunk regrouping
            content_filter_config: Optional config for content classification
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.include_tables = include_tables
        self.include_images = include_images
        self._regrouping_config = regrouping_config
        self._filter_config = content_filter_config or ContentFilterConfig()

    def _classify_is_content(self, chunk: Chunk) -> bool:
        """Determine if a chunk is relevant content or noise.

        Returns False for noise elements (page headers/footers, logos, etc.)
        so downstream consumers can filter on chunk.is_content.
        """
        cf = self._filter_config
        if not cf.enabled:
            return True
        # Noise element types
        if chunk.element_type in cf.noise_element_types:
            return False
        # Noise image classifications
        if chunk.element_type == "picture":
            classification = chunk.metadata.get("classification", "").lower()
            if classification in cf.noise_image_classifications:
                return False
        # Too short to be meaningful
        if len(chunk.text.strip()) < cf.min_text_length:
            return False
        return True

    def _classify_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Set is_content flag on all chunks. Returns the same list (mutated)."""
        noise_count = 0
        for chunk in chunks:
            chunk.is_content = self._classify_is_content(chunk)
            if not chunk.is_content:
                noise_count += 1
        if noise_count:
            logger.info(
                f"Content classification: {noise_count} noise chunks, "
                f"{len(chunks) - noise_count} content chunks"
            )
        return chunks

    def extract(self, document: ExtractedDocument) -> list[Chunk]:
        """
        Extract chunks from an ExtractedDocument.

        Args:
            document: Parsed document with page-organized content

        Returns:
            List of Chunk objects in reading order
        """
        chunks = []
        current_chapter = None
        current_section = None
        order = 0

        for page in document.pages:
            paragraph_number = 0

            # Process elements
            for element in page.elements:
                # Track chapter/section headers
                if element.element_type.value in ["title", "heading"]:
                    if element.element_type.value == "title":
                        current_chapter = element.content
                    else:
                        current_section = element.content

                # Skip if too short
                if len(element.content) < self.min_chunk_size:
                    continue

                # Create chunk(s)
                if len(element.content) <= self.max_chunk_size:
                    chunk = self._create_chunk(
                        text=element.content,
                        order=order,
                        element_type=element.element_type.value,
                        page=page.page_number,
                        bbox=element.bbox,
                        chapter=current_chapter,
                        section=current_section,
                        paragraph_number=paragraph_number,
                        section_path=element.section_path,
                        section_level=element.section_level,
                    )
                    chunks.append(chunk)
                    order += 1
                else:
                    # Split large chunks
                    sub_chunks = self._split_text(
                        text=element.content,
                        start_order=order,
                        element_type=element.element_type.value,
                        page=page.page_number,
                        bbox=element.bbox,
                        chapter=current_chapter,
                        section=current_section,
                        section_path=element.section_path,
                        section_level=element.section_level,
                    )
                    chunks.extend(sub_chunks)
                    order += len(sub_chunks)

                paragraph_number += 1

            # Process tables
            if self.include_tables:
                for table in page.tables:
                    # Use markdown representation
                    table_text = table.to_markdown_string() or table.markdown
                    if len(table_text) >= self.min_chunk_size:
                        chunk = self._create_chunk(
                            text=table_text,
                            order=order,
                            element_type="table",
                            page=page.page_number,
                            bbox=table.bbox,
                            chapter=current_chapter,
                            section=current_section,
                            metadata={"rows": table.num_rows, "cols": table.num_cols},
                            section_path=table.section_path,
                            section_level=table.section_level,
                            chunk_type="table",
                        )
                        chunks.append(chunk)
                        order += 1

            # Process images
            if self.include_images:
                for image in page.images:
                    # Use description if available
                    if image.description and len(image.description) >= self.min_chunk_size:
                        chunk = self._create_chunk(
                            text=f"[Image: {image.classification}] {image.description}",
                            order=order,
                            element_type="picture",
                            page=page.page_number,
                            bbox=image.bbox,
                            chapter=current_chapter,
                            section=current_section,
                            metadata={"classification": image.classification},
                        )
                        chunks.append(chunk)
                        order += 1

        logger.info(f"Extracted {len(chunks)} chunks from {len(document.pages)} pages")

        # Classify chunks as content vs noise
        self._classify_chunks(chunks)

        # Regroup if configured
        if self._regrouping_config and self._regrouping_config.strategy != RegroupingStrategy.NONE:
            from ingestion.chunking.regrouper import ChunkRegrouper
            regrouper = ChunkRegrouper(self._regrouping_config)
            original_count = len(chunks)
            chunks = regrouper.regroup(chunks)
            logger.info(f"Regrouped {original_count} → {len(chunks)} chunks (strategy={self._regrouping_config.strategy.value})")

        return chunks

    def extract_from_docling(self, doc: Any, result: Any = None) -> list[Chunk]:
        """
        Extract chunks directly from a Docling document.

        This method provides direct access to Docling's document structure
        for more precise chunk extraction.

        Args:
            doc: DoclingDocument object
            result: Optional ConversionResult for additional metadata

        Returns:
            List of Chunk objects
        """
        try:
            from docling_core.types.doc import TextItem, TableItem, PictureItem
        except ImportError:
            raise ImportError("docling-core not installed")

        chunks = []
        current_chapter = None
        current_section = None
        paragraph_counter: dict[int, int] = {}  # Track per-page paragraph numbers

        for idx, item_data in enumerate(doc.iterate_items()):
            # Handle both single items and (item, level) tuples
            if isinstance(item_data, tuple):
                item, _ = item_data
            else:
                item = item_data

            # Extract content based on type
            text = None
            element_type = "unknown"
            metadata: dict[str, Any] = {}

            if isinstance(item, TextItem):
                text = item.text
                element_type = str(item.label) if hasattr(item, "label") else "text"

                # Track headers
                if element_type in ["section_header", "title", "heading"]:
                    if element_type == "title":
                        current_chapter = text
                    else:
                        current_section = text

            elif isinstance(item, TableItem):
                if self.include_tables:
                    try:
                        table_df = item.export_to_dataframe()
                        text = table_df.to_markdown() if not table_df.empty else str(item)
                    except Exception:
                        text = str(item)
                    element_type = "table"

            elif isinstance(item, PictureItem):
                if self.include_images:
                    text = getattr(item, "caption", None) or f"[Picture {idx}]"
                    element_type = "picture"
                    # Store classification in metadata for content filtering
                    classification = "unknown"
                    if hasattr(item, "classification"):
                        classification = str(item.classification).lower()
                    metadata = {"classification": classification}

            else:
                text = getattr(item, "text", None)
                element_type = type(item).__name__

            # Skip empty or too short
            if not text or len(text.strip()) < self.min_chunk_size:
                continue

            # Get positions from provenance
            positions = []
            physical_page = 1

            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "bbox") and hasattr(prov, "page_no"):
                        page_no = prov.page_no
                        physical_page = page_no
                        bbox = prov.bbox

                        # Get page dimensions
                        page_width = 595.0  # A4 default
                        page_height = 842.0

                        if hasattr(doc, "pages") and doc.pages and page_no in doc.pages:
                            page_item = doc.pages[page_no]
                            if hasattr(page_item, "size") and page_item.size:
                                page_width = getattr(page_item.size, "width", page_width)
                                page_height = getattr(page_item.size, "height", page_height)

                        # Convert coordinates
                        position = self._convert_docling_bbox(
                            bbox, page_no, page_width, page_height
                        )
                        positions.append(position)

            # Track paragraph number
            if physical_page not in paragraph_counter:
                paragraph_counter[physical_page] = 0
            paragraph_counter[physical_page] += 1

            chunk = Chunk(
                text=text.strip(),
                order=idx,
                element_type=element_type,
                physical_page=physical_page,
                positions=positions,
                chapter=current_chapter,
                section=current_section,
                paragraph_number=paragraph_counter[physical_page],
                metadata=metadata,
            )
            chunks.append(chunk)

        logger.info(f"Extracted {len(chunks)} chunks from Docling document")

        # Classify chunks as content vs noise
        self._classify_chunks(chunks)

        return chunks

    def _create_chunk(
        self,
        text: str,
        order: int,
        element_type: str,
        page: int,
        bbox: Optional[BoundingBox],
        chapter: Optional[str] = None,
        section: Optional[str] = None,
        paragraph_number: Optional[int] = None,
        metadata: Optional[dict] = None,
        section_path: Optional[list[str]] = None,
        section_level: int = 0,
        chunk_type: str = "original",
    ) -> Chunk:
        """Create a single chunk."""
        positions = []
        if bbox:
            positions.append(ChunkPosition.from_bbox(bbox))

        return Chunk(
            text=text.strip(),
            order=order,
            element_type=element_type,
            physical_page=page,
            positions=positions,
            chapter=chapter,
            section=section,
            paragraph_number=paragraph_number,
            metadata=metadata or {},
            section_path=section_path or [],
            section_level=section_level,
            chunk_type=chunk_type,
        )

    def _split_text(
        self,
        text: str,
        start_order: int,
        element_type: str,
        page: int,
        bbox: Optional[BoundingBox],
        chapter: Optional[str] = None,
        section: Optional[str] = None,
        section_path: Optional[list[str]] = None,
        section_level: int = 0,
    ) -> list[Chunk]:
        """Split large text into smaller chunks."""
        chunks = []

        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        current_text = ""
        order = start_order

        for para in paragraphs:
            if len(current_text) + len(para) <= self.max_chunk_size:
                current_text = f"{current_text}\n\n{para}" if current_text else para
            else:
                if current_text:
                    chunk = self._create_chunk(
                        text=current_text,
                        order=order,
                        element_type=element_type,
                        page=page,
                        bbox=bbox,
                        chapter=chapter,
                        section=section,
                        section_path=section_path,
                        section_level=section_level,
                    )
                    chunks.append(chunk)
                    order += 1
                current_text = para

        # Don't forget the last chunk
        if current_text:
            chunk = self._create_chunk(
                text=current_text,
                order=order,
                element_type=element_type,
                page=page,
                bbox=bbox,
                chapter=chapter,
                section=section,
                section_path=section_path,
                section_level=section_level,
            )
            chunks.append(chunk)

        return chunks

    def _convert_docling_bbox(
        self,
        bbox: Any,
        page_no: int,
        page_width: float,
        page_height: float,
    ) -> ChunkPosition:
        """Convert Docling bounding box to ChunkPosition."""
        left = float(getattr(bbox, "l", 0))
        right = float(getattr(bbox, "r", 0))
        top = float(getattr(bbox, "t", 0))
        bottom = float(getattr(bbox, "b", 0))

        # Determine coordinate system
        coord_origin = "BOTTOMLEFT"
        if hasattr(bbox, "coord_origin"):
            coord_origin = str(bbox.coord_origin).upper()

        # Convert to TOPLEFT normalized coordinates
        if "BOTTOMLEFT" in coord_origin:
            # Convert y-coordinates
            y1 = (page_height - top) / page_height
            y2 = (page_height - bottom) / page_height
        else:
            y1 = top / page_height
            y2 = bottom / page_height

        # Normalize x-coordinates
        x1 = left / page_width
        x2 = right / page_width

        # Clamp to valid range
        x1 = max(0.0, min(1.0, x1))
        x2 = max(0.0, min(1.0, x2))
        y1 = max(0.0, min(1.0, y1))
        y2 = max(0.0, min(1.0, y2))

        return ChunkPosition(page=page_no, x1=x1, x2=x2, y1=y1, y2=y2)


def extract_chunks(
    document: ExtractedDocument,
    min_chunk_size: int = 50,
    max_chunk_size: int = 2000,
    regrouping_config: Optional[RegroupingConfig] = None,
    content_filter_config: Optional[ContentFilterConfig] = None,
) -> list[Chunk]:
    """
    Convenience function to extract chunks from a document.

    Args:
        document: Parsed document
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size in characters
        regrouping_config: Optional config for chunk regrouping
        content_filter_config: Optional config for content classification

    Returns:
        List of chunks (each has is_content flag set)
    """
    extractor = ChunkExtractor(
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        regrouping_config=regrouping_config,
        content_filter_config=content_filter_config,
    )
    return extractor.extract(document)
