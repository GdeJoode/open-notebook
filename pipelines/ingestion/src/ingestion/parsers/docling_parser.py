"""
Docling-based document parser.

Provides comprehensive document parsing with:
- PDF, DOCX, XLSX, PPTX, HTML support
- Table extraction with structure recognition
- Image extraction with VLM descriptions
- OCR for scanned documents
- LaTeX and formula extraction
- Per-page content organization
"""

import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from ingestion.config import DoclingConfig, ImageRefMode, TableFormat
from ingestion.models import (
    BoundingBox,
    ElementType,
    ExtractedDocument,
    ExtractedElement,
    ExtractedImage,
    ExtractedTable,
    PageContent,
)


class DoclingParser:
    """
    Document parser using Docling library.

    Features:
    - Full Docling configuration support
    - GPU acceleration (CUDA, MPS, CPU)
    - OCR with multiple engines
    - Table extraction with MD and CSV output
    - Image extraction with classification and VLM descriptions
    - Per-page content organization
    """

    def __init__(self, config: Optional[DoclingConfig] = None):
        """
        Initialize the Docling parser.

        Args:
            config: DoclingConfig instance. If None, uses defaults.
        """
        self.config = config or DoclingConfig()
        self._converter = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize the Docling converter."""
        if self._initialized:
            return

        logger.info("🔧 Initializing Docling document converter...")

        try:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError:
            raise ImportError(
                "Docling not installed. Install with: pip install docling>=2.58.0"
            )

        # Log configuration
        logger.info(f"📋 Configuration:")
        logger.info(f"   • Device: {self.config.device.value}")
        logger.info(f"   • OCR enabled: {self.config.do_ocr}")
        if self.config.do_ocr:
            logger.info(f"   • OCR engine: {self.config.ocr_engine.value}")
            logger.info(f"   • OCR languages: {', '.join(self.config.ocr_lang)}")
        logger.info(f"   • Table extraction: {self.config.do_table_structure}")
        if self.config.do_table_structure:
            logger.info(f"   • Table mode: {self.config.table_mode.value}")
        logger.info(f"   • Image extraction: {self.config.generate_picture_images}")
        logger.info(f"   • Image description (VLM): {self.config.do_picture_description}")
        if self.config.do_picture_description:
            logger.info(f"   • VLM model: {self.config.vlm_model}")

        # Get pipeline options from config
        logger.info("⚙️ Building pipeline options...")
        pipeline_options = self.config.to_docling_options()

        # Create converter with format options
        logger.info("🚀 Creating document converter...")
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self._initialized = True
        logger.success(f"✅ Docling parser initialized successfully")

    def parse(self, source_path: Path) -> ExtractedDocument:
        """
        Parse a document and extract all content.

        Args:
            source_path: Path to the document file

        Returns:
            ExtractedDocument with all extracted content organized by page
        """
        self._ensure_initialized()

        start_time = time.time()
        file_size_mb = source_path.stat().st_size / (1024 * 1024)
        logger.info(f"📄 Starting document processing: {source_path.name}")
        logger.info(f"   • File size: {file_size_mb:.2f} MB")
        logger.info(f"   • File type: {source_path.suffix}")

        # Convert document
        logger.info("🔍 Running Docling conversion pipeline...")
        logger.info("   • Layout analysis...")
        logger.info("   • Text extraction...")
        if self.config.do_ocr:
            logger.info(f"   • OCR processing ({self.config.ocr_engine.value})...")
        if self.config.do_table_structure:
            logger.info("   • Table structure recognition...")
        if self.config.do_picture_description:
            logger.info("   • Image analysis (VLM)...")

        result = self._converter.convert(str(source_path))
        doc = result.document

        conversion_time = time.time() - start_time
        logger.info(f"✅ Conversion completed in {conversion_time:.2f}s")

        # Get page count
        page_count = 0
        if hasattr(result, "input") and hasattr(result.input, "page_count"):
            page_count = result.input.page_count
        logger.info(f"📑 Document has {page_count} pages")

        # Extract content
        logger.info("📦 Extracting structured content...")
        pages = self._extract_pages(doc, result)
        all_tables = []
        all_images = []

        for page in pages:
            all_tables.extend(page.tables)
            all_images.extend(page.images)

        if all_tables:
            logger.info(f"   • Found {len(all_tables)} tables")
        if all_images:
            logger.info(f"   • Found {len(all_images)} images")

        # Count text elements
        total_elements = sum(len(p.elements) for p in pages)
        logger.info(f"   • Found {total_elements} text elements")

        # Export formats
        logger.info("📝 Generating markdown export...")
        full_markdown = doc.export_to_markdown()
        full_text = self._extract_plain_text(doc)

        extraction_time = time.time() - start_time

        extracted_doc = ExtractedDocument(
            source_path=source_path,
            title=self._extract_title(doc),
            page_count=page_count or len(pages),
            pages=pages,
            full_markdown=full_markdown,
            full_text=full_text,
            all_tables=all_tables,
            all_images=all_images,
            extraction_time_seconds=extraction_time,
            metadata={
                "docling_version": self._get_docling_version(),
                "config": {
                    "device": self.config.device.value,
                    "do_ocr": self.config.do_ocr,
                    "do_table_structure": self.config.do_table_structure,
                    "do_picture_description": self.config.do_picture_description,
                },
            },
        )

        logger.success(
            f"🎉 Document processed successfully!\n"
            f"   • Pages: {page_count}\n"
            f"   • Tables: {len(all_tables)}\n"
            f"   • Images: {len(all_images)}\n"
            f"   • Text elements: {total_elements}\n"
            f"   • Total time: {extraction_time:.2f}s"
        )

        return extracted_doc

    def _extract_pages(self, doc: Any, result: Any) -> list[PageContent]:
        """Extract content organized by page."""
        from docling_core.types.doc import TableItem, PictureItem

        pages_dict: dict[int, PageContent] = {}
        heading_stack: list[tuple[str, int]] = []  # (heading_text, level)

        # Pre-read page dimensions so BoundingBox.from_docling can normalize
        # to 0–1. prov.page_height is often missing, so we source the real
        # size from doc.pages[page_no].size (mirrors ChunkExtractor).
        page_sizes = self._get_page_sizes(doc)

        # Iterate through all items in document
        for item, level in doc.iterate_items():
            # Get page number from provenance
            page_num = self._get_page_number(item)
            page_width, page_height = page_sizes.get(page_num, (None, None))

            if page_num not in pages_dict:
                pages_dict[page_num] = PageContent(page_number=page_num)

            page = pages_dict[page_num]

            # Update heading stack on headings/titles
            element_type = self._get_element_type(item)
            if element_type in (ElementType.TITLE, ElementType.HEADING):
                text = getattr(item, "text", "") or ""
                # Pop headings at same or deeper level
                while heading_stack and heading_stack[-1][1] >= level:
                    heading_stack.pop()
                heading_stack.append((text.strip(), level))

            # Build section_path from stack
            section_path = [h[0] for h in heading_stack]

            # Handle different item types
            if isinstance(item, TableItem):
                table = self._extract_table(item, page_num, page_width, page_height)
                table.section_path = section_path
                table.section_level = level
                page.tables.append(table)
            elif isinstance(item, PictureItem):
                image = self._extract_image(
                    item, page_num, result, page_width, page_height
                )
                page.images.append(image)
            else:
                element = self._extract_element(
                    item, page_num, section_path, level, page_width, page_height
                )
                if element:
                    page.elements.append(element)

        # Sort pages by number
        sorted_pages = sorted(pages_dict.values(), key=lambda p: p.page_number)

        # Generate text content for each page
        for page in sorted_pages:
            page.text_content = page.to_markdown(include_images=False, include_tables=True)

        return sorted_pages

    def _get_page_sizes(self, doc: Any) -> dict[int, tuple[float, float]]:
        """Map page_no -> (width, height) in points from the DoclingDocument.

        Used to normalize bounding boxes to 0–1. Pages without a usable
        size are simply omitted; callers then emit a zero bbox for elements
        on those pages.
        """
        sizes: dict[int, tuple[float, float]] = {}
        pages = getattr(doc, "pages", None)
        if not pages:
            return sizes
        try:
            items = pages.items()
        except AttributeError:
            return sizes
        for page_no, page_item in items:
            size = getattr(page_item, "size", None)
            if not size:
                continue
            width = getattr(size, "width", None)
            height = getattr(size, "height", None)
            if width and height:
                sizes[int(page_no)] = (float(width), float(height))
        return sizes

    def _get_page_number(self, item: Any) -> int:
        """Get page number from item provenance."""
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                if hasattr(prov, "page_no"):
                    return prov.page_no
        return 1

    def _extract_element(
        self,
        item: Any,
        page_num: int,
        section_path: Optional[list[str]] = None,
        section_level: int = 0,
        page_width: Optional[float] = None,
        page_height: Optional[float] = None,
    ) -> Optional[ExtractedElement]:
        """Extract a generic element (text, heading, etc.)."""
        # Get element type
        element_type = self._get_element_type(item)
        if element_type is None:
            return None

        # Get content
        content = ""
        if hasattr(item, "text"):
            content = item.text
        elif hasattr(item, "export_to_markdown"):
            content = item.export_to_markdown()

        if not content or not content.strip():
            return None

        # Get bounding box
        bbox = None
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                bbox = BoundingBox.from_docling(
                    prov, page_num, page_width, page_height
                )
                break

        return ExtractedElement(
            content=content.strip(),
            element_type=element_type,
            page=page_num,
            bbox=bbox,
            section_path=section_path or [],
            section_level=section_level,
        )

    def _get_element_type(self, item: Any) -> Optional[ElementType]:
        """Map Docling item type to our ElementType."""
        type_name = type(item).__name__.lower()

        type_mapping = {
            "textitem": ElementType.TEXT,
            "titleitem": ElementType.TITLE,
            "sectionheaderitem": ElementType.HEADING,
            "paragraphitem": ElementType.PARAGRAPH,
            "listitem": ElementType.LIST_ITEM,
            "codeitem": ElementType.CODE,
            "formulaitem": ElementType.FORMULA,
            "captionitem": ElementType.CAPTION,
            "footnoteitem": ElementType.FOOTNOTE,
        }

        for key, value in type_mapping.items():
            if key in type_name:
                return value

        # Default to TEXT for unknown items with text content
        if hasattr(item, "text"):
            return ElementType.TEXT

        return None

    def _extract_table(
        self,
        item: Any,
        page_num: int,
        page_width: Optional[float] = None,
        page_height: Optional[float] = None,
    ) -> ExtractedTable:
        """Extract table with multiple format representations."""
        # Get markdown representation
        markdown = ""
        if hasattr(item, "export_to_markdown"):
            markdown = item.export_to_markdown()

        # Extract structured data
        headers = []
        rows = []

        if hasattr(item, "data") and item.data:
            data = item.data
            if hasattr(data, "table_cells"):
                # Build table from cells
                headers, rows = self._build_table_from_cells(data.table_cells)
            elif hasattr(data, "grid"):
                # Build from grid
                grid = data.grid
                if grid:
                    headers = [str(c) for c in grid[0]] if grid else []
                    rows = [[str(c) for c in row] for row in grid[1:]]

        # Get bounding box
        bbox = None
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                bbox = BoundingBox.from_docling(
                    prov, page_num, page_width, page_height
                )
                break

        table = ExtractedTable(
            content=markdown,
            element_type=ElementType.TABLE,
            page=page_num,
            bbox=bbox,
            markdown=markdown,
            headers=headers,
            rows=rows,
        )

        # Generate CSV
        table.csv = table.to_csv_string()

        return table

    def _build_table_from_cells(self, cells: Any) -> tuple[list[str], list[list[str]]]:
        """Build table structure from Docling cell data."""
        if not cells:
            return [], []

        # Find dimensions
        max_row = 0
        max_col = 0
        for cell in cells:
            if hasattr(cell, "row_index"):
                max_row = max(max_row, cell.row_index + 1)
            if hasattr(cell, "col_index"):
                max_col = max(max_col, cell.col_index + 1)

        if max_row == 0 or max_col == 0:
            return [], []

        # Build grid
        grid = [["" for _ in range(max_col)] for _ in range(max_row)]

        for cell in cells:
            row = getattr(cell, "row_index", 0)
            col = getattr(cell, "col_index", 0)
            text = getattr(cell, "text", "") or ""
            if 0 <= row < max_row and 0 <= col < max_col:
                grid[row][col] = text

        # First row as headers
        headers = grid[0] if grid else []
        rows = grid[1:] if len(grid) > 1 else []

        return headers, rows

    def _extract_image(
        self,
        item: Any,
        page_num: int,
        result: Any,
        page_width: Optional[float] = None,
        page_height: Optional[float] = None,
    ) -> ExtractedImage:
        """Extract image with classification and VLM description."""
        # Get classification
        classification = "unknown"
        if hasattr(item, "classification") and item.classification is not None:
            cls = item.classification
            # PictureClassificationClass is a Pydantic model with class_name/confidence
            if hasattr(cls, "class_name"):
                classification = str(cls.class_name)
            else:
                classification = str(cls)
        elif hasattr(item, "image_type"):
            classification = str(item.image_type)

        # Get VLM description
        description = ""
        if hasattr(item, "description"):
            description = item.description or ""
        elif hasattr(item, "caption"):
            description = item.caption or ""

        # Get image data
        image_data = None
        width = 0
        height = 0
        img_format = "png"

        if hasattr(item, "image"):
            img = item.image
            if hasattr(img, "uri") and img.uri:
                # Convert to str — Docling may return a Pydantic AnyUrl object
                image_data = str(img.uri)
            if hasattr(img, "size"):
                width, height = img.size
            if hasattr(img, "format"):
                img_format = img.format or "png"

        # Get bounding box
        bbox = None
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                bbox = BoundingBox.from_docling(
                    prov, page_num, page_width, page_height
                )
                break

        return ExtractedImage(
            content=description or f"[{classification}]",
            element_type=ElementType.IMAGE,
            page=page_num,
            bbox=bbox,
            image_data=image_data,
            classification=classification,
            description=description,
            width=width,
            height=height,
            format=img_format,
        )

    def _extract_title(self, doc: Any) -> str:
        """Extract document title."""
        if hasattr(doc, "name"):
            return doc.name or ""

        # Try to find first title element
        for item, level in doc.iterate_items():
            type_name = type(item).__name__.lower()
            if "title" in type_name and hasattr(item, "text"):
                return item.text or ""

        return ""

    def _extract_plain_text(self, doc: Any) -> str:
        """Extract plain text from document."""
        if hasattr(doc, "export_to_text"):
            return doc.export_to_text()

        # Fallback: extract text from all items
        texts = []
        for item, level in doc.iterate_items():
            if hasattr(item, "text") and item.text:
                texts.append(item.text)
        return "\n".join(texts)

    def _get_docling_version(self) -> str:
        """Get Docling version."""
        try:
            import docling
            return getattr(docling, "__version__", "unknown")
        except Exception:
            return "unknown"


def parse_document(
    source_path: Path,
    config: Optional[DoclingConfig] = None,
) -> ExtractedDocument:
    """
    Convenience function to parse a document.

    Args:
        source_path: Path to the document
        config: Optional DoclingConfig

    Returns:
        ExtractedDocument with all content
    """
    parser = DoclingParser(config)
    return parser.parse(source_path)
