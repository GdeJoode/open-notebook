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

        try:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError:
            raise ImportError(
                "Docling not installed. Install with: pip install docling>=2.58.0"
            )

        # Get pipeline options from config
        pipeline_options = self.config.to_docling_options()

        # Create converter with format options
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self._initialized = True
        logger.info(f"Docling parser initialized with device={self.config.device.value}")

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
        logger.info(f"Parsing document: {source_path}")

        # Convert document
        result = self._converter.convert(str(source_path))
        doc = result.document

        # Get page count
        page_count = 0
        if hasattr(result, "input") and hasattr(result.input, "page_count"):
            page_count = result.input.page_count

        # Extract content
        pages = self._extract_pages(doc, result)
        all_tables = []
        all_images = []

        for page in pages:
            all_tables.extend(page.tables)
            all_images.extend(page.images)

        # Export formats
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

        logger.info(
            f"Document parsed: {page_count} pages, "
            f"{len(all_tables)} tables, {len(all_images)} images "
            f"in {extraction_time:.2f}s"
        )

        return extracted_doc

    def _extract_pages(self, doc: Any, result: Any) -> list[PageContent]:
        """Extract content organized by page."""
        from docling_core.types.doc import TableItem, PictureItem

        pages_dict: dict[int, PageContent] = {}

        # Iterate through all items in document
        for item, level in doc.iterate_items():
            # Get page number from provenance
            page_num = self._get_page_number(item)

            if page_num not in pages_dict:
                pages_dict[page_num] = PageContent(page_number=page_num)

            page = pages_dict[page_num]

            # Handle different item types
            if isinstance(item, TableItem):
                table = self._extract_table(item, page_num)
                page.tables.append(table)
            elif isinstance(item, PictureItem):
                image = self._extract_image(item, page_num, result)
                page.images.append(image)
            else:
                element = self._extract_element(item, page_num)
                if element:
                    page.elements.append(element)

        # Sort pages by number
        sorted_pages = sorted(pages_dict.values(), key=lambda p: p.page_number)

        # Generate text content for each page
        for page in sorted_pages:
            page.text_content = page.to_markdown(include_images=False, include_tables=True)

        return sorted_pages

    def _get_page_number(self, item: Any) -> int:
        """Get page number from item provenance."""
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                if hasattr(prov, "page_no"):
                    return prov.page_no
        return 1

    def _extract_element(self, item: Any, page_num: int) -> Optional[ExtractedElement]:
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
                bbox = BoundingBox.from_docling(prov, page_num)
                break

        return ExtractedElement(
            content=content.strip(),
            element_type=element_type,
            page=page_num,
            bbox=bbox,
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

    def _extract_table(self, item: Any, page_num: int) -> ExtractedTable:
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
                bbox = BoundingBox.from_docling(prov, page_num)
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

    def _extract_image(self, item: Any, page_num: int, result: Any) -> ExtractedImage:
        """Extract image with classification and VLM description."""
        # Get classification
        classification = "unknown"
        if hasattr(item, "classification"):
            classification = str(item.classification)
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
            if hasattr(img, "uri"):
                # Base64 encoded
                image_data = img.uri
            if hasattr(img, "size"):
                width, height = img.size
            if hasattr(img, "format"):
                img_format = img.format or "png"

        # Get bounding box
        bbox = None
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                bbox = BoundingBox.from_docling(prov, page_num)
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
