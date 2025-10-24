"""
Extract chunks with bounding box positions from Docling documents.
"""

from typing import Any, Dict, List, Optional
from loguru import logger

try:
    from docling.document_converter import DocumentConverter, ConversionResult
    from docling_core.types.doc import DoclingDocument, TextItem, TableItem, PictureItem
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not available. Chunk extraction will be disabled.")


def extract_chunks_from_docling(
    source_path: str,
    output_format: str = "markdown"
) -> tuple[str, List[Dict[str, Any]], Optional[ConversionResult]]:
    """
    Extract both content and chunks with spatial information from a document using Docling.

    Args:
        source_path: Path to the document file or URL
        output_format: Output format for content (markdown, html, json)

    Returns:
        Tuple of (content_string, chunks_list, conversion_result)
        - content_string: The extracted content in the specified format
        - chunks_list: List of chunk dictionaries with bounding boxes
        - conversion_result: The full Docling ConversionResult for reference

    Raises:
        ImportError: If Docling is not installed
        ValueError: If source_path is invalid
    """
    if not DOCLING_AVAILABLE:
        raise ImportError(
            "Docling not installed. Install with: pip install content-core[docling]"
        )

    if not source_path:
        raise ValueError("No source path provided for chunk extraction")

    # Initialize Docling converter and convert document
    converter = DocumentConverter()
    result: ConversionResult = converter.convert(source_path)
    doc: DoclingDocument = result.document

    # Extract content in the desired format
    if output_format == "html":
        content = doc.export_to_html()
    elif output_format == "json":
        content = doc.export_to_json()
    else:
        content = doc.export_to_markdown()

    # Extract chunks with positions
    chunks = extract_chunks_with_positions(doc, result)

    return content, chunks, result


def extract_chunks_with_positions(
    doc: DoclingDocument,
    result: ConversionResult
) -> List[Dict[str, Any]]:
    """
    Extract text chunks with bounding box positions from Docling document.

    Args:
        doc: The DoclingDocument object
        result: The ConversionResult containing additional metadata

    Returns:
        List of chunks with format:
        {
            'text': str,
            'order': int,
            'physical_page': int,
            'printed_page': int | None,
            'chapter': str | None,
            'paragraph_number': int | None,
            'element_type': str,
            'positions': [[page_num, x1, x2, y1, y2], ...],
            'metadata': dict
        }
    """
    chunks = []
    current_chapter = None
    paragraph_counter = {}  # Track paragraph numbers per page

    # Iterate through document items in reading order
    for idx, item in enumerate(doc.iterate_items()):
        # Extract text content
        text = None
        element_type = "unknown"

        if isinstance(item, TextItem):
            text = item.text
            element_type = str(item.label) if hasattr(item, 'label') else 'text'

            # Track section headers for chapter context
            if element_type in ['section_header', 'title', 'heading']:
                current_chapter = text

        elif isinstance(item, TableItem):
            # For tables, export as markdown
            try:
                table_df = item.export_to_dataframe()
                text = table_df.to_markdown() if not table_df.empty else str(item)
                element_type = 'table'
            except Exception as e:
                logger.warning(f"Failed to export table to markdown: {e}")
                text = str(item)
                element_type = 'table'

        elif isinstance(item, PictureItem):
            # For pictures, use caption or description
            text = item.caption if hasattr(item, 'caption') else f"[Picture {idx}]"
            element_type = 'picture'

        else:
            # For other item types, try to get text representation
            text = getattr(item, 'text', None) or str(item)
            element_type = str(type(item).__name__)

        # Skip empty chunks
        if not text or not text.strip():
            continue

        # Extract bounding boxes from provenance
        positions = []
        physical_page = 0
        printed_page = None

        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'bbox') and hasattr(prov, 'page_no'):
                    page_no = prov.page_no
                    physical_page = page_no  # Use first page as primary page

                    # Get page size for normalization
                    if hasattr(result, 'document') and hasattr(result.document, 'pages'):
                        pages = result.document.pages
                        if page_no < len(pages):
                            page = pages[page_no]
                            bbox = prov.bbox

                            # Convert to top-left origin if needed
                            if hasattr(bbox, 'to_top_left_origin') and hasattr(page, 'size'):
                                bbox = bbox.to_top_left_origin(page_height=page.size.height)

                            # Normalize coordinates (0-1 range)
                            if hasattr(bbox, 'normalized') and hasattr(page, 'size'):
                                bbox = bbox.normalized(page.size)

                            # Extract coordinates
                            # Format: [page_number, x1, x2, y1, y2]
                            positions.append([
                                page_no,
                                float(bbox.l),  # x1 (left)
                                float(bbox.r),  # x2 (right)
                                float(bbox.t),  # y1 (top)
                                float(bbox.b),  # y2 (bottom)
                            ])

        # Track paragraph numbers per page
        if physical_page not in paragraph_counter:
            paragraph_counter[physical_page] = 0
        if element_type in ['paragraph', 'text']:
            paragraph_counter[physical_page] += 1
            paragraph_number = paragraph_counter[physical_page]
        else:
            paragraph_number = None

        # Try to extract printed page number from page labels if available
        # This is a placeholder - actual implementation would need PDF metadata
        # For now, fallback to physical_page + 1
        printed_page = physical_page + 1

        # Build chunk dictionary
        chunk = {
            'text': text,
            'order': idx,
            'physical_page': physical_page,
            'printed_page': printed_page,
            'chapter': current_chapter,
            'paragraph_number': paragraph_number,
            'element_type': element_type,
            'positions': positions,
            'metadata': {
                'has_spatial_data': len(positions) > 0,
                'num_locations': len(positions),
                'item_type': type(item).__name__,
            }
        }

        chunks.append(chunk)

    logger.info(f"Extracted {len(chunks)} chunks with spatial information")
    return chunks
