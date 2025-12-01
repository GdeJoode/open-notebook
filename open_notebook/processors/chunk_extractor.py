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


def extract_chunks_from_existing_document(
    doc: DoclingDocument
) -> List[Dict[str, Any]]:
    """
    Extract chunks from an existing DoclingDocument (no re-conversion needed).

    This is more efficient when you already have a DoclingDocument from processing.

    Args:
        doc: The DoclingDocument object

    Returns:
        List of chunks with spatial information

    Example:
        ```python
        from open_notebook.utils.docling_utils import reconstruct_docling_document
        from open_notebook.processors.chunk_extractor import extract_chunks_from_existing_document

        # Reconstruct document from metadata
        doc = reconstruct_docling_document(state.metadata)
        if doc:
            chunks = extract_chunks_from_existing_document(doc)
        ```
    """
    if not DOCLING_AVAILABLE:
        raise ImportError(
            "Docling not installed. Install with: pip install content-core[docling]"
        )

    # Extract chunks with positions (ConversionResult not required for basic extraction)
    chunks = extract_chunks_with_positions(doc, result=None)
    return chunks


def extract_chunks_with_positions(
    doc: DoclingDocument,
    result: Optional[ConversionResult] = None
) -> List[Dict[str, Any]]:
    """
    Extract text chunks with bounding box positions from Docling document.

    Args:
        doc: The DoclingDocument object
        result: Optional ConversionResult containing additional metadata

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
    # VERY VISIBLE LOG - confirms this function is being called with updated code
    logger.warning("=" * 60)
    logger.warning("extract_chunks_with_positions CALLED - VERSION 2024-11-30")
    logger.warning("=" * 60)

    chunks = []
    current_chapter = None
    paragraph_counter = {}  # Track paragraph numbers per page

    # Log page information once at the start
    if hasattr(doc, 'pages') and doc.pages:
        logger.info(f"Document has {len(doc.pages)} pages with dimensions:")
        for page_no, page_item in sorted(doc.pages.items())[:3]:  # Show first 3 pages
            if hasattr(page_item, 'size') and page_item.size:
                logger.info(f"  Page {page_no}: {page_item.size.width:.1f} x {page_item.size.height:.1f} points")
            else:
                logger.info(f"  Page {page_no}: No size information")
    else:
        logger.warning("Document has no page dimension information - using A4 defaults (595x842)")

    # Iterate through document items in reading order
    for idx, item_data in enumerate(doc.iterate_items()):
        # Handle both single items and (item, page_num) tuples
        if isinstance(item_data, tuple):
            item, _ = item_data
        else:
            item = item_data

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
            # For other item types, try to get text attribute first
            text = getattr(item, 'text', None)
            if not text:
                # Skip items without text content
                logger.debug(f"Skipping item without text: {type(item).__name__}")
                continue
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
                    bbox = prov.bbox

                    # Get page dimensions - doc.pages is Dict[int, PageItem] keyed by 1-based page number
                    page_width = 595.0   # A4 default width in points
                    page_height = 842.0  # A4 default height in points

                    # doc.pages is a dictionary keyed by page_no (1-based)
                    if hasattr(doc, 'pages') and doc.pages and page_no in doc.pages:
                        page_item = doc.pages[page_no]
                        if hasattr(page_item, 'size') and page_item.size:
                            page_width = page_item.size.width if hasattr(page_item.size, 'width') else page_width
                            page_height = page_item.size.height if hasattr(page_item.size, 'height') else page_height

                    # Get raw bbox coordinates
                    left = float(bbox.l)
                    right = float(bbox.r)
                    bbox_t = float(bbox.t)  # "top" edge y-coordinate
                    bbox_b = float(bbox.b)  # "bottom" edge y-coordinate

                    # Determine coordinate system
                    coord_origin = 'BOTTOMLEFT'  # Default assumption for PDFs
                    if hasattr(bbox, 'coord_origin'):
                        coord_origin = str(bbox.coord_origin).upper()

                    # Convert to TOPLEFT coordinates (y=0 at page top, y increases downward)
                    if 'BOTTOMLEFT' in coord_origin:
                        # In BOTTOMLEFT: y=0 at page bottom, y increases upward
                        # bbox.t is the TOP edge (higher y value, visually at top of box)
                        # bbox.b is the BOTTOM edge (lower y value, visually at bottom of box)
                        #
                        # To convert to TOPLEFT:
                        # - A point at y=page_height (top of page in BOTTOMLEFT) should become y=0
                        # - A point at y=0 (bottom of page in BOTTOMLEFT) should become y=page_height
                        # Formula: new_y = page_height - old_y
                        #
                        # The TOP edge of the box (visually higher) should have SMALLER y in TOPLEFT
                        # The BOTTOM edge of the box (visually lower) should have LARGER y in TOPLEFT
                        top_edge_topleft = page_height - bbox_t  # Visual top -> smaller y
                        bottom_edge_topleft = page_height - bbox_b  # Visual bottom -> larger y
                    else:
                        # Already TOPLEFT
                        top_edge_topleft = bbox_t
                        bottom_edge_topleft = bbox_b

                    # Normalize to 0-1 range
                    x_left = left / page_width
                    x_right = right / page_width

                    # Ensure y_top < y_bottom (defensive, should already be correct)
                    y_top = min(top_edge_topleft, bottom_edge_topleft) / page_height
                    y_bottom = max(top_edge_topleft, bottom_edge_topleft) / page_height

                    # Clamp values to 0-1 range
                    x_left = max(0.0, min(1.0, x_left))
                    x_right = max(0.0, min(1.0, x_right))
                    y_top = max(0.0, min(1.0, y_top))
                    y_bottom = max(0.0, min(1.0, y_bottom))

                    # Debug: Log coordinate conversion for first few chunks
                    if idx < 5 or (page_no == 22):  # Log first 5 chunks and all on page 22
                        logger.info(
                            f"Chunk {idx}: page={page_no}, origin={coord_origin}, "
                            f"raw_bbox(l={left:.1f}, r={right:.1f}, t={bbox_t:.1f}, b={bbox_b:.1f}), "
                            f"pageSize=({page_width:.1f}x{page_height:.1f}), "
                            f"→ TOPLEFT y_top={y_top:.3f} ({y_top*100:.1f}%), y_bottom={y_bottom:.3f} ({y_bottom*100:.1f}%)"
                        )

                    # Format: [page_number, x_left, x_right, y_top, y_bottom] normalized 0-1
                    positions.append([
                        page_no,
                        x_left,
                        x_right,
                        y_top,
                        y_bottom,
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
        # For now, use physical_page directly (Docling uses 1-based page numbers)
        printed_page = physical_page

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
