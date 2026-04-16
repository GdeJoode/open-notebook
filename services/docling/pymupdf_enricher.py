"""
PyMuPDF text enrichment for Docling documents.

Uses PyMuPDF to extract clean text from the PDF text layer and replaces
Docling element content where a reliable match is found. This gives us:
- Cleaner text (no OCR artifacts on native PDFs)
- A mapping from PyMuPDF text blocks to Docling elements (for annotation matching)
- Detection of which pages have a text layer (vs pure scanned images)

Docling provides the structure (bboxes, element types, sections, tables, images).
PyMuPDF provides the authoritative text content.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz
from loguru import logger

from ingestion.models.document import (
    BoundingBox,
    ElementType,
    ExtractedDocument,
    ExtractedElement,
    PageContent,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    """A PyMuPDF text block with page coordinates and font info."""

    text: str
    page: int  # 1-indexed
    x0: float
    y0: float
    x1: float
    y1: float
    # Font characteristics (dominant span in the block)
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False
    font_name: str = ""
    # After matching, which Docling element this maps to
    element_idx: int = -1

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class PageTextInfo:
    """PyMuPDF text extraction info for one page."""

    page_num: int  # 1-indexed
    has_text_layer: bool
    blocks: list[TextBlock] = field(default_factory=list)
    page_height: float = 0.0


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xad", "").replace("­", "")
    text = re.sub(r"-\s+", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# PyMuPDF text extraction
# ---------------------------------------------------------------------------


def _extract_text_blocks(pdf_path: Path) -> dict[int, PageTextInfo]:
    """Extract text blocks from every page using PyMuPDF.

    Returns a dict of page_num → PageTextInfo.
    """
    doc = fitz.open(str(pdf_path))
    pages: dict[int, PageTextInfo] = {}

    for fitz_page in doc:
        page_num = fitz_page.number + 1
        ph = fitz_page.rect.height

        td = fitz_page.get_text("dict")
        blocks: list[TextBlock] = []

        for block in td.get("blocks", []):
            if block.get("type", 0) != 0:  # Only text blocks (not images)
                continue
            lines = block.get("lines", [])
            text = " ".join(
                "".join(span["text"] for span in line.get("spans", []))
                for line in lines
            ).strip()
            if not text:
                continue
            bbox = block["bbox"]  # (x0, y0, x1, y1) TOPLEFT origin

            # Extract dominant font characteristics (longest span wins)
            font_size = 0.0
            is_bold = False
            is_italic = False
            font_name = ""
            max_span_len = 0
            for line in lines:
                for span in line.get("spans", []):
                    span_len = len(span.get("text", ""))
                    if span_len > max_span_len:
                        max_span_len = span_len
                        font_size = span.get("size", 0.0)
                        flags = span.get("flags", 0)
                        is_bold = bool(flags & 16)
                        is_italic = bool(flags & 2)
                        font_name = span.get("font", "")

            blocks.append(TextBlock(
                text=text,
                page=page_num,
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
                font_size=font_size,
                is_bold=is_bold,
                is_italic=is_italic,
                font_name=font_name,
            ))

        # A page has a text layer if we found meaningful text blocks
        total_text = " ".join(b.text for b in blocks)
        has_text_layer = len(total_text.strip()) > 20

        pages[page_num] = PageTextInfo(
            page_num=page_num,
            has_text_layer=has_text_layer,
            blocks=blocks,
            page_height=ph,
        )

    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Coordinate matching: PyMuPDF blocks ↔ Docling elements
# ---------------------------------------------------------------------------


def _pymupdf_y_to_docling_y(pymupdf_y: float, page_height: float) -> float:
    """Convert PyMuPDF TOPLEFT y-coordinate to Docling's coordinate space.

    Docling stores y = -(page_height - pymupdf_y_top).
    """
    return -(page_height - pymupdf_y)


def _vertical_overlap(
    block: TextBlock,
    elem_y: float,
    elem_height: float,
    page_height: float,
) -> float:
    """Calculate vertical overlap between a PyMuPDF block and a Docling element.

    Converts Docling coordinates to PyMuPDF space for comparison.
    Returns overlap ratio relative to the smaller height.
    """
    # Convert Docling y back to PyMuPDF TOPLEFT space
    elem_y_top = page_height + elem_y  # elem_y is negative
    elem_y_bottom = elem_y_top + elem_height

    # PyMuPDF block
    block_y_top = block.y0
    block_y_bottom = block.y1

    # Overlap
    overlap_top = max(block_y_top, elem_y_top)
    overlap_bottom = min(block_y_bottom, elem_y_bottom)

    if overlap_bottom <= overlap_top:
        return 0.0

    overlap_height = overlap_bottom - overlap_top
    min_height = min(block.height, elem_height)
    if min_height <= 0:
        return 0.0

    return overlap_height / min_height


def _horizontal_overlap(block: TextBlock, elem_x: float, elem_width: float) -> float:
    """Calculate horizontal overlap ratio."""
    block_x0, block_x1 = block.x0, block.x1
    elem_x0, elem_x1 = elem_x, elem_x + elem_width

    overlap_left = max(block_x0, elem_x0)
    overlap_right = min(block_x1, elem_x1)

    if overlap_right <= overlap_left:
        return 0.0

    overlap_width = overlap_right - overlap_left
    min_width = min(block.width, elem_width)
    if min_width <= 0:
        return 0.0

    return overlap_width / min_width


def _token_overlap(text_a: str, text_b: str) -> float:
    """Word-level overlap ratio."""
    words_a = set(_normalize(text_a).split())
    words_b = set(_normalize(text_b).split())
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


# ---------------------------------------------------------------------------
# Main enrichment
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Element type classification from font characteristics
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(
    r"^\s*(?:[-•·▪▸►◦‣⁃]|\(\d+\)|\d+[.)]\s|[a-zA-Z][.)]\s|[ivxlIVXL]+[.)]\s)"
)


def _classify_blocks(
    blocks: list[TextBlock],
) -> list[tuple[TextBlock, ElementType]]:
    """Classify element types from font characteristics.

    Strategy:
    1. Compute median font size across all blocks → body text baseline
    2. Blocks significantly larger or bold → HEADING or TITLE
    3. ALL CAPS short text → HEADING
    4. Text starting with bullet/number pattern → LIST_ITEM
    5. Short italic text at top/bottom of page → skip (running header/footer)
    6. Everything else → TEXT
    """
    if not blocks:
        return []

    # Compute median body font size
    sizes = [b.font_size for b in blocks if b.font_size > 0 and len(b.text) > 20]
    if not sizes:
        sizes = [b.font_size for b in blocks if b.font_size > 0]
    if not sizes:
        return [(b, ElementType.TEXT) for b in blocks]

    sizes.sort()
    median_size = sizes[len(sizes) // 2]

    result: list[tuple[TextBlock, ElementType]] = []

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue

        size_ratio = block.font_size / median_size if median_size > 0 else 1.0

        # Skip likely page numbers (very short, different font, at page edges)
        if len(text) <= 3 and text.isdigit():
            continue

        # Skip running headers/footers (short italic text)
        if block.is_italic and len(text) < 60 and size_ratio < 1.05:
            continue

        # TITLE: significantly larger (>1.5x median)
        if size_ratio > 1.5 and len(text) < 200:
            result.append((block, ElementType.TITLE))
            continue

        # HEADING: larger than body (>1.1x) or bold, and relatively short
        if (size_ratio > 1.1 or block.is_bold) and len(text) < 200:
            result.append((block, ElementType.HEADING))
            continue

        # HEADING: ALL CAPS short text (common for sub-headings in books)
        if text == text.upper() and len(text) > 5 and len(text) < 150 and text[0].isalpha():
            result.append((block, ElementType.HEADING))
            continue

        # LIST_ITEM: starts with bullet or number pattern
        if _BULLET_RE.match(text):
            result.append((block, ElementType.LIST_ITEM))
            continue

        # Default: body text
        result.append((block, ElementType.TEXT))

    return result


def _inherit_section_from_neighbors(
    document: ExtractedDocument,
    page_num: int,
) -> list[str]:
    """Inherit section_path from the last element on the nearest prior page.

    Looks backwards from page_num to find a page with Docling elements
    that have a section_path. Returns that path so new elements on this
    page start within the same section context.
    """
    # Look at pages before this one (up to 5 pages back)
    for offset in range(1, 6):
        prev_page = document.get_page(page_num - offset)
        if not prev_page or not prev_page.elements:
            continue
        # Take section_path from the last element on that page
        for elem in reversed(prev_page.elements):
            if elem.section_path:
                return list(elem.section_path)
    return []


# ---------------------------------------------------------------------------
# Main enrichment
# ---------------------------------------------------------------------------


def enrich_with_pymupdf_text(
    document: ExtractedDocument,
    pdf_path: Path,
    position_threshold: float = 0.3,
    text_confirm_threshold: float = 0.4,
    text_replace_threshold: float = 0.6,
) -> tuple[ExtractedDocument, dict[int, PageTextInfo]]:
    """Enrich Docling document with PyMuPDF text.

    For each Docling element (except tables and images), finds the matching
    PyMuPDF text block(s) by page position and text similarity. If a
    reliable match is found, replaces the Docling content with PyMuPDF text.

    Args:
        document: Docling-extracted document.
        pdf_path: Path to the original PDF.
        position_threshold: Minimum vertical overlap to consider a positional match.
        text_confirm_threshold: Minimum token overlap to confirm a match.
        text_replace_threshold: Minimum token overlap to replace content
            (below this, we keep Docling text as it might be from a different region).

    Returns:
        Tuple of (enriched document, page text info dict for annotation matching).
    """
    page_texts = _extract_text_blocks(pdf_path)

    replaced = 0
    kept_docling = 0
    no_text_layer = 0

    for page_content in document.pages:
        page_info = page_texts.get(page_content.page_number)
        if not page_info or not page_info.has_text_layer:
            no_text_layer += 1
            continue

        for elem_idx, elem in enumerate(page_content.elements):
            # Skip tables and images — Docling is better for these
            if elem.element_type in (ElementType.TABLE, ElementType.IMAGE):
                continue

            if not elem.bbox:
                continue

            # Find matching PyMuPDF blocks by position
            matching_blocks: list[TextBlock] = []
            for block in page_info.blocks:
                v_overlap = _vertical_overlap(
                    block, elem.bbox.y, elem.bbox.height, page_info.page_height
                )
                if v_overlap < position_threshold:
                    continue

                h_overlap = _horizontal_overlap(block, elem.bbox.x, elem.bbox.width)
                if h_overlap < position_threshold:
                    continue

                matching_blocks.append(block)

            if not matching_blocks:
                # No positional match — try text-based fallback
                best_block = None
                best_score = 0.0
                for block in page_info.blocks:
                    score = _token_overlap(elem.content, block.text)
                    if score > best_score:
                        best_score = score
                        best_block = block
                if best_block and best_score >= text_confirm_threshold:
                    matching_blocks = [best_block]

            if not matching_blocks:
                kept_docling += 1
                continue

            # Combine text from all matching blocks (reading order: top to bottom)
            matching_blocks.sort(key=lambda b: b.y0)
            pymupdf_text = " ".join(b.text for b in matching_blocks).strip()

            if not pymupdf_text:
                kept_docling += 1
                continue

            # Decide whether to replace
            overlap = _token_overlap(pymupdf_text, elem.content)

            if overlap >= text_replace_threshold:
                # Good match — replace with cleaner PyMuPDF text
                elem.content = pymupdf_text
                replaced += 1
            elif overlap >= text_confirm_threshold:
                # Partial match — replace only if PyMuPDF text is longer
                # (Docling may have truncated or missed parts)
                if len(pymupdf_text) > len(elem.content) * 1.1:
                    elem.content = pymupdf_text
                    replaced += 1
                else:
                    kept_docling += 1
            else:
                # Low overlap — keep Docling (might be different region)
                kept_docling += 1

            # Record mapping for annotation matching
            for block in matching_blocks:
                block.element_idx = elem_idx

    # --- Fill in missing pages/elements from PyMuPDF ---
    # For pages where Docling found few/no elements but PyMuPDF has text,
    # create new elements from PyMuPDF blocks.
    created = 0
    for page_num, page_info in page_texts.items():
        if not page_info.has_text_layer:
            continue

        page_content = document.get_page(page_num)

        # Check if page is missing or has very few elements
        existing_text_elems = 0
        if page_content:
            existing_text_elems = sum(
                1 for e in page_content.elements
                if e.element_type not in (ElementType.TABLE, ElementType.IMAGE)
            )

        if existing_text_elems >= 2:
            # Page has enough Docling elements — also map any unmatched blocks
            # to their nearest element for annotation matching
            for block in page_info.blocks:
                if block.element_idx >= 0:
                    continue  # Already mapped
                # Find nearest element by text overlap
                best_idx = -1
                best_score = 0.0
                for idx, elem in enumerate(page_content.elements):
                    score = _token_overlap(block.text, elem.content)
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_score >= 0.3 and best_idx >= 0:
                    block.element_idx = best_idx
            continue

        # Page missing or sparse — create structured elements from PyMuPDF blocks
        if not page_content:
            page_content = PageContent(page_number=page_num)
            document.pages.append(page_content)
            document.pages.sort(key=lambda p: p.page_number)

        # Step A: Classify each block's element type from font info
        classified_blocks = _classify_blocks(page_info.blocks)

        # Step B: Inherit section context from nearest Docling page
        inherited_section = _inherit_section_from_neighbors(document, page_num)

        # Step C: Build heading hierarchy and assign section paths
        heading_stack: list[tuple[str, int]] = []  # (title, level)
        if inherited_section:
            heading_stack = [(s, i + 1) for i, s in enumerate(inherited_section)]

        for block, elem_type in classified_blocks:
            if len(block.text.strip()) < 3:
                continue

            # Update heading stack if this block is a heading
            if elem_type in (ElementType.TITLE, ElementType.HEADING):
                level = 1 if elem_type == ElementType.TITLE else 2
                # Pop headings at same or deeper level
                while heading_stack and heading_stack[-1][1] >= level:
                    heading_stack.pop()
                heading_stack.append((block.text.strip(), level))

            section_path = [h[0] for h in heading_stack] if heading_stack else (
                inherited_section if inherited_section else []
            )
            section_level = len(heading_stack)

            docling_y = _pymupdf_y_to_docling_y(block.y0, page_info.page_height)
            bbox = BoundingBox(
                x=block.x0,
                y=docling_y,
                width=block.width,
                height=block.height,
                page=page_num,
            )
            new_elem = ExtractedElement(
                content=block.text,
                element_type=elem_type,
                page=page_num,
                bbox=bbox,
                confidence=0.9,
                source="pymupdf",
                section_path=section_path,
                section_level=section_level,
            )
            elem_idx = len(page_content.elements)
            page_content.elements.append(new_elem)
            block.element_idx = elem_idx
            created += 1

        # Update page text_content
        page_content.text_content = " ".join(
            e.content for e in page_content.elements if e.content
        )

    pages_with_text = sum(1 for p in page_texts.values() if p.has_text_layer)
    logger.info(
        f"PyMuPDF enrichment: {replaced} elements replaced, "
        f"{kept_docling} kept Docling text, "
        f"{created} new elements created from PyMuPDF, "
        f"{no_text_layer} pages without text layer, "
        f"{pages_with_text} pages with text layer"
    )

    return document, page_texts
