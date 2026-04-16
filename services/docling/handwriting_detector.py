"""
Handwriting detection for PDF documents.

Identifies regions that likely contain handwritten text by analyzing:
1. Pages with no/minimal docling output → fully handwritten (reMarkable-style)
2. Ink annotations without machine-readable text → handwritten margin notes
3. Chunks with large bbox but empty/minimal content → missed handwriting

Returns a list of regions that should be sent to a VLM for text recognition.
"""

from pathlib import Path
from typing import Optional

import fitz  # pymupdf
from loguru import logger

from ingestion.models.document import (
    BoundingBox,
    ExtractedDocument,
    PageContent,
)


def _page_image_to_bytes(pdf_path: Path, page_num: int, dpi: int = 200) -> Optional[bytes]:
    """Render a PDF page to PNG bytes."""
    try:
        doc = fitz.open(str(pdf_path))
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None
        page = doc[page_num - 1]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception as e:
        logger.warning(f"Failed to render page {page_num}: {e}")
        return None


def _crop_region_to_bytes(
    pdf_path: Path, page_num: int, bbox: BoundingBox, dpi: int = 200, padding: float = 0.02,
) -> Optional[bytes]:
    """Crop a region from a PDF page and return as PNG bytes."""
    try:
        doc = fitz.open(str(pdf_path))
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None

        page = doc[page_num - 1]
        pw, ph = page.rect.width, page.rect.height

        # bbox uses raw page coordinates (not normalized)
        x0 = max(0, bbox.x - padding * pw)
        y0 = max(0, bbox.y - padding * ph)
        x1 = min(pw, bbox.x2 + padding * pw)
        y1 = min(ph, bbox.y2 + padding * ph)

        clip = fitz.Rect(x0, y0, x1, y1)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception as e:
        logger.warning(f"Failed to crop region on page {page_num}: {e}")
        return None


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


class HandwritingRegion:
    """A region that likely contains handwritten text."""

    def __init__(
        self,
        page: int,
        bbox: Optional[BoundingBox],
        region_type: str,
        image_bytes: Optional[bytes] = None,
        context: str = "",
    ):
        """
        Args:
            page: 1-indexed page number.
            bbox: Bounding box of the region (None = full page).
            region_type: "full_page", "ink_annotation", "empty_chunk".
            image_bytes: Pre-rendered PNG of the region.
            context: Any context text (e.g., nearby content) to help VLM.
        """
        self.page = page
        self.bbox = bbox
        self.region_type = region_type
        self.image_bytes = image_bytes
        self.context = context


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


def detect_handwriting(
    document: ExtractedDocument,
    pdf_path: Path,
    unmatched_annotations: list[dict],
    min_elements_per_page: int = 2,
    min_content_length: int = 10,
    dpi: int = 200,
) -> list[HandwritingRegion]:
    """Detect regions likely containing handwritten text.

    Three detection strategies:

    1. **Full-page handwriting**: Pages where docling found fewer than
       ``min_elements_per_page`` text elements but the page is not blank
       (has ink annotations or non-zero content area). Typical for
       reMarkable notes or fully handwritten pages.

    2. **Ink annotations**: Unmatched ink/freetext annotations from the
       annotation extractor that have no machine-readable text. These are
       handwritten margin notes or drawings with text.

    3. **Empty chunks**: Elements where docling detected a region (has bbox)
       but extracted very little content (< ``min_content_length`` chars).
       This suggests OCR failed, possibly due to handwriting.

    Args:
        document: The docling-extracted document.
        pdf_path: Path to the original PDF.
        unmatched_annotations: Annotations that didn't match any chunk
            (output from annotation_extractor).
        min_elements_per_page: Pages with fewer elements are candidates.
        min_content_length: Content shorter than this in a large bbox is suspect.
        dpi: Resolution for rendering cropped regions.

    Returns:
        List of HandwritingRegion objects ready for VLM processing.
    """
    regions: list[HandwritingRegion] = []

    # --- Strategy 1: Full-page handwriting ---
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    for page_num in range(1, total_pages + 1):
        page_content = document.get_page(page_num)
        element_count = page_content.element_count if page_content else 0

        if element_count < min_elements_per_page:
            # Check if page is actually blank (no visible content at all)
            fitz_page = doc[page_num - 1]
            page_text = fitz_page.get_text().strip()
            has_annotations = len(list(fitz_page.annots())) > 0
            has_drawings = len(fitz_page.get_drawings()) > 0

            # Check if page is visually blank (scanned blank pages)
            is_visually_blank = False
            if not page_text and not has_annotations and not has_drawings:
                is_visually_blank = True
            elif not page_text and not has_annotations:
                # Has images but no text/annotations — likely a scanned blank page.
                # Use a simple heuristic: if there are only background images
                # and no drawings/ink, treat as blank.
                try:
                    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(0.25, 0.25))
                    data = pix.samples_mv
                    # Sample every 10th pixel for speed
                    n = pix.n
                    total = 0
                    white = 0
                    for i in range(0, len(data), n * 10):
                        total += 1
                        if all(data[i + c] > 235 for c in range(min(n, 3))):
                            white += 1
                    white_ratio = white / total if total > 0 else 1.0
                    if white_ratio > 0.92:
                        is_visually_blank = True
                        logger.debug(f"Page {page_num}: ~{white_ratio:.0%} white, blank")
                except Exception as e:
                    logger.debug(f"Page {page_num}: pixel check failed ({e}), assuming not blank")

            if not is_visually_blank and (page_text or has_annotations or has_drawings):
                img = _page_image_to_bytes(pdf_path, page_num, dpi)
                regions.append(HandwritingRegion(
                    page=page_num,
                    bbox=None,  # Full page
                    region_type="full_page",
                    image_bytes=img,
                    context=f"Page {page_num}: docling found {element_count} elements "
                            f"but page has visual content",
                ))
                logger.info(f"Page {page_num}: detected as handwritten (full page)")

    doc.close()

    # --- Strategy 2: Unmatched ink/freetext annotations ---
    for annot in unmatched_annotations:
        if annot["type"] in ("ink", "freetext") and not annot["has_text"]:
            img = _crop_region_to_bytes(pdf_path, annot["page"], annot["bbox"], dpi)
            regions.append(HandwritingRegion(
                page=annot["page"],
                bbox=annot["bbox"],
                region_type="ink_annotation",
                image_bytes=img,
                context=f"Ink/freetext annotation on page {annot['page']} "
                        f"without machine-readable text",
            ))
            logger.info(
                f"Page {annot['page']}: detected handwritten ink annotation"
            )

    # --- Strategy 3: Empty chunks with large bbox ---
    for page_content in document.pages:
        for elem in page_content.elements:
            if not elem.bbox:
                continue
            content_len = len(elem.content.strip())
            bbox_area = elem.bbox.area

            # Large area but minimal content → suspect handwriting
            if bbox_area > 0.01 and content_len < min_content_length:
                img = _crop_region_to_bytes(
                    pdf_path, elem.page, elem.bbox, dpi
                )
                regions.append(HandwritingRegion(
                    page=elem.page,
                    bbox=elem.bbox,
                    region_type="empty_chunk",
                    image_bytes=img,
                    context=f"Element on page {elem.page} with bbox area "
                            f"{bbox_area:.4f} but only {content_len} chars content",
                ))

    logger.info(
        f"Handwriting detection: {len(regions)} regions found "
        f"({sum(1 for r in regions if r.region_type == 'full_page')} full pages, "
        f"{sum(1 for r in regions if r.region_type == 'ink_annotation')} ink annotations, "
        f"{sum(1 for r in regions if r.region_type == 'empty_chunk')} empty chunks)"
    )

    return regions
