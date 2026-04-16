"""
PDF annotation extraction and chunk matching via PyMuPDF.

Extracts highlights, underlines, strikeouts, sticky notes, ink, and freetext
annotations. Matches annotations to Docling elements using PyMuPDF text blocks
as the bridge — both annotations and text blocks live in the same PyMuPDF
coordinate space, so matching is reliable.

When a single annotation spans multiple Docling elements, those elements are merged.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import fitz
from loguru import logger

from ingestion.models.document import (
    AnnotationInfo,
    BoundingBox,
    ExtractedDocument,
    ExtractedElement,
    PageContent,
)
from pymupdf_enricher import PageTextInfo, TextBlock

# ---------------------------------------------------------------------------
# PyMuPDF annotation type mapping
# ---------------------------------------------------------------------------

ANNOT_TYPE_MAP = {
    0: "text",           # Sticky note
    8: "highlight",
    9: "underline",
    10: "squiggly",
    11: "strikeout",
    15: "ink",           # Freehand drawing / handwritten
    19: "freetext",      # Typed text annotation
}

_COLOR_NAMES = {
    (1.0, 1.0, 0.0): "yellow",
    (1.0, 0.9, 0.0): "yellow",
    (0.0, 1.0, 0.0): "green",
    (0.0, 0.8, 0.0): "green",
    (0.0, 0.5, 0.0): "green",
    (1.0, 0.0, 0.0): "red",
    (0.8, 0.0, 0.0): "red",
    (0.0, 0.0, 1.0): "blue",
    (0.0, 0.0, 0.8): "blue",
    (0.3, 0.5, 1.0): "blue",
    (1.0, 0.6, 0.8): "pink",
    (1.0, 0.4, 0.7): "pink",
    (1.0, 0.75, 0.8): "pink",
    (1.0, 0.5, 0.0): "orange",
    (1.0, 0.65, 0.0): "orange",
    (0.5, 0.0, 0.5): "purple",
    (0.6, 0.2, 0.8): "purple",
    (0.8, 0.3, 1.0): "purple",
    (0.5, 0.5, 0.5): "gray",
}


def _rgb_to_color_name(rgb: tuple) -> str:
    """Map an RGB tuple to the nearest named color."""
    if not rgb or len(rgb) < 3:
        return "unknown"
    r, g, b = round(rgb[0], 1), round(rgb[1], 1), round(rgb[2], 1)
    if (r, g, b) in _COLOR_NAMES:
        return _COLOR_NAMES[(r, g, b)]
    best_name = "unknown"
    best_dist = float("inf")
    for (cr, cg, cb), name in _COLOR_NAMES.items():
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xad", "").replace("­", "")
    text = re.sub(r"-\s+", "", text)
    return text.strip()


def _rect_overlap(r1: fitz.Rect, r2_x0, r2_y0, r2_x1, r2_y1) -> float:
    """Overlap ratio between a fitz.Rect and a raw rect (same coordinate space)."""
    ix0 = max(r1.x0, r2_x0)
    iy0 = max(r1.y0, r2_y0)
    ix1 = min(r1.x1, r2_x1)
    iy1 = min(r1.y1, r2_y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area1 = r1.width * r1.height
    area2 = (r2_x1 - r2_x0) * (r2_y1 - r2_y0)
    min_area = min(area1, area2)
    if min_area <= 0:
        return 0.0
    return intersection / min_area


# ---------------------------------------------------------------------------
# Annotation extraction
# ---------------------------------------------------------------------------


def _extract_annotations(pdf_path: Path) -> dict[int, list[dict]]:
    """Extract annotations grouped by page number (1-indexed)."""
    doc = fitz.open(str(pdf_path))
    by_page: dict[int, list[dict]] = defaultdict(list)

    for fitz_page in doc:
        page_num = fitz_page.number + 1
        for annot in fitz_page.annots():
            atype = annot.type[0]
            type_name = ANNOT_TYPE_MAP.get(atype)
            if not type_name:
                continue

            colors = annot.colors
            rgb = colors.get("stroke") or colors.get("fill") or ()
            color_name = _rgb_to_color_name(rgb) if rgb else "unknown"
            comment = (annot.info.get("content") or "").strip()

            # Get highlighted text from the page text layer
            highlighted_text = ""
            if type_name in ("highlight", "underline", "squiggly", "strikeout"):
                highlighted_text = fitz_page.get_textbox(annot.rect).strip()

            by_page[page_num].append({
                "type": type_name,
                "color": list(rgb) if rgb else [],
                "color_name": color_name,
                "comment": comment,
                "highlighted_text": highlighted_text,
                "rect": annot.rect,  # fitz.Rect in PyMuPDF coordinates
                "page": page_num,
            })

    total = sum(len(v) for v in by_page.values())
    doc.close()
    logger.info(f"Extracted {total} annotations from {pdf_path.name}")
    return dict(by_page)


# ---------------------------------------------------------------------------
# Annotation → Docling element matching via PyMuPDF text blocks
# ---------------------------------------------------------------------------


def _find_element_for_annotation(
    annot: dict,
    page_info: PageTextInfo,
    page_content: PageContent,
) -> list[int]:
    """Find Docling element indices that match an annotation.

    Strategy:
    1. Find PyMuPDF text blocks that overlap with the annotation rect
       (same coordinate space — reliable)
    2. Follow the block → element mapping to find the Docling element
    3. If no mapped block found, fall back to text matching against elements

    Returns list of element indices (may be multiple if annotation spans chunks).
    """
    annot_rect = annot["rect"]

    # Step 1: Find overlapping PyMuPDF text blocks
    matched_element_indices = set()
    for block in page_info.blocks:
        overlap = _rect_overlap(annot_rect, block.x0, block.y0, block.x1, block.y1)
        if overlap >= 0.3:
            if block.element_idx >= 0 and block.element_idx < len(page_content.elements):
                matched_element_indices.add(block.element_idx)

    if matched_element_indices:
        return sorted(matched_element_indices)

    # Step 2: Fall back to text matching on highlighted text
    if annot["highlighted_text"]:
        hl_norm = _normalize(annot["highlighted_text"])
        hl_words = set(hl_norm.split())
        if not hl_words:
            return []

        best_idx = -1
        best_score = 0.0
        for idx, elem in enumerate(page_content.elements):
            if not elem.content:
                continue
            elem_words = set(_normalize(elem.content).split())
            overlap = len(hl_words & elem_words) / len(hl_words)
            if overlap > best_score:
                best_score = overlap
                best_idx = idx

        if best_score >= 0.3 and best_idx >= 0:
            return [best_idx]

    return []


def _merge_elements(elements: list[ExtractedElement]) -> ExtractedElement:
    """Merge multiple elements into one."""
    if len(elements) == 1:
        return elements[0]

    sorted_elems = sorted(elements, key=lambda e: (e.bbox.y if e.bbox else 0))
    combined = "\n".join(e.content for e in sorted_elems if e.content)

    bbox = sorted_elems[0].bbox
    for elem in sorted_elems[1:]:
        if elem.bbox and bbox:
            bbox = BoundingBox.union(bbox, elem.bbox)

    merged = ExtractedElement(
        content=combined,
        element_type=sorted_elems[0].element_type,
        page=sorted_elems[0].page,
        bbox=bbox,
        confidence=min(e.confidence for e in sorted_elems),
        metadata=sorted_elems[0].metadata,
        section_path=sorted_elems[0].section_path,
        section_level=sorted_elems[0].section_level,
        annotations=[a for e in sorted_elems for a in e.annotations],
        source=sorted_elems[0].source,
    )
    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def enrich_document_with_annotations(
    document: ExtractedDocument,
    pdf_path: Path,
    page_texts: dict[int, PageTextInfo],
) -> tuple[ExtractedDocument, list[dict]]:
    """Extract annotations and attach them to Docling elements.

    Uses the PyMuPDF text block → element mapping from pymupdf_enricher
    for reliable coordinate-space matching.

    Args:
        document: Docling document to enrich.
        pdf_path: Path to the original PDF.
        page_texts: PyMuPDF page text info (from pymupdf_enricher).

    Returns:
        Tuple of (enriched document, list of unmatched annotations).
    """
    annots_by_page = _extract_annotations(pdf_path)
    if not annots_by_page:
        return document, []

    unmatched: list[dict] = []

    for page_num, page_annots in annots_by_page.items():
        page_content = document.get_page(page_num)
        page_info = page_texts.get(page_num)

        if not page_content:
            unmatched.extend(page_annots)
            continue

        if not page_info:
            # No PyMuPDF info — fall back to pure text matching
            page_info = PageTextInfo(page_num=page_num, has_text_layer=False)

        # Phase 1: Find matches for all annotations (without modifying elements)
        single_matches: list[tuple[dict, int]] = []
        merge_groups: dict[tuple, list[dict]] = defaultdict(list)

        for annot in page_annots:
            indices = _find_element_for_annotation(annot, page_info, page_content)
            if not indices:
                unmatched.append(annot)
            elif len(indices) == 1:
                single_matches.append((annot, indices[0]))
            else:
                merge_groups[tuple(sorted(indices))].append(annot)

        # Phase 2: Apply merges
        for indices_key, annots in merge_groups.items():
            indices = list(indices_key)
            # Validate indices still in range
            if any(i >= len(page_content.elements) for i in indices):
                unmatched.extend(annots)
                continue
            matched_elements = [page_content.elements[i] for i in indices]
            merged = _merge_elements(matched_elements)
            for annot in annots:
                merged.annotations.append(AnnotationInfo(
                    type=annot["type"], color=annot["color"],
                    color_name=annot["color_name"], comment=annot["comment"],
                    highlighted_text=annot["highlighted_text"],
                ))
            for idx in sorted(indices, reverse=True):
                page_content.elements.pop(idx)
            page_content.elements.insert(min(indices), merged)

        # Phase 3: Apply single-chunk annotations (re-find after merges)
        for annot, _orig_idx in single_matches:
            annotation_info = AnnotationInfo(
                type=annot["type"], color=annot["color"],
                color_name=annot["color_name"], comment=annot["comment"],
                highlighted_text=annot["highlighted_text"],
            )
            # Re-find element by text match (indices may have shifted after merges)
            attached = False
            if annot["highlighted_text"]:
                hl_norm = _normalize(annot["highlighted_text"])
                hl_words = set(hl_norm.split())
                for elem in page_content.elements:
                    if not elem.content:
                        continue
                    elem_words = set(_normalize(elem.content).split())
                    if hl_words and len(hl_words & elem_words) / len(hl_words) >= 0.3:
                        elem.annotations.append(annotation_info)
                        attached = True
                        break
            if not attached:
                unmatched.append(annot)

    total = sum(len(v) for v in annots_by_page.values())
    matched = total - len(unmatched)
    logger.info(f"Annotations: {matched}/{total} matched ({matched/total:.0%}), "
                f"{len(unmatched)} unmatched")

    return document, unmatched
