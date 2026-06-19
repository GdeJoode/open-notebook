"""
MinerU layout-to-ExtractedDocument translation.

MinerU 2.x writes three structured-data files per input document:

- ``<stem>.md`` — primary markdown rendering
- ``<stem>_content_list.json`` — flat reading-order list of elements
  with per-element ``bbox``, ``page_idx``, ``type``, etc.
- ``<stem>_middle.json`` — nested block/line/span tree (carries the
  ``_backend`` discriminator we need for bbox normalisation)

This parser ingests the two JSON files plus the markdown and produces
an ``ExtractedDocument`` whose ``ExtractedElement`` instances carry
populated ``BoundingBox`` data — required for ``PdfChunkViewer``
compatibility per Track A's Q-A-5 resolution.

Schema reference: ``docs/tracks/A-mineru/MINERU_OUTPUT_SPIKE.md``.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

from ingestion.models import (
    BoundingBox,
    ElementType,
    ExtractedDocument,
    ExtractedElement,
    ExtractedImage,
    ExtractedTable,
    PageContent,
)


class MineruLayoutParseError(RuntimeError):
    """Raised when the MinerU output cannot be reconstructed into an ExtractedDocument."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MinerU normalises bboxes to a 0-1000 range for the pipeline/office
# backends and to a 0-1 range for the vlm backend. We re-normalise to
# 0-1 to stay consistent with the rest of the codebase (BoundingBox
# treats x/y/width/height as floats in 0-1).
_PIPELINE_BBOX_FACTOR = 1000.0
_VLM_BBOX_FACTOR = 1.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_mineru_output(
    *,
    source_path: Path,
    output_dir: Path,
    content_list_path: Path,
    markdown_path: Optional[Path] = None,
    middle_json_path: Optional[Path] = None,
    processing_seconds: float = 0.0,
) -> ExtractedDocument:
    """Reconstruct an ExtractedDocument from MinerU's structured output.

    Parameters
    ----------
    source_path
        Absolute path to the original input file (PDF/image/etc.). Used
        to populate ``ExtractedDocument.source_path``.
    output_dir
        Directory MinerU wrote its outputs into (``<output>/<stem>/<method>/``).
        Image paths in ``content_list.json`` are resolved relative to
        this directory.
    content_list_path
        Path to ``<stem>_content_list.json``. Required — this is the
        flat reading-order element list we translate.
    markdown_path
        Optional path to ``<stem>.md``. When provided, its contents
        populate ``full_markdown`` and ``full_text``.
    middle_json_path
        Optional path to ``<stem>_middle.json``. We only inspect it for
        the ``_backend`` discriminator (drives bbox normalisation).
    processing_seconds
        Total CLI processing time reported by the MinerU service.

    Raises
    ------
    MineruLayoutParseError
        If ``content_list.json`` is missing, malformed, or yields zero
        usable elements. We refuse to build a half-empty
        ExtractedDocument because downstream chunking would silently
        degrade.
    """
    if not content_list_path.exists():
        raise MineruLayoutParseError(
            f"MinerU content_list.json not found at {content_list_path}"
        )

    try:
        with content_list_path.open(encoding="utf-8") as f:
            raw_elements = json.load(f)
    except json.JSONDecodeError as exc:
        raise MineruLayoutParseError(
            f"MinerU content_list.json at {content_list_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw_elements, list):
        raise MineruLayoutParseError(
            f"MinerU content_list.json at {content_list_path} expected to be a JSON array, "
            f"got {type(raw_elements).__name__}"
        )

    bbox_factor = _detect_bbox_factor(middle_json_path)

    # Translate flat element list into (page_idx, element) tuples grouped
    # by page.
    grouped: dict[int, list[ExtractedElement]] = {}
    for raw in raw_elements:
        if not isinstance(raw, dict):
            continue
        page_idx = int(raw.get("page_idx", 0) or 0)
        element = _translate_element(raw, output_dir=output_dir, bbox_factor=bbox_factor)
        if element is None:
            continue
        grouped.setdefault(page_idx, []).append(element)

    if not grouped:
        raise MineruLayoutParseError(
            f"MinerU content_list.json at {content_list_path} yielded zero elements"
        )

    pages = _build_pages(grouped)

    full_markdown = ""
    if markdown_path and markdown_path.exists():
        full_markdown = markdown_path.read_text(encoding="utf-8")

    full_text = _build_full_text(pages, fallback_markdown=full_markdown)
    title = _infer_title(grouped, fallback=source_path.stem)
    page_count = max(grouped.keys()) + 1 if grouped else 0

    all_tables = [t for p in pages for t in p.tables]
    all_images = [i for p in pages for i in p.images]

    return ExtractedDocument(
        source_path=source_path,
        title=title,
        page_count=page_count,
        pages=pages,
        full_markdown=full_markdown,
        full_text=full_text,
        all_tables=all_tables,
        all_images=all_images,
        extraction_time_seconds=float(processing_seconds),
        extracted_at=datetime.now(),
        metadata={"parser": "mineru"},
    )


# ---------------------------------------------------------------------------
# Element translation
# ---------------------------------------------------------------------------


def _translate_element(
    raw: dict[str, Any],
    *,
    output_dir: Path,
    bbox_factor: float,
) -> Optional[ExtractedElement]:
    """Map a single MinerU record to an ExtractedElement subclass.

    Returns ``None`` for records we deliberately drop (currently none —
    we keep page-furniture too because downstream chunking can filter
    by ``element_type``).
    """
    mineru_type = str(raw.get("type", "text"))
    page = int(raw.get("page_idx", 0) or 0) + 1  # MinerU is 0-indexed; we are 1-indexed
    bbox = _bbox_from_mineru(raw.get("bbox"), page=page, factor=bbox_factor)
    base_kwargs: dict[str, Any] = {
        "page": page,
        "bbox": bbox,
        "confidence": 1.0,
        "source": "mineru",
        "metadata": {"mineru_type": mineru_type},
    }

    if mineru_type == "text":
        return _build_text_element(raw, base_kwargs)
    if mineru_type == "equation":
        return _build_equation_element(raw, base_kwargs)
    if mineru_type == "code":
        return _build_code_element(raw, base_kwargs)
    if mineru_type == "list":
        return _build_list_element(raw, base_kwargs)
    if mineru_type == "table":
        return _build_table_element(raw, base_kwargs, output_dir=output_dir)
    if mineru_type in {"image", "chart"}:
        return _build_image_element(
            raw,
            base_kwargs,
            output_dir=output_dir,
            classification="chart" if mineru_type == "chart" else "image",
        )
    if mineru_type in {"header"}:
        return _build_furniture_element(raw, base_kwargs, ElementType.HEADER)
    if mineru_type in {"footer", "page_number"}:
        return _build_furniture_element(raw, base_kwargs, ElementType.FOOTER)
    if mineru_type == "page_footnote":
        return _build_furniture_element(raw, base_kwargs, ElementType.FOOTNOTE)
    if mineru_type == "aside_text":
        return _build_furniture_element(raw, base_kwargs, ElementType.CAPTION)

    # Unknown type — fall back to generic TEXT so we don't drop content.
    return _build_text_element(raw, base_kwargs, fallback_type=ElementType.TEXT)


def _build_text_element(
    raw: dict[str, Any],
    base_kwargs: dict[str, Any],
    *,
    fallback_type: ElementType = ElementType.PARAGRAPH,
) -> Optional[ExtractedElement]:
    content = (raw.get("text") or "").strip()
    if not content:
        return None
    text_level = int(raw.get("text_level", 0) or 0)
    if text_level == 1:
        element_type = ElementType.TITLE
    elif text_level >= 2:
        element_type = ElementType.HEADING
    else:
        element_type = fallback_type

    return ExtractedElement(
        content=content,
        element_type=element_type,
        section_level=text_level,
        **base_kwargs,
    )


def _build_equation_element(
    raw: dict[str, Any],
    base_kwargs: dict[str, Any],
) -> Optional[ExtractedElement]:
    content = (raw.get("text") or "").strip()
    if not content:
        return None
    metadata = {**base_kwargs.pop("metadata", {}), "text_format": raw.get("text_format", "latex")}
    img_path = raw.get("img_path")
    if img_path:
        metadata["img_path"] = img_path
    return ExtractedElement(
        content=content,
        element_type=ElementType.FORMULA,
        metadata=metadata,
        **base_kwargs,
    )


def _build_code_element(
    raw: dict[str, Any],
    base_kwargs: dict[str, Any],
) -> Optional[ExtractedElement]:
    content = (raw.get("code_body") or raw.get("text") or "").rstrip()
    if not content:
        return None
    metadata = {**base_kwargs.pop("metadata", {})}
    sub_type = raw.get("sub_type")
    if sub_type:
        metadata["sub_type"] = sub_type
    caption = raw.get("code_caption") or []
    if caption:
        metadata["caption"] = caption
    return ExtractedElement(
        content=content,
        element_type=ElementType.CODE,
        metadata=metadata,
        **base_kwargs,
    )


def _build_list_element(
    raw: dict[str, Any],
    base_kwargs: dict[str, Any],
) -> Optional[ExtractedElement]:
    items = raw.get("list_items") or []
    items = [str(i).strip() for i in items if str(i).strip()]
    if not items:
        return None
    content = "\n".join(f"- {item}" for item in items)
    metadata = {**base_kwargs.pop("metadata", {}), "list_items": items}
    sub_type = raw.get("sub_type")
    if sub_type:
        metadata["sub_type"] = sub_type
    return ExtractedElement(
        content=content,
        element_type=ElementType.LIST_ITEM,
        metadata=metadata,
        **base_kwargs,
    )


def _build_table_element(
    raw: dict[str, Any],
    base_kwargs: dict[str, Any],
    *,
    output_dir: Path,
) -> Optional[ExtractedElement]:
    table_body = raw.get("table_body") or ""
    headers, rows = _parse_html_table(table_body)
    markdown = _rows_to_markdown(headers, rows) if (headers or rows) else table_body.strip()
    csv_text = _rows_to_csv(headers, rows) if (headers or rows) else ""
    metadata = {**base_kwargs.pop("metadata", {})}
    captions = raw.get("table_caption") or []
    footnotes = raw.get("table_footnote") or []
    if captions:
        metadata["caption"] = captions
    if footnotes:
        metadata["footnote"] = footnotes
    if table_body:
        metadata["html"] = table_body
    img_path = raw.get("img_path")
    if img_path:
        metadata["img_path"] = img_path

    # Use the markdown body as ExtractedElement.content so existing
    # full-text consumers see something useful.
    content = markdown or table_body or ""
    return ExtractedTable(
        content=content,
        element_type=ElementType.TABLE,
        page=base_kwargs["page"],
        bbox=base_kwargs["bbox"],
        confidence=base_kwargs["confidence"],
        metadata=metadata,
        section_path=[],
        section_level=0,
        source=base_kwargs["source"],
        markdown=markdown,
        csv=csv_text,
        headers=headers,
        rows=rows,
    )


def _build_image_element(
    raw: dict[str, Any],
    base_kwargs: dict[str, Any],
    *,
    output_dir: Path,
    classification: str,
) -> Optional[ExtractedElement]:
    img_path_str = raw.get("img_path") or ""
    image_path: Optional[Path] = None
    if img_path_str:
        candidate = Path(img_path_str)
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        image_path = candidate

    metadata = {**base_kwargs.pop("metadata", {})}
    captions = raw.get("image_caption") or []
    footnotes = raw.get("image_footnote") or []
    description = " ".join(captions).strip() if captions else ""
    if footnotes:
        metadata["footnote"] = footnotes
    sub_type = raw.get("sub_type")
    if sub_type:
        metadata["sub_type"] = sub_type

    return ExtractedImage(
        content=description,
        element_type=ElementType.IMAGE,
        page=base_kwargs["page"],
        bbox=base_kwargs["bbox"],
        confidence=base_kwargs["confidence"],
        metadata=metadata,
        section_path=[],
        section_level=0,
        source=base_kwargs["source"],
        image_path=image_path,
        classification=classification,
        description=description,
    )


def _build_furniture_element(
    raw: dict[str, Any],
    base_kwargs: dict[str, Any],
    element_type: ElementType,
) -> Optional[ExtractedElement]:
    content = (raw.get("text") or "").strip()
    if not content:
        return None
    return ExtractedElement(
        content=content,
        element_type=element_type,
        section_level=0,
        **base_kwargs,
    )


# ---------------------------------------------------------------------------
# Bbox + backend helpers
# ---------------------------------------------------------------------------


def _detect_bbox_factor(middle_json_path: Optional[Path]) -> float:
    """Pick the divisor for MinerU's bbox coordinates.

    Pipeline / office backends emit ``[x0, y0, x1, y1]`` in a 0-1000
    range. VLM backend emits floats in 0-1. When ``middle_json_path``
    is unreadable, default to the pipeline factor — that's the most
    common case and a wrong guess only ends up rescaling bboxes,
    never crashes the parser.
    """
    if middle_json_path is None or not middle_json_path.exists():
        return _PIPELINE_BBOX_FACTOR
    try:
        with middle_json_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return _PIPELINE_BBOX_FACTOR

    backend = str(data.get("_backend", "")).lower() if isinstance(data, dict) else ""
    if backend.startswith("vlm"):
        return _VLM_BBOX_FACTOR
    return _PIPELINE_BBOX_FACTOR


def _bbox_from_mineru(
    raw_bbox: Any,
    *,
    page: int,
    factor: float,
) -> Optional[BoundingBox]:
    """Build a normalised BoundingBox from MinerU's ``[x0, y0, x1, y1]``.

    Returns ``None`` for missing / malformed bboxes; downstream
    consumers (e.g. PdfChunkViewer) already handle missing bboxes by
    falling back to text-only highlighting.
    """
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return None
    try:
        x0 = float(raw_bbox[0]) / factor
        y0 = float(raw_bbox[1]) / factor
        x1 = float(raw_bbox[2]) / factor
        y1 = float(raw_bbox[3]) / factor
    except (TypeError, ValueError):
        return None

    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)

    # Dev-mode sanity check: MinerU coords should already be 0–1 after
    # division by `factor`. This is the shared BoundingBox convention
    # (see ingestion.models.document.BoundingBox). Guarded by __debug__
    # so production runs (python -O) skip it; it never fires on valid
    # MinerU output, only flags a future factor/back-end regression.
    if __debug__:
        for coord in (x0, y0, x1, y1):
            assert -0.01 <= coord <= 1.01, (
                f"MinerU bbox coord {coord} outside 0–1 after factor {factor}; "
                f"raw_bbox={raw_bbox}"
            )

    return BoundingBox(x=x0, y=y0, width=width, height=height, page=page)


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------


def _build_pages(grouped: dict[int, list[ExtractedElement]]) -> list[PageContent]:
    """Group translated elements into PageContent objects, sorted by page."""
    pages: list[PageContent] = []
    for page_idx in sorted(grouped.keys()):
        elements: list[ExtractedElement] = []
        tables: list[ExtractedTable] = []
        images: list[ExtractedImage] = []
        text_parts: list[str] = []
        for elem in grouped[page_idx]:
            if isinstance(elem, ExtractedTable):
                tables.append(elem)
            elif isinstance(elem, ExtractedImage):
                images.append(elem)
            else:
                elements.append(elem)
                if elem.content:
                    text_parts.append(elem.content)

        pages.append(
            PageContent(
                page_number=page_idx + 1,
                elements=elements,
                tables=tables,
                images=images,
                text_content="\n\n".join(text_parts),
            )
        )
    return pages


def _build_full_text(pages: list[PageContent], *, fallback_markdown: str) -> str:
    """Best-effort plain-text rendering for downstream consumers.

    Joins per-page text with double newlines. Falls back to the
    markdown when no per-page text could be assembled (e.g. all
    elements were tables or images).
    """
    chunks = [p.text_content for p in pages if p.text_content]
    if chunks:
        return "\n\n".join(chunks)
    return fallback_markdown


def _infer_title(
    grouped: dict[int, list[ExtractedElement]],
    *,
    fallback: str,
) -> str:
    """Use the first level-1 heading on page 0 (if any) as the document title."""
    page0 = grouped.get(0) or []
    for elem in page0:
        if elem.element_type == ElementType.TITLE and elem.content:
            return elem.content
    return fallback


# ---------------------------------------------------------------------------
# Minimal HTML table -> headers/rows extraction
# ---------------------------------------------------------------------------


class _HTMLTableParser(HTMLParser):
    """Pulls headers + rows out of MinerU's ``<html><body><table>...`` blob.

    Loses merged-cell information (colspan/rowspan) — that's acceptable
    for V1; the original HTML is preserved on the element metadata so
    we can render richer views later if needed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._is_header_cell = False
        self._cell_buf: list[str] = []
        self._current_row: list[str] = []
        self.rows: list[list[str]] = []
        self.headers: list[str] = []
        self._header_set = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_row and tag in ("td", "th"):
            self._in_cell = True
            self._is_header_cell = tag == "th"
            self._cell_buf = []
        elif tag == "br" and self._in_cell:
            self._cell_buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "table":
            self._in_table = False
        elif tag == "tr" and self._in_row:
            self._in_row = False
            if not self._current_row:
                return
            if not self._header_set and self._is_header_cell:
                self.headers = self._current_row
                self._header_set = True
            else:
                # First non-header row promotes to headers if no <th> seen
                if not self._header_set and not self.headers:
                    self.headers = self._current_row
                    self._header_set = True
                else:
                    self.rows.append(self._current_row)
        elif tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            cell = "".join(self._cell_buf).strip()
            self._current_row.append(cell)
            self._cell_buf = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_buf.append(data)


def _parse_html_table(html: str) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows) extracted from a MinerU table HTML blob."""
    if not html or "<table" not in html.lower():
        return ([], [])
    parser = _HTMLTableParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return ([], [])
    return (parser.headers, parser.rows)


_BAR_PIPE = re.compile(r"\|")


def _safe_cell(value: str) -> str:
    """Escape pipe chars so we don't break markdown table rendering."""
    if not value:
        return ""
    return _BAR_PIPE.sub(r"\|", value.replace("\n", " ").strip())


def _rows_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    if not headers and not rows:
        return ""
    cols = max(
        (len(headers) if headers else 0),
        *(len(r) for r in rows),
    )
    if cols == 0:
        return ""
    padded_headers = (headers + [""] * cols)[:cols] if headers else [""] * cols
    md_lines: list[str] = []
    md_lines.append("| " + " | ".join(_safe_cell(h) for h in padded_headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * cols) + " |")
    for row in rows:
        padded = (row + [""] * cols)[:cols]
        md_lines.append("| " + " | ".join(_safe_cell(c) for c in padded) + " |")
    return "\n".join(md_lines)


def _rows_to_csv(headers: list[str], rows: list[list[str]]) -> str:
    if not headers and not rows:
        return ""
    output = io.StringIO()
    writer = csv.writer(output)
    if headers:
        writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue()
