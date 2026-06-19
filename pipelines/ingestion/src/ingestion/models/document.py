"""
Document extraction models.

Models for representing extracted document content including:
- Text elements with bounding boxes
- Tables with MD and CSV representations
- Images with classification and VLM descriptions
- Page-organized content structure
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ElementType(str, Enum):
    """Type of extracted element."""
    TEXT = "text"
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    CODE = "code"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"


@dataclass
class BoundingBox:
    """
    Bounding box for element location on page.

    Canonical coordinate convention (shared by every parser):
    ``x``, ``y``, ``width``, ``height`` are normalized to 0.0–1.0 relative
    to page dimensions, with a TOPLEFT origin. Every ``from_*`` constructor
    MUST emit coordinates in this convention so that ``chunk.positions``
    is parser-agnostic downstream (frontend overlay, structure graph).
    A future vision parser (Track H) must follow the same convention.
    """
    x: float  # Left edge
    y: float  # Top edge
    width: float
    height: float
    page: int  # 1-indexed page number

    @property
    def x2(self) -> float:
        """Right edge."""
        return self.x + self.width

    @property
    def y2(self) -> float:
        """Bottom edge."""
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point of bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        """Area of bounding box."""
        return self.width * self.height

    def overlap_ratio(self, other: "BoundingBox") -> float:
        """Calculate intersection-over-minimum-area overlap with another bbox.

        Returns a value between 0.0 (no overlap) and 1.0 (full overlap).
        Uses the smaller area as denominator so a small highlight fully
        inside a large chunk still scores 1.0.
        """
        if self.page != other.page:
            return 0.0
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        min_area = min(self.area, other.area)
        if min_area <= 0:
            return 0.0
        return intersection / min_area

    @staticmethod
    def union(a: "BoundingBox", b: "BoundingBox") -> "BoundingBox":
        """Create the smallest bbox that contains both a and b."""
        x = min(a.x, b.x)
        y = min(a.y, b.y)
        x2 = max(a.x2, b.x2)
        y2 = max(a.y2, b.y2)
        return BoundingBox(x=x, y=y, width=x2 - x, height=y2 - y, page=a.page)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "page": self.page,
        }

    @classmethod
    def from_docling(
        cls,
        prov: Any,
        page: int,
        page_width: Optional[float] = None,
        page_height: Optional[float] = None,
    ) -> "BoundingBox":
        """
        Create a normalized BoundingBox from a Docling provenance object.

        Coordinates are returned in the 0.0–1.0 normalized, TOPLEFT-origin
        convention shared by every ``BoundingBox.from_*`` constructor (see
        the class docstring). Docling reports raw PDF points with a
        BOTTOMLEFT origin, so we divide by the page dimensions and flip the
        y-axis here, at extraction time, rather than guessing downstream.

        ``page_width``/``page_height`` MUST be the true page size in points
        (from ``doc.pages[page_no].size``). The provenance object's own
        ``page_height`` attribute is frequently absent, which is why callers
        thread the real dimensions in. When dimensions are unknown or
        non-positive we cannot normalize, so we fall back to a zero box
        (downstream consumers treat a zero box as "no spatial data").
        """
        if not prov or not hasattr(prov, "bbox"):
            return cls(x=0, y=0, width=0, height=0, page=page)

        bbox = prov.bbox

        # Fall back to the provenance's own page_height only when the caller
        # could not supply one (legacy callers). Width has no such fallback.
        if page_height is None:
            page_height = getattr(prov, "page_height", None)

        # Without real dimensions we can't produce normalized coordinates;
        # emit a zero box rather than raw points masquerading as 0–1.
        if not page_width or not page_height or page_width <= 0 or page_height <= 0:
            return cls(x=0, y=0, width=0, height=0, page=page)

        left = float(getattr(bbox, "l", 0) or 0)
        bottom = float(getattr(bbox, "b", 0) or 0)
        right = float(getattr(bbox, "r", 0) or 0)
        top = float(getattr(bbox, "t", 0) or 0)

        # Docling defaults to BOTTOMLEFT origin; honour an explicit
        # coord_origin when present (mirrors ChunkExtractor._convert_docling_bbox).
        coord_origin = "BOTTOMLEFT"
        if hasattr(bbox, "coord_origin"):
            coord_origin = str(bbox.coord_origin).upper()

        if "TOPLEFT" in coord_origin:
            y_top = top / page_height
            y_bottom = bottom / page_height
        else:
            # BOTTOMLEFT: flip so the smaller image-space y is the top edge.
            y_top = (page_height - top) / page_height
            y_bottom = (page_height - bottom) / page_height

        x1 = left / page_width
        x2 = right / page_width

        # Clamp to the valid normalized range.
        x1 = max(0.0, min(1.0, x1))
        x2 = max(0.0, min(1.0, x2))
        y_top = max(0.0, min(1.0, y_top))
        y_bottom = max(0.0, min(1.0, y_bottom))

        return cls(
            x=x1,
            y=y_top,
            width=max(0.0, x2 - x1),
            height=max(0.0, y_bottom - y_top),
            page=page,
        )


@dataclass
class AnnotationInfo:
    """
    PDF annotation attached to an extracted element.

    Represents highlights, underlines, strikeouts, sticky notes, ink
    annotations, and freetext annotations extracted via PyMuPDF.
    """
    type: str  # "highlight", "underline", "strikeout", "squiggly", "sticky_note", "ink", "freetext"
    color: list[float] = field(default_factory=list)  # RGB [0.0-1.0]
    color_name: str = ""  # "yellow", "green", "red", "blue", "pink", "orange", "purple"
    comment: str = ""  # Text content of sticky note or popup comment
    highlighted_text: str = ""  # The text covered by the annotation
    handwritten_text: str = ""  # VLM-recognized text if the annotation was handwritten

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "type": self.type,
            "color": self.color,
            "color_name": self.color_name,
        }
        if self.comment:
            d["comment"] = self.comment
        if self.highlighted_text:
            d["highlighted_text"] = self.highlighted_text
        if self.handwritten_text:
            d["handwritten_text"] = self.handwritten_text
        return d


@dataclass
class ExtractedElement:
    """
    Base class for extracted document elements.

    All elements have:
    - Content (text, markdown, etc.)
    - Type classification
    - Optional bounding box for spatial location
    - Page number for organization
    - Annotations from PDF markup (highlights, notes, etc.)
    - Source indicating how the content was extracted
    """
    content: str
    element_type: ElementType
    page: int  # 1-indexed page number
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Section hierarchy from Docling
    section_path: list[str] = field(default_factory=list)
    section_level: int = 0
    # Annotations from PDF markup
    annotations: list[AnnotationInfo] = field(default_factory=list)
    # How the content was extracted
    source: str = "native"  # "native", "ocr", "vlm_handwriting"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "content": self.content,
            "type": self.element_type.value,
            "page": self.page,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "section_path": self.section_path,
            "section_level": self.section_level,
        }
        if self.annotations:
            d["annotations"] = [a.to_dict() for a in self.annotations]
        if self.source != "native":
            d["source"] = self.source
        return d


@dataclass
class ExtractedTable(ExtractedElement):
    """
    Extracted table with multiple format representations.

    Includes:
    - Markdown representation for human readability
    - CSV representation for data processing
    - Structured data for programmatic access
    - Original cell structure
    """
    markdown: str = ""
    csv: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0

    def __post_init__(self):
        self.element_type = ElementType.TABLE
        if self.rows:
            self.num_rows = len(self.rows)
            self.num_cols = max(len(row) for row in self.rows) if self.rows else 0

    def to_csv_string(self) -> str:
        """Generate CSV string from table data."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        if self.headers:
            writer.writerow(self.headers)
        writer.writerows(self.rows)
        return output.getvalue()

    def to_markdown_string(self) -> str:
        """Generate Markdown table string."""
        if not self.rows and not self.headers:
            return self.markdown or ""

        lines = []
        all_rows = [self.headers] + self.rows if self.headers else self.rows

        if not all_rows:
            return ""

        # Calculate column widths
        col_widths = [
            max(len(str(row[i])) if i < len(row) else 0 for row in all_rows)
            for i in range(self.num_cols)
        ]

        # Header row
        if self.headers:
            header = "| " + " | ".join(
                str(h).ljust(col_widths[i]) for i, h in enumerate(self.headers)
            ) + " |"
            separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
            lines.extend([header, separator])

        # Data rows
        for row in self.rows:
            line = "| " + " | ".join(
                str(row[i] if i < len(row) else "").ljust(col_widths[i])
                for i in range(self.num_cols)
            ) + " |"
            lines.append(line)

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "markdown": self.markdown or self.to_markdown_string(),
            "csv": self.csv or self.to_csv_string(),
            "headers": self.headers,
            "rows": self.rows,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
        })
        return base


@dataclass
class ExtractedImage(ExtractedElement):
    """
    Extracted image with classification and description.

    Includes:
    - Image file path or base64 data
    - Classification (chart, diagram, photo, formula)
    - VLM-generated description
    - Original dimensions
    """
    image_path: Optional[Path] = None
    image_data: Optional[str] = None  # Base64 encoded
    classification: str = "unknown"  # chart, diagram, photo, formula
    description: str = ""
    width: int = 0
    height: int = 0
    format: str = "png"

    def __post_init__(self):
        self.element_type = ElementType.IMAGE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "image_path": str(self.image_path) if self.image_path else None,
            "has_image_data": bool(self.image_data),
            "classification": self.classification,
            "description": self.description,
            "width": self.width,
            "height": self.height,
            "format": self.format,
        })
        return base


@dataclass
class PageContent:
    """
    Content from a single page, organized by element type.

    Provides page-level organization for document content.
    """
    page_number: int  # 1-indexed
    elements: list[ExtractedElement] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    text_content: str = ""  # Combined text content for the page

    @property
    def element_count(self) -> int:
        """Total number of elements on this page."""
        return len(self.elements) + len(self.tables) + len(self.images)

    def to_markdown(self, include_images: bool = True, include_tables: bool = True) -> str:
        """
        Export page content as Markdown.

        Args:
            include_images: Include image references/descriptions
            include_tables: Include table markdown

        Returns:
            Markdown string for the page
        """
        parts = []

        # Sort all elements by vertical position
        all_items: list[tuple[float, str]] = []

        for elem in self.elements:
            y_pos = elem.bbox.y if elem.bbox else 0
            all_items.append((y_pos, elem.content))

        if include_tables:
            for table in self.tables:
                y_pos = table.bbox.y if table.bbox else 0
                md = table.to_markdown_string() or table.markdown
                all_items.append((y_pos, md))

        if include_images:
            for img in self.images:
                y_pos = img.bbox.y if img.bbox else 0
                if img.image_path:
                    ref = f"![{img.description or 'Image'}]({img.image_path})"
                else:
                    ref = f"<!-- Image: {img.description or img.classification} -->"
                all_items.append((y_pos, ref))

        # Sort by y position and join
        all_items.sort(key=lambda x: x[0])
        parts = [item[1] for item in all_items]

        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "page_number": self.page_number,
            "element_count": self.element_count,
            "elements": [e.to_dict() for e in self.elements],
            "tables": [t.to_dict() for t in self.tables],
            "images": [i.to_dict() for i in self.images],
            "text_content": self.text_content,
        }


@dataclass
class ExtractedDocument:
    """
    Complete extracted document with page-organized content.

    Provides:
    - Full document markdown
    - Per-page content organization
    - Aggregated tables and images
    - Document metadata
    """
    source_path: Path
    title: str = ""
    page_count: int = 0
    pages: list[PageContent] = field(default_factory=list)
    full_markdown: str = ""
    full_text: str = ""

    # Aggregated content
    all_tables: list[ExtractedTable] = field(default_factory=list)
    all_images: list[ExtractedImage] = field(default_factory=list)

    # Metadata
    extracted_at: datetime = field(default_factory=datetime.now)
    extraction_time_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def table_count(self) -> int:
        """Total number of tables."""
        return len(self.all_tables)

    @property
    def image_count(self) -> int:
        """Total number of images."""
        return len(self.all_images)

    def get_page(self, page_number: int) -> Optional[PageContent]:
        """Get content for a specific page (1-indexed)."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def to_markdown(self, include_page_markers: bool = True) -> str:
        """
        Export full document as Markdown.

        Args:
            include_page_markers: Include page number markers

        Returns:
            Full document Markdown
        """
        if self.full_markdown:
            return self.full_markdown

        parts = []
        if self.title:
            parts.append(f"# {self.title}")

        for page in self.pages:
            if include_page_markers:
                parts.append(f"\n---\n## Page {page.page_number}\n")
            parts.append(page.to_markdown())

        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_path": str(self.source_path),
            "title": self.title,
            "page_count": self.page_count,
            "table_count": self.table_count,
            "image_count": self.image_count,
            "pages": [p.to_dict() for p in self.pages],
            "extracted_at": self.extracted_at.isoformat(),
            "extraction_time_seconds": self.extraction_time_seconds,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
