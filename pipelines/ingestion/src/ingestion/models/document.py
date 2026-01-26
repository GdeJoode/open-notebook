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

    Coordinates are normalized (0.0 to 1.0) relative to page dimensions.
    Origin is top-left corner.
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
    def from_docling(cls, prov: Any, page: int) -> "BoundingBox":
        """
        Create from Docling provenance object.

        Docling uses BOTTOMLEFT origin, we convert to TOPLEFT.
        """
        if not prov or not hasattr(prov, "bbox"):
            return cls(x=0, y=0, width=0, height=0, page=page)

        bbox = prov.bbox
        # Docling bbox: (left, bottom, right, top) with BOTTOMLEFT origin
        # Convert to (x, y, width, height) with TOPLEFT origin
        page_height = getattr(prov, "page_height", 1.0)

        left = getattr(bbox, "l", 0) or 0
        bottom = getattr(bbox, "b", 0) or 0
        right = getattr(bbox, "r", 0) or 0
        top = getattr(bbox, "t", 0) or 0

        # Convert to normalized coordinates with TOPLEFT origin
        return cls(
            x=left,
            y=page_height - top if page_height else top,
            width=right - left,
            height=top - bottom,
            page=page,
        )


@dataclass
class ExtractedElement:
    """
    Base class for extracted document elements.

    All elements have:
    - Content (text, markdown, etc.)
    - Type classification
    - Optional bounding box for spatial location
    - Page number for organization
    """
    content: str
    element_type: ElementType
    page: int  # 1-indexed page number
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "type": self.element_type.value,
            "page": self.page,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


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
