"""
HTTP client for the docling service.

Routes document parsing to the GPU-accelerated docling service (port 8100)
instead of running docling in-process. Uses a shared volume so the file
doesn't need to travel over HTTP:

    host path          → /data/input   (mounted in both containers)
    host path          → /data/output  (mounted in both containers)

The service writes its JSON/markdown/image output under /data/output/<stem>/
and we load document.json back into the existing ExtractedDocument dataclass
so the rest of the pipeline (chunking, embedding, classification) is
unchanged.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from loguru import logger

from ingestion.models import (
    BoundingBox,
    ElementType,
    ExtractedDocument,
    ExtractedElement,
    ExtractedImage,
    ExtractedTable,
    PageContent,
)
from ingestion.models.source import IngestionResult, SourceMetadata, SourceType


DEFAULT_SERVICE_URL = "http://docling:8100"
SHARED_INPUT_DIR = Path("/data/input")
SHARED_OUTPUT_DIR = Path("/data/output")
PROCESS_TIMEOUT_SECONDS = 1800  # 30 min — large PDFs with VLM can be slow


class DoclingServiceError(RuntimeError):
    """Raised when the docling service returns an error response."""


class DoclingHttpClient:
    """Runs docling parsing via HTTP to the docling service container."""

    def __init__(
        self,
        service_url: Optional[str] = None,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        timeout: float = PROCESS_TIMEOUT_SECONDS,
    ):
        self.service_url = (
            service_url
            or os.environ.get("DOCLING_SERVICE_URL")
            or DEFAULT_SERVICE_URL
        ).rstrip("/")
        self.input_dir = Path(input_dir or SHARED_INPUT_DIR)
        self.output_dir = Path(output_dir or SHARED_OUTPUT_DIR)
        self.timeout = timeout

    async def process(self, source_path: Path) -> IngestionResult:
        """Process a document via the docling service.

        Copies source_path into the shared input dir, calls the service,
        then loads the service's JSON output into an IngestionResult.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        metadata = SourceMetadata.from_path(source_path, SourceType.DOCUMENT)
        result = IngestionResult(source_metadata=metadata)

        staged_path: Optional[Path] = None
        try:
            staged_path = self._stage_input(source_path)
            logger.info(
                f"[docling-http] Staged {source_path.name} at {staged_path}; "
                f"calling {self.service_url}/process"
            )

            file_result = await self._call_service(staged_path)

            output_dir = Path(file_result["output_dir"])
            json_path = Path(file_result["json_path"])
            markdown_path = Path(file_result["markdown_path"])

            document = self._load_document(
                source_path=source_path,
                json_path=json_path,
                markdown_path=markdown_path,
                service_result=file_result,
            )

            result.document = document
            metadata.page_count = document.page_count
            result.output_directory = output_dir
            result.markdown_path = markdown_path
            result.json_path = json_path
            result.original_path = (
                output_dir / source_path.name
                if (output_dir / source_path.name).exists()
                else None
            )
            result.table_paths = sorted(
                (output_dir / "extracted_info" / "tables").glob("*.md")
            ) if (output_dir / "extracted_info" / "tables").exists() else []
            result.image_paths = sorted(
                (output_dir / "extracted_info" / "images").glob("*.png")
            ) if (output_dir / "extracted_info" / "images").exists() else []

            return result.complete(success=True)

        except Exception as e:
            logger.error(f"[docling-http] Failed for {source_path.name}: {e}")
            return result.complete(success=False, error=str(e))
        finally:
            # Clean up the staged input copy; service output stays for downstream use
            if staged_path and staged_path.exists():
                try:
                    staged_path.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stage_input(self, source_path: Path) -> Path:
        """Copy source file into the shared input dir with a unique name."""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        # Preserve extension, prefix with UUID to avoid collisions across jobs
        unique_name = f"{uuid.uuid4().hex[:12]}__{source_path.name}"
        staged = self.input_dir / unique_name
        shutil.copy2(source_path, staged)
        return staged

    async def _call_service(self, staged_path: Path) -> dict[str, Any]:
        """POST to /process and return the first (and only) FileResult."""
        payload = {
            "input_path": str(staged_path),
            "output_path": str(self.output_dir),
            "copy_original": True,
            "extract_annotations": True,
            "read_handwriting": True,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.service_url}/process", json=payload
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results") or []
        if not results:
            raise DoclingServiceError(
                f"Service returned no results: {data}"
            )
        file_result = results[0]
        if not file_result.get("success"):
            raise DoclingServiceError(
                f"Service reported failure: {file_result.get('error')}"
            )
        return file_result

    def _load_document(
        self,
        source_path: Path,
        json_path: Path,
        markdown_path: Path,
        service_result: dict[str, Any],
    ) -> ExtractedDocument:
        """Reconstruct an ExtractedDocument from the service's JSON output."""
        with json_path.open() as f:
            data = json.load(f)

        pages = [_page_from_dict(p) for p in data.get("pages", [])]

        full_markdown = ""
        if markdown_path.exists():
            full_markdown = markdown_path.read_text(encoding="utf-8")

        full_text = data.get("full_text") or ""
        if not full_text and full_markdown:
            full_text = full_markdown

        all_tables = [t for page in pages for t in page.tables]
        all_images = [i for page in pages for i in page.images]

        return ExtractedDocument(
            source_path=source_path,
            title=data.get("title", "") or source_path.stem,
            page_count=data.get("page_count", len(pages)),
            pages=pages,
            full_markdown=full_markdown,
            full_text=full_text,
            all_tables=all_tables,
            all_images=all_images,
            extraction_time_seconds=float(
                service_result.get("processing_seconds", 0.0)
            ),
            extracted_at=datetime.now(),
            metadata=data.get("metadata", {}) or {},
        )


# ---------------------------------------------------------------------------
# JSON → dataclass reconstruction
# ---------------------------------------------------------------------------


def _bbox_from_dict(d: Optional[dict]) -> Optional[BoundingBox]:
    if not d:
        return None
    return BoundingBox(
        x=float(d.get("x", 0.0)),
        y=float(d.get("y", 0.0)),
        width=float(d.get("width", 0.0)),
        height=float(d.get("height", 0.0)),
        page=int(d.get("page", 1)),
    )


def _element_type_from_str(s: str) -> ElementType:
    try:
        return ElementType(s)
    except ValueError:
        return ElementType.TEXT


def _element_from_dict(d: dict) -> ExtractedElement:
    return ExtractedElement(
        content=d.get("content", ""),
        element_type=_element_type_from_str(d.get("type", "text")),
        page=int(d.get("page", 1)),
        bbox=_bbox_from_dict(d.get("bbox")),
        confidence=float(d.get("confidence", 1.0)),
        metadata=d.get("metadata") or {},
        section_path=d.get("section_path") or [],
        section_level=int(d.get("section_level", 0)),
        source=d.get("source", "native"),
    )


def _table_from_dict(d: dict) -> ExtractedTable:
    return ExtractedTable(
        content=d.get("content", ""),
        element_type=ElementType.TABLE,
        page=int(d.get("page", 1)),
        bbox=_bbox_from_dict(d.get("bbox")),
        confidence=float(d.get("confidence", 1.0)),
        metadata=d.get("metadata") or {},
        section_path=d.get("section_path") or [],
        section_level=int(d.get("section_level", 0)),
        source=d.get("source", "native"),
        markdown=d.get("markdown", ""),
        csv=d.get("csv", ""),
        headers=d.get("headers") or [],
        rows=d.get("rows") or [],
    )


def _coerce_number(value: Any, default: float = 0.0) -> float:
    """Accept int, float, or exporter quirks like ['width', 436.0]."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (int, float)):
                return float(item)
    return default


def _image_from_dict(d: dict) -> ExtractedImage:
    image_path_str = d.get("image_path")
    return ExtractedImage(
        content=d.get("content", ""),
        element_type=ElementType.IMAGE,
        page=int(_coerce_number(d.get("page", 1), 1.0)),
        bbox=_bbox_from_dict(d.get("bbox")),
        confidence=float(_coerce_number(d.get("confidence", 1.0), 1.0)),
        metadata=d.get("metadata") or {},
        section_path=d.get("section_path") or [],
        section_level=int(_coerce_number(d.get("section_level", 0), 0.0)),
        source=d.get("source", "native"),
        image_path=Path(image_path_str) if image_path_str else None,
        classification=d.get("classification", "unknown"),
        description=d.get("description", ""),
        width=int(_coerce_number(d.get("width", 0))),
        height=int(_coerce_number(d.get("height", 0))),
        format=d.get("format", "png"),
    )


def _page_from_dict(d: dict) -> PageContent:
    return PageContent(
        page_number=int(d.get("page_number", 1)),
        elements=[_element_from_dict(e) for e in d.get("elements", [])],
        tables=[_table_from_dict(t) for t in d.get("tables", [])],
        images=[_image_from_dict(i) for i in d.get("images", [])],
        text_content=d.get("text_content", ""),
    )
