"""
HTTP client for the MinerU service.

Routes document parsing to the GPU-accelerated MinerU service (port
8104) instead of running MinerU in-process. Symmetric with
``DoclingHttpClient`` — same call surface (``process(source_path) ->
IngestionResult``), same shared-volume input staging pattern, same
JSON-on-disk loading approach.

The MinerU service writes output under
``/data/mineru_output/<stem>/<method>/`` where ``<method>`` is the
MinerU CLI parsing method (default ``auto``). We load
``<stem>_content_list.json`` through ``mineru_layout_parser`` to
rebuild an ``ExtractedDocument`` with populated bboxes so downstream
chunking + ``PdfChunkViewer`` work identically to the docling path
(Q-A-5).
"""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
from loguru import logger

from ingestion.models import IngestionResult, SourceMetadata, SourceType

from app_main.services.parsing import (
    MineruLayoutParseError,
    parse_mineru_output,
)


DEFAULT_SERVICE_URL = "http://mineru:8104"
SHARED_INPUT_DIR = Path("/data/input")
SHARED_OUTPUT_DIR = Path("/data/mineru_output")
PROCESS_TIMEOUT_SECONDS = 1800  # 30 min — large PDFs with VLM can be slow
HEALTH_CHECK_TIMEOUT_SECONDS = 2.0


class MineruServiceError(RuntimeError):
    """Raised when the MinerU service returns an error response."""


@dataclass(frozen=True)
class MineruHealthResult:
    """Result of a MinerU service health probe.

    ``healthy`` is the only field guaranteed to be set. ``version`` is
    populated from the service's own ``/health`` response payload when
    available; ``error`` carries the failure reason on a non-healthy
    probe so the UI can surface it in a tooltip.
    """

    healthy: bool
    version: Optional[str] = None
    error: Optional[str] = None


class MineruHttpClient:
    """Runs MinerU parsing via HTTP to the mineru service container."""

    def __init__(
        self,
        service_url: Optional[str] = None,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        timeout: float = PROCESS_TIMEOUT_SECONDS,
    ) -> None:
        self.service_url = (
            service_url
            or os.environ.get("MINERU_SERVICE_URL")
            or DEFAULT_SERVICE_URL
        ).rstrip("/")
        self.input_dir = Path(input_dir or SHARED_INPUT_DIR)
        self.output_dir = Path(output_dir or SHARED_OUTPUT_DIR)
        self.timeout = timeout

    async def process(self, source_path: Path) -> IngestionResult:
        """Process a document via the MinerU service.

        Copies ``source_path`` into the shared input dir, calls
        ``POST /process``, then rebuilds an ``IngestionResult`` from
        the on-disk JSON output. The staged input copy is removed in
        ``finally``; MinerU's per-document output directory stays
        intact so downstream chunking and image rendering can read
        from it.
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
                f"[mineru-http] Staged {source_path.name} at {staged_path}; "
                f"calling {self.service_url}/process"
            )

            file_result = await self._call_service(staged_path)

            output_dir = Path(file_result["output_dir"])
            content_list_path = Path(file_result["content_list_path"])
            markdown_path_str = file_result.get("markdown_path")
            markdown_path = Path(markdown_path_str) if markdown_path_str else None
            middle_json_path_str = file_result.get("middle_json_path")
            middle_json_path = Path(middle_json_path_str) if middle_json_path_str else None

            document = parse_mineru_output(
                source_path=source_path,
                output_dir=output_dir,
                content_list_path=content_list_path,
                markdown_path=markdown_path,
                middle_json_path=middle_json_path,
                processing_seconds=float(file_result.get("processing_seconds", 0.0)),
            )

            result.document = document
            metadata.page_count = document.page_count
            result.output_directory = output_dir
            result.markdown_path = markdown_path
            result.json_path = content_list_path

            # The mineru service writes an `images/` folder next to the
            # markdown when images are present.
            images_dir = output_dir / "images"
            if images_dir.exists():
                result.image_paths = sorted(images_dir.glob("*"))

            # MinerU does not write per-table markdown files (tables are
            # inline HTML in content_list.json), so table_paths stays
            # empty. The ExtractedTable instances on the document still
            # carry the rendered markdown.
            return result.complete(success=True)

        except Exception as exc:
            logger.error(f"[mineru-http] Failed for {source_path.name}: {exc}")
            return result.complete(success=False, error=str(exc))
        finally:
            # Clean up the staged input copy; service output stays for downstream use.
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
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.service_url}/process", json=payload
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results") or []
        if not results:
            raise MineruServiceError(f"Service returned no results: {data}")
        file_result = results[0]
        if not file_result.get("success"):
            raise MineruServiceError(
                f"Service reported failure: {file_result.get('error')}"
            )

        # Sanity check — the layout parser will explode without these.
        if not file_result.get("content_list_path"):
            raise MineruServiceError(
                f"Service response missing content_list_path: {file_result}"
            )

        return file_result

    # ------------------------------------------------------------------
    # Health probing
    # ------------------------------------------------------------------

    async def health_check(
        self, *, timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS
    ) -> MineruHealthResult:
        """Probe the MinerU service's ``/health`` endpoint.

        Used by the ``GET /api/health/mineru`` router to back the UI
        chip. The 2-second default timeout keeps the page-load impact
        small even when the container is unreachable.

        Never raises. Connection errors, HTTP errors, malformed JSON —
        all are surfaced as ``MineruHealthResult(healthy=False,
        error=...)``. The caller only needs to inspect ``healthy``.
        """
        url = f"{self.service_url}/health"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
            if resp.status_code != 200:
                return MineruHealthResult(
                    healthy=False,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )
            try:
                data = resp.json()
            except ValueError:
                # 200 with non-JSON body — treat as healthy without version
                return MineruHealthResult(healthy=True)
            version = data.get("version") if isinstance(data, dict) else None
            return MineruHealthResult(healthy=True, version=version)
        except httpx.HTTPError as exc:
            return MineruHealthResult(healthy=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 — endpoint must never raise
            return MineruHealthResult(healthy=False, error=str(exc))

    # ------------------------------------------------------------------
    # Convenience accessors (used by tests and the auto-fallback logic in A.1c)
    # ------------------------------------------------------------------

    @classmethod
    def is_supported_extension(cls, source_path: Path) -> bool:
        """Return True if MinerU is known to accept this file extension.

        Mirrors ``services/mineru/api.py::SUPPORTED_EXTENSIONS``. Kept
        on the client so callers can pre-flight without touching the
        service or env vars.
        """
        return source_path.suffix.lower() in _MINERU_SUPPORTED_EXTENSIONS


# Local copy of the service-side supported-extensions set. We keep two
# copies (one server-side, one client-side) deliberately because the
# server enforces the contract and the client wants to avoid a network
# round-trip for an obvious mismatch.
_MINERU_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tiff",
}


__all__ = [
    "DEFAULT_SERVICE_URL",
    "HEALTH_CHECK_TIMEOUT_SECONDS",
    "MineruHealthResult",
    "MineruHttpClient",
    "MineruServiceError",
    "PROCESS_TIMEOUT_SECONDS",
    "SHARED_INPUT_DIR",
    "SHARED_OUTPUT_DIR",
]
