"""
MinerU document processing service.

Standalone HTTP API for document parsing via MinerU 2.x. Mirrors the
shape of services/docling/api.py so the two parsers can be swapped
behind the parser_engine routing in app-main without touching call
sites.

The service is a thin wrapper around the `mineru` CLI: each /process
call shells out to `mineru -p <input> -o <output> -b <backend>`. The
CLI internally drives MinerU's full layout/OCR/formula/table pipeline
and writes a per-document directory tree under <output>/<stem>/<method>/
with markdown, content_list.json, middle.json and an images/ subfolder.

We chose to wrap the CLI rather than expose MinerU's bundled
mineru-api directly because:

- the CLI guarantees synchronous, blocking behaviour with predictable
  on-disk output (no task-queue polling needed);
- the request/response shape stays symmetric with the docling service
  so DoclingHttpClient and MineruHttpClient share their call pattern;
- we don't depend on MinerU's task-state retention semantics which
  evict tasks after 24h.

All MinerU runtime knobs are configurable via environment variables
(MINERU_BACKEND, MINERU_METHOD, MINERU_LANG, MINERU_DEVICE,
MINERU_FORMULA_ENABLE, MINERU_TABLE_ENABLE) — see services/mineru/README.md.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field


app = FastAPI(
    title="MinerU Processing Service",
    description="Document parsing API powered by MinerU 2.x",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
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


def _bool_env(key: str, default: bool) -> bool:
    val = os.getenv(key, "").lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _str_env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val else default


def _runtime_config() -> dict[str, str]:
    """Snapshot of the env-driven runtime config (logged on first call)."""
    return {
        "device": _str_env("MINERU_DEVICE", "cuda"),
        "backend": _str_env("MINERU_BACKEND", "pipeline"),
        "method": _str_env("MINERU_METHOD", "auto"),
        "lang": _str_env("MINERU_LANG", "en"),
        "formula_enable": "true" if _bool_env("MINERU_FORMULA_ENABLE", True) else "false",
        "table_enable": "true" if _bool_env("MINERU_TABLE_ENABLE", True) else "false",
    }


# ---------------------------------------------------------------------------
# Request / response models (mirrored from services/docling/api.py)
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    input_path: str = Field(
        ...,
        description="Path to a file or directory to process (inside mounted volume)",
    )
    output_path: str = Field(
        ...,
        description="Base output directory. Each file gets its own <stem>/ subdirectory.",
    )
    copy_original: bool = Field(
        default=True,
        description="Copy the original file into the output directory",
    )
    # MinerU does not expose a per-call backend toggle through our wrapper
    # because pipeline vs vlm requires very different model preloading.
    # Use the MINERU_BACKEND env var for that.


class FileResult(BaseModel):
    source: str
    success: bool
    output_dir: Optional[str] = None
    markdown_path: Optional[str] = None
    content_list_path: Optional[str] = None
    middle_json_path: Optional[str] = None
    page_count: int = 0
    table_count: int = 0
    image_count: int = 0
    processing_seconds: float = 0.0
    error: Optional[str] = None
    backend: Optional[str] = None
    method: Optional[str] = None


class ProcessResponse(BaseModel):
    total_files: int
    succeeded: int
    failed: int
    results: list[FileResult]
    total_seconds: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "mineru"}


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest) -> ProcessResponse:
    """Process a file (or directory of files) through MinerU.

    Each input file is shelled out to the `mineru` CLI independently so a
    single failing file doesn't take the whole batch down.
    """
    input_path = Path(req.input_path)
    output_base = Path(req.output_path)

    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Input path not found: {req.input_path}",
        )

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            f
            for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Input path is neither file nor directory: {req.input_path}",
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files found in {req.input_path}",
        )

    output_base.mkdir(parents=True, exist_ok=True)
    config = _runtime_config()
    logger.info(f"[mineru-api] runtime config: {config}")

    results: list[FileResult] = []
    t0 = time.time()
    for file_path in files:
        results.append(
            _process_single(
                file_path=file_path,
                output_base=output_base,
                copy_original=req.copy_original,
                config=config,
            )
        )

    total_seconds = time.time() - t0
    succeeded = sum(1 for r in results if r.success)

    return ProcessResponse(
        total_files=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
        total_seconds=round(total_seconds, 2),
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _process_single(
    file_path: Path,
    output_base: Path,
    copy_original: bool,
    config: dict[str, str],
) -> FileResult:
    """Run the MinerU CLI for a single file and locate its outputs.

    The CLI writes everything under `<output_base>/<stem>/<method>/`. We
    leave that layout intact and report back the canonical paths the
    client expects (markdown_path, content_list_path, middle_json_path).
    """
    logger.info(f"[mineru-api] processing {file_path}")
    t0 = time.time()
    stem = file_path.stem
    method = config["method"]
    backend = config["backend"]

    try:
        cmd = [
            "mineru",
            "-p",
            str(file_path),
            "-o",
            str(output_base),
            "-b",
            backend,
        ]
        # `-m` is only meaningful for pipeline / hybrid backends; mineru
        # rejects it for pure vlm backend. Pass conservatively.
        if backend.startswith("pipeline") or backend.startswith("hybrid"):
            cmd.extend(["-m", method])
            cmd.extend(["-l", config["lang"]])

        cmd.extend(["-f", config["formula_enable"]])
        cmd.extend(["-t", config["table_enable"]])

        # Inherit the parent env (HF_HOME, MINERU_MODEL_SOURCE, etc.).
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("MINERU_TIMEOUT_SECONDS", "1800")),
        )

        if completed.returncode != 0:
            err_tail = (completed.stderr or completed.stdout or "")[-2000:]
            logger.error(
                f"[mineru-api] mineru exited {completed.returncode} for "
                f"{file_path.name}: {err_tail}"
            )
            return FileResult(
                source=str(file_path),
                success=False,
                error=f"mineru CLI exit {completed.returncode}: {err_tail.strip()}",
                processing_seconds=round(time.time() - t0, 2),
                backend=backend,
                method=method,
            )

        output_dir = output_base / stem / method
        markdown_path = output_dir / f"{stem}.md"
        content_list_path = output_dir / f"{stem}_content_list.json"
        middle_json_path = output_dir / f"{stem}_middle.json"

        if not markdown_path.exists() or not content_list_path.exists():
            return FileResult(
                source=str(file_path),
                success=False,
                error=(
                    f"mineru completed but expected outputs missing under "
                    f"{output_dir} (md={markdown_path.exists()}, "
                    f"content_list={content_list_path.exists()})"
                ),
                processing_seconds=round(time.time() - t0, 2),
                backend=backend,
                method=method,
            )

        if copy_original:
            target = output_base / stem / file_path.name
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(file_path, target)
            except OSError as exc:
                # Non-fatal — original-copy is best-effort.
                logger.warning(f"[mineru-api] could not copy original: {exc}")

        page_count, table_count, image_count = _summarise_content_list(content_list_path)

        return FileResult(
            source=str(file_path),
            success=True,
            output_dir=str(output_dir),
            markdown_path=str(markdown_path),
            content_list_path=str(content_list_path),
            middle_json_path=str(middle_json_path) if middle_json_path.exists() else None,
            page_count=page_count,
            table_count=table_count,
            image_count=image_count,
            processing_seconds=round(time.time() - t0, 2),
            backend=backend,
            method=method,
        )

    except subprocess.TimeoutExpired:
        return FileResult(
            source=str(file_path),
            success=False,
            error="mineru CLI timed out",
            processing_seconds=round(time.time() - t0, 2),
            backend=backend,
            method=method,
        )
    except Exception as exc:  # pragma: no cover - defensive top-level
        logger.exception(f"[mineru-api] unexpected failure on {file_path}")
        return FileResult(
            source=str(file_path),
            success=False,
            error=str(exc),
            processing_seconds=round(time.time() - t0, 2),
            backend=backend,
            method=method,
        )


def _summarise_content_list(content_list_path: Path) -> tuple[int, int, int]:
    """Return (page_count, table_count, image_count) read from content_list.json.

    Failure here is non-fatal — we just return zeros so the FileResult
    can still report the markdown/content_list paths.
    """
    try:
        import json

        with content_list_path.open() as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            f"[mineru-api] could not summarise {content_list_path}: {exc}"
        )
        return (0, 0, 0)

    if not isinstance(data, list):
        return (0, 0, 0)

    page_ids = {int(item.get("page_idx", 0)) for item in data if isinstance(item, dict)}
    tables = sum(1 for item in data if isinstance(item, dict) and item.get("type") == "table")
    images = sum(
        1
        for item in data
        if isinstance(item, dict) and item.get("type") in {"image", "chart"}
    )
    page_count = (max(page_ids) + 1) if page_ids else 0
    return (page_count, tables, images)
