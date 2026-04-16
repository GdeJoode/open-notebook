"""
Docling document processing service.

Standalone HTTP API for document parsing via Docling.
Accepts file or directory paths (via mounted volumes) and writes
structured output (markdown, JSON, images, tables) to an output directory.

All pipeline options are configured via environment variables.
"""

import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from ingestion.config import (
    AcceleratorDevice,
    DoclingConfig,
    IngestionConfig,
    OcrEngine,
    OutputConfig,
    TableFormat,
    TableMode,
)
from ingestion.parsers import DoclingParser
from ingestion.exporters import DocumentExporter

from pymupdf_enricher import enrich_with_pymupdf_text
from annotation_extractor import enrich_document_with_annotations
from handwriting_detector import detect_handwriting
from vlm_reader import read_handwriting_regions

app = FastAPI(
    title="Docling Processing Service",
    description="Document parsing API powered by Docling",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Environment-based configuration
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".pptx", ".ppt", ".html", ".htm", ".txt", ".md",
}


def _bool_env(key: str, default: bool) -> bool:
    val = os.getenv(key, "").lower()
    if not val:
        return default
    return val in ("1", "true", "yes")


def _float_env(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val else default


def _int_env(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val else default


def build_docling_config() -> DoclingConfig:
    """Build DoclingConfig from environment variables."""
    cfg = DoclingConfig()

    # GPU / accelerator
    if device := os.getenv("DOCLING_DEVICE"):
        cfg.device = AcceleratorDevice(device)
    cfg.num_threads = _int_env("DOCLING_NUM_THREADS", cfg.num_threads)

    # OCR
    cfg.do_ocr = _bool_env("DOCLING_DO_OCR", cfg.do_ocr)
    if ocr_engine := os.getenv("DOCLING_OCR_ENGINE"):
        cfg.ocr_engine = OcrEngine(ocr_engine)
    if ocr_lang := os.getenv("DOCLING_OCR_LANG"):
        cfg.ocr_lang = [lang.strip() for lang in ocr_lang.split(",")]

    # Tables
    cfg.do_table_structure = _bool_env("DOCLING_DO_TABLE_STRUCTURE", cfg.do_table_structure)
    if table_mode := os.getenv("DOCLING_TABLE_MODE"):
        cfg.table_mode = TableMode(table_mode)

    # Images
    cfg.generate_page_images = _bool_env("DOCLING_GENERATE_PAGE_IMAGES", cfg.generate_page_images)
    cfg.generate_picture_images = _bool_env("DOCLING_GENERATE_PICTURE_IMAGES", cfg.generate_picture_images)
    cfg.images_scale = _float_env("DOCLING_IMAGES_SCALE", cfg.images_scale)

    # Image classification & description
    cfg.do_picture_classification = _bool_env("DOCLING_DO_PICTURE_CLASSIFICATION", cfg.do_picture_classification)
    cfg.do_picture_description = _bool_env("DOCLING_DO_PICTURE_DESCRIPTION", cfg.do_picture_description)
    if vlm_model := os.getenv("DOCLING_VLM_MODEL"):
        cfg.vlm_model = vlm_model
    if vlm_prompt := os.getenv("DOCLING_VLM_PROMPT"):
        cfg.vlm_prompt = vlm_prompt

    # Formula & code
    cfg.do_formula_extraction = _bool_env("DOCLING_DO_FORMULA_EXTRACTION", cfg.do_formula_extraction)
    cfg.do_code_enrichment = _bool_env("DOCLING_DO_CODE_ENRICHMENT", cfg.do_code_enrichment)

    # Output organisation
    cfg.organize_by_page = _bool_env("DOCLING_ORGANIZE_BY_PAGE", cfg.organize_by_page)

    return cfg


# Lazy-initialised parser (heavy init, do once)
_parser: Optional[DoclingParser] = None


def get_parser() -> DoclingParser:
    global _parser
    if _parser is None:
        cfg = build_docling_config()
        _parser = DoclingParser(cfg)
    return _parser


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    input_path: str = Field(
        ...,
        description="Path to a file or directory to process (inside mounted volume)",
    )
    output_path: str = Field(
        ...,
        description="Base output directory. Each file gets its own subdirectory.",
    )
    copy_original: bool = Field(
        default=True,
        description="Copy the original file into the output directory",
    )
    extract_annotations: bool = Field(
        default=True,
        description="Extract PDF annotations (highlights, notes) and attach to chunks",
    )
    read_handwriting: bool = Field(
        default=True,
        description="Use VLM to read handwritten text (reMarkable notes, margin notes)",
    )


class FileResult(BaseModel):
    source: str
    success: bool
    output_dir: Optional[str] = None
    markdown_path: Optional[str] = None
    json_path: Optional[str] = None
    table_count: int = 0
    image_count: int = 0
    page_count: int = 0
    annotation_count: int = 0
    handwriting_regions: int = 0
    processing_seconds: float = 0.0
    error: Optional[str] = None


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
def health():
    return {"status": "ok", "service": "docling"}


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    """Process a file or directory of files."""
    input_path = Path(req.input_path)
    output_base = Path(req.output_path)

    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Input path not found: {req.input_path}")

    # Collect files to process
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        raise HTTPException(status_code=400, detail=f"Input path is neither file nor directory: {req.input_path}")

    if not files:
        raise HTTPException(status_code=400, detail=f"No supported files found in {req.input_path}")

    parser = get_parser()
    output_cfg = OutputConfig(base_dir=output_base)
    exporter = DocumentExporter(output_cfg)

    results: list[FileResult] = []
    t0 = time.time()

    for file_path in files:
        result = _process_single(
            file_path, parser, exporter, req.copy_original,
            extract_annotations=req.extract_annotations,
            read_handwriting=req.read_handwriting,
        )
        results.append(result)

    total_seconds = time.time() - t0
    succeeded = sum(1 for r in results if r.success)

    return ProcessResponse(
        total_files=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
        total_seconds=round(total_seconds, 2),
    )


def _process_single(
    file_path: Path,
    parser: DoclingParser,
    exporter: DocumentExporter,
    copy_original: bool,
    extract_annotations: bool = True,
    read_handwriting: bool = True,
) -> FileResult:
    """Process a single file through the full enrichment pipeline.

    Pipeline order:
    1. Docling parse → structure (bboxes, types, sections, tables, images)
    2. PyMuPDF text enrichment → replace element text with cleaner PyMuPDF text
    3. Annotation extraction → match via PyMuPDF text blocks (same coordinate space)
    4. Handwriting detection + VLM → fill in text for handwritten/scanned content
    """
    logger.info(f"Processing: {file_path}")
    t0 = time.time()

    annotation_count = 0
    handwriting_region_count = 0
    is_pdf = file_path.suffix.lower() == ".pdf"

    try:
        # Step 1: Docling parsing (structure + OCR text)
        document = parser.parse(file_path)

        # Step 2: PyMuPDF text enrichment (PDF only)
        page_texts = {}
        if is_pdf:
            logger.info("Running PyMuPDF text enrichment...")
            document, page_texts = enrich_with_pymupdf_text(document, file_path)

        # Step 3: Annotation extraction (PDF only, uses PyMuPDF block→element map)
        unmatched_annotations = []
        if extract_annotations and is_pdf:
            logger.info("Running annotation extraction...")
            document, unmatched_annotations = enrich_document_with_annotations(
                document, file_path, page_texts
            )
            annotation_count = sum(
                len(elem.annotations)
                for page in document.pages
                for elem in page.elements
            )

        # Step 4: Handwriting detection + VLM reading (PDF only)
        if read_handwriting and is_pdf:
            try:
                logger.info("Running handwriting detection...")
                regions = detect_handwriting(
                    document, file_path, unmatched_annotations
                )
                handwriting_region_count = len(regions)

                if regions:
                    logger.info(f"Sending {len(regions)} regions to VLM...")
                    document = read_handwriting_regions(regions, document)
            except Exception as e:
                logger.warning(f"Handwriting detection/VLM failed (non-fatal): {e}")
                handwriting_region_count = 0

        # Step 4: Export enriched document
        exported = exporter.export(
            document=document,
            source_name=file_path.stem,
            copy_original=copy_original,
            table_format=TableFormat.BOTH,
        )

        elapsed = time.time() - t0

        return FileResult(
            source=str(file_path),
            success=True,
            output_dir=str(exported.get("root", "")),
            markdown_path=str(exported.get("markdown", "")),
            json_path=str(exported.get("json", "")),
            table_count=len(exported.get("tables", [])),
            image_count=len(exported.get("images", [])),
            page_count=document.page_count,
            annotation_count=annotation_count,
            handwriting_regions=handwriting_region_count,
            processing_seconds=round(elapsed, 2),
        )

    except Exception as exc:
        logger.error(f"Failed to process {file_path}: {exc}")
        return FileResult(
            source=str(file_path),
            success=False,
            error=str(exc),
            processing_seconds=round(time.time() - t0, 2),
        )
