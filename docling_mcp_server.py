"""
Docling MCP Server — GraniteDocling VLM pipeline.

Uses IBM's Granite-Docling-258M as a single-pass VLM for document
conversion. Replaces the ensemble pipeline (separate OCR, layout,
table models) with one compact model that handles everything.

Requirements on your PC:
  - Python 3.12 with the open-notebook .venv activated
  - Docling >= 2.58.0 with VLM extras:  pip install "docling[vlm]"
  - mcp[cli] installed:  pip install "mcp[cli]"
  - NVIDIA GPU with CUDA (uses your 5070 Ti)

Usage in claude_desktop_config.json:
{
  "mcpServers": {
    "docling-local": {
      "command": "E:/Repos/Private/open-notebook/.venv/Scripts/python.exe",
      "args": ["E:/Repos/Private/open-notebook/docling_mcp_server.py"],
      "env": {
        "INGESTION_OUTPUT_DIR": "E:/Regiodeals/pipeline_outputs"
      }
    }
  }
}
"""

import asyncio
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ── MCP server setup ──────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP(
    "Docling Local (GraniteDocling)",
    description="Local document parser using GraniteDocling VLM pipeline",
)

# Thread pool for blocking Docling calls (GPU inference + file I/O)
_executor = ThreadPoolExecutor(max_workers=2)

# ── Configuration ─────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = os.environ.get(
    "INGESTION_OUTPUT_DIR", "E:/Regiodeals/pipeline_outputs"
)

# Lazy-initialized converter (heavy — loads VLM on first use)
_converter = None


def _get_converter():
    """Get or create the DocumentConverter with GraniteDocling VLM pipeline."""
    global _converter
    if _converter is not None:
        return _converter

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel import vlm_model_specs
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    # GraniteDocling with Transformers framework (CUDA on your 5070 Ti)
    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_model_specs.GRANITEDOCLING,
    )

    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )
    return _converter


def _export_document(doc, source_path: Path, output_dir: str) -> dict:
    """
    Export a DoclingDocument to a structured directory.

    Output structure:
        [output_dir]/[filename]/
            original_file/          # Copy of original
            output/extracted_info/
                document.md         # Full markdown
                document.json       # Structured JSON (DoclingDocument)
                metadata.json       # Extraction metadata
                tables/             # Tables as .md and .csv
                images/             # Extracted images
    """
    source_name = source_path.stem
    root = Path(output_dir) / source_name
    info_dir = root / "output" / "extracted_info"
    tables_dir = info_dir / "tables"
    images_dir = info_dir / "images"
    original_dir = root / "original_file"

    # Create directories
    for d in [info_dir, tables_dir, images_dir, original_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy original file
    shutil.copy2(source_path, original_dir / source_path.name)

    # Export markdown
    md_content = doc.export_to_markdown()
    md_path = info_dir / "document.md"
    md_path.write_text(md_content, encoding="utf-8")

    # Export structured JSON
    json_path = info_dir / "document.json"
    json_path.write_text(
        doc.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # Extract tables
    table_files = []
    from docling_core.types.doc import TableItem
    for i, (item, _level) in enumerate(doc.iterate_items()):
        if isinstance(item, TableItem):
            tbl_name = f"table_{i:03d}"
            # Markdown format
            try:
                md_table = item.export_to_markdown()
                tbl_md = tables_dir / f"{tbl_name}.md"
                tbl_md.write_text(md_table, encoding="utf-8")
                table_files.append(str(tbl_md))
            except Exception:
                pass
            # CSV format
            try:
                from io import StringIO
                import csv
                if hasattr(item, "data") and item.data:
                    df_data = item.data
                    tbl_csv = tables_dir / f"{tbl_name}.csv"
                    # Export table grid to CSV
                    if hasattr(df_data, "grid"):
                        grid = df_data.grid
                        with open(tbl_csv, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            for row in grid:
                                writer.writerow([cell.text if hasattr(cell, 'text') else str(cell) for cell in row])
                        table_files.append(str(tbl_csv))
            except Exception:
                pass

    # Extract images
    image_files = []
    from docling_core.types.doc import PictureItem
    for i, (item, _level) in enumerate(doc.iterate_items()):
        if isinstance(item, PictureItem):
            if hasattr(item, "image") and item.image is not None:
                try:
                    img_path = images_dir / f"image_{i:03d}.png"
                    pil_img = item.image.pil_image
                    if pil_img is not None:
                        pil_img.save(str(img_path))
                        image_files.append(str(img_path))
                except Exception:
                    pass

    # Count stats
    table_count = sum(1 for item, _ in doc.iterate_items() if isinstance(item, TableItem))
    image_count = sum(1 for item, _ in doc.iterate_items() if isinstance(item, PictureItem))
    page_count = doc.num_pages() if hasattr(doc, "num_pages") else 0

    # Write metadata
    metadata = {
        "source": str(source_path),
        "source_name": source_name,
        "page_count": page_count,
        "table_count": table_count,
        "image_count": image_count,
        "pipeline": "GraniteDocling-258M (VlmPipeline)",
    }
    meta_path = info_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "root": str(root),
        "markdown": str(md_path),
        "json": str(json_path),
        "metadata": str(meta_path),
        "tables": table_files,
        "images": image_files,
        "page_count": page_count,
        "table_count": table_count,
        "image_count": image_count,
    }


def _do_convert_and_export(source_path: str, output_dir: str) -> dict:
    """Blocking conversion + export — runs in a thread via executor."""
    source = Path(source_path)
    start = time.time()
    converter = _get_converter()
    result = converter.convert(source=str(source))
    doc = result.document
    elapsed = time.time() - start
    exported = _export_document(doc, source, output_dir)
    return {**exported, "processing_time_seconds": round(elapsed, 2)}


# ── MCP Tools ─────────────────────────────────────────────────────


@mcp.tool()
async def convert_document(
    source_path: str,
    ctx: Context,
    output_dir: str | None = None,
) -> str:
    """
    Convert a document (PDF) using GraniteDocling VLM pipeline.

    Single-pass conversion: layout, OCR, tables, equations and images
    are all handled by one compact 258M-parameter model running on
    your local GPU.

    Output structure:
        [output_dir]/[filename]/
            original_file/          # Copy of original
            output/extracted_info/
                document.md         # Full markdown
                document.json       # Structured JSON
                metadata.json       # Extraction metadata
                tables/             # Tables as .md and .csv
                images/             # Extracted images

    Args:
        source_path: Full path to the source document.
        output_dir:  Output directory (default: INGESTION_OUTPUT_DIR env var).

    Returns:
        JSON summary with paths and statistics (no document content).
    """
    source = Path(source_path)
    if not source.exists():
        return json.dumps({"error": f"File not found: {source_path}"})

    out_dir = output_dir or DEFAULT_OUTPUT_DIR

    await ctx.info(f"Starting conversion: {source.name}")

    try:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            _executor, _do_convert_and_export, str(source), out_dir,
        )

        # Heartbeat every 5s — ctx.info() always sends a notification over
        # stdio (unlike report_progress which requires a progressToken from
        # the client).  This keeps the connection alive during long GPU runs.
        elapsed = 0
        while not future.done():
            await asyncio.sleep(5)
            elapsed += 5
            await ctx.info(f"Converting {source.name}... ({elapsed}s elapsed)")

        exported = await future
        await ctx.info(f"Conversion complete: {source.name}")

        summary = {
            "status": "success",
            "source": str(source),
            "pages": exported["page_count"],
            "tables": exported["table_count"],
            "images": exported["image_count"],
            "processing_time_seconds": exported["processing_time_seconds"],
            "output_directory": exported["root"],
            "markdown_path": exported["markdown"],
            "json_path": exported["json"],
            "table_files": exported["tables"],
            "image_files": exported["images"],
        }
        return json.dumps(summary, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "source": str(source)})


@mcp.tool()
async def batch_convert(
    source_directory: str,
    ctx: Context,
    output_dir: str | None = None,
    pattern: str = "*.pdf",
) -> str:
    """
    Convert all matching documents in a directory.

    Args:
        source_directory: Directory containing documents.
        output_dir:       Output directory (default: INGESTION_OUTPUT_DIR).
        pattern:          Glob pattern to match files (default: *.pdf).

    Returns:
        JSON summary per file.
    """
    source_dir = Path(source_directory)
    if not source_dir.is_dir():
        return json.dumps({"error": f"Directory not found: {source_directory}"})

    files = sorted(source_dir.glob(pattern))
    if not files:
        return json.dumps({"error": f"No files matching '{pattern}' in {source_directory}"})

    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    total_files = len(files)
    await ctx.info(f"Batch converting {total_files} file(s) from {source_dir.name}")

    loop = asyncio.get_running_loop()
    results = []

    for idx, f in enumerate(files):
        await ctx.info(f"[{idx + 1}/{total_files}] Converting {f.name}")

        try:
            future = loop.run_in_executor(
                _executor, _do_convert_and_export, str(f), out_dir,
            )

            # Heartbeat while this file converts
            elapsed = 0
            while not future.done():
                await asyncio.sleep(5)
                elapsed += 5
                await ctx.info(
                    f"[{idx + 1}/{total_files}] {f.name}... ({elapsed}s elapsed)"
                )

            exported = await future
            await ctx.info(
                f"[{idx + 1}/{total_files}] {f.name} done "
                f"({exported['processing_time_seconds']}s)"
            )

            results.append({
                "status": "success",
                "source": str(f),
                "pages": exported["page_count"],
                "tables": exported["table_count"],
                "images": exported["image_count"],
                "processing_time_seconds": exported["processing_time_seconds"],
                "output_directory": exported["root"],
            })
        except Exception as e:
            results.append({"status": "error", "source": str(f), "error": str(e)})
    return json.dumps({"processed": len(results), "results": results}, indent=2)


@mcp.tool()
def list_outputs(output_dir: str | None = None) -> str:
    """
    List previously converted documents in the output directory.

    Args:
        output_dir: Output directory to inspect (default: INGESTION_OUTPUT_DIR).

    Returns:
        JSON list of converted documents with their output paths.
    """
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    if not out_dir.exists():
        return json.dumps({"error": f"Output directory not found: {out_dir}"})

    entries = []
    for d in sorted(out_dir.iterdir()):
        if d.is_dir():
            md_path = d / "output" / "extracted_info" / "document.md"
            meta_path = d / "output" / "extracted_info" / "metadata.json"
            entry = {"name": d.name, "path": str(d), "has_markdown": md_path.exists()}

            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    entry["pages"] = meta.get("page_count")
                    entry["tables"] = meta.get("table_count")
                    entry["images"] = meta.get("image_count")
                    entry["pipeline"] = meta.get("pipeline")
                except Exception:
                    pass

            entries.append(entry)

    return json.dumps({"output_dir": str(out_dir), "documents": entries}, indent=2)


@mcp.tool()
def read_extracted_markdown(source_name: str, output_dir: str | None = None) -> str:
    """
    Read the extracted markdown for a previously converted document.

    Args:
        source_name: Name of the document (directory name in output).
        output_dir:  Output directory (default: INGESTION_OUTPUT_DIR).

    Returns:
        The markdown content, or error message.
    """
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    md_path = out_dir / source_name / "output" / "extracted_info" / "document.md"

    if not md_path.exists():
        return json.dumps({"error": f"Markdown not found: {md_path}"})

    return md_path.read_text(encoding="utf-8")


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")
