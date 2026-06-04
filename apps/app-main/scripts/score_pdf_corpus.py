"""Threshold-tuning CLI: score a directory (or a single PDF) through Docling.

Run via:

    uv run --project apps/app-main \
        python apps/app-main/scripts/score_pdf_corpus.py <pdf-or-dir>

Prints one Markdown table row per PDF with the overall confidence, the
six per-signal scores, and the @0.95 decision. Output is paste-ready
for ``docs/tracks/A-mineru/threshold-tuning.md``.

The script picks Docling automatically:

* If ``USE_DOCLING_SERVICE`` is truthy and the docling container is up
  on ``DOCLING_SERVICE_URL``, requests are routed there.
* Otherwise it runs Docling in-process via ``IngestionWorkflow`` —
  slower but no container needed.

MinerU is **not** invoked; we only need the Docling-side score (the
confidence threshold is what we're tuning).

This module is side-effect-free except for stdout; the helper functions
(`format_row`, `format_header`, `dominant_signal`) are imported by
``tests/test_score_pdf_corpus.py`` so the row format stays stable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

# Allow ``python apps/app-main/scripts/score_pdf_corpus.py`` to import
# the workspace package without installing it as a script entry point.
_APP_MAIN_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_APP_MAIN_SRC) not in sys.path:
    sys.path.insert(0, str(_APP_MAIN_SRC))

from app_main.services.parsing.confidence import (  # noqa: E402
    DEFAULT_THRESHOLD,
    DoclingConfidenceScore,
    SIGNAL_WEIGHTS,
    score_docling_extraction,
)


# Order of columns in the Markdown table — matches threshold-tuning.md.
SIGNAL_ORDER: tuple[str, ...] = (
    "ocr_confidence",
    "text_density",
    "heading_rate",
    "table_success",
    "image_text_ratio",
    "unknown_element_ratio",
)


def format_header(threshold: float = DEFAULT_THRESHOLD) -> str:
    """Return the Markdown header + separator rows for the score table."""
    cols = (
        ["fixture", "overall"]
        + [_short_signal_name(name) for name in SIGNAL_ORDER]
        + [f"decision @{threshold:.2f}", "dominant_signal"]
    )
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    return f"{header}\n{separator}"


def format_row(name: str, score: DoclingConfidenceScore) -> str:
    """Render a single PDF's score as a Markdown table row."""
    signal_cells = [
        f"{score.signals.get(sig, 0.0):.2f}" for sig in SIGNAL_ORDER
    ]
    return (
        f"| {name} "
        f"| {score.overall:.3f} "
        + "".join(f"| {cell} " for cell in signal_cells)
        + f"| {score.decision} "
        f"| {dominant_signal(score)} |"
    )


def dominant_signal(score: DoclingConfidenceScore) -> str:
    """Identify the lowest-scoring weighted signal.

    Threshold-tuning workflows care most about *which* signal dragged
    the score down, not the precise number. We multiply each signal by
    its weight and return the smallest contributor.
    """
    if not score.signals:
        return "n/a"
    weighted = {
        name: SIGNAL_WEIGHTS.get(name, 0.0) * score.signals.get(name, 0.0)
        for name in score.signals
    }
    # Lowest weighted contribution = biggest "missing" potential.
    return min(weighted, key=weighted.get)


def _short_signal_name(name: str) -> str:
    """Compact column names for the printed Markdown."""
    return {
        "ocr_confidence": "ocr",
        "text_density": "density",
        "heading_rate": "heading",
        "table_success": "table",
        "image_text_ratio": "image",
        "unknown_element_ratio": "unknown",
    }.get(name, name)


def _iter_pdf_paths(target: Path) -> Iterable[Path]:
    """Yield PDFs to score, sorted for deterministic output."""
    if target.is_file():
        yield target
        return
    yield from sorted(p for p in target.glob("*.pdf") if p.is_file())


async def _run_docling_via_host_bridge(path: Path) -> Any:
    """Stage on host, POST container path to docling, parse result.

    Defaults assume the docker-compose layout shipped with the repo:
    host ``docling_input/`` mounted as ``/data/input`` and host
    ``docling_output/`` mounted as ``/data/output``. Overridable via
    env vars so non-default layouts still work.
    """
    import shutil
    import uuid

    import httpx

    from ingestion.models import IngestionResult, SourceMetadata, SourceType

    host_input = Path(
        os.environ.get(
            "DOCLING_HOST_INPUT_DIR",
            str(Path.cwd() / "docling_input"),
        )
    )
    host_output = Path(
        os.environ.get(
            "DOCLING_HOST_OUTPUT_DIR",
            str(Path.cwd() / "docling_output"),
        )
    )
    container_input = Path(
        os.environ.get("DOCLING_CONTAINER_INPUT_DIR", "/data/input")
    )
    container_output = Path(
        os.environ.get("DOCLING_CONTAINER_OUTPUT_DIR", "/data/output")
    )
    service_url = (
        os.environ.get("DOCLING_SERVICE_URL", "http://localhost:8100")
        .rstrip("/")
    )

    host_input.mkdir(parents=True, exist_ok=True)
    staged_name = f"score_{uuid.uuid4().hex[:8]}__{path.name}"
    host_staged = host_input / staged_name
    container_staged = container_input / staged_name
    shutil.copy2(path, host_staged)

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{service_url}/process",
                json={
                    "input_path": str(container_staged),
                    "output_path": str(container_output),
                    "copy_original": False,
                    "extract_annotations": False,
                    "read_handwriting": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results") or []
        if not results or not results[0].get("success"):
            return None
        file_result = results[0]

        # Translate container paths in the response back to host paths
        # so the existing IngestionResult-loader can read the JSON.
        from app_main.services.docling_http_client import DoclingHttpClient

        client_for_loader = DoclingHttpClient(
            service_url=service_url,
            input_dir=host_input,
            output_dir=host_output,
        )
        container_output_dir = Path(file_result["output_dir"])
        host_output_dir = host_output / container_output_dir.relative_to(
            container_output
        )
        json_path = host_output / Path(file_result["json_path"]).relative_to(
            container_output
        )
        markdown_path = host_output / Path(
            file_result["markdown_path"]
        ).relative_to(container_output)

        metadata = SourceMetadata.from_path(path, SourceType.DOCUMENT)
        ingestion_result = IngestionResult(source_metadata=metadata)
        document = client_for_loader._load_document(
            source_path=path,
            json_path=json_path,
            markdown_path=markdown_path,
            service_result=file_result,
        )
        ingestion_result.document = document
        ingestion_result.output_directory = host_output_dir
        ingestion_result.markdown_path = markdown_path
        ingestion_result.json_path = json_path
        metadata.page_count = document.page_count
        return ingestion_result.complete(success=True)
    finally:
        if host_staged.exists():
            try:
                host_staged.unlink()
            except OSError:
                pass


async def _score_one(path: Path, threshold: float) -> Optional[DoclingConfidenceScore]:
    """Run docling on a single PDF and return its score, or ``None`` on
    extraction failure (logged to stderr)."""
    # Lazy import — these pull docling + httpx + a heavy chain of
    # ingestion modules. Keep the CLI fast for ``--help``.
    docling_result = await _run_docling(path)
    if docling_result is None or not getattr(docling_result, "success", False):
        return None
    return score_docling_extraction(docling_result, threshold=threshold)


async def _run_docling(path: Path) -> Any:
    """Run Docling either via the HTTP service or in-process workflow.

    When running on the docker host (where the script lives) against
    a docling container with bind-mounts ``host:docling_input → /data/input``
    and ``host:docling_output → /data/output``, the staging step needs
    to write to the *host* side of the bind-mount while the service
    needs to see the *container* path. Use ``DoclingHostRunner`` for
    that bridge; the upstream ``DoclingHttpClient`` assumes both ends
    see the same path (the docker-network case) and would fail here.
    """
    if os.environ.get("USE_DOCLING_SERVICE", "").lower() in {
        "1", "true", "yes", "on"
    }:
        return await _run_docling_via_host_bridge(path)

    # In-process: instantiate the same IngestionWorkflow SourceExtractor
    # falls back to. We need a ContentSettings to drive it; the model
    # default is fine for tuning (we only care about the score signals,
    # not chunking or embeddings).
    from ingestion.workflow import IngestionWorkflow
    from shared.models.settings import ContentSettings

    from app_main.services.source_extractor import build_ingestion_config

    config = build_ingestion_config(ContentSettings())
    workflow = IngestionWorkflow(config)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, workflow.process, path)


async def _score_corpus(target: Path, threshold: float) -> int:
    """Iterate PDFs, print the Markdown table to stdout.

    Returns process exit code (0 on full success, 1 if any PDF failed).
    """
    pdf_paths = list(_iter_pdf_paths(target))
    if not pdf_paths:
        print(f"No PDFs found under {target}", file=sys.stderr)
        return 1

    print(format_header(threshold))
    any_failure = False
    for pdf in pdf_paths:
        try:
            score = await _score_one(pdf, threshold)
        except Exception as exc:  # pragma: no cover — surfaced to stderr
            print(f"# {pdf.name}: extraction crashed — {exc}", file=sys.stderr)
            any_failure = True
            continue
        if score is None:
            print(f"# {pdf.name}: docling reported failure", file=sys.stderr)
            any_failure = True
            continue
        print(format_row(pdf.name, score))
    return 1 if any_failure else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score a PDF or directory of PDFs through Docling and print "
            "a Markdown row per file for threshold tuning."
        )
    )
    parser.add_argument(
        "target",
        type=Path,
        help="A single PDF file or a directory containing *.pdf files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Confidence threshold for the decision column (default {DEFAULT_THRESHOLD}).",
    )
    args = parser.parse_args(argv)

    if not args.target.exists():
        print(f"Target does not exist: {args.target}", file=sys.stderr)
        return 1

    return asyncio.run(_score_corpus(args.target, args.threshold))


if __name__ == "__main__":
    raise SystemExit(main())
