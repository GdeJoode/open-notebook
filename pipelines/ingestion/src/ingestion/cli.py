"""
Ingestion Pipeline CLI.

Command-line interface for the ingestion pipeline.

Usage:
    # Ingest a single document
    ingestion ingest document.pdf

    # Ingest with custom output directory
    ingestion ingest document.pdf --output ./my_output

    # Ingest audio/video
    ingestion ingest recording.mp3

    # Batch ingest a directory
    ingestion batch ./documents/ --recursive

    # Interactive mode with destination prompt
    ingestion ingest document.pdf --ask-destination
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from ingestion.config import (
    DestinationType,
    IngestionConfig,
    TableFormat,
    WhisperModel,
)
from ingestion.workflow import IngestionWorkflow, ingest


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    logger.remove()  # Remove default handler
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    )


def prompt_destination() -> DestinationType:
    """Prompt user for destination choice."""
    print("\nWhere should this content be stored?")
    print("  1. Project (associate with a specific project)")
    print("  2. Knowledge Base (general knowledge)")
    print()

    while True:
        choice = input("Enter choice [1/2]: ").strip()
        if choice == "1":
            return DestinationType.PROJECT
        elif choice == "2":
            return DestinationType.KNOWLEDGE_BASE
        else:
            print("Invalid choice. Please enter 1 or 2.")


def prompt_project_id() -> str:
    """Prompt user for project ID."""
    while True:
        project_id = input("Enter project ID: ").strip()
        if project_id:
            return project_id
        print("Project ID cannot be empty.")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ingestion",
        description="Ingestion Pipeline - Document parsing and audio/video transcription",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a single file",
    )
    ingest_parser.add_argument(
        "source",
        type=Path,
        help="Path to source file (PDF, DOCX, MP3, MP4, etc.)",
    )
    ingest_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: ./ingestion_output)",
    )
    ingest_parser.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="Custom name for output directory",
    )
    ingest_parser.add_argument(
        "--ask-destination",
        action="store_true",
        help="Prompt for destination (project vs knowledge base)",
    )
    ingest_parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project ID (implies --destination project)",
    )
    ingest_parser.add_argument(
        "--destination",
        choices=["project", "knowledge_base"],
        default=None,
        help="Destination type",
    )
    ingest_parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR for documents",
    )
    ingest_parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable image extraction",
    )
    ingest_parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="large-v3",
        help="WhisperX model size for audio/video",
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch ingest files from a directory",
    )
    batch_parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing source files",
    )
    batch_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory",
    )
    batch_parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process subdirectories recursively",
    )
    batch_parser.add_argument(
        "--concurrent",
        type=int,
        default=3,
        help="Maximum concurrent operations",
    )
    batch_parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="File pattern to match (e.g., '*.pdf')",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a file",
    )
    info_parser.add_argument(
        "source",
        type=Path,
        help="Path to source file",
    )

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    """Handle the ingest command."""
    source = args.source

    if not source.exists():
        logger.error(f"Source file not found: {source}")
        return 1

    # Build config
    config = IngestionConfig.from_env()

    if args.output:
        config.output.base_dir = args.output

    if args.no_ocr:
        config.docling.do_ocr = False

    if args.no_images:
        config.docling.generate_picture_images = False

    if args.whisper_model:
        config.whisperx.model_size = WhisperModel(args.whisper_model)

    # Determine destination
    destination = None
    project_id = None

    if args.project:
        destination = DestinationType.PROJECT
        project_id = args.project
    elif args.destination:
        destination = DestinationType(args.destination)
    elif args.ask_destination:
        destination = prompt_destination()
        if destination == DestinationType.PROJECT:
            project_id = prompt_project_id()

    # Process
    logger.info(f"Processing: {source}")

    workflow = IngestionWorkflow(config)
    result = workflow.process(
        source_path=source,
        destination=destination,
        project_id=project_id,
        source_name=args.name,
    )

    # Report result
    if result.success:
        logger.success(f"Ingestion complete: {result.output_directory}")
        print(f"\nOutput files:")
        print(f"  Markdown: {result.markdown_path}")
        print(f"  JSON:     {result.json_path}")
        if result.original_path:
            print(f"  Original: {result.original_path}")
        if result.table_paths:
            print(f"  Tables:   {len(result.table_paths)} files")
        if result.image_paths:
            print(f"  Images:   {len(result.image_paths)} files")
        print(f"\nProcessing time: {result.processing_time_seconds:.2f}s")
        return 0
    else:
        logger.error(f"Ingestion failed: {result.error_message}")
        return 1


def cmd_batch(args: argparse.Namespace) -> int:
    """Handle the batch command."""
    source_dir = args.source_dir

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return 1

    # Build config
    config = IngestionConfig.from_env()
    config.max_concurrent_files = args.concurrent

    if args.output:
        config.output.base_dir = args.output

    # Find files
    if args.recursive:
        files = list(source_dir.rglob(args.pattern))
    else:
        files = list(source_dir.glob(args.pattern))

    # Filter to supported types
    supported_files = [f for f in files if f.is_file() and config.is_supported(f)]

    if not supported_files:
        logger.warning(f"No supported files found in {source_dir}")
        return 0

    logger.info(f"Found {len(supported_files)} files to process")

    # Process
    workflow = IngestionWorkflow(config)

    def on_progress(current: int, total: int, result):
        status = "✓" if result.success else "✗"
        logger.info(f"[{current}/{total}] {status} {result.source_metadata.filename}")

    results = asyncio.run(workflow.process_batch(supported_files, on_progress))

    # Summary
    success_count = sum(1 for r in results if r.success)
    fail_count = len(results) - success_count

    print(f"\nBatch complete: {success_count} succeeded, {fail_count} failed")

    if fail_count > 0:
        print("\nFailed files:")
        for r in results:
            if not r.success:
                print(f"  - {r.source_metadata.filename}: {r.error_message}")

    return 0 if fail_count == 0 else 1


def cmd_info(args: argparse.Namespace) -> int:
    """Handle the info command."""
    source = args.source

    if not source.exists():
        logger.error(f"Source file not found: {source}")
        return 1

    config = IngestionConfig()

    print(f"\nFile Information: {source.name}")
    print("-" * 40)
    print(f"Path:      {source}")
    print(f"Size:      {source.stat().st_size / 1024:.1f} KB")
    print(f"Extension: {source.suffix}")

    if config.is_document(source):
        print(f"Type:      Document")
        print(f"Processor: Docling")
    elif config.is_audio(source):
        print(f"Type:      Audio")
        print(f"Processor: WhisperX")
    elif config.is_video(source):
        print(f"Type:      Video")
        print(f"Processor: WhisperX (with audio extraction)")
    else:
        print(f"Type:      Unsupported")
        return 1

    print(f"Supported: {'Yes' if config.is_supported(source) else 'No'}")

    return 0


def main() -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "batch":
        return cmd_batch(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
