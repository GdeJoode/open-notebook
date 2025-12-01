"""
File management utilities for Open Notebook.

Handles file operations, directory organization, and naming schemes for the
file management system.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple

from loguru import logger


def generate_output_filename(
    original_filename: str,
    naming_scheme: Literal["timestamp_prefix", "date_prefix", "datetime_suffix", "original"],
) -> str:
    """
    Generate output filename based on the configured naming scheme.

    Args:
        original_filename: Original filename with extension
        naming_scheme: Naming scheme to use

    Returns:
        Generated filename

    Examples:
        >>> generate_output_filename("document.pdf", "date_prefix")
        "2025-11-05_document.pdf"
        >>> generate_output_filename("report.docx", "timestamp_prefix")
        "20251105_143022_report.docx"
    """
    stem = Path(original_filename).stem
    suffix = Path(original_filename).suffix
    now = datetime.now()

    if naming_scheme == "timestamp_prefix":
        # Format: YYYYMMDD_HHMMSS_filename.ext
        prefix = now.strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{stem}{suffix}"

    elif naming_scheme == "date_prefix":
        # Format: YYYY-MM-DD_filename.ext
        prefix = now.strftime("%Y-%m-%d")
        return f"{prefix}_{stem}{suffix}"

    elif naming_scheme == "datetime_suffix":
        # Format: filename_YYYYMMDD_HHMMSS.ext
        suffix_time = now.strftime("%Y%m%d_%H%M%S")
        return f"{stem}_{suffix_time}{suffix}"

    else:  # "original"
        return original_filename


def ensure_directory(directory_path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory_path: Path to directory

    Returns:
        Path object for the directory

    Raises:
        OSError: If directory creation fails
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def copy_file(
    source_path: str,
    destination_dir: str,
    new_filename: str | None = None,
    overwrite: bool = False,
) -> Tuple[Path, bool]:
    """
    Copy file to destination directory with optional renaming.

    Args:
        source_path: Source file path
        destination_dir: Destination directory path
        new_filename: Optional new filename (uses original if None)
        overwrite: Whether to overwrite existing files

    Returns:
        Tuple of (destination_path, was_copied)
        was_copied is False if file already exists and overwrite=False

    Raises:
        FileNotFoundError: If source file doesn't exist
        OSError: If copy operation fails
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Ensure destination directory exists
    dest_dir = ensure_directory(destination_dir)

    # Determine destination filename
    filename = new_filename or source.name
    destination = dest_dir / filename

    # Handle existing files
    if destination.exists() and not overwrite:
        logger.info(f"File already exists, skipping copy: {destination}")
        return destination, False

    # Copy file
    shutil.copy2(source, destination)
    logger.info(f"✅ Copied file: {source} → {destination}")
    return destination, True


def move_file(
    source_path: str,
    destination_dir: str,
    new_filename: str | None = None,
    overwrite: bool = False,
) -> Tuple[Path, bool]:
    """
    Move file to destination directory with optional renaming.

    Args:
        source_path: Source file path
        destination_dir: Destination directory path
        new_filename: Optional new filename (uses original if None)
        overwrite: Whether to overwrite existing files

    Returns:
        Tuple of (destination_path, was_moved)
        was_moved is False if file already exists and overwrite=False

    Raises:
        FileNotFoundError: If source file doesn't exist
        OSError: If move operation fails
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Ensure destination directory exists
    dest_dir = ensure_directory(destination_dir)

    # Determine destination filename
    filename = new_filename or source.name
    destination = dest_dir / filename

    # Handle existing files
    if destination.exists() and not overwrite:
        logger.info(f"File already exists, skipping move: {destination}")
        return destination, False

    # Move file
    shutil.move(str(source), str(destination))
    logger.info(f"✅ Moved file: {source} → {destination}")
    return destination, True


def create_document_subdirectory(
    base_markdown_dir: str,
    document_name: str,
) -> Path:
    """
    Create subdirectory structure for a document's markdown and assets.

    Creates:
        base_markdown_dir/document_name/
        base_markdown_dir/document_name/images/
        base_markdown_dir/document_name/tables/

    Args:
        base_markdown_dir: Base markdown directory path
        document_name: Name of the document (used as subdirectory name)

    Returns:
        Path to the document subdirectory

    Raises:
        OSError: If directory creation fails
    """
    # Create main document directory
    doc_dir = ensure_directory(os.path.join(base_markdown_dir, document_name))

    # Create subdirectories for assets
    ensure_directory(doc_dir / "images")
    ensure_directory(doc_dir / "tables")

    logger.info(f"✅ Created document subdirectory structure: {doc_dir}")
    return doc_dir


def generate_unique_document_name(base_name: str, timestamp: bool = True) -> str:
    """
    Generate a unique document name for subdirectory creation.

    Args:
        base_name: Base name for the document (e.g., filename without extension)
        timestamp: Whether to append timestamp for uniqueness

    Returns:
        Unique document name suitable for directory creation

    Examples:
        >>> generate_unique_document_name("my_document", timestamp=True)
        "my_document_20251105_143022"
        >>> generate_unique_document_name("report", timestamp=False)
        "report"
    """
    # Sanitize base name for filesystem
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name)

    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_name}_{ts}"
    else:
        return safe_name


def organize_file(
    file_path: str,
    input_dir: str,
    output_dir: str,
    file_operation: Literal["copy", "move", "none"],
    naming_scheme: Literal["timestamp_prefix", "date_prefix", "datetime_suffix", "original"],
) -> Tuple[Path | None, Path | None]:
    """
    Organize a file according to the file management settings.

    Workflow:
    1. Copy/Move to INPUT directory (if file_operation is not "none")
    2. Copy to OUTPUT directory with naming scheme

    Args:
        file_path: Path to the file to organize
        input_dir: Input directory path
        output_dir: Output directory path
        file_operation: Operation to perform (copy, move, or none)
        naming_scheme: Naming scheme for output file

    Returns:
        Tuple of (input_path, output_path)
        input_path is None if file_operation is "none"
        output_path is the file in OUTPUT directory with renamed filename

    Raises:
        FileNotFoundError: If source file doesn't exist
        OSError: If file operations fail
    """
    source = Path(file_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    input_path = None
    output_path = None

    # Step 1: Organize into INPUT directory
    if file_operation == "copy":
        input_path, _ = copy_file(str(source), input_dir)
        logger.info(f"📁 Copied to INPUT: {input_path}")
    elif file_operation == "move":
        input_path, _ = move_file(str(source), input_dir)
        logger.info(f"📁 Moved to INPUT: {input_path}")
    else:  # "none"
        logger.info(f"📁 Keeping file in original location: {source}")
        input_path = None

    # Step 2: Copy to OUTPUT directory with naming scheme
    output_filename = generate_output_filename(source.name, naming_scheme)
    output_path, _ = copy_file(str(source), output_dir, new_filename=output_filename)
    logger.info(f"📄 Saved to OUTPUT: {output_path}")

    return input_path, output_path
