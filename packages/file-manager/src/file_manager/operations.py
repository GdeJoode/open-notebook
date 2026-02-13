"""
File operations - copy, move, delete with async support.
"""

import asyncio
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Tuple

import aiofiles
import aiofiles.os
from loguru import logger


def generate_output_filename(
    original_filename: str,
    naming_scheme: Literal["timestamp_prefix", "date_prefix", "datetime_suffix", "original"],
) -> str:
    """
    Generate output filename based on the configured naming scheme.

    Args:
        original_filename: Original filename with extension.
        naming_scheme: Naming scheme to use.

    Returns:
        Generated filename.

    Examples:
        >>> generate_output_filename("document.pdf", "date_prefix")
        "2025-01-26_document.pdf"
    """
    stem = Path(original_filename).stem
    suffix = Path(original_filename).suffix
    now = datetime.now()

    if naming_scheme == "timestamp_prefix":
        prefix = now.strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{stem}{suffix}"
    elif naming_scheme == "date_prefix":
        prefix = now.strftime("%Y-%m-%d")
        return f"{prefix}_{stem}{suffix}"
    elif naming_scheme == "datetime_suffix":
        suffix_time = now.strftime("%Y%m%d_%H%M%S")
        return f"{stem}_{suffix_time}{suffix}"
    else:  # "original"
        return original_filename


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe filesystem usage.

    Args:
        filename: Original filename.

    Returns:
        Sanitized filename safe for filesystem.
    """
    # Replace problematic characters
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
    sanitized = "".join(c if c in safe_chars else "_" for c in filename)

    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip("_.")

    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed"

    return sanitized


def generate_unique_name(base_name: str, add_timestamp: bool = True) -> str:
    """
    Generate a unique name suitable for directory/file creation.

    Args:
        base_name: Base name (e.g., filename without extension).
        add_timestamp: Whether to append timestamp for uniqueness.

    Returns:
        Unique name safe for filesystem.
    """
    safe_name = sanitize_filename(base_name)

    if add_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_name}_{ts}"
    return safe_name


async def ensure_directory(directory_path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't (async).

    Args:
        directory_path: Path to directory.

    Returns:
        Path object for the directory.
    """
    path = Path(directory_path)
    await aiofiles.os.makedirs(str(path), exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def ensure_directory_sync(directory_path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't (sync).

    Args:
        directory_path: Path to directory.

    Returns:
        Path object for the directory.
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


async def copy_file(
    source_path: str,
    destination_dir: str,
    new_filename: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[Path, bool]:
    """
    Copy file to destination directory with optional renaming (async).

    Args:
        source_path: Source file path.
        destination_dir: Destination directory path.
        new_filename: Optional new filename (uses original if None).
        overwrite: Whether to overwrite existing files.

    Returns:
        Tuple of (destination_path, was_copied).

    Raises:
        FileNotFoundError: If source file doesn't exist.
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    dest_dir = await ensure_directory(destination_dir)
    filename = new_filename or source.name
    destination = dest_dir / filename

    if destination.exists() and not overwrite:
        logger.info(f"File already exists, skipping copy: {destination}")
        return destination, False

    # Use thread pool for blocking I/O
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, shutil.copy2, source, destination)
    logger.info(f"Copied file: {source} → {destination}")
    return destination, True


async def move_file(
    source_path: str,
    destination_dir: str,
    new_filename: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[Path, bool]:
    """
    Move file to destination directory with optional renaming (async).

    Args:
        source_path: Source file path.
        destination_dir: Destination directory path.
        new_filename: Optional new filename (uses original if None).
        overwrite: Whether to overwrite existing files.

    Returns:
        Tuple of (destination_path, was_moved).

    Raises:
        FileNotFoundError: If source file doesn't exist.
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    dest_dir = await ensure_directory(destination_dir)
    filename = new_filename or source.name
    destination = dest_dir / filename

    if destination.exists() and not overwrite:
        logger.info(f"File already exists, skipping move: {destination}")
        return destination, False

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, shutil.move, str(source), str(destination))
    logger.info(f"Moved file: {source} → {destination}")
    return destination, True


async def delete_file(file_path: str) -> bool:
    """
    Delete a file (async).

    Args:
        file_path: Path to file to delete.

    Returns:
        True if deleted, False if file didn't exist.
    """
    path = Path(file_path)
    if not path.exists():
        return False

    try:
        await aiofiles.os.remove(str(path))
        logger.info(f"Deleted file: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete file {path}: {e}")
        raise


async def delete_directory(dir_path: str, recursive: bool = False) -> bool:
    """
    Delete a directory (async).

    Args:
        dir_path: Path to directory to delete.
        recursive: Whether to delete contents recursively.

    Returns:
        True if deleted, False if directory didn't exist.
    """
    path = Path(dir_path)
    if not path.exists():
        return False

    try:
        loop = asyncio.get_event_loop()
        if recursive:
            await loop.run_in_executor(None, shutil.rmtree, str(path))
        else:
            await aiofiles.os.rmdir(str(path))
        logger.info(f"Deleted directory: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete directory {path}: {e}")
        raise


async def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes (async).

    Args:
        file_path: Path to file.

    Returns:
        File size in bytes.
    """
    stat = await aiofiles.os.stat(file_path)
    return stat.st_size


async def list_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = False,
) -> list[Path]:
    """
    List files in a directory (async).

    Args:
        directory: Directory path.
        pattern: Glob pattern for matching files.
        recursive: Whether to search recursively.

    Returns:
        List of file paths.
    """
    path = Path(directory)
    if not path.exists():
        return []

    loop = asyncio.get_event_loop()
    if recursive:
        files = await loop.run_in_executor(
            None, lambda: list(path.rglob(pattern))
        )
    else:
        files = await loop.run_in_executor(
            None, lambda: list(path.glob(pattern))
        )

    return [f for f in files if f.is_file()]


async def read_file(file_path: str, mode: str = "r") -> str | bytes:
    """
    Read file contents (async).

    Args:
        file_path: Path to file.
        mode: Read mode ('r' for text, 'rb' for binary).

    Returns:
        File contents.
    """
    async with aiofiles.open(file_path, mode=mode) as f:
        return await f.read()


async def write_file(file_path: str, content: str | bytes, mode: str = "w") -> Path:
    """
    Write content to file (async).

    Args:
        file_path: Path to file.
        content: Content to write.
        mode: Write mode ('w' for text, 'wb' for binary).

    Returns:
        Path to written file.
    """
    path = Path(file_path)
    await ensure_directory(str(path.parent))

    async with aiofiles.open(file_path, mode=mode) as f:
        await f.write(content)

    logger.debug(f"Wrote file: {path}")
    return path
