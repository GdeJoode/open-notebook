"""
File Manager - File organization and storage management.

This package provides async file operations, storage management,
and MIME type utilities for document processing workflows.
"""

from file_manager.config import FileManagerConfig, get_config, set_config
from file_manager.mime import (
    get_file_category,
    get_mime_type,
    is_document,
    is_image,
    validate_extension,
)
from file_manager.operations import (
    copy_file,
    delete_directory,
    delete_file,
    ensure_directory,
    generate_output_filename,
    generate_unique_name,
    get_file_size,
    list_files,
    move_file,
    read_file,
    sanitize_filename,
    write_file,
)
from file_manager.storage import StorageManager

__all__ = [
    # Config
    "FileManagerConfig",
    "get_config",
    "set_config",
    # Storage
    "StorageManager",
    # Operations
    "copy_file",
    "move_file",
    "delete_file",
    "delete_directory",
    "read_file",
    "write_file",
    "list_files",
    "get_file_size",
    "ensure_directory",
    "generate_output_filename",
    "generate_unique_name",
    "sanitize_filename",
    # MIME
    "get_mime_type",
    "get_file_category",
    "is_document",
    "is_image",
    "validate_extension",
]
