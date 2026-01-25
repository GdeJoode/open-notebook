"""
MIME type detection and file type utilities.
"""

import mimetypes
from pathlib import Path
from typing import Optional, Tuple

# Initialize mimetypes database
mimetypes.init()

# Custom MIME type mappings for common document types
CUSTOM_MIME_TYPES = {
    ".md": "text/markdown",
    ".epub": "application/epub+zip",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

# Document type categories
DOCUMENT_CATEGORIES = {
    "pdf": [".pdf"],
    "word": [".doc", ".docx"],
    "spreadsheet": [".xls", ".xlsx", ".csv"],
    "presentation": [".ppt", ".pptx"],
    "text": [".txt", ".md", ".rst"],
    "html": [".html", ".htm"],
    "ebook": [".epub", ".mobi"],
    "image": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"],
}


def get_mime_type(file_path: str) -> str:
    """
    Get MIME type for a file.

    Args:
        file_path: Path to the file.

    Returns:
        MIME type string (defaults to application/octet-stream if unknown).
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    # Check custom mappings first
    if extension in CUSTOM_MIME_TYPES:
        return CUSTOM_MIME_TYPES[extension]

    # Fall back to mimetypes library
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def get_extension_for_mime(mime_type: str) -> Optional[str]:
    """
    Get file extension for a MIME type.

    Args:
        mime_type: MIME type string.

    Returns:
        File extension (including dot) or None if unknown.
    """
    # Check custom mappings first (reverse lookup)
    for ext, mime in CUSTOM_MIME_TYPES.items():
        if mime == mime_type:
            return ext

    # Fall back to mimetypes library
    return mimetypes.guess_extension(mime_type)


def get_file_category(file_path: str) -> Optional[str]:
    """
    Get document category for a file.

    Args:
        file_path: Path to the file.

    Returns:
        Category name (pdf, word, spreadsheet, etc.) or None if not categorized.
    """
    extension = Path(file_path).suffix.lower()
    for category, extensions in DOCUMENT_CATEGORIES.items():
        if extension in extensions:
            return category
    return None


def is_document(file_path: str) -> bool:
    """
    Check if file is a document type (not image).

    Args:
        file_path: Path to the file.

    Returns:
        True if file is a document type.
    """
    category = get_file_category(file_path)
    return category is not None and category != "image"


def is_image(file_path: str) -> bool:
    """
    Check if file is an image.

    Args:
        file_path: Path to the file.

    Returns:
        True if file is an image.
    """
    return get_file_category(file_path) == "image"


def validate_extension(file_path: str, allowed_extensions: list[str]) -> Tuple[bool, str]:
    """
    Validate file extension against allowed list.

    Args:
        file_path: Path to the file.
        allowed_extensions: List of allowed extensions (with dots, e.g., [".pdf", ".docx"]).

    Returns:
        Tuple of (is_valid, error_message).
    """
    extension = Path(file_path).suffix.lower()
    if not extension:
        return False, "File has no extension"

    allowed_lower = [ext.lower() for ext in allowed_extensions]
    if extension not in allowed_lower:
        return False, f"Extension {extension} not allowed. Allowed: {', '.join(allowed_extensions)}"

    return True, ""


def get_content_type_header(file_path: str) -> str:
    """
    Get Content-Type header value for file (for HTTP responses).

    Args:
        file_path: Path to the file.

    Returns:
        Content-Type header value with charset for text types.
    """
    mime_type = get_mime_type(file_path)

    # Add charset for text types
    if mime_type.startswith("text/"):
        return f"{mime_type}; charset=utf-8"

    return mime_type
