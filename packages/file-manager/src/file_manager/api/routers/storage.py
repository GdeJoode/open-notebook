"""
Storage management API routes.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from file_manager.config import get_config
from file_manager.mime import validate_extension
from file_manager.storage import StorageManager

router = APIRouter()


class OrganizeRequest(BaseModel):
    """File organization request."""

    file_path: str
    file_operation: Optional[Literal["copy", "move", "none"]] = None
    naming_scheme: Optional[
        Literal["timestamp_prefix", "date_prefix", "datetime_suffix", "original"]
    ] = None


class OrganizeResponse(BaseModel):
    """File organization response."""

    input_path: Optional[str]
    output_path: str


class DocumentDirectoryResponse(BaseModel):
    """Document directory creation response."""

    root: str
    images: str
    tables: str


class StorageStatsResponse(BaseModel):
    """Storage statistics response."""

    uploads: Dict[str, Any]
    input: Dict[str, Any]
    output: Dict[str, Any]
    markdown: Dict[str, Any]
    temp: Dict[str, Any]


@router.post("/setup")
async def setup_storage() -> Dict[str, str]:
    """
    Set up storage directories.

    Creates all configured directories if they don't exist.

    Returns:
        Confirmation message.
    """
    config = get_config()
    manager = StorageManager(config)
    await manager.setup()
    return {"status": "Storage directories initialized"}


@router.post("/organize", response_model=OrganizeResponse)
async def organize_file(request: OrganizeRequest) -> OrganizeResponse:
    """
    Organize a file according to file management settings.

    Copies/moves file to input directory and creates output copy
    with configured naming scheme.

    Args:
        request: Organization request with file path and options.

    Returns:
        Paths to input and output files.
    """
    config = get_config()

    # Validate file extension
    is_valid, error = validate_extension(
        request.file_path, config.allowed_extensions_list
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    manager = StorageManager(config)
    input_path, output_path = await manager.organize_file(
        request.file_path,
        file_operation=request.file_operation,
        naming_scheme=request.naming_scheme,
    )

    return OrganizeResponse(
        input_path=str(input_path) if input_path else None,
        output_path=str(output_path),
    )


@router.post("/document-directory", response_model=DocumentDirectoryResponse)
async def create_document_directory(
    document_name: str,
    add_timestamp: bool = True,
) -> DocumentDirectoryResponse:
    """
    Create subdirectory structure for a document's markdown and assets.

    Creates:
        markdown_dir/document_name/
        markdown_dir/document_name/images/
        markdown_dir/document_name/tables/

    Args:
        document_name: Name of the document.
        add_timestamp: Whether to add timestamp for uniqueness.

    Returns:
        Paths to created directories.
    """
    config = get_config()
    manager = StorageManager(config)
    paths = await manager.create_document_directory(document_name, add_timestamp)

    return DocumentDirectoryResponse(
        root=str(paths["root"]),
        images=str(paths["images"]),
        tables=str(paths["tables"]),
    )


@router.get("/stats", response_model=StorageStatsResponse)
async def get_storage_stats() -> StorageStatsResponse:
    """
    Get storage statistics.

    Returns:
        Statistics for each storage directory.
    """
    config = get_config()
    manager = StorageManager(config)
    stats = await manager.get_storage_stats()

    return StorageStatsResponse(**stats)


@router.post("/cleanup-temp")
async def cleanup_temp() -> Dict[str, Any]:
    """
    Clean up temporary files.

    Returns:
        Number of files deleted.
    """
    config = get_config()
    manager = StorageManager(config)
    count = await manager.cleanup_temp()

    return {"deleted_count": count}


@router.get("/config")
async def get_storage_config() -> Dict[str, Any]:
    """
    Get current storage configuration.

    Returns:
        Storage configuration values.
    """
    config = get_config()
    return {
        "upload_dir": config.upload_dir,
        "input_dir": config.input_dir,
        "output_dir": config.output_dir,
        "markdown_dir": config.markdown_dir,
        "temp_dir": config.temp_dir,
        "default_file_operation": config.default_file_operation,
        "default_naming_scheme": config.default_naming_scheme,
        "max_file_size_mb": config.max_file_size_mb,
        "max_upload_size_mb": config.max_upload_size_mb,
        "allowed_extensions": config.allowed_extensions_list,
    }
