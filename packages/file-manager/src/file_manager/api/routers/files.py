"""
File operations API routes.
"""

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from file_manager.config import get_config
from file_manager.mime import get_file_category, get_mime_type, validate_extension
from file_manager.operations import (
    delete_file,
    get_file_size,
    list_files,
    read_file,
    write_file,
)

router = APIRouter()


class FileInfo(BaseModel):
    """File information response."""

    name: str
    path: str
    size_bytes: int
    mime_type: str
    category: str | None


class FileListResponse(BaseModel):
    """File list response."""

    files: List[FileInfo]
    total: int


class WriteFileRequest(BaseModel):
    """Write file request."""

    path: str
    content: str


@router.get("/list/{directory:path}", response_model=FileListResponse)
async def list_directory_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = False,
) -> FileListResponse:
    """
    List files in a directory.

    Args:
        directory: Directory path to list.
        pattern: Glob pattern for filtering files.
        recursive: Whether to search recursively.

    Returns:
        List of file information.
    """
    config = get_config()

    # Resolve directory relative to configured paths
    dir_mapping = {
        "uploads": config.upload_dir,
        "input": config.input_dir,
        "output": config.output_dir,
        "markdown": config.markdown_dir,
        "temp": config.temp_dir,
    }

    if directory in dir_mapping:
        dir_path = dir_mapping[directory]
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown directory: {directory}. Use: {', '.join(dir_mapping.keys())}",
        )

    files = await list_files(dir_path, pattern, recursive)

    file_infos = []
    for f in files:
        size = await get_file_size(str(f))
        file_infos.append(
            FileInfo(
                name=f.name,
                path=str(f),
                size_bytes=size,
                mime_type=get_mime_type(str(f)),
                category=get_file_category(str(f)),
            )
        )

    return FileListResponse(files=file_infos, total=len(file_infos))


@router.get("/info", response_model=FileInfo)
async def get_file_info(path: str) -> FileInfo:
    """
    Get information about a file.

    Args:
        path: Full path to the file.

    Returns:
        File information.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    size = await get_file_size(path)

    return FileInfo(
        name=file_path.name,
        path=str(file_path),
        size_bytes=size,
        mime_type=get_mime_type(path),
        category=get_file_category(path),
    )


@router.get("/read")
async def read_file_content(path: str) -> Dict[str, Any]:
    """
    Read file content.

    Args:
        path: Full path to the file.

    Returns:
        File content (text).
    """
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        content = await read_file(path, mode="r")
        return {"path": path, "content": content}
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not a text file")


@router.post("/write")
async def write_file_content(request: WriteFileRequest) -> Dict[str, Any]:
    """
    Write content to a file.

    Args:
        request: Write file request with path and content.

    Returns:
        Confirmation with path.
    """
    result_path = await write_file(request.path, request.content)
    return {"path": str(result_path), "success": True}


@router.delete("/delete")
async def delete_file_endpoint(path: str) -> Dict[str, Any]:
    """
    Delete a file.

    Args:
        path: Full path to the file.

    Returns:
        Confirmation of deletion.
    """
    deleted = await delete_file(path)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")
    return {"path": path, "deleted": True}


@router.post("/validate")
async def validate_file(path: str) -> Dict[str, Any]:
    """
    Validate a file against allowed extensions.

    Args:
        path: Path to validate.

    Returns:
        Validation result.
    """
    config = get_config()
    is_valid, error = validate_extension(path, config.allowed_extensions_list)

    return {
        "path": path,
        "valid": is_valid,
        "error": error if not is_valid else None,
        "mime_type": get_mime_type(path),
        "category": get_file_category(path),
    }
