"""
Source Folders API routes.

Provides endpoints for source-centric file management: per-source directories,
original file storage, pipeline cache, and duplicate detection.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from file_manager.services.duplicate_detector import DuplicateDetector
from file_manager.services.pipeline_cache import PipelineCacheService
from file_manager.services.source_folder import SourceFolderService
from shared.models.file_tracking import PipelineCacheEntry, SourceFolder
from surrealdb_service.repositories.file_tracking import (
    PipelineCacheRepository,
    SourceFolderRepository,
)

router = APIRouter()

# Lazy-init singletons
_folder_service: Optional[SourceFolderService] = None
_cache_service: Optional[PipelineCacheService] = None
_duplicate_detector: Optional[DuplicateDetector] = None


def _get_storage_root() -> str:
    return os.environ.get("STORAGE_ROOT", os.path.expanduser("~/.open-notebook"))


def _get_folder_service() -> SourceFolderService:
    global _folder_service
    if _folder_service is None:
        _folder_service = SourceFolderService(
            repo=SourceFolderRepository(),
            storage_root=_get_storage_root(),
        )
    return _folder_service


def _get_cache_service() -> PipelineCacheService:
    global _cache_service
    if _cache_service is None:
        _cache_service = PipelineCacheService(
            cache_repo=PipelineCacheRepository(),
            folder_repo=SourceFolderRepository(),
        )
    return _cache_service


def _get_duplicate_detector() -> DuplicateDetector:
    global _duplicate_detector
    if _duplicate_detector is None:
        _duplicate_detector = DuplicateDetector(
            repo=SourceFolderRepository(),
        )
    return _duplicate_detector


# --- Request/Response Models ---


class SourceFolderCreate(BaseModel):
    """Create source folder request."""
    title: str = Field(description="Human-readable title")
    source_type: str = Field(default="document", description="Source type")
    original_filename: Optional[str] = Field(None, description="Original filename")
    mime_type: Optional[str] = Field(None, description="MIME type")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SourceFolderResponse(BaseModel):
    """Source folder response model."""
    id: str
    source_id: str
    source_record_id: Optional[str] = None
    title: str
    slug: str
    folder_path: str
    original_filename: Optional[str] = None
    original_file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    source_type: str
    total_cache_entries: int = 0
    total_folder_size_bytes: int = 0
    metadata: Dict[str, Any] = {}
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class SourceFolderListResponse(BaseModel):
    """Source folder list response."""
    folders: List[SourceFolderResponse]
    total: int


class StoreOriginalRequest(BaseModel):
    """Store original file request."""
    source_file_path: str = Field(description="Path to source file")
    copy_file: bool = Field(default=True, description="Copy (True) or move (False)")


class LinkSourceRequest(BaseModel):
    """Link to Source record request."""
    source_record_id: str = Field(description="Domain Source record ID")


class DuplicateCheckRequest(BaseModel):
    """Duplicate check request."""
    filename: str = Field(description="Filename to check")
    file_size_bytes: Optional[int] = Field(None, description="File size")
    mime_type: Optional[str] = Field(None, description="MIME type")


class DuplicateCheckResponse(BaseModel):
    """Duplicate check response."""
    match_type: str
    message: str
    matches: List[Dict[str, Any]] = []


class CacheSaveRequest(BaseModel):
    """Save pipeline cache request."""
    data: Any = Field(description="Data to cache (dict, list, or string)")
    file_format: str = Field(default="json", description="File format")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    config_used: Optional[Dict[str, Any]] = Field(None, description="Config for change detection")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class CacheEntryResponse(BaseModel):
    """Pipeline cache entry response."""
    id: str
    source_folder_id: str
    source_id: str
    pipeline_step: str
    version: int
    is_current: bool
    file_path: str
    file_format: str
    file_size_bytes: Optional[int] = None
    processing_time_seconds: Optional[float] = None
    config_hash: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class CacheContentResponse(BaseModel):
    """Pipeline cache content response."""
    pipeline_step: str
    version: int
    file_format: str
    content: str


# --- Converters ---


def _folder_to_response(folder: SourceFolder) -> SourceFolderResponse:
    return SourceFolderResponse(
        id=str(folder.id),
        source_id=folder.source_id,
        source_record_id=folder.source_record_id,
        title=folder.title,
        slug=folder.slug,
        folder_path=folder.folder_path,
        original_filename=folder.original_filename,
        original_file_path=folder.original_file_path,
        file_size_bytes=folder.file_size_bytes,
        mime_type=folder.mime_type,
        source_type=folder.source_type,
        total_cache_entries=folder.total_cache_entries,
        total_folder_size_bytes=folder.total_folder_size_bytes,
        metadata=folder.metadata,
        created=folder.created,
        updated=folder.updated,
    )


def _cache_to_response(entry: PipelineCacheEntry) -> CacheEntryResponse:
    return CacheEntryResponse(
        id=str(entry.id),
        source_folder_id=entry.source_folder_id,
        source_id=entry.source_id,
        pipeline_step=entry.pipeline_step,
        version=entry.version,
        is_current=entry.is_current,
        file_path=entry.file_path,
        file_format=entry.file_format,
        file_size_bytes=entry.file_size_bytes,
        processing_time_seconds=entry.processing_time_seconds,
        config_hash=entry.config_hash,
        metadata=entry.metadata,
        created=entry.created,
        updated=entry.updated,
    )


# --- Endpoints ---


@router.post("", response_model=SourceFolderResponse)
async def create_source_folder(request: SourceFolderCreate) -> SourceFolderResponse:
    """Create a new source folder with directory structure."""
    service = _get_folder_service()
    folder = await service.create_folder(
        title=request.title,
        source_type=request.source_type,
        original_filename=request.original_filename,
        mime_type=request.mime_type,
        file_size_bytes=request.file_size_bytes,
        metadata=request.metadata,
    )
    return _folder_to_response(folder)


@router.get("", response_model=SourceFolderListResponse)
async def list_source_folders(
    limit: int = Query(50, description="Maximum results"),
    offset: int = Query(0, description="Offset for pagination"),
) -> SourceFolderListResponse:
    """List source folders with pagination."""
    service = _get_folder_service()
    folders = await service.list_folders(limit=limit, offset=offset)
    return SourceFolderListResponse(
        folders=[_folder_to_response(f) for f in folders],
        total=len(folders),
    )


@router.get("/by-source-id/{source_id}", response_model=SourceFolderResponse)
async def get_folder_by_source_id(source_id: str) -> SourceFolderResponse:
    """Lookup a source folder by its short random ID."""
    service = _get_folder_service()
    folder = await service.get_folder_by_source_id(source_id)
    if not folder:
        raise HTTPException(status_code=404, detail="Source folder not found")
    return _folder_to_response(folder)


@router.get("/{folder_id}", response_model=SourceFolderResponse)
async def get_source_folder(folder_id: str) -> SourceFolderResponse:
    """Get a source folder by its DB ID."""
    service = _get_folder_service()
    folder = await service.get_folder(folder_id)
    if not folder:
        raise HTTPException(status_code=404, detail="Source folder not found")
    return _folder_to_response(folder)


@router.get("/{folder_id}/contents")
async def get_folder_contents(folder_id: str) -> Dict[str, Any]:
    """Get filesystem listing of a source folder."""
    service = _get_folder_service()
    return await service.get_folder_contents(folder_id)


@router.post("/{folder_id}/original", response_model=SourceFolderResponse)
async def store_original_file(
    folder_id: str,
    request: StoreOriginalRequest,
) -> SourceFolderResponse:
    """Store the original source file into the folder."""
    service = _get_folder_service()
    try:
        folder = await service.store_original(
            folder_id=folder_id,
            source_file_path=request.source_file_path,
            copy=request.copy_file,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _folder_to_response(folder)


@router.post("/{folder_id}/link-source", response_model=SourceFolderResponse)
async def link_to_source_record(
    folder_id: str,
    request: LinkSourceRequest,
) -> SourceFolderResponse:
    """Link a source folder to a domain Source record."""
    service = _get_folder_service()
    try:
        folder = await service.link_to_source(folder_id, request.source_record_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _folder_to_response(folder)


@router.delete("/{folder_id}")
async def delete_source_folder(
    folder_id: str,
    delete_physical: bool = Query(True, description="Delete physical directory"),
) -> Dict[str, Any]:
    """Delete a source folder."""
    service = _get_folder_service()
    deleted = await service.delete_folder(folder_id, delete_physical=delete_physical)
    if not deleted:
        raise HTTPException(status_code=404, detail="Source folder not found")
    return {"success": True, "folder_id": folder_id, "physical_deleted": delete_physical}


@router.post("/check-duplicate", response_model=DuplicateCheckResponse)
async def check_duplicate(request: DuplicateCheckRequest) -> DuplicateCheckResponse:
    """Check for potential duplicate files."""
    detector = _get_duplicate_detector()
    result = await detector.check_duplicate(
        filename=request.filename,
        file_size_bytes=request.file_size_bytes,
        mime_type=request.mime_type,
    )
    return DuplicateCheckResponse(
        match_type=result.match_type,
        message=result.message,
        matches=[
            {
                "source_folder_id": str(m.source_folder.id),
                "source_id": m.source_folder.source_id,
                "original_filename": m.source_folder.original_filename,
                "similarity_score": round(m.similarity_score, 3),
                "matched_criteria": m.matched_criteria,
            }
            for m in result.matches
        ],
    )


# --- Cache endpoints ---


@router.post("/{folder_id}/cache/{step}", response_model=CacheEntryResponse)
async def save_pipeline_cache(
    folder_id: str,
    step: str,
    request: CacheSaveRequest,
) -> CacheEntryResponse:
    """Save a pipeline step output to cache."""
    service = _get_cache_service()
    try:
        entry = await service.save_output(
            source_folder_id=folder_id,
            pipeline_step=step,
            data=request.data,
            file_format=request.file_format,
            processing_time=request.processing_time,
            config_used=request.config_used,
            metadata=request.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _cache_to_response(entry)


@router.get("/{folder_id}/cache/{step}", response_model=CacheContentResponse)
async def get_pipeline_cache(
    folder_id: str,
    step: str,
    version: Optional[int] = Query(None, description="Specific version"),
) -> CacheContentResponse:
    """Load a pipeline step's cached output."""
    service = _get_cache_service()
    content = await service.get_output(folder_id, step, version=version)
    if content is None:
        raise HTTPException(status_code=404, detail="Cache entry not found")

    entry = await service.get_cache_entry(folder_id, step)
    return CacheContentResponse(
        pipeline_step=step,
        version=entry.version if entry else 1,
        file_format=entry.file_format if entry else "json",
        content=content,
    )


@router.get("/{folder_id}/cache", response_model=List[CacheEntryResponse])
async def list_cache_entries(folder_id: str) -> List[CacheEntryResponse]:
    """List all current cache entries for a source folder."""
    service = _get_cache_service()
    entries = await service.list_cache_entries(folder_id)
    return [_cache_to_response(e) for e in entries]


@router.get(
    "/{folder_id}/cache/{step}/history",
    response_model=List[CacheEntryResponse],
)
async def get_cache_version_history(
    folder_id: str,
    step: str,
) -> List[CacheEntryResponse]:
    """Get version history for a pipeline step's cache."""
    service = _get_cache_service()
    entries = await service.get_version_history(folder_id, step)
    return [_cache_to_response(e) for e in entries]
