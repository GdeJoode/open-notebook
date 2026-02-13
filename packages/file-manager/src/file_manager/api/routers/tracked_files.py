"""Tracked Files API routes."""

import hashlib
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from shared.types.enums import FileStatus, StorageLocation
from surrealdb_service.repositories.file_tracking import (
    KnowledgeBaseRepository,
    ProjectRepository,
    TrackedFileRepository,
)

router = APIRouter()

_file_repo: Optional[TrackedFileRepository] = None
_project_repo: Optional[ProjectRepository] = None
_kb_repo: Optional[KnowledgeBaseRepository] = None


def _get_storage_root() -> str:
    return os.environ.get("STORAGE_ROOT", os.path.expanduser("~/.open-notebook"))


def _get_file_repo() -> TrackedFileRepository:
    global _file_repo
    if _file_repo is None:
        _file_repo = TrackedFileRepository()
    return _file_repo


def _get_project_repo() -> ProjectRepository:
    global _project_repo
    if _project_repo is None:
        _project_repo = ProjectRepository()
    return _project_repo


def _get_kb_repo() -> KnowledgeBaseRepository:
    global _kb_repo
    if _kb_repo is None:
        _kb_repo = KnowledgeBaseRepository()
    return _kb_repo


class FileCreateRequest(BaseModel):
    filename: str
    file_path: str


class StatusUpdateRequest(BaseModel):
    status: str


class FileMoveRequest(BaseModel):
    target_type: str
    target_id: str


# --- Static path routes (before /{file_id} to avoid capture) ---


@router.get("")
async def list_tracked_files(
    status: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    kb_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    repo = _get_file_repo()

    if status:
        try:
            file_status = FileStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid status: {status}"
            )
        files = await repo.get_by_status(file_status)
    elif project_id:
        files = await repo.get_by_project(project_id)
    elif kb_id:
        files = await repo.get_by_knowledge_base(kb_id)
    else:
        files = await repo.get_all()

    return {"total": len(files), "files": [f.model_dump() for f in files]}


@router.post("")
async def create_tracked_file(request: FileCreateRequest) -> Dict[str, Any]:
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=400, detail=f"File does not exist: {request.file_path}"
        )

    repo = _get_file_repo()

    existing = await repo.get_by_path(request.file_path)
    if existing:
        raise HTTPException(
            status_code=400, detail="File path is already tracked"
        )

    file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

    existing_hash = await repo.get_by_hash(file_hash)
    if existing_hash:
        raise HTTPException(
            status_code=400, detail="Duplicate file (same content hash)"
        )

    file_size = os.path.getsize(request.file_path)
    mime_type = mimetypes.guess_type(request.filename)[0]

    data = {
        "filename": request.filename,
        "file_path": request.file_path,
        "file_hash": file_hash,
        "file_size_bytes": file_size,
        "mime_type": mime_type,
        "status": FileStatus.PENDING.value,
        "storage_location": StorageLocation.TEMP.value,
    }
    tracked = await repo.create(data)
    return tracked.model_dump()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    storage_root = _get_storage_root()
    temp_dir = os.path.join(storage_root, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    content = await file.read()
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(content)

    file_hash = hashlib.sha256(content).hexdigest()

    repo = _get_file_repo()
    existing = await repo.get_by_hash(file_hash)
    if existing:
        os.unlink(file_path)
        raise HTTPException(
            status_code=400, detail="Duplicate file (same content hash)"
        )

    mime_type = file.content_type or mimetypes.guess_type(file.filename)[0]

    data = {
        "filename": file.filename,
        "file_path": file_path,
        "file_hash": file_hash,
        "file_size_bytes": len(content),
        "mime_type": mime_type,
        "status": FileStatus.PENDING.value,
        "storage_location": StorageLocation.TEMP.value,
    }
    tracked = await repo.create(data)
    return tracked.model_dump()


@router.get("/stats")
async def get_file_stats() -> Dict[str, Any]:
    repo = _get_file_repo()
    return await repo.get_stats()


@router.get("/pending")
async def get_pending_files() -> Dict[str, Any]:
    repo = _get_file_repo()
    files = await repo.get_pending_files()
    return {"total": len(files), "files": [f.model_dump() for f in files]}


@router.get("/by-path")
async def get_file_by_path(path: str = Query(...)) -> Dict[str, Any]:
    repo = _get_file_repo()
    tracked = await repo.get_by_path(path)
    if not tracked:
        raise HTTPException(status_code=404, detail="Tracked file not found")
    return tracked.model_dump()


@router.get("/by-hash/{hash}")
async def get_file_by_hash(hash: str) -> Dict[str, Any]:
    repo = _get_file_repo()
    tracked = await repo.get_by_hash(hash)
    if not tracked:
        raise HTTPException(status_code=404, detail="Tracked file not found")
    return tracked.model_dump()


# --- Dynamic path routes ---


@router.get("/{file_id}")
async def get_tracked_file(file_id: str) -> Dict[str, Any]:
    repo = _get_file_repo()
    tracked = await repo.get(file_id)
    if not tracked:
        raise HTTPException(status_code=404, detail="Tracked file not found")
    return tracked.model_dump()


@router.patch("/{file_id}/status")
async def update_file_status(
    file_id: str, request: StatusUpdateRequest
) -> Dict[str, Any]:
    repo = _get_file_repo()
    tracked = await repo.get(file_id)
    if not tracked:
        raise HTTPException(status_code=404, detail="Tracked file not found")

    try:
        file_status = FileStatus(request.status)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid status: {request.status}"
        )

    updated = await repo.update_status(file_id, file_status)
    return updated.model_dump()


@router.post("/{file_id}/link-source")
async def link_to_source(
    file_id: str, source_id: str = Query(...)
) -> Dict[str, Any]:
    repo = _get_file_repo()
    tracked = await repo.get(file_id)
    if not tracked:
        raise HTTPException(status_code=404, detail="Tracked file not found")

    updated = await repo.link_to_source(file_id, source_id)
    return updated.model_dump()


@router.post("/{file_id}/link-transcription")
async def link_to_transcription(
    file_id: str, transcription_id: str = Query(...)
) -> Dict[str, Any]:
    repo = _get_file_repo()
    tracked = await repo.get(file_id)
    if not tracked:
        raise HTTPException(status_code=404, detail="Tracked file not found")

    updated = await repo.link_to_transcription(file_id, transcription_id)
    return updated.model_dump()


@router.post("/{file_id}/move")
async def move_file(file_id: str, request: FileMoveRequest) -> Dict[str, Any]:
    repo = _get_file_repo()
    tracked = await repo.get(file_id)
    if not tracked:
        raise HTTPException(status_code=404, detail="Tracked file not found")

    if request.target_type not in ("project", "knowledge_base"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target_type: {request.target_type}. Must be 'project' or 'knowledge_base'",
        )

    if request.target_type == "project":
        project_repo = _get_project_repo()
        project = await project_repo.get(request.target_id)
        if not project:
            raise HTTPException(status_code=404, detail="Target project not found")

        target_dir = project.input_path
        os.makedirs(target_dir, exist_ok=True)
        new_path = os.path.join(target_dir, tracked.filename)

        if Path(tracked.file_path).exists():
            shutil.move(tracked.file_path, new_path)

        updated = await repo.move_to_project(file_id, request.target_id)
        await repo.update(file_id, {"file_path": new_path})
    else:
        kb_repo = _get_kb_repo()
        kb = await kb_repo.get(request.target_id)
        if not kb:
            raise HTTPException(
                status_code=404, detail="Target knowledge base not found"
            )

        target_dir = kb.sources_path
        os.makedirs(target_dir, exist_ok=True)
        new_path = os.path.join(target_dir, tracked.filename)

        if Path(tracked.file_path).exists():
            shutil.move(tracked.file_path, new_path)

        updated = await repo.move_to_knowledge_base(file_id, request.target_id)
        await repo.update(file_id, {"file_path": new_path})

    return updated.model_dump()


@router.delete("/{file_id}")
async def delete_tracked_file(
    file_id: str, delete_physical: bool = Query(False)
) -> Dict[str, Any]:
    repo = _get_file_repo()
    tracked = await repo.get(file_id)
    if not tracked:
        raise HTTPException(status_code=404, detail="Tracked file not found")

    physical_deleted = False
    if delete_physical:
        fp = Path(tracked.file_path)
        if fp.exists():
            fp.unlink()
            physical_deleted = True

    await repo.delete(file_id)
    return {"success": True, "physical_deleted": physical_deleted}
