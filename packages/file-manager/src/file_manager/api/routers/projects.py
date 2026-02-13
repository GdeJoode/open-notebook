"""Projects API routes."""

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from surrealdb_service.repositories.file_tracking import (
    ProjectRepository,
    TrackedFileRepository,
)

router = APIRouter()

_project_repo: Optional[ProjectRepository] = None
_file_repo: Optional[TrackedFileRepository] = None


def _get_storage_root() -> str:
    return os.environ.get("STORAGE_ROOT", os.path.expanduser("~/.open-notebook"))


def _get_project_repo() -> ProjectRepository:
    global _project_repo
    if _project_repo is None:
        _project_repo = ProjectRepository()
    return _project_repo


def _get_file_repo() -> TrackedFileRepository:
    global _file_repo
    if _file_repo is None:
        _file_repo = TrackedFileRepository()
    return _file_repo


def _make_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class ProjectCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router.get("")
async def list_projects(
    include_empty: bool = Query(False),
) -> Dict[str, Any]:
    repo = _get_project_repo()
    if include_empty:
        projects = await repo.get_all()
    else:
        projects = await repo.get_active_projects()
    return {"total": len(projects), "projects": [p.model_dump() for p in projects]}


@router.post("")
async def create_project(request: ProjectCreateRequest) -> Dict[str, Any]:
    repo = _get_project_repo()
    slug = _make_slug(request.name)

    existing = await repo.get_by_slug(slug)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Project with slug '{slug}' already exists",
        )

    storage_root = _get_storage_root()
    base_path = os.path.join(storage_root, "projects", slug)
    input_path = os.path.join(base_path, "input")
    output_path = os.path.join(base_path, "output")
    exports_path = os.path.join(base_path, "exports")

    for d in [base_path, input_path, output_path, exports_path]:
        os.makedirs(d, exist_ok=True)

    data = {
        "name": request.name,
        "slug": slug,
        "description": request.description,
        "base_path": base_path,
        "input_path": input_path,
        "output_path": output_path,
        "exports_path": exports_path,
    }
    project = await repo.create(data)
    return project.model_dump()


@router.get("/by-slug/{slug}")
async def get_project_by_slug(slug: str) -> Dict[str, Any]:
    repo = _get_project_repo()
    project = await repo.get_by_slug(slug)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.model_dump()


@router.get("/{project_id}")
async def get_project(project_id: str) -> Dict[str, Any]:
    repo = _get_project_repo()
    project = await repo.get(project_id)
    if not project:
        project = await repo.get_by_slug(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.model_dump()


@router.patch("/{project_id}")
async def update_project(
    project_id: str, request: ProjectUpdateRequest
) -> Dict[str, Any]:
    repo = _get_project_repo()
    project = await repo.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    update_data = request.model_dump(exclude_none=True)
    updated = await repo.update(project_id, update_data)
    return updated.model_dump()


@router.delete("/{project_id}")
async def delete_project(
    project_id: str, delete_files: bool = Query(False)
) -> Dict[str, Any]:
    repo = _get_project_repo()
    project = await repo.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result: Dict[str, Any] = {"success": True}

    if delete_files:
        base = Path(project.base_path)
        files_removed = 0
        if base.exists():
            files_removed = sum(1 for f in base.rglob("*") if f.is_file())
            shutil.rmtree(base)
        result["files_removed"] = files_removed

    await repo.delete(project_id)
    return result


@router.get("/{project_id}/files")
async def list_project_files(project_id: str) -> Dict[str, Any]:
    repo = _get_project_repo()
    project = await repo.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    file_repo = _get_file_repo()
    files = await file_repo.get_by_project(project_id)
    return {"total": len(files), "files": [f.model_dump() for f in files]}


@router.post("/{project_id}/processing/start")
async def start_processing(project_id: str) -> Dict[str, Any]:
    repo = _get_project_repo()
    project = await repo.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    updated = await repo.update(project_id, {"is_processing": True})
    return updated.model_dump()


@router.post("/{project_id}/processing/complete")
async def complete_processing(project_id: str) -> Dict[str, Any]:
    repo = _get_project_repo()
    project = await repo.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    updated = await repo.update(project_id, {"is_processing": False})
    return updated.model_dump()


@router.post("/{project_id}/sync-stats")
async def sync_project_stats(project_id: str) -> Dict[str, Any]:
    repo = _get_project_repo()
    project = await repo.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    base = Path(project.base_path)
    file_count = 0
    total_size = 0
    if base.exists():
        for f in base.rglob("*"):
            if f.is_file():
                file_count += 1
                total_size += f.stat().st_size

    updated = await repo.update(
        project_id, {"file_count": file_count, "total_size_bytes": total_size}
    )
    return updated.model_dump()
