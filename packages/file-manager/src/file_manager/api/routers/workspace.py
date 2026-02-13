"""Workspace API routes."""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

STANDARD_DIRS = ["projects", "knowledgebases", "temp"]


def _get_storage_root() -> str:
    return os.environ.get("STORAGE_ROOT", os.path.expanduser("~/.open-notebook"))


class DirectoryRequest(BaseModel):
    path: str
    create_if_missing: bool = False


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    storage_root = _get_storage_root()
    paths = {
        "storage_root": storage_root,
        "projects_path": os.path.join(storage_root, "projects"),
        "knowledgebases_path": os.path.join(storage_root, "knowledgebases"),
        "temp_path": os.path.join(storage_root, "temp"),
    }
    for key in ["projects_path", "knowledgebases_path", "temp_path"]:
        os.makedirs(paths[key], exist_ok=True)
    return paths


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    storage_root = _get_storage_root()
    projects_path = Path(storage_root) / "projects"
    kb_path = Path(storage_root) / "knowledgebases"
    temp_path = Path(storage_root) / "temp"

    projects_count = 0
    if projects_path.exists():
        projects_count = sum(1 for d in projects_path.iterdir() if d.is_dir())

    kb_count = 0
    if kb_path.exists():
        kb_count = sum(1 for d in kb_path.iterdir() if d.is_dir())

    temp_count = 0
    total_size = 0
    if temp_path.exists():
        for f in temp_path.rglob("*"):
            if f.is_file():
                temp_count += 1
                total_size += f.stat().st_size

    # Also add sizes from projects and kb dirs
    for d in [projects_path, kb_path]:
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size

    return {
        "projects_count": projects_count,
        "knowledgebases_count": kb_count,
        "temp_files_count": temp_count,
        "total_size_bytes": total_size,
    }


@router.post("/initialize")
async def initialize_workspace() -> Dict[str, Any]:
    storage_root = _get_storage_root()
    created = []
    for name in STANDARD_DIRS:
        path = os.path.join(storage_root, name)
        os.makedirs(path, exist_ok=True)
        created.append(name)
    return {
        "success": True,
        "storage_root": storage_root,
        "created_directories": created,
    }


@router.get("/directories")
async def list_directories() -> List[Dict[str, Any]]:
    storage_root = _get_storage_root()
    result = []
    for name in STANDARD_DIRS:
        path = os.path.join(storage_root, name)
        result.append({
            "name": name,
            "exists": os.path.isdir(path),
            "path": path,
        })
    return result


@router.delete("/temp/cleanup")
async def cleanup_temp() -> Dict[str, Any]:
    storage_root = _get_storage_root()
    temp_path = Path(storage_root) / "temp"

    deleted_count = 0
    deleted_size = 0

    if temp_path.exists():
        for f in temp_path.iterdir():
            if f.is_file():
                deleted_size += f.stat().st_size
                f.unlink()
                deleted_count += 1
            elif f.is_dir():
                for sub in f.rglob("*"):
                    if sub.is_file():
                        deleted_size += sub.stat().st_size
                        deleted_count += 1
                shutil.rmtree(f)

    return {
        "success": True,
        "deleted_count": deleted_count,
        "deleted_size_bytes": deleted_size,
    }


@router.post("/directory")
async def set_directory(request: DirectoryRequest) -> Dict[str, Any]:
    if os.path.isdir(request.path):
        return {"status": "selected", "exists": True, "path": request.path}

    if request.create_if_missing:
        os.makedirs(request.path, exist_ok=True)
        return {"status": "created", "exists": True, "path": request.path}

    return {"status": "error", "exists": False, "path": request.path}
