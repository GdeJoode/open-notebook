"""Knowledge Bases API routes."""

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from surrealdb_service.repositories.file_tracking import (
    KnowledgeBaseRepository,
    TrackedFileRepository,
)

router = APIRouter()

_kb_repo: Optional[KnowledgeBaseRepository] = None
_file_repo: Optional[TrackedFileRepository] = None


def _get_storage_root() -> str:
    return os.environ.get("STORAGE_ROOT", os.path.expanduser("~/.open-notebook"))


def _get_kb_repo() -> KnowledgeBaseRepository:
    global _kb_repo
    if _kb_repo is None:
        _kb_repo = KnowledgeBaseRepository()
    return _kb_repo


def _get_file_repo() -> TrackedFileRepository:
    global _file_repo
    if _file_repo is None:
        _file_repo = TrackedFileRepository()
    return _file_repo


def _make_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class KBCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None


class KBUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router.get("")
async def list_knowledge_bases() -> Dict[str, Any]:
    repo = _get_kb_repo()
    kbs = await repo.get_all()
    return {"total": len(kbs), "knowledge_bases": [kb.model_dump() for kb in kbs]}


@router.post("")
async def create_knowledge_base(request: KBCreateRequest) -> Dict[str, Any]:
    repo = _get_kb_repo()
    slug = _make_slug(request.name)

    existing = await repo.get_by_slug(slug)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Knowledge base with slug '{slug}' already exists",
        )

    storage_root = _get_storage_root()
    base_path = os.path.join(storage_root, "knowledgebases", slug)
    os.makedirs(base_path, exist_ok=True)

    data = {
        "name": request.name,
        "slug": slug,
        "description": request.description,
        "base_path": base_path,
        "sources_path": os.path.join(base_path, "sources"),
        "exports_path": os.path.join(base_path, "exports"),
        "media_path": os.path.join(base_path, "media"),
    }
    kb = await repo.create(data)
    return kb.model_dump()


@router.get("/default")
async def get_default_kb() -> Dict[str, Any]:
    repo = _get_kb_repo()
    kb = await repo.get_or_create_default(storage_root=_get_storage_root())
    return kb.model_dump()


@router.get("/by-slug/{slug}")
async def get_kb_by_slug(slug: str) -> Dict[str, Any]:
    repo = _get_kb_repo()
    kb = await repo.get_by_slug(slug)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb.model_dump()


@router.get("/{kb_id}")
async def get_knowledge_base(kb_id: str) -> Dict[str, Any]:
    repo = _get_kb_repo()
    kb = await repo.get(kb_id)
    if not kb:
        kb = await repo.get_by_slug(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb.model_dump()


@router.patch("/{kb_id}")
async def update_knowledge_base(
    kb_id: str, request: KBUpdateRequest
) -> Dict[str, Any]:
    repo = _get_kb_repo()
    kb = await repo.get(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    update_data = request.model_dump(exclude_none=True)
    updated = await repo.update(kb_id, update_data)
    return updated.model_dump()


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str, delete_files: bool = Query(False)
) -> Dict[str, Any]:
    repo = _get_kb_repo()
    kb = await repo.get(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if kb.slug == "_default":
        raise HTTPException(
            status_code=400, detail="Cannot delete the default knowledge base"
        )

    result: Dict[str, Any] = {"success": True}

    if delete_files:
        base = Path(kb.base_path)
        files_removed = 0
        if base.exists():
            files_removed = sum(1 for f in base.rglob("*") if f.is_file())
            shutil.rmtree(base)
        result["files_removed"] = files_removed

    await repo.delete(kb_id)
    return result


@router.get("/{kb_id}/files")
async def list_kb_files(kb_id: str) -> Dict[str, Any]:
    repo = _get_kb_repo()
    kb = await repo.get(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    file_repo = _get_file_repo()
    files = await file_repo.get_by_knowledge_base(kb_id)
    return {"total": len(files), "files": [f.model_dump() for f in files]}


@router.post("/{kb_id}/sync-stats")
async def sync_kb_stats(kb_id: str) -> Dict[str, Any]:
    repo = _get_kb_repo()
    kb = await repo.get(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    base = Path(kb.base_path)
    file_count = 0
    total_size = 0
    if base.exists():
        for f in base.rglob("*"):
            if f.is_file():
                file_count += 1
                total_size += f.stat().st_size

    updated = await repo.update(
        kb_id, {"file_count": file_count, "total_size_bytes": total_size}
    )
    return updated.model_dump()
