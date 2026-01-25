"""
Settings API routes.
"""

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from surrealdb_service.repositories import ContentSettingsRepository

router = APIRouter()


class SettingsUpdate(BaseModel):
    """Partial settings update."""
    model_config = {"extra": "allow"}


@router.get("", response_model=Dict[str, Any])
async def get_settings():
    """Get current content settings."""
    repo = ContentSettingsRepository()
    settings = await repo.get()
    return settings.model_dump()


@router.patch("", response_model=Dict[str, Any])
async def update_settings(data: SettingsUpdate):
    """Update content settings."""
    repo = ContentSettingsRepository()
    updates = data.model_dump(exclude_unset=True)
    settings = await repo.patch(updates)
    return settings.model_dump()


@router.put("", response_model=Dict[str, Any])
async def replace_settings(data: SettingsUpdate):
    """Replace all content settings."""
    repo = ContentSettingsRepository()
    settings = await repo.update(data.model_dump())
    return settings.model_dump()
