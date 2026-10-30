"""Settings router - application settings management."""

from fastapi import APIRouter, Depends
from loguru import logger

from app_main.api.schemas import SettingsResponse, SettingsUpdate
from app_main.dependencies import get_settings_service
from app_main.services.settings_service import SettingsService
from app_main.api.agent_auth import RECHT_BEHEERDER

router = APIRouter(prefix="/settings", tags=["settings"], dependencies=[RECHT_BEHEERDER])


def _build_settings_response(settings) -> SettingsResponse:
    """Build a SettingsResponse from a ContentSettings model."""
    data = settings.model_dump()
    return SettingsResponse(**data)


@router.get("", response_model=SettingsResponse)
async def get_settings(
    settings_service: SettingsService = Depends(get_settings_service),
):
    """Get current application settings."""
    settings = await settings_service.get()
    return _build_settings_response(settings)


@router.put("", response_model=SettingsResponse)
async def update_settings(
    body: SettingsUpdate,
    settings_service: SettingsService = Depends(get_settings_service),
):
    """Update application settings."""
    update_data = body.model_dump(exclude_none=True)
    settings = await settings_service.update(update_data)
    logger.info("Updated settings with {} field(s)", len(update_data))
    return _build_settings_response(settings)
