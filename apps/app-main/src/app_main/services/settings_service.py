"""
Settings service - business logic for application settings.
"""

from typing import Any, Dict

from shared.models import ContentSettings
from surrealdb_service.repositories import ContentSettingsRepository


class SettingsService:
    """Service for settings business logic."""

    def __init__(self, settings_repo: ContentSettingsRepository):
        self.settings_repo = settings_repo

    async def get(self) -> ContentSettings:
        """Get current settings."""
        return await self.settings_repo.get()

    async def update(self, data: Dict[str, Any]) -> ContentSettings:
        """Update settings."""
        return await self.settings_repo.update(data)

    async def patch(self, updates: Dict[str, Any]) -> ContentSettings:
        """Patch specific settings fields."""
        return await self.settings_repo.patch(updates)
