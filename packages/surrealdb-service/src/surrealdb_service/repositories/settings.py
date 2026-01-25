"""
Settings repository for singleton configuration records.
"""

from typing import Optional

from shared.models import ContentSettings
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.repositories.base import RecordRepository


class ContentSettingsRepository(RecordRepository[ContentSettings]):
    """Repository for ContentSettings singleton."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(ContentSettings, config)
