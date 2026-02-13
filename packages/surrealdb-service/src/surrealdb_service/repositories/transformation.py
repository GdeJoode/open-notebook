"""
Transformation repositories.
"""

from typing import List, Optional

from shared.models.transformation import DefaultPrompts, Transformation
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.repositories.base import BaseRepository, RecordRepository


class TransformationRepository(BaseRepository[Transformation]):
    """Repository for Transformation CRUD operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Transformation, config)

    async def get_defaults(self) -> List[Transformation]:
        """Get all transformations marked as apply_default."""
        return await self.query("apply_default = true")

    async def get_by_name(self, name: str) -> Optional[Transformation]:
        """Get a transformation by name."""
        results = await self.query("name = $name", {"name": name}, limit=1)
        return results[0] if results else None


class DefaultPromptsRepository(RecordRepository[DefaultPrompts]):
    """Repository for the DefaultPrompts singleton."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(DefaultPrompts, config)
