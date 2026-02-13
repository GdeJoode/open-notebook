"""
Model repositories for AI model configuration and defaults.
"""

from typing import List, Optional

from shared.models.llm import DefaultModels, Model
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.repositories.base import BaseRepository, RecordRepository


class ModelRepository(BaseRepository[Model]):
    """Repository for Model CRUD operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Model, config)

    async def get_by_type(self, model_type: str) -> List[Model]:
        """Get all models of a specific type."""
        return await self.query("type = $type", {"type": model_type})

    async def get_enabled(self) -> List[Model]:
        """Get all enabled models."""
        return await self.query("is_enabled = true")

    async def get_by_provider(self, provider: str) -> List[Model]:
        """Get all models for a specific provider."""
        return await self.query("provider = $provider", {"provider": provider})


class DefaultModelsRepository(RecordRepository[DefaultModels]):
    """Repository for the DefaultModels singleton."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(DefaultModels, config)
