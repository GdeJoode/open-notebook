"""
Model service - business logic for AI model management.
"""

from typing import Any, Dict, List, Optional

from llm_manager import ModelManager
from shared.models import DefaultModels, Model
from surrealdb_service.repositories import DefaultModelsRepository, ModelRepository


class ModelService:
    """Service for model management business logic."""

    def __init__(
        self,
        model_repo: ModelRepository,
        defaults_repo: DefaultModelsRepository,
        model_manager: ModelManager,
    ):
        self.model_repo = model_repo
        self.defaults_repo = defaults_repo
        self.model_manager = model_manager

    async def get(self, model_id: str) -> Optional[Model]:
        """Get a model by ID."""
        return await self.model_repo.get(model_id)

    async def get_all(self) -> List[Model]:
        """Get all models."""
        return await self.model_repo.get_all()

    async def get_by_type(self, model_type: str) -> List[Model]:
        """Get models by type."""
        return await self.model_repo.get_by_type(model_type)

    async def create(self, data: Dict[str, Any]) -> Model:
        """Create a new model."""
        return await self.model_repo.create(data)

    async def delete(self, model_id: str) -> bool:
        """Delete a model."""
        return await self.model_repo.delete(model_id)

    async def get_defaults(self) -> DefaultModels:
        """Get default model assignments."""
        return await self.defaults_repo.get()

    async def update_defaults(self, data: Dict[str, Any]) -> DefaultModels:
        """Update default model assignments and refresh manager cache."""
        result = await self.defaults_repo.update(data)
        self.model_manager.set_defaults(result)
        self.model_manager.clear_cache()
        return result
