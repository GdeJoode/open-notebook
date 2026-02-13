"""
Transformation service - business logic for text transformations.
"""

from typing import Any, Dict, List, Optional

from shared.models import Transformation
from shared.models.transformation import DefaultPrompts
from surrealdb_service.repositories import (
    DefaultPromptsRepository,
    TransformationRepository,
)


class TransformationService:
    """Service for transformation business logic."""

    def __init__(
        self,
        transformation_repo: TransformationRepository,
        default_prompts_repo: DefaultPromptsRepository,
    ):
        self.transformation_repo = transformation_repo
        self.default_prompts_repo = default_prompts_repo

    async def get(self, transformation_id: str) -> Optional[Transformation]:
        """Get a transformation by ID."""
        return await self.transformation_repo.get(transformation_id)

    async def get_all(self, order_by: str = "name asc") -> List[Transformation]:
        """Get all transformations."""
        return await self.transformation_repo.get_all(order_by=order_by)

    async def create(self, data: Dict[str, Any]) -> Transformation:
        """Create a new transformation."""
        return await self.transformation_repo.create(data)

    async def update(
        self, transformation_id: str, data: Dict[str, Any],
    ) -> Transformation:
        """Update a transformation."""
        return await self.transformation_repo.update(transformation_id, data)

    async def delete(self, transformation_id: str) -> bool:
        """Delete a transformation."""
        return await self.transformation_repo.delete(transformation_id)

    async def get_default_prompts(self) -> DefaultPrompts:
        """Get default prompts."""
        return await self.default_prompts_repo.get()

    async def update_default_prompts(self, data: Dict[str, Any]) -> DefaultPrompts:
        """Update default prompts."""
        return await self.default_prompts_repo.update(data)
