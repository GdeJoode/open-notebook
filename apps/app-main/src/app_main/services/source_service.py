"""
Source service - business logic for source operations.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import Chunk, Source, SourceEmbedding, SourceInsight
from surrealdb_service.repositories import (
    ChunkRepository,
    SourceEmbeddingRepository,
    SourceInsightRepository,
    SourceRepository,
)


class SourceService:
    """Service for source business logic."""

    def __init__(
        self,
        source_repo: SourceRepository,
        chunk_repo: ChunkRepository,
        insight_repo: SourceInsightRepository,
        embedding_repo: SourceEmbeddingRepository,
    ):
        self.source_repo = source_repo
        self.chunk_repo = chunk_repo
        self.insight_repo = insight_repo
        self.embedding_repo = embedding_repo

    async def get(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        return await self.source_repo.get(source_id)

    async def get_all(self, order_by: str = "updated DESC") -> List[Source]:
        """Get all sources."""
        return await self.source_repo.get_all(order_by=order_by)

    async def create(self, data: Dict[str, Any]) -> Source:
        """Create a new source."""
        return await self.source_repo.create(data)

    async def update(self, source_id: str, data: Dict[str, Any]) -> Source:
        """Update a source."""
        return await self.source_repo.update(source_id, data)

    async def delete(self, source_id: str) -> bool:
        """Delete a source and its related data."""
        return await self.source_repo.delete(source_id)

    async def get_chunks(self, source_id: str) -> List[Chunk]:
        """Get all chunks for a source."""
        return await self.source_repo.get_chunks(source_id)

    async def get_insights(self, source_id: str) -> List[SourceInsight]:
        """Get all insights for a source."""
        return await self.source_repo.get_insights(source_id)

    async def get_embeddings(self, source_id: str) -> List[SourceEmbedding]:
        """Get all embeddings for a source."""
        return await self.source_repo.get_embeddings(source_id)

    async def get_embedding_count(self, source_id: str) -> int:
        """Get embedding count for a source."""
        return await self.source_repo.get_embedding_count(source_id)

    async def delete_embeddings(self, source_id: str) -> int:
        """Delete all embeddings for a source."""
        return await self.source_repo.delete_embeddings(source_id)

    async def add_to_notebook(self, source_id: str, notebook_id: str) -> bool:
        """Add a source to a notebook."""
        return await self.source_repo.add_to_notebook(source_id, notebook_id)

    async def add_insight(
        self,
        source_id: str,
        insight_type: str,
        content: str,
        embedding: Optional[List[float]] = None,
    ) -> SourceInsight:
        """Add an insight to a source."""
        return await self.source_repo.add_insight(
            source_id, insight_type, content, embedding
        )

    async def add_embedding(
        self,
        source_id: str,
        content: str,
        order: int,
        embedding: List[float],
    ) -> SourceEmbedding:
        """Add an embedding to a source."""
        return await self.source_repo.add_embedding(
            source_id, content, order, embedding
        )
