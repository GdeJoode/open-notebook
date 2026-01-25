"""
Source repository with specialized operations.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import Asset, Chunk, Source, SourceEmbedding, SourceInsight
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.base import BaseRepository


class SourceRepository(BaseRepository[Source]):
    """Repository for Source operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Source, config)

    async def add_to_notebook(self, source_id: str, notebook_id: str) -> bool:
        """
        Add a source to a notebook.

        Args:
            source_id: Source ID.
            notebook_id: Notebook ID.

        Returns:
            True if successful.
        """
        try:
            await self.relate(source_id, "reference", notebook_id)
            return True
        except Exception as e:
            logger.error(f"Failed to add source to notebook: {e}")
            return False

    async def get_insights(self, source_id: str) -> List[SourceInsight]:
        """
        Get all insights for a source.

        Args:
            source_id: Source ID.

        Returns:
            List of source insights.
        """
        try:
            result = await execute_query(
                "SELECT * FROM source_insight WHERE source=$id",
                {"id": ensure_record_id(source_id)},
                self.config,
            )
            return [SourceInsight(**item) for item in result]
        except Exception as e:
            logger.error(f"Failed to get insights for source {source_id}: {e}")
            return []

    async def get_chunks(self, source_id: str) -> List[Chunk]:
        """
        Get all chunks for a source, ordered by position.

        Args:
            source_id: Source ID.

        Returns:
            List of chunks ordered by order field.
        """
        try:
            result = await execute_query(
                "SELECT * FROM chunk WHERE source=$id ORDER BY order ASC",
                {"id": ensure_record_id(source_id)},
                self.config,
            )
            return [Chunk(**item) for item in result]
        except Exception as e:
            logger.error(f"Failed to get chunks for source {source_id}: {e}")
            return []

    async def get_embeddings(self, source_id: str) -> List[SourceEmbedding]:
        """
        Get all embeddings for a source.

        Args:
            source_id: Source ID.

        Returns:
            List of source embeddings ordered by order field.
        """
        try:
            result = await execute_query(
                "SELECT * FROM source_embedding WHERE source=$id ORDER BY order ASC",
                {"id": ensure_record_id(source_id)},
                self.config,
            )
            return [SourceEmbedding(**item) for item in result]
        except Exception as e:
            logger.error(f"Failed to get embeddings for source {source_id}: {e}")
            return []

    async def get_embedding_count(self, source_id: str) -> int:
        """
        Get the count of embeddings for a source.

        Args:
            source_id: Source ID.

        Returns:
            Number of embeddings.
        """
        try:
            result = await execute_query(
                "SELECT count() AS chunks FROM source_embedding WHERE source=$id GROUP ALL",
                {"id": ensure_record_id(source_id)},
                self.config,
            )
            if result:
                return result[0].get("chunks", 0)
            return 0
        except Exception as e:
            logger.error(f"Failed to count embeddings for source {source_id}: {e}")
            return 0

    async def delete_embeddings(self, source_id: str) -> int:
        """
        Delete all embeddings for a source.

        Args:
            source_id: Source ID.

        Returns:
            Number of deleted embeddings.
        """
        try:
            result = await execute_query(
                "DELETE source_embedding WHERE source=$source_id",
                {"source_id": ensure_record_id(source_id)},
                self.config,
            )
            return len(result) if result else 0
        except Exception as e:
            logger.error(f"Failed to delete embeddings for source {source_id}: {e}")
            return 0

    async def add_insight(
        self,
        source_id: str,
        insight_type: str,
        content: str,
        embedding: Optional[List[float]] = None,
    ) -> SourceInsight:
        """
        Add an insight to a source.

        Args:
            source_id: Source ID.
            insight_type: Type of insight.
            content: Insight content.
            embedding: Optional embedding vector.

        Returns:
            Created insight.
        """
        try:
            result = await execute_query(
                """
                CREATE source_insight CONTENT {
                    "source": $source_id,
                    "insight_type": $insight_type,
                    "content": $content,
                    "embedding": $embedding
                }
                """,
                {
                    "source_id": ensure_record_id(source_id),
                    "insight_type": insight_type,
                    "content": content,
                    "embedding": embedding or [],
                },
                self.config,
            )
            if result:
                return SourceInsight(**result[0])
            raise RuntimeError("Failed to create insight")
        except Exception as e:
            logger.exception(f"Failed to add insight to source {source_id}: {e}")
            raise

    async def add_embedding(
        self,
        source_id: str,
        content: str,
        order: int,
        embedding: List[float],
    ) -> SourceEmbedding:
        """
        Add an embedding to a source.

        Args:
            source_id: Source ID.
            content: Text content.
            order: Order in document.
            embedding: Embedding vector.

        Returns:
            Created embedding.
        """
        try:
            result = await execute_query(
                """
                CREATE source_embedding CONTENT {
                    "source": $source_id,
                    "order": $order,
                    "content": $content,
                    "embedding": $embedding
                }
                """,
                {
                    "source_id": ensure_record_id(source_id),
                    "order": order,
                    "content": content,
                    "embedding": embedding,
                },
                self.config,
            )
            if result:
                return SourceEmbedding(**result[0])
            raise RuntimeError("Failed to create embedding")
        except Exception as e:
            logger.exception(f"Failed to add embedding to source {source_id}: {e}")
            raise


class ChunkRepository(BaseRepository[Chunk]):
    """Repository for Chunk operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Chunk, config)

    async def get_by_source(
        self,
        source_id: str,
        page: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Get chunks for a source, optionally filtered by page.

        Args:
            source_id: Source ID.
            page: Optional physical page number.

        Returns:
            List of chunks.
        """
        where = "source=$source_id"
        params: Dict[str, Any] = {"source_id": ensure_record_id(source_id)}

        if page is not None:
            where += " AND physical_page=$page"
            params["page"] = page

        return await self.query(where, params, order_by="order ASC")

    async def delete_by_source(self, source_id: str) -> int:
        """
        Delete all chunks for a source.

        Args:
            source_id: Source ID.

        Returns:
            Number of deleted chunks.
        """
        try:
            result = await execute_query(
                "DELETE chunk WHERE source=$source_id",
                {"source_id": ensure_record_id(source_id)},
                self.config,
            )
            return len(result) if result else 0
        except Exception as e:
            logger.error(f"Failed to delete chunks for source {source_id}: {e}")
            return 0

    async def bulk_create(self, chunks: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Create multiple chunks at once.

        Args:
            chunks: List of chunk data.

        Returns:
            List of created chunks.
        """
        try:
            result = await execute_query(
                "INSERT INTO chunk $chunks",
                {"chunks": chunks},
                self.config,
            )
            return [Chunk(**item) for item in result] if result else []
        except Exception as e:
            logger.exception(f"Failed to bulk create chunks: {e}")
            raise


class SourceInsightRepository(BaseRepository[SourceInsight]):
    """Repository for SourceInsight operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(SourceInsight, config)

    async def get_by_source(self, source_id: str) -> List[SourceInsight]:
        """Get all insights for a source."""
        return await self.query(
            "source=$source_id",
            {"source_id": ensure_record_id(source_id)},
        )

    async def get_source(self, insight_id: str) -> Optional[Source]:
        """
        Get the source for an insight.

        Args:
            insight_id: Insight ID.

        Returns:
            Source or None.
        """
        try:
            result = await execute_query(
                "SELECT source.* FROM $id FETCH source",
                {"id": ensure_record_id(insight_id)},
                self.config,
            )
            if result and result[0].get("source"):
                return Source(**result[0]["source"])
            return None
        except Exception as e:
            logger.error(f"Failed to get source for insight {insight_id}: {e}")
            return None


class SourceEmbeddingRepository(BaseRepository[SourceEmbedding]):
    """Repository for SourceEmbedding operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(SourceEmbedding, config)

    async def get_by_source(self, source_id: str) -> List[SourceEmbedding]:
        """Get all embeddings for a source."""
        return await self.query(
            "source=$source_id",
            {"source_id": ensure_record_id(source_id)},
            order_by="order ASC",
        )

    async def get_source(self, embedding_id: str) -> Optional[Source]:
        """
        Get the source for an embedding.

        Args:
            embedding_id: Embedding ID.

        Returns:
            Source or None.
        """
        try:
            result = await execute_query(
                "SELECT source.* FROM $id FETCH source",
                {"id": ensure_record_id(embedding_id)},
                self.config,
            )
            if result and result[0].get("source"):
                return Source(**result[0]["source"])
            return None
        except Exception as e:
            logger.error(f"Failed to get source for embedding {embedding_id}: {e}")
            return None
