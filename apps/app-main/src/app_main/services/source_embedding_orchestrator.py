"""
SourceEmbeddingOrchestrator — triggers vector embedding for a source.

Delegates the actual work to the embeddings pipeline service obtained
from the DI container. Lives here so handlers/routers can inject just
this concern instead of the whole SourceProcessingService god.

Pulled out of SourceProcessingService in Phase 3 of the refactor.
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from surrealdb_service.repositories import SourceRepository


class SourceEmbeddingOrchestrator:
    """Triggers embedding for a source's chunks."""

    def __init__(self, source_repo: SourceRepository) -> None:
        self.source_repo = source_repo

    async def embed_source(self, source_id: str) -> Dict[str, Any]:
        """Generate embeddings for a source's chunks.

        Raises ValueError when the source does not exist. Returns a dict
        with ``source_id`` and ``embedded_chunks`` count.
        """
        source = await self.source_repo.get(source_id)
        if source is None:
            raise ValueError(f"Source '{source_id}' not found")

        logger.debug("Embedding content for vector search")
        # Lazy import to avoid pulling embeddings deps at module load.
        from app_main.dependencies import get_embedding_service

        embedding_service = await get_embedding_service()
        await embedding_service.embed_source(source_id)

        embedded_chunks = await self.source_repo.get_embedding_count(source_id)
        return {
            "source_id": str(source.id),
            "embedded_chunks": embedded_chunks,
        }
