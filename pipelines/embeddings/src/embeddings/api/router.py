"""
Standalone FastAPI router for embedding operations.

Service is injected via a dependency function so the router can be
mounted in any FastAPI app (app-main, standalone, etc.).
"""

from typing import Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from embeddings.config import EmbeddingConfig
from embeddings.service import EmbeddingService


# --- Response schemas (local to this router) ---

class EmbedResultResponse(BaseModel):
    id: str
    embeddings_created: int
    chunks_used: int
    used_structural_chunks: bool
    errors: list[str]


class RebuildResultResponse(BaseModel):
    sources_processed: int
    notes_processed: int
    insights_processed: int
    total_embeddings: int
    errors: list[str]


class RebuildRequest(BaseModel):
    include_sources: bool = True
    include_notes: bool = True
    include_insights: bool = True
    mode: str = "all"


class ConfigResponse(BaseModel):
    batch_size: int
    max_concurrent: int
    retry_attempts: int
    retry_delay: float
    fallback_chunk_size: int
    fallback_chunk_overlap: int


def create_router(
    get_embedding_service: Callable[..., EmbeddingService],
) -> APIRouter:
    """Create the embeddings router with the given service dependency.

    Args:
        get_embedding_service: A FastAPI dependency callable that returns
            an ``EmbeddingService`` instance.
    """
    router = APIRouter(tags=["embeddings"])

    @router.post(
        "/embed/source/{source_id}",
        response_model=EmbedResultResponse,
    )
    async def embed_source(
        source_id: str,
        service: EmbeddingService = Depends(get_embedding_service),
    ):
        """Generate embeddings for a single source."""
        result = await service.embed_source(source_id)
        return EmbedResultResponse(
            id=result.id,
            embeddings_created=result.embeddings_created,
            chunks_used=result.chunks_used,
            used_structural_chunks=result.used_structural_chunks,
            errors=result.errors,
        )

    @router.post(
        "/embed/note/{note_id}",
        response_model=EmbedResultResponse,
    )
    async def embed_note(
        note_id: str,
        service: EmbeddingService = Depends(get_embedding_service),
    ):
        """Generate embeddings for a single note."""
        result = await service.embed_note(note_id)
        return EmbedResultResponse(
            id=result.id,
            embeddings_created=result.embeddings_created,
            chunks_used=result.chunks_used,
            used_structural_chunks=result.used_structural_chunks,
            errors=result.errors,
        )

    @router.post(
        "/rebuild",
        response_model=RebuildResultResponse,
    )
    async def rebuild_embeddings(
        body: RebuildRequest,
        service: EmbeddingService = Depends(get_embedding_service),
    ):
        """Rebuild all embeddings."""
        result = await service.rebuild_embeddings(
            include_sources=body.include_sources,
            include_notes=body.include_notes,
            include_insights=body.include_insights,
            mode=body.mode,
        )
        return RebuildResultResponse(
            sources_processed=result.sources_processed,
            notes_processed=result.notes_processed,
            insights_processed=result.insights_processed,
            total_embeddings=result.total_embeddings,
            errors=result.errors,
        )

    @router.get("/config", response_model=ConfigResponse)
    async def get_config(
        service: EmbeddingService = Depends(get_embedding_service),
    ):
        """Return the current embedding configuration."""
        cfg = service.config
        return ConfigResponse(
            batch_size=cfg.batch_size,
            max_concurrent=cfg.max_concurrent,
            retry_attempts=cfg.retry_attempts,
            retry_delay=cfg.retry_delay,
            fallback_chunk_size=cfg.fallback_chunk_size,
            fallback_chunk_overlap=cfg.fallback_chunk_overlap,
        )

    return router
