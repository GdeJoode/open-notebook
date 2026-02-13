"""
Source API routes.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from shared.models import Asset, Chunk, Source, SourceEmbedding, SourceInsight
from surrealdb_service.repositories import (
    ChunkRepository,
    SourceEmbeddingRepository,
    SourceInsightRepository,
    SourceRepository,
)

router = APIRouter()


# Request/Response models
class SourceCreate(BaseModel):
    title: Optional[str] = None
    asset: Optional[Dict[str, Any]] = None
    topics: List[str] = []
    full_text: Optional[str] = None
    notebook_id: Optional[str] = None


class SourceUpdate(BaseModel):
    title: Optional[str] = None
    topics: Optional[List[str]] = None
    full_text: Optional[str] = None


class ChunkCreate(BaseModel):
    source: str
    text: str
    order: int
    physical_page: int
    printed_page: Optional[int] = None
    chapter: Optional[str] = None
    paragraph_number: Optional[int] = None
    element_type: str = "paragraph"
    positions: List[List[float]] = []
    metadata: Dict[str, Any] = {}


class InsightCreate(BaseModel):
    insight_type: str
    content: str
    embedding: Optional[List[float]] = None


class EmbeddingCreate(BaseModel):
    content: str
    order: int
    embedding: List[float]


# Source endpoints
@router.get("", response_model=List[Dict[str, Any]])
async def list_sources(limit: Optional[int] = None, offset: Optional[int] = None):
    """List all sources."""
    repo = SourceRepository()
    sources = await repo.get_all(order_by="updated DESC", limit=limit, offset=offset)
    # Exclude full_text for list view
    return [{k: v for k, v in s.model_dump().items() if k != "full_text"} for s in sources]


@router.post("", response_model=Dict[str, Any])
async def create_source(data: SourceCreate):
    """Create a new source."""
    repo = SourceRepository()
    source_data = {k: v for k, v in data.model_dump().items() if k != "notebook_id"}
    source = await repo.create(source_data)

    if data.notebook_id:
        await repo.add_to_notebook(source.id, data.notebook_id)

    return source.model_dump()


@router.get("/{source_id}", response_model=Dict[str, Any])
async def get_source(source_id: str, include_full_text: bool = False):
    """Get a source by ID."""
    repo = SourceRepository()
    source = await repo.get(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    data = source.model_dump()
    if not include_full_text:
        data.pop("full_text", None)
    return data


@router.patch("/{source_id}", response_model=Dict[str, Any])
async def update_source(source_id: str, data: SourceUpdate):
    """Update a source."""
    repo = SourceRepository()
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    source = await repo.update(source_id, updates)
    return source.model_dump()


@router.delete("/{source_id}")
async def delete_source(source_id: str):
    """Delete a source and related data."""
    source_repo = SourceRepository()
    chunk_repo = ChunkRepository()

    # Delete related chunks and embeddings
    await chunk_repo.delete_by_source(source_id)
    await source_repo.delete_embeddings(source_id)

    # Delete the source
    await source_repo.delete(source_id)
    return {"status": "deleted"}


@router.post("/{source_id}/notebook/{notebook_id}")
async def add_source_to_notebook(source_id: str, notebook_id: str):
    """Add a source to a notebook."""
    repo = SourceRepository()
    success = await repo.add_to_notebook(source_id, notebook_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add source to notebook")
    return {"status": "added"}


# Insights endpoints
@router.get("/{source_id}/insights", response_model=List[Dict[str, Any]])
async def get_source_insights(source_id: str):
    """Get all insights for a source."""
    repo = SourceRepository()
    insights = await repo.get_insights(source_id)
    return [i.model_dump() for i in insights]


@router.post("/{source_id}/insights", response_model=Dict[str, Any])
async def add_source_insight(source_id: str, data: InsightCreate):
    """Add an insight to a source."""
    repo = SourceRepository()
    insight = await repo.add_insight(
        source_id,
        data.insight_type,
        data.content,
        data.embedding,
    )
    return insight.model_dump()


@router.delete("/insights/{insight_id}")
async def delete_insight(insight_id: str):
    """Delete an insight."""
    repo = SourceInsightRepository()
    await repo.delete(insight_id)
    return {"status": "deleted"}


# Chunks endpoints
@router.get("/{source_id}/chunks", response_model=List[Dict[str, Any]])
async def get_source_chunks(source_id: str, page: Optional[int] = None):
    """Get all chunks for a source."""
    repo = ChunkRepository()
    chunks = await repo.get_by_source(source_id, page)
    return [c.model_dump() for c in chunks]


@router.post("/{source_id}/chunks", response_model=Dict[str, Any])
async def create_chunk(source_id: str, data: ChunkCreate):
    """Create a chunk for a source."""
    repo = ChunkRepository()
    chunk_data = data.model_dump()
    chunk_data["source"] = source_id
    chunk = await repo.create(chunk_data)
    return chunk.model_dump()


@router.post("/{source_id}/chunks/bulk", response_model=List[Dict[str, Any]])
async def create_chunks_bulk(source_id: str, chunks: List[ChunkCreate]):
    """Create multiple chunks at once."""
    repo = ChunkRepository()
    chunk_data = [
        {**c.model_dump(), "source": source_id}
        for c in chunks
    ]
    created = await repo.bulk_create(chunk_data)
    return [c.model_dump() for c in created]


@router.delete("/chunks/{chunk_id}")
async def delete_chunk(chunk_id: str):
    """Delete a chunk."""
    repo = ChunkRepository()
    await repo.delete(chunk_id)
    return {"status": "deleted"}


# Embeddings endpoints
@router.get("/{source_id}/embeddings", response_model=List[Dict[str, Any]])
async def get_source_embeddings(source_id: str):
    """Get all embeddings for a source."""
    repo = SourceRepository()
    embeddings = await repo.get_embeddings(source_id)
    return [e.model_dump() for e in embeddings]


@router.get("/{source_id}/embeddings/count", response_model=Dict[str, int])
async def get_source_embedding_count(source_id: str):
    """Get the count of embeddings for a source."""
    repo = SourceRepository()
    count = await repo.get_embedding_count(source_id)
    return {"count": count}


@router.post("/{source_id}/embeddings", response_model=Dict[str, Any])
async def add_source_embedding(source_id: str, data: EmbeddingCreate):
    """Add an embedding to a source."""
    repo = SourceRepository()
    embedding = await repo.add_embedding(
        source_id,
        data.content,
        data.order,
        data.embedding,
    )
    return embedding.model_dump()


@router.delete("/{source_id}/embeddings")
async def delete_source_embeddings(source_id: str):
    """Delete all embeddings for a source."""
    repo = SourceRepository()
    count = await repo.delete_embeddings(source_id)
    return {"status": "deleted", "count": count}
