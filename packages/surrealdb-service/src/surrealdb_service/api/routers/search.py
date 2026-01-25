"""
Search API routes.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from surrealdb_service.repositories import SearchRepository

router = APIRouter()


class TextSearchRequest(BaseModel):
    keyword: str
    results: int = 10
    include_sources: bool = True
    include_notes: bool = True


class VectorSearchRequest(BaseModel):
    embedding: List[float]
    results: int = 10
    include_sources: bool = True
    include_notes: bool = True
    minimum_score: float = 0.2


class HybridSearchRequest(BaseModel):
    keyword: str
    embedding: List[float]
    results: int = 10
    include_sources: bool = True
    include_notes: bool = True
    minimum_score: float = 0.2
    text_weight: float = 0.5


@router.post("/text", response_model=List[Dict[str, Any]])
async def text_search(request: TextSearchRequest):
    """Perform a text search across sources and notes."""
    repo = SearchRepository()
    return await repo.text_search(
        request.keyword,
        request.results,
        request.include_sources,
        request.include_notes,
    )


@router.post("/vector", response_model=List[Dict[str, Any]])
async def vector_search(request: VectorSearchRequest):
    """Perform a vector similarity search."""
    repo = SearchRepository()
    return await repo.vector_search(
        request.embedding,
        request.results,
        request.include_sources,
        request.include_notes,
        request.minimum_score,
    )


@router.post("/hybrid", response_model=List[Dict[str, Any]])
async def hybrid_search(request: HybridSearchRequest):
    """Perform a hybrid text + vector search."""
    repo = SearchRepository()
    return await repo.hybrid_search(
        request.keyword,
        request.embedding,
        request.results,
        request.include_sources,
        request.include_notes,
        request.minimum_score,
        request.text_weight,
    )
