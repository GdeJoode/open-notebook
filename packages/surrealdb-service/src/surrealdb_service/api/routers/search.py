"""
Search API routes.

Deze losse API heeft geen sleutel; zoeken krijgt daarom het strengste bereik
met inhoud (F2-05, ADR 0002 C). Zoeken leest alleen fragmenten (ADR 0004).
"""

from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel
from shared.models.classification import ClassificationScope

from surrealdb_service.repositories import SearchRepository

router = APIRouter()

_SCOPE = ClassificationScope.EXTERN


class TextSearchRequest(BaseModel):
    keyword: str
    results: int = 10


class VectorSearchRequest(BaseModel):
    embedding: List[float]
    results: int = 10
    minimum_score: float = 0.2


class HybridSearchRequest(BaseModel):
    keyword: str
    embedding: List[float]
    results: int = 10
    minimum_score: float = 0.2
    text_weight: float = 0.5


@router.post("/text", response_model=List[Dict[str, Any]])
async def text_search(request: TextSearchRequest):
    """Perform a text search across source chunks."""
    repo = SearchRepository()
    return await repo.text_search(request.keyword, request.results, scope=_SCOPE)


@router.post("/vector", response_model=List[Dict[str, Any]])
async def vector_search(request: VectorSearchRequest):
    """Perform a vector similarity search."""
    repo = SearchRepository()
    return await repo.vector_search(
        request.embedding, request.results, request.minimum_score, scope=_SCOPE
    )


@router.post("/hybrid", response_model=List[Dict[str, Any]])
async def hybrid_search(request: HybridSearchRequest):
    """Perform a hybrid text + vector search."""
    repo = SearchRepository()
    return await repo.hybrid_search(
        request.keyword,
        request.embedding,
        request.results,
        request.minimum_score,
        request.text_weight,
        scope=_SCOPE,
    )
