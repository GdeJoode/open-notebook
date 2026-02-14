"""Knowledge graph router - entity browsing and graph visualization."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app_main.dependencies import get_knowledge_graph_service
from app_main.services.knowledge_graph_service import KnowledgeGraphService

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])


@router.get("/entities", response_model=Dict[str, Any])
async def list_entities(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    entity_type: Optional[str] = None,
    svc: KnowledgeGraphService = Depends(get_knowledge_graph_service),
):
    """List entities with pagination and optional type filter."""
    items = await svc.list_entities(limit=limit, offset=offset, entity_type=entity_type)
    total = await svc.count_entities(entity_type=entity_type)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/entities/{entity_id:path}", response_model=Dict[str, Any])
async def get_entity(
    entity_id: str,
    svc: KnowledgeGraphService = Depends(get_knowledge_graph_service),
):
    """Get a single entity with relations and source metadata."""
    result = await svc.get_entity(entity_id)
    if not result:
        raise HTTPException(status_code=404, detail="Entity not found")
    return result


@router.get("/entity-types", response_model=List[Dict[str, Any]])
async def get_entity_types(
    svc: KnowledgeGraphService = Depends(get_knowledge_graph_service),
):
    """Get entity type summary with counts."""
    return await svc.get_entity_types_summary()


@router.get("/graph", response_model=Dict[str, Any])
async def get_graph(
    entity_type: Optional[str] = None,
    limit: int = Query(5000, ge=1, le=50000),
    svc: KnowledgeGraphService = Depends(get_knowledge_graph_service),
):
    """Get graph data (nodes + edges) for visualization."""
    return await svc.get_graph_data(entity_type=entity_type, limit=limit)


@router.get("/search", response_model=List[Dict[str, Any]])
async def search_entities(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    svc: KnowledgeGraphService = Depends(get_knowledge_graph_service),
):
    """Search entities by text."""
    return await svc.search_entities(query=q, limit=limit)
