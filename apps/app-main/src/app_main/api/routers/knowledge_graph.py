"""Knowledge graph router - entity browsing and graph visualization."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app_main.dependencies import (
    get_knowledge_graph_service,
    get_mentions_projection_service,
)
from app_main.services.knowledge_graph_service import KnowledgeGraphService
from app_main.services.mentions_projection_service import (
    MentionsProjectionService,
)

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])


@router.get("/entities", response_model=Dict[str, Any])
async def list_entities(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    entity_type: Optional[str] = None,
    status: Optional[str] = Query(
        None,
        description="Triage-status filter: 'active' | 'reference' | 'archived'.",
    ),
    source_id: Optional[str] = Query(
        None,
        description=(
            "Source-scoped filter: only entities whose source_documents "
            "contains this source record id (e.g. 'source:abc')."
        ),
    ),
    svc: KnowledgeGraphService = Depends(get_knowledge_graph_service),
):
    """List entities with pagination and optional type/status/source filters.

    The optional ``status`` filter (Q.5) is backed by ``idx_entity_status`` so a
    status-filtered page stays cheap; rows carry ``structural_degree`` /
    ``doc_count`` for the KG table. The optional ``source_id`` filter scopes the
    listing to a single source for the source-detail Entities tab.
    """
    items = await svc.list_entities(
        limit=limit,
        offset=offset,
        entity_type=entity_type,
        status=status,
        source_id=source_id,
    )
    total = await svc.count_entities(
        entity_type=entity_type, status=status, source_id=source_id
    )
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
    limit: int = Query(500, ge=1, le=50000),
    svc: KnowledgeGraphService = Depends(get_knowledge_graph_service),
):
    """Get graph data (nodes + edges) for visualization.

    ``limit`` is a soft cap on the node count, not a hard slice of the
    entity table: the endpoint is edge-first and active-only, so every
    returned edge is guaranteed to have both endpoints in ``nodes``.
    """
    return await svc.get_graph_data(entity_type=entity_type, limit=limit)


@router.get("/search", response_model=List[Dict[str, Any]])
async def search_entities(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    svc: KnowledgeGraphService = Depends(get_knowledge_graph_service),
):
    """Search entities by text."""
    return await svc.search_entities(query=q, limit=limit)


# ---------------------------------------------------------------------------
# Track U.2 — document↔entity (`mentions`) edge projection
# ---------------------------------------------------------------------------


@router.get("/document-graph", response_model=Dict[str, Any])
async def get_document_graph(
    min_weight: float = Query(
        0.0,
        ge=0.0,
        description=(
            "Per-edge weight floor. Default 0.0 returns every df>=2 edge; raise "
            "to 0.3 for the named-only overview skeleton (U.1 preset)."
        ),
    ),
    source_id: Optional[str] = Query(
        None,
        description=(
            "Scope to edges incident on a single source (the document's "
            "neighbourhood). 'source:abc' record id."
        ),
    ),
    svc: MentionsProjectionService = Depends(get_mentions_projection_service),
):
    """Fetch the materialized document↔entity (`mentions`) edges (feeds U.4 viz).

    Each edge carries its endpoints (``source`` / ``target``), the projection
    ``weight`` (R.2 salience × rarity), and the per-edge "why" — the shared
    concept name/type/df. Edges are read from the regenerated ``mentions`` table;
    run ``POST /document-graph/regenerate`` first to (re)build them.
    """
    edges = await svc.get_document_entity_edges(
        min_weight=min_weight, source_id=source_id
    )
    return {"edges": edges, "count": len(edges)}


@router.post("/document-graph/regenerate", response_model=Dict[str, Any])
async def regenerate_document_graph(
    min_weight: float = Query(
        0.0,
        ge=0.0,
        description="Per-edge weight cutoff applied during regeneration.",
    ),
    named_only: bool = Query(
        False,
        description="Use the 0.3 named-only preset (overview skeleton).",
    ),
    drop_singletons: bool = Query(
        True,
        description=(
            "Drop df==1 singleton concepts (the mandatory R.6 legibility step). "
            "Leave true for the drawable graph."
        ),
    ),
    dry_run: bool = Query(
        False,
        description=(
            "Compute the projected edge set and report its size WITHOUT writing "
            "(the gated-live-write workflow: count first, persist on confirm)."
        ),
    ),
    svc: MentionsProjectionService = Depends(get_mentions_projection_service),
):
    """Regenerate the ``mentions`` document↔entity edges from the projection.

    Idempotent: clears the existing edges then RELATEs the freshly-projected set
    from ``entity.source_documents`` (R.2 weight × R.6 normalization), so a re-run
    yields the same edge set with no duplicates. The canonical ``entity`` /
    ``source`` rows are never mutated — only the derived ``mentions`` table.

    With ``dry_run=true`` nothing is written: the projected edge count is
    returned so a live regenerate can be confirmed before it persists.
    """
    if dry_run:
        edges, n_sources = await svc.project_edges(
            drop_singletons=drop_singletons,
            min_weight=min_weight,
            named_only=named_only,
        )
        return {
            "dry_run": True,
            "would_create": len(edges),
            "n_sources": n_sources,
        }

    result = await svc.regenerate(
        drop_singletons=drop_singletons,
        min_weight=min_weight,
        named_only=named_only,
    )
    return {
        "dry_run": False,
        "cleared": result.cleared,
        "created": result.created,
        "failed": result.failed,
        "active_entities": result.active_entities,
        "emitted_concepts": result.emitted_concepts,
        "n_sources": result.n_sources,
        "min_weight": result.min_weight,
    }
