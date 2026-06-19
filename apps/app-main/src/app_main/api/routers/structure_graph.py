"""Structure-graph router — document structure subgraph for the inspect UI (I.F).

Exposes ``GET /api/sources/{source_id}/structure-graph`` returning a
Sigma/graphology-friendly ``{nodes, edges}`` payload. The page-limit is capped at
500 (Q-I-F-2); a request above the cap is rejected with 422.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from app_main.services.graph.structure_graph_reader import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    read_structure_graph,
)

router = APIRouter(prefix="/sources", tags=["structure-graph"])


@router.get("/{source_id}/structure-graph", response_model=Dict[str, Any])
async def get_structure_graph(
    source_id: str,
    page: Optional[int] = Query(
        None, ge=0, description="Optional physical-page filter (0-indexed)."
    ),
    page_limit: int = Query(
        DEFAULT_PAGE_LIMIT,
        ge=1,
        description=(
            f"Max doc_node rows to return. Hard cap {MAX_PAGE_LIMIT}; "
            "higher requests are rejected with 422."
        ),
    ),
) -> Dict[str, Any]:
    """Return the structure subgraph (nodes + edges) for a source.

    ``page_limit`` defaults to 100 and is capped at 500. A value above the cap
    is a client error (422) rather than a silent clamp, so callers learn they
    asked for more than the server will serve.
    """
    if page_limit > MAX_PAGE_LIMIT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"page_limit exceeds the maximum of {MAX_PAGE_LIMIT} "
                f"(requested {page_limit})."
            ),
        )

    return await read_structure_graph(
        source_id, page=page, page_limit=page_limit
    )
