"""Versioned agent capability API (Track G.1).

The ``/api/v1/agents`` surface external agents call with an ``X-API-Key``. Every
route is gated by :func:`require_agent_key` (the SOLE authenticator — these paths
are excluded from the shared-password middleware, G-D1) and rate-limited per key.

G.1 ships one real capability — ``POST /extract-entities`` (typed entities from
raw text, ``read`` scope, no DB) — proving the auth + limit + audit path end to
end; later phases add the ingest façade and the other capabilities.
"""

from fastapi import APIRouter, Depends, Request, Response
from shared.models.agents import (
    ExtractEntitiesRequest,
    ExtractEntitiesResponse,
    ExtractedEntity,
)

from app_main.api.agent_auth import AgentKeyContext, require_agent_key
from app_main.api.agent_rate_limit import agent_default_limit, agent_key_func
from app_main.api.rate_limit import limiter

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


@router.post("/extract-entities", response_model=ExtractEntitiesResponse)
@limiter.limit(agent_default_limit, key_func=agent_key_func)
async def extract_entities(
    request: Request,
    response: Response,
    body: ExtractEntitiesRequest,
    key: AgentKeyContext = Depends(require_agent_key("read")),
) -> ExtractEntitiesResponse:
    """Extract typed entities from raw text (stateless — no source is created)."""
    from app_main.services.entity_extraction_service import extract_from_text

    entities = await extract_from_text(body.text, body.ontology_name)
    return ExtractEntitiesResponse(
        entities=[
            ExtractedEntity(
                name=e.name, entity_type=e.entity_type, confidence=e.confidence
            )
            for e in entities
        ]
    )


@router.get("/openapi.json")
async def agent_openapi(
    request: Request,
    key: AgentKeyContext = Depends(require_agent_key("read")),
) -> dict:
    """The OpenAPI spec filtered to just the agent surface (auto-generated)."""
    full = request.app.openapi()
    agent_paths = {
        path: item
        for path, item in full.get("paths", {}).items()
        if path.startswith("/api/v1/agents")
    }
    return {
        "openapi": full.get("openapi", "3.1.0"),
        "info": {
            **full.get("info", {}),
            "title": "Open Notebook Agent API",
        },
        "paths": agent_paths,
        "components": full.get("components", {}),
    }
