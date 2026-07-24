"""Versioned agent capability API (Track G.1).

The ``/api/v1/agents`` surface external agents call with an ``X-API-Key``. The
whole router is gated at the ROUTER level by two dependencies, so EVERY current
and future route is protected without opting in per-route (the exclusion from the
shared-password middleware is subtree-wide — a route that forgot the gate would
be fully open):

  1. :func:`agent_ip_throttle` — a per-IP requests/min cap that runs BEFORE auth,
     so a flood of invalid keys from one host is bounded (each such request
     otherwise costs a DB read + an audit write);
  2. :func:`require_agent_key` (``read`` baseline) — the SOLE authenticator, fail-
     closed and password-independent. A route needing a higher scope adds its own
     ``Depends(require_agent_key("write"|"admin"))``.

G.1 ships one real capability — ``POST /extract-entities`` (typed entities from
raw text, no DB) — proving the throttle + auth + per-key limit + audit path end to
end; later phases add the ingest façade and the other capabilities.
"""

from fastapi import APIRouter, Depends, Request, Response
from shared.models.agents import (
    ExtractEntitiesRequest,
    ExtractEntitiesResponse,
    ExtractedEntity,
)

from app_main.api.agent_auth import require_agent_key
from app_main.api.agent_rate_limit import (
    agent_default_limit,
    agent_ip_throttle,
    agent_key_func,
)
from app_main.api.rate_limit import limiter

router = APIRouter(
    prefix="/api/v1/agents",
    tags=["agents"],
    # Router-level gate: per-IP pre-auth throttle THEN the read-scope key check,
    # applied to every route on this router (in order).
    dependencies=[
        Depends(agent_ip_throttle()),
        Depends(require_agent_key("read")),
    ],
)


@router.post("/extract-entities", response_model=ExtractEntitiesResponse)
@limiter.limit(agent_default_limit, key_func=agent_key_func)
async def extract_entities(
    request: Request,
    response: Response,
    body: ExtractEntitiesRequest,
) -> ExtractEntitiesResponse:
    """Extract typed entities from raw text (stateless — no source is created).

    Gated by the router-level read-scope dependency; the per-key slowapi limit
    below bounds authenticated abuse (the pre-auth throttle bounds floods).
    """
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
async def agent_openapi(request: Request) -> dict:
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
