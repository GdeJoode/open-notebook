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

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from shared.models.agents import (
    ExtractEntitiesRequest,
    ExtractEntitiesResponse,
    ExtractedEntity,
    GenerateSummaryRequest,
    GenerateSummaryResponse,
    JobStatusResponse,
    ProcessSourceResponse,
    ProcessUrlRequest,
)

from app_main.api.agent_auth import AgentKeyContext, require_agent_key
from app_main.api.agent_rate_limit import (
    agent_default_limit,
    agent_ip_throttle,
    agent_key_func,
)
from app_main.api.rate_limit import limiter
from app_main.dependencies import (
    get_notebook_service,
    get_source_service,
    get_summarization_service,
    get_transformation_service,
)
from app_main.services.summarization_service import SummarizationService

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


@router.post("/generate-summary", response_model=GenerateSummaryResponse)
@limiter.limit(agent_default_limit, key_func=agent_key_func)
async def generate_summary(
    request: Request,
    response: Response,
    body: GenerateSummaryRequest,
    _key=Depends(require_agent_key("write")),
    service: SummarizationService = Depends(get_summarization_service),
) -> GenerateSummaryResponse:
    """Summarize raw text (stateless — no source/summary is persisted).

    WRITE scope: the router-level read gate resolves the key; this adds the
    write-scope check (reusing the cached key — no second lookup). An unknown /
    unimplemented strategy is a 422.
    """
    try:
        result = await service.summarize_text(
            body.text, strategy=body.strategy, config=body.config
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return GenerateSummaryResponse(
        summary=result.get("summary", ""), strategy=result.get("strategy", "")
    )


@router.post("/process-url", response_model=ProcessSourceResponse)
@limiter.limit(agent_default_limit, key_func=agent_key_func)
async def process_url(
    request: Request,
    response: Response,
    body: ProcessUrlRequest,
    _key=Depends(require_agent_key("write")),
    source_svc=Depends(get_source_service),
    notebook_svc=Depends(get_notebook_service),
    transformation_svc=Depends(get_transformation_service),
) -> ProcessSourceResponse:
    """Ingest a URL headlessly via the SAME process_source chain as the UI.

    A thin façade over `_create_source_impl` (type="link") — no parallel pipeline.
    Returns the enqueued job id to poll via GET /agents/jobs/{id}.
    (process-document / process-audio, which take a multipart upload + reuse
    enforce_upload_guards, are deferred to G.3b.)
    """
    from app_main.api.routers.sources_upload import _create_source_impl
    from app_main.api.schemas import SourceCreate

    source_create = SourceCreate(
        type="link",
        url=body.url,
        notebook_id=body.notebook_id,
        transformations=body.transformations,
    )
    result = await _create_source_impl(
        source_create, None, source_svc, notebook_svc, transformation_svc
    )
    # The enqueued process_source command id is stashed on the source as `command`.
    src = await source_svc.get(result.id)
    job_id = str(getattr(src, "command", "") or "")
    return ProcessSourceResponse(
        job_id=job_id, source_id=str(result.id), status="queued"
    )


@router.get("/jobs/{job_id:path}", response_model=JobStatusResponse)
async def get_job_status(
    request: Request,
    job_id: str,
    _key=Depends(require_agent_key("read")),
) -> JobStatusResponse:
    """Poll a job's status — verbatim CommandService.get_command_status.

    An unknown id returns the ``status:"unknown"`` shape (200, not 500).
    """
    from app_main.services.command_service import CommandService

    status = await CommandService.get_command_status(job_id)
    return JobStatusResponse(
        job_id=str(status.get("job_id", job_id)),
        status=str(status.get("status", "unknown")),
        result=status.get("result"),
        error=status.get("error"),
    )


@router.get("/audit-log")
async def get_audit_log(
    request: Request,
    agent_id: Optional[str] = None,
    limit: int = 100,
    key: AgentKeyContext = Depends(require_agent_key("read")),
) -> dict:
    """The calling key's own agent_audit_log entries (newest first).

    Scoped to the caller's ``agent_id`` — a ``read`` key passing a different
    ``?agent_id=`` is still scoped to itself (no cross-agent leak). Only an
    ``admin`` key may read another agent's trail via ``?agent_id=``.
    """
    from app_main.services.agents.audit_service import AgentAuditService

    target = key.agent_id
    if agent_id and key.permission == "admin":
        target = agent_id
    limit = max(1, min(limit, 500))
    entries = await AgentAuditService().list_for_agent(target, limit=limit)
    return {"agent_id": target, "entries": entries}


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
