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

import ipaddress
import socket
import unicodedata
from typing import Optional
from urllib.parse import urlparse

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
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

_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}

#: Codepoints the fetcher's UTS-46/IDNA transform collapses that NFKC does NOT,
#: folded here so the guard classifies the SAME host the fetcher connects to. An
#: exhaustive 0x110000 scan of UTS-46 vs NFKC found exactly two divergent classes:
#:  - the ideographic dot separators U+3002 / U+FF61 (mapped to "."); NFKC already
#:    handles U+FF0E + fullwidth digits. Else http://169。254。169。254/ (ASCII
#:    digits, ideographic dots) slips past the numeric check to metadata.
#:  - the Unicode-16 "outlined digits" U+1CCF0–U+1CCF9 (the ONLY codepoints UTS-46
#:    maps to an ASCII digit that NFKC leaves alone); folded so an outlined-digit
#:    IP can't dodge the numeric check on the Chromium fetch path.
_HOST_FOLD = {0x3002: ".", 0xFF61: ".", **{0x1CCF0 + i: str(i) for i in range(10)}}


def _normalize_host(raw: str) -> str:
    """Collapse a URL host to the ASCII form the OS resolver / fetcher will see.

    NFKC-folds fullwidth variants + translates the UTS-46/NFKC-divergent dot and
    digit codepoints (:data:`_HOST_FOLD`), so a homoglyph-obfuscated numeric IP
    (fullwidth/outlined digits, ideographic dots) normalizes to the same
    dotted-quad the guard then classifies. A genuine IDN hostname (e.g.
    ``münchen.de``) normalizes but stays non-numeric → allowed.
    """
    h = unicodedata.normalize("NFKC", (raw or "").strip())
    return h.translate(_HOST_FOLD).lower().rstrip(".")


def _host_to_ip(host: str):
    """Coerce a URL host to a canonical IP the way the OS resolver would for a
    NUMERIC host — including the non-canonical IPv4 encodings that
    ``ipaddress.ip_address`` REJECTS but glibc ``getaddrinfo`` accepts and
    resolves to the same address: decimal (``2130706433`` → 127.0.0.1), octal
    (``0177.0.0.1``), hex (``0x7f.0.0.1``), and short forms (``127.1``). Those
    encodings were the SSRF bypass in the round-1 guard.

    Returns an ``ipaddress`` address, or ``None`` for a real (non-numeric)
    hostname — DNS resolution is deliberately NOT performed here, so a hostname
    that RESOLVES to a private IP (DNS-rebinding) stays out of scope; that deeper
    fix belongs in the shared link-fetch layer the password-gated UI also uses.
    """
    try:
        return ipaddress.ip_address(host)  # canonical IPv4 / IPv6
    except ValueError:
        pass
    try:
        # inet_aton parses the SAME lenient numeric-IPv4 forms getaddrinfo does
        # (decimal / octal / hex / short) → 4 packed bytes → canonical IPv4.
        return ipaddress.ip_address(socket.inet_aton(host))
    except (OSError, ValueError):
        return None


def _reject_ssrf_url(url: str) -> None:
    """Reject SSRF targets before the link fetcher runs (write-key surface).

    Blocks non-http(s) schemes and any host that normalizes to a loopback /
    private / link-local / reserved / multicast / unspecified IP — INCLUDING the
    decimal/octal/hex/short-form and trailing-dot encodings (see
    :func:`_host_to_ip`) AND the Unicode homoglyph-dot / fullwidth encodings (see
    :func:`_normalize_host`), both of which earlier guards let through to the
    cloud metadata endpoint. Defense-in-depth on the AGENT key surface
    (externally distributed, unlike the shared UI password).

    Out of scope here (shared fetch-layer follow-ups, since the password-gated UI
    shares that layer): hostname→private-IP DNS-rebinding, and a public URL that
    30x-REDIRECTS to a private target (the fetcher follows redirects; a pre-flight
    guard can't see the hop). Raises 422 on a disallowed URL.
    """
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid URL")
    if (parsed.scheme or "").lower() not in ("http", "https"):
        raise HTTPException(status_code=422, detail="URL scheme must be http or https")
    # Normalize Unicode homoglyph dots / fullwidth forms AND strip a trailing
    # FQDN dot (127.0.0.1. → 127.0.0.1) so the host can't dodge the numeric check
    # while still resolving to the same address downstream.
    host = _normalize_host(parsed.hostname or "")
    if not host:
        raise HTTPException(status_code=422, detail="URL has no host")
    if host in _BLOCKED_HOSTNAMES:
        raise HTTPException(status_code=422, detail="URL host is not allowed")
    ip = _host_to_ip(host)
    if ip is not None and (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        raise HTTPException(status_code=422, detail="URL host is not allowed")


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
    Returns the enqueued job id to poll via GET /agents/jobs/{id}. The multipart
    upload siblings are `POST /process-document` and `POST /process-audio` (G.3b).
    """
    from app_main.api.routers.sources_upload import _create_source_impl
    from app_main.api.schemas import SourceCreate

    _reject_ssrf_url(body.url)  # block loopback / private / metadata targets

    source_create = SourceCreate(
        type="link",
        url=body.url,
        notebook_id=body.notebook_id,
        transformations=body.transformations,
        # async: enqueue a process_source JOB (sets source.command) instead of
        # parsing synchronously in-request. Without this the parse blocks the
        # request for minutes and no pollable job id is produced.
        async_processing=True,
    )
    result = await _create_source_impl(
        source_create, None, source_svc, notebook_svc, transformation_svc
    )
    # The enqueued process_source command id is stashed on the source as `command`.
    src = await source_svc.get(result.id)
    job_id = str(getattr(src, "command", "") or "")
    # Record the (agent, job) ownership so GET /jobs/{id} can authorize the caller
    # (jobs carry no owner column). The audit middleware reads request.state.job_id.
    # Caveat: this couples poll-authz to the best-effort audit write — if that
    # CREATE fails (the failure the middleware's try/except swallows), the caller
    # gets a valid job_id but 404s on their own poll. That is FAIL-SAFE (deny, not
    # leak); a dedicated ownership store would decouple it (deployment follow-up).
    request.state.job_id = job_id
    return ProcessSourceResponse(
        job_id=job_id, source_id=str(result.id), status="queued"
    )


def _parse_transformations(raw: Optional[str]) -> Optional[list]:
    """A multipart ``transformations`` form field → a list of names.

    Accepts a JSON array (``["t1","t2"]``) or a comma-separated string; empty →
    None (run the notebook default). The chain validates each name exists (404).
    """
    if not raw or not raw.strip():
        return None
    import json

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(t) for t in parsed]
    except (ValueError, TypeError):
        pass
    return [t.strip() for t in raw.split(",") if t.strip()]


async def _ingest_upload(
    request: Request,
    file: UploadFile,
    notebook_id: str,
    transformations: Optional[str],
    source_svc,
    notebook_svc,
    transformation_svc,
) -> ProcessSourceResponse:
    """Shared multipart-upload ingest for process-document / process-audio.

    A thin façade over the SAME ``_create_source_impl`` (``type="upload"``) the UI
    uses — it runs the size + page-count upload guards, saves the file, and
    enqueues the ``process_source`` job (async → a real pollable ``job_id``). The
    ``process_source`` pipeline routes by file type, so documents and audio share
    this path (the two routes differ only in their documented intent).
    """
    from app_main.api.routers.sources_upload import _create_source_impl
    from app_main.api.schemas import SourceCreate

    source_create = SourceCreate(
        type="upload",
        notebook_id=notebook_id,
        transformations=_parse_transformations(transformations),
        async_processing=True,
    )
    result = await _create_source_impl(
        source_create, file, source_svc, notebook_svc, transformation_svc
    )
    src = await source_svc.get(result.id)
    job_id = str(getattr(src, "command", "") or "")
    # Stamp the enqueued job so GET /jobs/{id} can authorize this caller (same
    # ownership binding as process-url — jobs carry no owner column).
    request.state.job_id = job_id
    return ProcessSourceResponse(
        job_id=job_id, source_id=str(result.id), status="queued"
    )


@router.post("/process-document", response_model=ProcessSourceResponse)
@limiter.limit(agent_default_limit, key_func=agent_key_func)
async def process_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    notebook_id: str = Form(...),
    transformations: Optional[str] = Form(None),
    _key=Depends(require_agent_key("write")),
    source_svc=Depends(get_source_service),
    notebook_svc=Depends(get_notebook_service),
    transformation_svc=Depends(get_transformation_service),
) -> ProcessSourceResponse:
    """Ingest an uploaded document headlessly (multipart) via process_source.

    WRITE scope. Reuses the shipped upload path incl. `enforce_upload_guards`
    (413 oversized / 422 over-paged); returns a pollable `job_id`. Audio files
    are accepted here too, but `POST /process-audio` documents that intent.
    """
    return await _ingest_upload(
        request,
        file,
        notebook_id,
        transformations,
        source_svc,
        notebook_svc,
        transformation_svc,
    )


@router.post("/process-audio", response_model=ProcessSourceResponse)
@limiter.limit(agent_default_limit, key_func=agent_key_func)
async def process_audio(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    notebook_id: str = Form(...),
    transformations: Optional[str] = Form(None),
    _key=Depends(require_agent_key("write")),
    source_svc=Depends(get_source_service),
    notebook_svc=Depends(get_notebook_service),
    transformation_svc=Depends(get_transformation_service),
) -> ProcessSourceResponse:
    """Ingest an uploaded audio file headlessly (multipart) via process_source.

    WRITE scope. Same upload chain as process-document (the process_source
    pipeline transcribes/parses by file type); returns a pollable `job_id`.
    """
    return await _ingest_upload(
        request,
        file,
        notebook_id,
        transformations,
        source_svc,
        notebook_svc,
        transformation_svc,
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
@limiter.limit(agent_default_limit, key_func=agent_key_func)
async def get_job_status(
    request: Request,
    response: Response,
    job_id: str,
    key: AgentKeyContext = Depends(require_agent_key("read")),
) -> JobStatusResponse:
    """Poll a job's status — but ONLY a job THIS agent enqueued.

    Jobs carry no owner column, so ownership is authorized via the agent's own
    audit trail (process-url stamps the job_id there). A job the calling agent
    did not enqueue → 404 (not the status), so this is neither an IDOR nor an
    existence oracle. An admin key may poll any job. On a hit, the payload is
    verbatim CommandService.get_command_status; an unknown id → status:"unknown".
    """
    from app_main.services.agents.audit_service import AgentAuditService
    from app_main.services.command_service import CommandService

    if key.permission != "admin":
        owns = await AgentAuditService().agent_owns_job(key.agent_id, job_id)
        if not owns:
            raise HTTPException(status_code=404, detail="Job not found")

    status = await CommandService.get_command_status(job_id)
    return JobStatusResponse(
        job_id=str(status.get("job_id", job_id)),
        status=str(status.get("status", "unknown")),
        result=status.get("result"),
        # CommandService returns the failure text under `error_message`; a failed
        # job would otherwise always surface error=null.
        error=status.get("error_message") or status.get("error"),
    )


@router.get("/audit-log")
@limiter.limit(agent_default_limit, key_func=agent_key_func)
async def get_audit_log(
    request: Request,
    response: Response,
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
