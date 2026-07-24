"""Notebook export router (Track D Phase D.3 onwards).

D.3 lands the NetworkX 7-format export endpoint here; D.1 (Obsidian)
and D.2 (JSONL) extend the same router so all Track-D export surfaces
share one prefix + tag and the FE can pull them through one client.

Design notes
------------

* All endpoints live under ``/api/notebooks/{notebook_id}/...`` so the
  notebook-scoped auth pattern (look up the notebook, 404 if missing)
  applies uniformly. The pattern mirrors B.2b's ``schema.ttl`` route.
* Every export is a *download*: ``Response`` with ``Content-Disposition:
  attachment`` + a format-specific ``Content-Type``. No streaming -- the
  NetworkX writers buffer the whole graph anyway, so a streaming response
  would just add a layer of indirection without changing the worst-case
  memory profile.
* Telemetry lives inside the service (not the router) so command-line
  scripts that wire ``NetworkxExportService`` directly still emit the
  metric.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from shared.models.export import (
    ExportFilter,
    JsonlExportRequest,
    NetworkxExportRequest,
    ObsidianExportRequest,
)
from surrealdb_service.repositories import EntityRepository

from app_main.dependencies import (
    get_entity_repo,
    get_jsonl_export_service,
    get_networkx_export_service,
    get_notebook_service,
    get_obsidian_export_service,
    get_okf_export_service,
    get_okf_import_service,
)
from app_main.services.command_service import CommandService
from app_main.services.jsonl_export_service import JsonlExportService
from app_main.services.networkx_export_service import NetworkxExportService
from app_main.services.notebook_service import NotebookService
from app_main.services.obsidian_export_service import (
    EXCLUDED_ENTITY_STATUSES,
    ObsidianExportService,
    VaultPathNotConfigured,
)
from app_main.services.okf_export_service import OkfExportService
from app_main.services.okf_import_service import OkfImportService

router = APIRouter(prefix="/notebooks/{notebook_id}", tags=["exports"])

# Above this surviving-entity count an OKF export is deferred to the job
# queue rather than built inline on the request path (OKF.2 constraint §5).
# A few thousand entities renders + zips in well under 10s inline; beyond
# that the async path keeps the request from blocking the worker.
OKF_ASYNC_ENTITY_THRESHOLD = 2000


# ---------------------------------------------------------------------------
# Filename helpers (lifted from B.2b's TTL exporter -- same sanitisation
# rules apply to every Content-Disposition the API emits).
# ---------------------------------------------------------------------------

_FILENAME_UNSAFE_RE = re.compile(r'[:/\\\r\n\t\x00"\']')


# Per-format MIME type + filename extension. Centralised so the
# router never falls out of sync with the writer side.
_FORMAT_TABLE: Dict[str, Tuple[str, str]] = {
    "graphml": ("application/xml", "graphml"),
    "gexf": ("application/xml", "gexf"),
    "gml": ("text/plain", "gml"),
    "json-tree": ("application/json", "json"),
    "edge-list": ("text/plain", "edges"),
    "adjacency-list": ("text/plain", "adj"),
    "pickle": ("application/octet-stream", "pkl"),
}


def _safe_filename(notebook_id: str, ext: str) -> str:
    """Build a download-safe filename: ``<notebook_id>.<ext>``.

    Same character class as the B.2b TTL exporter -- colons, slashes,
    quotes and control characters get replaced with underscores so the
    string survives ``Content-Disposition`` and common OS save dialogs.

    Example: ``("notebook:abc-123", "graphml") -> "notebook_abc-123.graphml"``.
    """
    safe_id = _FILENAME_UNSAFE_RE.sub("_", notebook_id)
    return f"{safe_id}.{ext}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/export-networkx",
    response_class=Response,
    responses={
        200: {
            "content": {
                "application/xml": {},
                "application/json": {},
                "application/octet-stream": {},
                "text/plain": {},
            },
            "description": "Serialised NetworkX graph in the requested format.",
        },
        404: {"description": "Notebook not found."},
        422: {"description": "Invalid format in request body."},
    },
)
async def export_notebook_networkx(
    notebook_id: str,
    request: NetworkxExportRequest,
    notebook_service: NotebookService = Depends(get_notebook_service),
    export_service: NetworkxExportService = Depends(get_networkx_export_service),
) -> Response:
    """Serialise the notebook's filtered graph in one of 7 NetworkX formats.

    The body shape (``NetworkxExportRequest``) enforces the closed set of
    format literals at the Pydantic layer, so a malformed ``format``
    triggers 422 before this handler runs.

    Filter knobs flow through to the D.0 repository methods, which gate
    on connection count + confidence + entity types. See
    ``shared.models.export.ExportFilter`` for the defaults.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        payload, _report = await export_service.export(notebook_id, request)
    except ValueError as exc:
        # Defence in depth -- Pydantic should reject unknown formats,
        # but the service raises ValueError if the dispatch ever sees
        # something unexpected.
        logger.warning("export-networkx rejected: {}", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    media_type, ext = _FORMAT_TABLE[request.format]
    filename = _safe_filename(notebook_id, ext)

    return Response(
        content=payload,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# ---------------------------------------------------------------------------
# Obsidian export (D.1a)
# ---------------------------------------------------------------------------


@router.post(
    "/export-obsidian",
    responses={
        200: {
            "content": {
                "application/zip": {},
                "application/json": {},
            },
            "description": (
                "Obsidian vault: streamed back as a zip archive "
                "(mode=zip) or written to Settings.vault_path with an "
                "ExportReport returned as JSON (mode=vault_path, D.1b)."
            ),
        },
        400: {
            "description": (
                "vault_path mode requested but Settings.vault_path is not "
                "configured (D.1b)."
            ),
        },
        404: {"description": "Notebook not found."},
        422: {"description": "Invalid request body (e.g. unknown mode)."},
        500: {
            "description": (
                "Filesystem failure during vault_path write. Body carries "
                "entities_written so the client knows where the batch "
                "stopped."
            ),
        },
    },
)
async def export_notebook_obsidian(
    notebook_id: str,
    request: ObsidianExportRequest,
    notebook_service: NotebookService = Depends(get_notebook_service),
    export_service: ObsidianExportService = Depends(get_obsidian_export_service),
) -> Response:
    """Build the notebook's filtered Obsidian vault.

    Two delivery modes:

    * ``mode="zip"`` -- streams the in-memory archive back as
      ``application/zip`` (D.1a).
    * ``mode="vault_path"`` -- writes each file directly to
      ``<Settings.vault_path>/<vault_entities_folder>/`` using
      POSIX atomic-rename per file, then returns the
      :class:`shared.models.export.ExportReport` as JSON (D.1b).

    Filter knobs (``ExportFilter``) flow through the service to the
    D.0 repository methods, then through the D.1 post-filter for
    ``min_connections`` + ``Entity.status``. The synchronous request
    shape matches D.3 / D.2 -- typical export wall-clock is under 10s
    for the V1 filter defaults, so no job-queue indirection on the
    request path (the async ``JobType.EXPORT_OBSIDIAN`` exists for
    auto-pipeline triggers, not user-initiated exports).
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        artifact = await export_service.export(notebook_id, request)
    except VaultPathNotConfigured as exc:
        # Friendly 400 -- the user/operator can fix this by setting
        # vault_path in Settings. We surface the service-side message
        # verbatim because it's already user-readable.
        logger.warning("export-obsidian vault_path rejected: {}", exc)
        raise HTTPException(
            status_code=400,
            detail=str(exc) or "Configure vault_path in Settings first",
        ) from exc
    except ValueError as exc:
        # Defense-in-depth path-traversal / non-absolute / not-writable
        # rejections from _write_to_vault. Surface as 500 with the
        # partial state if available -- a misconfigured vault is an
        # operator problem, not a 400-class user-input problem.
        logger.error("export-obsidian filesystem rejection: {}", exc)
        partial = _extract_partial_state(exc)
        body: Dict[str, Any] = {"error": str(exc)}
        body.update(partial)
        raise HTTPException(status_code=500, detail=body) from exc
    except OSError as exc:
        # Mid-batch filesystem failure (disk full, permission flip
        # mid-write, etc.). Partial state is attached by
        # _write_to_vault via the exception's trailing dict arg.
        logger.error("export-obsidian filesystem failure: {}", exc)
        partial = _extract_partial_state(exc)
        body = {"error": str(exc)}
        body.update(partial)
        raise HTTPException(status_code=500, detail=body) from exc

    if artifact.mode == "vault_path":
        # D.1b: return the report as JSON, no file body. The artifact
        # carries the resolved target dir for debugging but we don't
        # echo it back to the client -- the path is on the server and
        # the client doesn't need it.
        return Response(
            content=artifact.report.model_dump_json(),
            media_type="application/json",
            status_code=200,
        )

    # mode="zip": stream via BytesIO per Q-D-7 -- the zip already lives
    # in memory because zipfile.ZipFile needs random access, but the
    # StreamingResponse wrapper keeps the response object lazy so
    # FastAPI doesn't double-buffer.
    safe_name = _safe_filename(notebook_id, "zip")
    headers = {
        "Content-Disposition": f'attachment; filename="{safe_name}"',
    }
    return StreamingResponse(
        io.BytesIO(artifact.zip_bytes or b""),
        media_type="application/zip",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# JSONL streaming export (D.2)
# ---------------------------------------------------------------------------


@router.post(
    "/export-jsonl",
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": (
                "Streamed zip containing entities.jsonl + relations.jsonl. "
                "Each line is a JSON object shaped for Neo4j apoc.load.json "
                "and LangChain RAG-loader compatibility."
            ),
        },
        404: {"description": "Notebook not found."},
        422: {"description": "Invalid filter shape in request body."},
        500: {"description": "Streaming/serialisation failure mid-export."},
    },
)
async def export_notebook_jsonl(
    notebook_id: str,
    request: JsonlExportRequest,
    notebook_service: NotebookService = Depends(get_notebook_service),
    export_service: JsonlExportService = Depends(get_jsonl_export_service),
) -> StreamingResponse:
    """Stream the notebook's filtered graph as JSONL inside a zip.

    The response body is a zip archive containing two newline-delimited
    JSON files:

    * ``entities.jsonl`` -- one entity per line, shaped as
      ``{id, canonical_name, entity_type, type_tags, primary_type,
      confidence, properties, source_documents, extracted_at}``.
      Embedding is deliberately excluded (Q-D-1) -- it would balloon
      the wire size + leak the vector privacy axis with zero downstream
      value.
    * ``relations.jsonl`` -- one relation per line, shaped as
      ``{id, source_entity, target_entity, relation_type, confidence,
      properties, source_documents}``. Relations whose endpoint failed
      the entity filter are silently dropped (Q-D-4).

    The service streams entity-by-entity into the in-memory zip so peak
    memory stays bounded -- a 5000-entity export holds the compressed
    archive (~10MB) and one Python row at a time, not the raw JSONL
    stream (Q-D-7 build-then-stream).

    Filename uses the same B.2b-derived sanitisation as the rest of the
    router so the ``notebook:abc`` ID survives ``Content-Disposition``.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    # ``ExportFilter`` Pydantic validation runs at the request boundary,
    # so malformed filters are 422'd by FastAPI before this handler
    # runs. ``stream_jsonl`` returns an async generator -- the actual
    # work happens during iteration inside StreamingResponse, so we
    # don't wrap the call in try/except here (errors raised inside the
    # generator propagate through Starlette and abort the response
    # naturally, with the failure path metric still firing in the
    # service's finally block).
    generator = export_service.stream_jsonl(notebook_id, request)

    safe_name = _safe_filename(notebook_id, "jsonl.zip")
    headers = {
        "Content-Disposition": f'attachment; filename="{safe_name}"',
    }
    return StreamingResponse(
        generator,
        media_type="application/zip",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# OKF (Open Knowledge Format) bundle export (OKF.2)
# ---------------------------------------------------------------------------


@router.post(
    "/export/okf",
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": (
                "Zipped OKF v0.1 Knowledge Bundle (markdown concepts + "
                "index.md). The ExportReport (incl. the omitted-field ledger) "
                "is surfaced in the ``X-OKF-Export-Report`` response header."
            ),
        },
        202: {
            "content": {"application/json": {}},
            "description": (
                "Large notebook: the export was deferred to the job queue. "
                "Body carries ``{job_id, status, entity_count}``; poll the "
                "job-status surface for completion."
            ),
        },
        404: {"description": "Notebook not found."},
        422: {"description": "Invalid filter shape in request body."},
    },
)
async def export_notebook_okf(
    notebook_id: str,
    request: ObsidianExportRequest,
    notebook_service: NotebookService = Depends(get_notebook_service),
    export_service: OkfExportService = Depends(get_okf_export_service),
) -> Response:
    """Export the notebook's curated KG projection as an OKF v0.1 bundle.

    Reuses :class:`ObsidianExportRequest` for the ``filter`` knob set (the
    OKF exporter shares the Track-D filter contract and only ever produces a
    zip). Normal notebooks build + zip inline and stream the archive back;
    notebooks whose surviving-entity count exceeds
    :data:`OKF_ASYNC_ENTITY_THRESHOLD` are deferred to the job queue
    (``JobType.EXPORT_OKF``) and return ``202`` with a pollable job id.

    Auth is the notebook-scoped lookup shared by every export route (G-Q1);
    no new auth surface is introduced.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    # Cheap size probe (projected, embedding-free query) to decide inline vs
    # deferred. The projection reused here is exactly what the export applies,
    # so the threshold reflects the real bundle size.
    entity_count = await export_service.count_entities(notebook_id, request.filter)
    if entity_count > OKF_ASYNC_ENTITY_THRESHOLD:
        job_id = await CommandService.submit_command_job(
            module_name="open_notebook.commands",
            command_name="export_okf",
            command_args={
                "notebook_id": notebook_id,
                "filter": request.filter.model_dump(mode="json"),
            },
        )
        logger.info(
            "okf export deferred: notebook={nb} entities={n} job={job}",
            nb=notebook_id,
            n=entity_count,
            job=job_id,
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "queued",
                "entity_count": entity_count,
                "threshold": OKF_ASYNC_ENTITY_THRESHOLD,
            },
        )

    artifact = await export_service.export(notebook_id, request)

    # Surface the ExportReport (incl. the OKF-D3 omitted-field ledger) in a
    # response header so the zip body stays a clean archive download.
    # ``model_dump_json`` is single-line compact JSON — safe as a header value.
    report_json = artifact.report.model_dump_json()
    safe_name = _safe_filename(notebook_id, "okf.zip")
    return StreamingResponse(
        io.BytesIO(artifact.zip_bytes or b""),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
            "X-OKF-Export-Report": report_json,
        },
    )


# ---------------------------------------------------------------------------
# OKF (Open Knowledge Format) bundle import (OKF.4)
# ---------------------------------------------------------------------------


@router.post(
    "/import/okf",
    responses={
        200: {
            "content": {"application/json": {}},
            "description": (
                "Import complete. Body is the ``OkfImportReport`` "
                "(imported/matched/skipped counts + the non-silent "
                "skip/dangling ledger)."
            ),
        },
        404: {"description": "Notebook not found."},
        422: {
            "description": (
                "Malformed bundle container (not a .zip / unreadable "
                "archive). A malformed *concept* inside a valid bundle is "
                "skipped non-silently and reported in the 200 body, never "
                "a 422."
            ),
        },
    },
)
async def import_notebook_okf(
    notebook_id: str,
    file: UploadFile = File(
        ..., description="An OKF v0.1 Knowledge Bundle as a .zip archive."
    ),
    apply_dedup: bool = Query(
        True,
        description=(
            "Run the K.5 propose + K.3 auto-merge dedup pass after "
            "persistence so an imported entity is matched against the "
            "existing graph, not duplicated. Idempotent."
        ),
    ),
    notebook_service: NotebookService = Depends(get_notebook_service),
    import_service: OkfImportService = Depends(get_okf_import_service),
) -> JSONResponse:
    """Import an uploaded OKF v0.1 bundle into this notebook (the OKF.3 seam).

    The inverse of ``POST /export/okf`` and the REST surface OKF.3 deferred to
    this phase. The multipart-uploaded zip is handed to
    :meth:`OkfImportService.import_bundle` (notebook-scoped), which parses the
    concepts, upserts entities via the deterministic ``(name, type)`` dedup +
    the injection-safe RELATE primitive, and creates notes/sources
    idempotently — all provenance-tagged ``okf-import``.

    Auth is the same notebook-scoped lookup every export route uses (G-Q1); no
    new auth surface. A malformed *bundle container* (non-zip / unreadable
    archive) is a 422; a malformed *concept* within a valid bundle is skipped
    non-silently and surfaced in the report — never a half-written graph.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    data = await file.read()

    try:
        report = await import_service.import_bundle(
            data, notebook_id=notebook_id, apply_dedup=apply_dedup
        )
    except (ValueError, FileNotFoundError) as exc:
        # Malformed bundle CONTAINER (not a zip, unreadable archive). A bad
        # concept never reaches here — it is skip-recorded inside the service.
        logger.warning("okf import rejected bundle for {nb}: {e}", nb=notebook_id, e=exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info(
        "okf import: notebook={nb} entities +{ec}/~{em} relations +{rc} "
        "notes +{nc}/~{nm} sources +{sc}/~{sm} skipped={sk}",
        nb=notebook_id,
        ec=report.entities_created,
        em=report.entities_matched,
        rc=report.relations_created,
        nc=report.notes_created,
        nm=report.notes_matched,
        sc=report.sources_created,
        sm=report.sources_matched,
        sk=len(report.skipped),
    )
    return JSONResponse(content=report.to_dict())


# ---------------------------------------------------------------------------
# Preview-counts endpoint (D.1c)
# ---------------------------------------------------------------------------


class ExportPreviewCounts(BaseModel):
    """Minimal response shape for the D.1c preview endpoint.

    Counts only -- entity/relation IDs MUST NOT leak through this
    surface so the dialog can poll on every slider change without
    exposing per-notebook PII to the request log.
    """

    entity_count: int = Field(
        ge=0,
        description="Number of entities that would survive the filter.",
    )
    relation_count: int = Field(
        ge=0,
        description=(
            "Number of relations that would survive the filter "
            "(both endpoints retained after the min_connections "
            "post-filter)."
        ),
    )


@router.get(
    "/export-preview",
    response_model=ExportPreviewCounts,
    responses={
        200: {"description": "Filtered entity + relation counts."},
        400: {"description": "Filter knob out of range."},
        404: {"description": "Notebook not found."},
    },
)
async def export_preview_counts(
    notebook_id: str,
    min_connections: int = Query(
        5, ge=0, description="Minimum entity degree (matches ExportFilter)."
    ),
    min_confidence: float = Query(
        0.9,
        description="Minimum entity confidence in [0.0, 1.0].",
    ),
    min_relation_confidence: float = Query(
        0.9,
        description="Minimum relation confidence in [0.0, 1.0].",
    ),
    entity_types: Optional[List[str]] = Query(
        None,
        description=(
            "Repeatable; allow-list of entity types. Omit for all types."
        ),
    ),
    include_orphans: bool = Query(False),
    include_archived: bool = Query(False),
    notebook_service: NotebookService = Depends(get_notebook_service),
    entity_repo: EntityRepository = Depends(get_entity_repo),
) -> ExportPreviewCounts:
    """Return surviving entity + relation counts for the given filter.

    Powers the D.1c dialog's debounced ``ExportPreviewCounts`` widget so
    the operator sees what the export would contain BEFORE clicking
    "Export". The endpoint mirrors the filter pipeline that
    ``ObsidianExportService._collect`` applies, in order:

    1. SurrealQL gate via ``ExportFilter`` (notebook scope, confidence,
       orphan_status, entity_types).
    2. ``status`` post-filter (drop ``archived`` + ``merged``) -- shared
       via ``EXCLUDED_ENTITY_STATUSES``.
    3. ``min_connections`` post-filter -- delegated to
       ``ObsidianExportService._apply_min_connections_filter`` so there
       is exactly one source of truth for degree counting.
    4. Q-D-4 silent-drop of relations whose endpoints didn't survive.

    Performance: counts only -- never serialises entities through the
    response. For a 1000-entity notebook the cost is two SurrealQL
    SELECTs (one per repo method, both already projected by D.0 to skip
    the 768-float ``embedding`` column) plus an O(n+m) Python pass over
    the in-memory entity + relation lists. The dialog debounces at
    300ms so even rapid slider drags don't hammer the backend.

    Args:
        notebook_id: Path parameter; looked up via NotebookService.
        min_connections: ExportFilter.min_connections analogue.
        min_confidence: ExportFilter.min_confidence analogue. Pydantic
            does NOT enforce the bound here because Query() decoupling
            from the body model means the bound has to be validated
            manually -- see the 400 branch below.
        min_relation_confidence: ExportFilter.min_relation_confidence
            analogue. Same manual bound check as min_confidence.
        entity_types: ExportFilter.entity_types analogue; FastAPI
            represents repeatable query params as a list.
        include_orphans / include_archived: ExportFilter analogues.
        notebook_service: Notebook lookup (404 path).
        entity_repo: D.0 list_entities_for_notebook +
            list_relations_for_notebook.

    Returns:
        ExportPreviewCounts: ``{entity_count, relation_count}``.
    """
    # Manual bound checks -- Query() doesn't auto-apply Pydantic Field
    # constraints from ExportFilter, so we validate here to match the
    # ExportFilter contract (Pydantic raises on the POST path).
    if not 0.0 <= min_confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="min_confidence must be in [0.0, 1.0]",
        )
    if not 0.0 <= min_relation_confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="min_relation_confidence must be in [0.0, 1.0]",
        )

    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    # Build the same ExportFilter the POST endpoint would receive so
    # the SurrealQL path is identical -- prevents preview/actual drift.
    export_filter = ExportFilter(
        min_connections=min_connections,
        min_confidence=min_confidence,
        min_relation_confidence=min_relation_confidence,
        entity_types=entity_types,
        include_orphans=include_orphans,
        include_archived=include_archived,
    )

    entities = await entity_repo.list_entities_for_notebook(
        notebook_id, export_filter
    )
    relations = await entity_repo.list_relations_for_notebook(
        notebook_id, export_filter
    )

    # Status post-filter MUST run BEFORE min_connections -- the service
    # applies them in the same order (see
    # ``ObsidianExportService._collect``). If the preview counted
    # archived/merged entities, the user would see "42 entities" but the
    # actual export would deliver fewer, breaking the dialog's contract.
    #
    # ``status`` defaults to "active" in the Entity model; the
    # ``or "active"`` keeps us safe against legacy rows where the field
    # is null. The set is the public symbol from the service module so
    # there's only one source of truth for "what's excluded".
    entities = [
        e
        for e in entities
        if (e.status or "active") not in EXCLUDED_ENTITY_STATUSES
    ]

    # min_connections post-filter -- delegate to the service's
    # @staticmethod so any future tuning of the degree algorithm lands
    # on both code paths simultaneously (Q-D-1c parity). The method has
    # no instance state, hence the un-instantiated call.
    surviving_entities = ObsidianExportService._apply_min_connections_filter(
        entities, relations, min_connections
    )

    # Drop relations whose endpoints didn't survive the post-filter --
    # otherwise the preview's relation count would overstate what the
    # exporter will emit (Q-D-4 silent-drop). Build the surviving-id
    # set once, then membership-test both ends.
    surviving_ids = {str(e.id) for e in surviving_entities if e.id}
    surviving_relation_count = 0
    for relation in relations:
        src = str(relation.in_entity) if relation.in_entity else None
        dst = str(relation.out_entity) if relation.out_entity else None
        if src and dst and src in surviving_ids and dst in surviving_ids:
            surviving_relation_count += 1

    return ExportPreviewCounts(
        entity_count=len(surviving_entities),
        relation_count=surviving_relation_count,
    )


def _extract_partial_state(exc: Exception) -> Dict[str, Any]:
    """Pull the partial-state dict that ``_write_to_vault`` attaches.

    ``_write_to_vault`` raises with ``(message, {"entities_written":
    N})`` as ``exc.args`` so the router can surface "how many files
    landed before we stopped". This helper hunts for that dict in
    ``exc.args`` and returns an empty dict if none is attached
    (path-traversal-on-config errors don't carry partial state because
    no writes happened).
    """
    for arg in getattr(exc, "args", ()):
        if isinstance(arg, dict) and "entities_written" in arg:
            return {
                "entities_written": int(arg["entities_written"]),
                **{k: v for k, v in arg.items() if k != "entities_written"},
            }
    return {}


__all__ = ["router"]
