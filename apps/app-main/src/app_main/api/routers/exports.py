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
from typing import Dict, Tuple

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from app_main.dependencies import (
    get_networkx_export_service,
    get_notebook_service,
    get_obsidian_export_service,
)
from app_main.services.networkx_export_service import NetworkxExportService
from app_main.services.notebook_service import NotebookService
from app_main.services.obsidian_export_service import ObsidianExportService
from shared.models.export import NetworkxExportRequest, ObsidianExportRequest


router = APIRouter(prefix="/notebooks/{notebook_id}", tags=["exports"])


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
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": "Obsidian vault streamed back as a zip archive.",
        },
        404: {"description": "Notebook not found."},
        422: {"description": "Invalid request body (e.g. unknown mode)."},
        501: {"description": "Mode not yet implemented (e.g. vault_path before D.1b)."},
    },
)
async def export_notebook_obsidian(
    notebook_id: str,
    request: ObsidianExportRequest,
    notebook_service: NotebookService = Depends(get_notebook_service),
    export_service: ObsidianExportService = Depends(get_obsidian_export_service),
) -> StreamingResponse:
    """Stream the notebook's filtered Obsidian vault back as a zip.

    Only ``mode="zip"`` is implemented in D.1a; ``mode="vault_path"``
    lands in D.1b and returns 501 here. The synchronous request shape
    matches D.3 / D.2 -- typical export wall-clock is under 10s for the
    V1 filter defaults, so no job-queue indirection.

    Filter knobs (``ExportFilter``) flow through the service to the
    D.0 repository methods, then through the D.1a post-filter for
    ``min_connections`` + ``Entity.status``.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    if request.mode == "vault_path":
        # D.1b implements the disk-write side. Surface as 501 (Not
        # Implemented) so a confused FE can show a clear error rather
        # than a generic 500.
        raise HTTPException(
            status_code=501,
            detail=(
                "Obsidian vault-path mode is not implemented yet "
                "(arrives in D.1b). Use mode='zip' for D.1a."
            ),
        )

    try:
        artifact = await export_service.export(notebook_id, request)
    except NotImplementedError as exc:
        # Defence in depth -- the mode check above should cover this,
        # but the service raises NotImplementedError too so we'd rather
        # convert it to 501 than leak as 500.
        logger.warning("export-obsidian rejected: {}", exc)
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    # Stream via BytesIO per Q-D-7 -- the zip already lives in memory
    # because zipfile.ZipFile needs random access, but the
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


__all__ = ["router"]
