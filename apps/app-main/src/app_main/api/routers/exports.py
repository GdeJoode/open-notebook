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

import re
from typing import Dict, Tuple

from fastapi import APIRouter, Depends, HTTPException, Response
from loguru import logger

from app_main.dependencies import (
    get_networkx_export_service,
    get_notebook_service,
)
from app_main.services.networkx_export_service import NetworkxExportService
from app_main.services.notebook_service import NotebookService
from shared.models.export import NetworkxExportRequest


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


__all__ = ["router"]
