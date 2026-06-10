"""Notebook-event read/acknowledge endpoints (Phase B.3c).

Surfaces the ``notebook_event`` rows emitted by the B.1e Pass-1
orchestrator (``extension_suggested`` for coverage 80-95%,
``schema_mismatch`` for coverage <80%) so the frontend
``SchemaSoftNudge`` banner can react to them.

Two endpoints:

* ``GET  /api/notebooks/{id}/events`` — list recent unread events,
  optionally filtered by type. The frontend polls this every 30s and
  shows a banner when the response is non-empty AND the notebook's
  ``soft_nudge_dismissed`` flag is false (the dismiss flag is fetched
  from the schema endpoint, not duplicated here).
* ``POST /api/notebooks/{id}/events/{event_id}/mark_read`` — set
  ``read_at`` on a single event so it stops appearing in the list.

Cross-track dependency
----------------------

The ``notebook_event`` table + :class:`NotebookEventRepository` are
created by **B.3b** (parallel branch ``track/b-schema-edit-ops``,
migration 46). This router imports the repo locally inside each
handler so that:

* Tests can patch the import path with ``unittest.mock`` and verify
  endpoint behaviour without the real repo existing on the branch.
* When B.3b lands on main first, no code change is needed here — the
  import resolves at runtime.

The repo contract this router assumes (documented for B.3b)::

    class NotebookEventRepository:
        async def list_for_notebook(
            self,
            notebook_id: str,
            event_types: Optional[List[str]] = None,
            unread_only: bool = False,
            limit: int = 20,
        ) -> List[NotebookEvent]: ...

        async def mark_read(self, event_id: str) -> bool: ...

If B.3b ships a different surface, adapt the calls here in a small
follow-up PR — the wire contract (JSON shape returned to the frontend)
is decoupled from the repo signature.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from app_main.dependencies import get_notebook_service
from app_main.services.notebook_service import NotebookService


router = APIRouter(prefix="/notebooks/{notebook_id}", tags=["notebook-events"])


# Event types the soft-nudge UI knows how to render. The frontend always
# requests one or both of these; the backend tolerates "unknown" types
# (it just returns whatever exists in the table) so future event types
# don't require coordinated deploys.
_KNOWN_NUDGE_EVENT_TYPES = frozenset({"extension_suggested", "schema_mismatch"})


class NotebookEventView(BaseModel):
    """Trimmed projection of a ``notebook_event`` row.

    The DB row carries more fields (``source_id``, ``payload`` dict,
    etc.); the UI banner only needs id + type + a human message + the
    timestamp for "X minutes ago" rendering.
    """

    id: str
    event_type: str = Field(
        description=(
            "One of ``extension_suggested`` (coverage 80-95%) or "
            "``schema_mismatch`` (coverage <80%) when surfaced by the "
            "soft-nudge UI. Other event types pass through unchanged."
        )
    )
    message: Optional[str] = None
    source_id: Optional[str] = None
    created_at: Optional[str] = None
    read_at: Optional[str] = None


class MarkReadResponse(BaseModel):
    """Echo response for ``POST .../mark_read``."""

    event_id: str
    success: bool


def _project_event(row: object) -> NotebookEventView:
    """Coerce a ``NotebookEvent`` DB row into the view-model.

    Defensive against missing attributes so a stale DB schema (e.g.
    B.3b ships an extra field before this router knows about it) doesn't
    500 the soft-nudge poll.
    """
    def _attr_or_get(obj: object, name: str) -> object:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name)
        return None

    event_id = _attr_or_get(row, "id")
    event_type = _attr_or_get(row, "event_type") or _attr_or_get(row, "type") or ""
    message = _attr_or_get(row, "message")
    source_id = _attr_or_get(row, "source_id") or _attr_or_get(row, "source")
    created_at = _attr_or_get(row, "created_at") or _attr_or_get(row, "created")
    read_at = _attr_or_get(row, "read_at")

    # Coerce datetime → isoformat string defensively.
    def _maybe_iso(v: object) -> Optional[str]:
        if v is None:
            return None
        if hasattr(v, "isoformat"):
            return v.isoformat()  # type: ignore[no-any-return]
        return str(v)

    return NotebookEventView(
        id=str(event_id) if event_id else "",
        event_type=str(event_type),
        message=str(message) if message else None,
        source_id=str(source_id) if source_id else None,
        created_at=_maybe_iso(created_at),
        read_at=_maybe_iso(read_at),
    )


@router.get(
    "/events",
    response_model=List[NotebookEventView],
    responses={404: {"description": "Notebook not found."}},
)
async def list_notebook_events(
    notebook_id: str,
    type: Optional[str] = Query(
        default=None,
        description=(
            "Comma-separated event types to filter on (e.g. "
            "``extension_suggested,schema_mismatch``). Omit to receive "
            "all event types."
        ),
    ),
    unread: bool = Query(
        default=False,
        description="If true, only events with ``read_at IS NULL`` are returned.",
    ),
    limit: int = Query(default=20, ge=1, le=100),
    notebook_service: NotebookService = Depends(get_notebook_service),
) -> List[NotebookEventView]:
    """List notebook events for the soft-nudge banner (B.3c).

    Returns newest-first. Default ``unread=false`` so callers can fetch
    history for an admin view; the soft-nudge poll always passes
    ``unread=true``.

    When the notebook exists but the ``NotebookEventRepository`` import
    fails (B.3b hasn't landed yet on this branch), we return an empty
    list with a warning rather than a 500. That way the frontend's
    polling loop stays quiet until the table exists.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    event_types: Optional[List[str]] = None
    if type:
        event_types = [t.strip() for t in type.split(",") if t.strip()]
        # Surface unknown types in the logs so we notice if the
        # frontend starts asking for something we don't yet emit.
        unknown = [t for t in event_types if t not in _KNOWN_NUDGE_EVENT_TYPES]
        if unknown:
            logger.debug(
                "Client asked for unknown nudge event types: {!r}", unknown
            )

    try:
        # Local import so the absence of the B.3b repo doesn't break
        # module load — see module docstring for rationale.
        from surrealdb_service.repositories.notebook_event import (  # type: ignore[import-not-found]
            NotebookEventRepository,
        )
    except ImportError:
        logger.warning(
            "NotebookEventRepository unavailable (B.3b not merged?); "
            "returning empty event list for notebook {}",
            notebook_id,
        )
        return []

    repo = NotebookEventRepository()
    try:
        rows = await repo.list_for_notebook(
            notebook_id,
            event_types=event_types,
            unread_only=unread,
            limit=limit,
        )
    except Exception as e:
        logger.error(
            "Failed to list notebook events for {}: {}", notebook_id, e
        )
        # Same rationale as above — don't break the polling loop. A 500
        # would cause the React Query retry loop to back off and the
        # banner to never appear even after the underlying problem
        # clears.
        return []

    return [_project_event(r) for r in rows]


@router.post(
    "/events/{event_id}/mark_read",
    response_model=MarkReadResponse,
    responses={
        404: {"description": "Notebook or event not found."},
    },
)
async def mark_event_read(
    notebook_id: str,
    event_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
) -> MarkReadResponse:
    """Mark a single notebook event as read (B.3c).

    Idempotent — re-marking an already-read event still returns
    ``success: true`` so the frontend retry path is safe.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        from surrealdb_service.repositories.notebook_event import (  # type: ignore[import-not-found]
            NotebookEventRepository,
        )
    except ImportError:
        logger.warning(
            "NotebookEventRepository unavailable (B.3b not merged?); "
            "treating mark_read for event {} as no-op",
            event_id,
        )
        # Best-effort: return success so the client UI doesn't loop
        # retrying. When B.3b lands, the second click would persist.
        return MarkReadResponse(event_id=event_id, success=True)

    repo = NotebookEventRepository()
    try:
        ok = await repo.mark_read(event_id)
    except Exception as e:
        logger.error("Failed to mark event {} read: {}", event_id, e)
        raise HTTPException(
            status_code=500, detail="Failed to mark event read"
        )

    return MarkReadResponse(event_id=event_id, success=bool(ok))
