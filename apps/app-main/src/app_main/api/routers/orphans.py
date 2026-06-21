"""Orphan-prune dashboard router (Phase B.5b).

Exposes the per-notebook orphans-dashboard endpoints:

* ``GET /api/notebooks/{notebook_id}/orphans`` returns the pending +
  archived entity rows plus their counts. Drives the OrphansDashboard
  table in the Schema tab.

* ``POST /api/notebooks/{notebook_id}/orphans/{entity_id}/reconnect``
  triggers a one-shot manual retry of B.5a's orphan-connector for a
  single entity. Returns the per-orphan retry outcome so the frontend
  can update the row in place.

Design notes
------------

* **Read endpoint is keyed by notebook**, not by source. The dashboard
  presents the user's per-notebook view ("how many orphans are still
  pending for THIS notebook?"). The cross-source case is implicit:
  every source the user imported into the notebook contributes.

* **Manual reconnect runs synchronously.** B.5b plan acceptance #5
  reads "Manual [Reconnect] action queues a job that retries that
  specific orphan". We honour that contract by running the
  ``orphan_connector.run`` invocation inline because (a) the work
  scope is bounded (one orphan, at most ``max_proposals_per_orphan``
  LLM calls) and (b) the response payload is what the UI needs to
  refresh. A background job worker would buy nothing here. Future
  scale-out can swap this for a job-queue submission without changing
  the API.

* **Pagination is deliberately omitted** in the first cut. Orphan
  counts in production notebooks are expected to be small (a few
  dozen at most); the dashboard renders the whole list comfortably.
  When sizes routinely exceed ~200 rows we'll add ``?limit/offset``
  on the GET endpoint -- the response shape already carries the total
  count fields needed for the pager.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app_main.dependencies import (
    get_entity_repo,
    get_notebook_service,
    get_source_repo,
)
from app_main.services.notebook_service import NotebookService

from entity_filtering.resolution.orphan_prune import (
    STATUS_ARCHIVED,
    STATUS_PENDING_RECONNECT,
    retry_pending_reconnects,
)
from surrealdb_service.repositories import EntityRepository, SourceRepository


router = APIRouter(prefix="/notebooks/{notebook_id}", tags=["orphans"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class OrphanRow(BaseModel):
    """View-model row for one orphan entity in the dashboard.

    Normalises the repository's raw dict surface so the frontend gets a
    predictable shape regardless of how the DB row evolves. Missing
    fields fall back to safe defaults so the table renders even on
    legacy rows.
    """

    id: str
    canonical_name: str
    entity_type: Optional[str] = None
    orphan_status: str
    reconnect_attempts: int = 0
    first_orphaned_at: Optional[str] = None
    last_reconnect_attempt_at: Optional[str] = None


class OrphansResponse(BaseModel):
    """JSON contract for ``GET /api/notebooks/{id}/orphans``.

    The frontend renders the counts in the section header and the
    ``items`` list in the table body. The ``items`` list contains both
    pending and archived rows so the user can toggle between tabs
    client-side without an additional round-trip.
    """

    notebook_id: str
    pending_count: int = 0
    archived_count: int = 0
    items: List[OrphanRow] = Field(default_factory=list)


class ReconnectResponse(BaseModel):
    """Result of a single-orphan manual reconnect.

    The frontend uses ``new_status`` to flip the row in place: when the
    reconnect succeeds, the row clears (``new_status = None``); when it
    fails, the row stays in the same state and ``attempted = 1,
    reconnected = 0``.
    """

    entity_id: str
    attempted: int
    reconnected: int
    new_status: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_row(row: Dict[str, Any]) -> OrphanRow:
    """Coerce a raw entity dict to an :class:`OrphanRow`."""
    first = row.get("first_orphaned_at")
    last = row.get("last_reconnect_attempt_at")
    return OrphanRow(
        id=str(row.get("id") or ""),
        canonical_name=str(row.get("canonical_name") or ""),
        entity_type=row.get("entity_type"),
        orphan_status=str(row.get("orphan_status") or ""),
        reconnect_attempts=int(row.get("reconnect_attempts") or 0),
        first_orphaned_at=first.isoformat() if hasattr(first, "isoformat") else (
            str(first) if first else None
        ),
        last_reconnect_attempt_at=last.isoformat() if hasattr(last, "isoformat") else (
            str(last) if last else None
        ),
    )


async def _ensure_notebook_exists(
    notebook_id: str, notebook_service: NotebookService
) -> None:
    """Common 404 guard -- every handler runs this first."""
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")


# ---------------------------------------------------------------------------
# GET /orphans
# ---------------------------------------------------------------------------


@router.get(
    "/orphans",
    response_model=OrphansResponse,
    responses={404: {"description": "Notebook not found."}},
)
async def get_notebook_orphans(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    entity_repo: EntityRepository = Depends(get_entity_repo),
) -> OrphansResponse:
    """Return pending + archived orphan entities for a notebook.

    Returns 404 when the notebook does not exist. Returns 200 with
    empty lists when the notebook has no orphans (the dashboard's
    empty state).
    """
    await _ensure_notebook_exists(notebook_id, notebook_service)

    pending_rows = await entity_repo.list_orphans_with_status(
        notebook_id, STATUS_PENDING_RECONNECT
    )
    archived_rows = await entity_repo.list_orphans_with_status(
        notebook_id, STATUS_ARCHIVED
    )

    items = [
        _normalise_row(row)
        for row in (pending_rows + archived_rows)
    ]

    return OrphansResponse(
        notebook_id=notebook_id,
        pending_count=len(pending_rows),
        archived_count=len(archived_rows),
        items=items,
    )


# ---------------------------------------------------------------------------
# POST /orphans/{entity_id}/reconnect
# ---------------------------------------------------------------------------


@router.post(
    "/orphans/{entity_id}/reconnect",
    response_model=ReconnectResponse,
    responses={
        404: {"description": "Notebook or entity not found."},
        409: {"description": "Entity has no source documents to retry from."},
    },
)
async def manual_reconnect_orphan(
    notebook_id: str,
    entity_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    entity_repo: EntityRepository = Depends(get_entity_repo),
    source_repo: SourceRepository = Depends(get_source_repo),
) -> ReconnectResponse:
    """Trigger a one-shot retry of B.5a for a single orphan.

    Steps:

    1. Load the entity to verify it exists and is in a retryable state
       (``pending_reconnect``). Already-archived entities return 409.
    2. Fetch the chunks of its first source (the source whose import
       originally produced the orphan).
    3. Call :func:`retry_pending_reconnects` with a scoped one-entity
       notebook view (the repo's notebook-id filter is honoured by
       the underlying query, so we don't need to scope manually).
    4. Return the per-orphan retry outcome.

    The manual reconnect intentionally re-uses the same
    ``retry_pending_reconnects`` code path the service runs after every
    extraction so the UI behaviour matches the auto-retry behaviour.
    """
    await _ensure_notebook_exists(notebook_id, notebook_service)

    entity = await entity_repo.get_entity_detail(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    status = entity.get("orphan_status")
    if status != STATUS_PENDING_RECONNECT:
        # Archived entities are intentionally not retried -- the user
        # would need to first un-archive them via a future admin path.
        # Healthy entities don't need retrying at all.
        raise HTTPException(
            status_code=409,
            detail=(
                f"Entity is not in pending_reconnect (current status: "
                f"{status or 'none'}); cannot reconnect."
            ),
        )

    source_docs = entity.get("source_documents") or []
    if not source_docs:
        raise HTTPException(
            status_code=409,
            detail="Entity has no source documents; cannot retry.",
        )

    source_id = str(source_docs[0])
    # The chunks come from the original source -- B.5a uses them to
    # propose new partner relationships.
    try:
        chunks = await source_repo.get_chunks(source_id)
    except Exception as e:
        logger.warning(
            f"manual_reconnect_orphan: failed to fetch chunks for "
            f"source {source_id}: {e}"
        )
        chunks = []

    # Convert to the dict-shape orphan_connector expects.
    chunk_dicts: List[Dict[str, Any]] = []
    for c in chunks or []:
        if not getattr(c, "text", None):
            continue
        chunk_dicts.append({"text": c.text, "id": str(c.id), "entities": []})

    # Wire the LLM caller -- match the service's pattern. Failure to
    # build a caller is non-fatal; the retry simply records the attempt
    # without confirming a relation.
    llm_caller = None
    try:
        from app_main.services.entity_extraction_service import (
            make_default_llm_caller,
        )

        # Same KG orphan-reconnect routine as the in-service B.5b retry —
        # use the extraction model so auto- and user-triggered runs match.
        llm_caller = await make_default_llm_caller(
            default_field="default_extraction_model"
        )
    except Exception as e:
        logger.warning(
            f"manual_reconnect_orphan: failed to wire LLM caller ({e}); "
            "retry will record an attempt but cannot confirm."
        )

    # Attempt-2 M4 fix: pass entity_id_filter so the retry runs ONLY
    # for the clicked orphan rather than fanning out across every
    # pending orphan in the notebook (cost amplification + side-effect
    # surprise per AC #5).
    outcome = await retry_pending_reconnects(
        notebook_id,
        entity_repo,
        chunk_dicts,
        llm_caller=llm_caller,
        entity_id_filter=entity_id,
    )

    # Re-read the orphan after the retry to surface its current status.
    # The outcome counts are now scoped to the filter (1 attempted at
    # most), but we still re-read so the UI can refresh the row from
    # the authoritative DB shape rather than reconstructing it from
    # the outcome.
    updated = await entity_repo.get_entity_detail(entity_id)
    new_status = updated.get("orphan_status") if updated else status
    reconnected = 1 if new_status is None else 0

    return ReconnectResponse(
        entity_id=entity_id,
        attempted=1 if outcome.attempted else 0,
        reconnected=reconnected,
        new_status=new_status,
    )
