"""Triage router — review queue + operator decisions + backbone warnings (Q.4).

Mirrors ``routers/entity_resolution.py`` (the K.5 candidate-queue precedent) in
shape and dependency-injection. Three endpoints:

* ``GET  /api/triage/queue`` — the B5 review queue read-model (open rows by
  default; ``?all=true`` includes decided rows).
* ``POST /api/triage/queue/{id}/decision`` — an operator's decision: pin the
  entity's status by hand (``manual_override``) and close the queue row. Because
  the pin sets ``manual_override = true``, a subsequent pipeline run does NOT
  overwrite it (Q.4 AC7 — override-wins).
* ``GET  /api/triage/backbone-warnings`` — recompute (read-time) the weak-edge
  warnings for a supplied set of entity ids.

These are thin HTTP shells over the Q.3/Q.4 services; no business logic lives in
the router.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from surrealdb_service.repositories import EntityRepository

from app_main.dependencies import get_entity_repo, get_triage_queue_repo
from app_main.services.triage.backbone_check import BackboneCheckService
from app_main.services.triage.signals_service import TriageSignalsService
from app_main.services.triage.status_assignment_service import (
    STATUS_ACTIVE,
    STATUS_REFERENCE,
)
from app_main.services.triage.triage_queue import (
    DECISION_OPEN,
    QueueItem,
    TriageQueueRepository,
)

router = APIRouter(prefix="/triage", tags=["triage"])


# --- Response / request schemas -----------------------------------------------


class QueueItemOut(BaseModel):
    """One triage-queue row (the B5 read-model, AC4 fields)."""

    id: str
    entity: str
    name: str
    type: str
    structural_degree: int
    doc_count: int
    reason: str
    decision: str
    batch_id: str

    @classmethod
    def from_item(cls, item: QueueItem) -> "QueueItemOut":
        return cls(
            id=item.id,
            entity=item.entity,
            name=item.name,
            type=item.type,
            structural_degree=item.structural_degree,
            doc_count=item.doc_count,
            reason=item.reason,
            decision=item.decision,
            batch_id=item.batch_id,
        )


class QueueResponse(BaseModel):
    count: int
    items: List[QueueItemOut]


class DecisionIn(BaseModel):
    """An operator decision on a queued entity.

    ``status`` is the status to pin (``active`` to promote, ``reference`` to keep
    out of the active KG). ``decision`` is a free label recorded on the queue row
    (defaults to the status). The decision ALWAYS sets ``manual_override`` so
    automation never re-derives the entity's status afterwards.
    """

    status: str = Field(
        ..., description="Status to pin: 'active' or 'reference'."
    )
    decision: Optional[str] = Field(
        default=None,
        description="Decision label for the queue row; defaults to the status.",
    )


class DecisionResponse(BaseModel):
    queue_id: str
    entity: str
    status: str
    manual_override: bool
    decision: str


class WeakEdgeOut(BaseModel):
    other: str
    relation_type: str
    doc_count: int


class BackboneWarningOut(BaseModel):
    entity_id: str
    structural_degree: int
    weak_edges: List[WeakEdgeOut]


class BackboneWarningsResponse(BaseModel):
    count: int
    warnings: List[BackboneWarningOut]


# --- Endpoints ----------------------------------------------------------------


@router.get("/queue", response_model=QueueResponse)
async def get_queue(
    include_all: bool = Query(
        default=False, alias="all", description="Include decided rows too."
    ),
    queue: TriageQueueRepository = Depends(get_triage_queue_repo),
) -> QueueResponse:
    """Return the triage review queue (open rows by default)."""
    items = await (queue.list_all() if include_all else queue.list_open())
    return QueueResponse(
        count=len(items), items=[QueueItemOut.from_item(i) for i in items]
    )


@router.post("/queue/{queue_id:path}/decision", response_model=DecisionResponse)
async def decide(
    queue_id: str,
    body: DecisionIn,
    queue: TriageQueueRepository = Depends(get_triage_queue_repo),
    entity_repo: EntityRepository = Depends(get_entity_repo),
) -> DecisionResponse:
    """Apply an operator decision: pin the entity status + close the queue row.

    Sets ``manual_override = true`` on the entity (the load-bearing override-wins
    contract) so a later :meth:`TriagePipeline.run` leaves the operator's status
    untouched. The queue row's ``decision`` moves off ``open`` so automation no
    longer refreshes it.
    """
    if body.status not in (STATUS_ACTIVE, STATUS_REFERENCE):
        raise HTTPException(
            status_code=422,
            detail=f"status must be '{STATUS_ACTIVE}' or '{STATUS_REFERENCE}'",
        )

    item = await queue.get(queue_id)
    if item is None:
        raise HTTPException(status_code=404, detail="queue row not found")

    # Pin the entity by hand: status + override in one write. Unconditional —
    # the operator may always (re)pin (set_manual_override has no override guard).
    wrote = await entity_repo.set_manual_override(
        item.entity, status=body.status, manual_override=True
    )
    if not wrote:
        raise HTTPException(
            status_code=404,
            detail=f"entity {item.entity} not found for pin",
        )

    decision_label = body.decision or body.status
    updated = await queue.set_decision(queue_id, decision_label)
    if updated is None:  # pragma: no cover - get() already proved it exists
        raise HTTPException(status_code=404, detail="queue row not found")

    return DecisionResponse(
        queue_id=updated.id,
        entity=item.entity,
        status=body.status,
        manual_override=True,
        decision=decision_label,
    )


@router.get("/backbone-warnings", response_model=BackboneWarningsResponse)
async def backbone_warnings(
    entity_ids: List[str] = Query(
        default_factory=list,
        alias="entity_id",
        description="Entity ids to check (repeat the param).",
    ),
    entity_repo: EntityRepository = Depends(get_entity_repo),
) -> BackboneWarningsResponse:
    """Recompute weak-edge backbone warnings for the supplied entity ids.

    Read-only: degree is recomputed via the Q.2 signals helper and weak edges via
    the Q.4 backbone check; nothing is written.
    """
    if not entity_ids:
        return BackboneWarningsResponse(count=0, warnings=[])

    signals = TriageSignalsService(entity_repo)
    degree_by_id = await signals._batch_effective_degree(list(entity_ids))

    backbone = BackboneCheckService(entity_repo)
    warnings = await backbone.warnings_for(list(entity_ids), degree_by_id)

    return BackboneWarningsResponse(
        count=len(warnings),
        warnings=[
            BackboneWarningOut(
                entity_id=w.entity_id,
                structural_degree=w.structural_degree,
                weak_edges=[
                    WeakEdgeOut(
                        other=e.other,
                        relation_type=e.relation_type,
                        doc_count=e.doc_count,
                    )
                    for e in w.weak_edges
                ],
            )
            for w in warnings
        ],
    )
