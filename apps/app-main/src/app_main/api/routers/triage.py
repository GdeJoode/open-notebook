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

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field
from surrealdb_service.repositories import EntityRepository

from app_main.dependencies import get_entity_repo, get_triage_queue_repo
from app_main.services.triage import queue_sources
from app_main.services.triage.backbone_check import BackboneCheckService
from app_main.services.triage.signals_service import TriageSignalsService
from app_main.services.triage.status_assignment_service import (
    STATUS_ACTIVE,
    STATUS_REFERENCE,
)
from app_main.services.triage.triage_queue import (
    DECISION_DISMISSED,
    DECISION_LINKED,
    DECISION_OPEN,
    KIND_STATUS,
    KIND_TOOI,
    QueueItem,
    TriageQueueRepository,
)
from app_main.api.agent_auth import RECHT_BEHEERDER

router = APIRouter(prefix="/triage", tags=["triage"], dependencies=[RECHT_BEHEERDER])


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
    #: ``status`` (Q.4: decide with /decision) or ``tooi`` (D2: decide with
    #: /tooi-decision). Migration 83.
    kind: str = KIND_STATUS
    #: A TOOI row's authority records: uri, label, valid_until, confidence,
    #: matched_on, current_names.
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    #: Q-B1: the documents behind *Docs* — ``{id, title}``, read NOW, while
    #: ``doc_count`` is the triage run's snapshot (the two can differ).
    sources: List[Dict[str, str]] = Field(default_factory=list)
    #: WB.3: the entity's canonical type, read now. ``type`` is the display
    #: label the triage run stored; the triage table shows it second.
    entity_type: str = ""

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
            kind=item.kind,
            candidates=list(item.candidates),
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


class TooiDecisionIn(BaseModel):
    """A curator's answer to a TOOI row: link ONE of its candidates, or dismiss.

    Neither touches the entity's status or ``manual_override`` — a TOOI row is a
    different decision from a status row (migration 83).
    """

    action: Literal["link", "dismiss"]
    uri: Optional[str] = Field(
        default=None, description="For `link`: one of the row's candidate URIs."
    )


class TooiDecisionResponse(BaseModel):
    queue_id: str
    entity: str
    decision: str
    uri: Optional[str] = None


class DecisionResponse(BaseModel):
    queue_id: str
    entity: str
    status: str
    manual_override: bool
    decision: str


class EntityPinIn(BaseModel):
    """An operator pin from the KG detail panel (Q.5 manual_override toggle).

    ``manual_override`` toggles the pin: ``true`` pins the entity (optionally
    setting ``status``); ``false`` releases it back to automation. Unlike the
    queue decision, this is keyed on the ENTITY id directly so the operator can
    pin any entity from the detail panel, not only queued ones.
    """

    manual_override: bool = Field(
        ..., description="Pin (true) or release (false) the entity."
    )
    status: Optional[str] = Field(
        default=None,
        description=(
            "Status to pin alongside the override ('active'/'reference'); "
            "omit to leave the current status untouched."
        ),
    )


class EntityPinResponse(BaseModel):
    entity: str
    status: Optional[str]
    manual_override: bool
    closed_queue_rows: int


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
    # One batched lookup for the whole queue (Q-B1): the hover on *Docs*. It is
    # an enrichment, so it degrades to "no documents" rather than failing the
    # queue a curator needs to work.
    try:
        sources = await queue_sources.sources_for_entities([i.entity for i in items])
    except Exception as exc:
        logger.warning("triage queue: document lookup failed, serving without: {e}", e=exc)
        sources = {}
    try:
        types = await queue_sources.entity_types([i.entity for i in items])
    except Exception as exc:
        logger.warning("triage queue: type lookup failed, serving without: {e}", e=exc)
        types = {}
    out = []
    for i in items:
        row = QueueItemOut.from_item(i)
        row.sources = sources.get(i.entity, [])
        row.entity_type = types.get(i.entity, "")
        out.append(row)
    return QueueResponse(count=len(out), items=out)


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
    if item.kind != KIND_STATUS:
        # A TOOI row is not a status question: pinning status here is exactly
        # the side effect migration 83 separated the rows to prevent.
        raise HTTPException(
            status_code=409,
            detail=f"queue row is kind '{item.kind}'; decide it with /tooi-decision",
        )

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


@router.post("/queue/{queue_id:path}/tooi-decision", response_model=TooiDecisionResponse)
async def decide_tooi(
    queue_id: str,
    body: TooiDecisionIn,
    queue: TriageQueueRepository = Depends(get_triage_queue_repo),
    entity_repo: EntityRepository = Depends(get_entity_repo),
) -> TooiDecisionResponse:
    """Answer a TOOI row (D2): link one candidate, or dismiss the row.

    ``link`` writes the chosen URI and that record's current names onto the
    entity — what an automatic link writes — and fails loudly if the write is
    refused, rather than closing the row over a link that does not exist.
    Either answer closes the row, and it stays closed until the authority offers
    a different candidate set. Status and ``manual_override`` are untouched.
    """
    item = await queue.get(queue_id)
    if item is None:
        raise HTTPException(status_code=404, detail="queue row not found")
    if item.kind != KIND_TOOI:
        raise HTTPException(
            status_code=409,
            detail=f"queue row is kind '{item.kind}'; decide it with /decision",
        )
    if item.decision != DECISION_OPEN:
        raise HTTPException(status_code=409, detail=f"already decided: {item.decision}")

    if body.action == "dismiss":
        await queue.set_decision(queue_id, DECISION_DISMISSED)
        return TooiDecisionResponse(queue_id=item.id, entity=item.entity,
                                    decision=DECISION_DISMISSED)

    chosen = next((c for c in item.candidates if c.get("uri") == body.uri), None)
    if chosen is None:
        raise HTTPException(status_code=422, detail="uri must be one of the row's candidates")
    names = list(chosen.get("current_names") or [chosen.get("label")])
    wrote = await entity_repo.update_external_ids(item.entity, [chosen["uri"]], names)
    if not wrote:
        raise HTTPException(
            status_code=502,
            detail=f"the link was not written for {item.entity}; the row stays open",
        )
    await queue.set_decision(queue_id, DECISION_LINKED)
    return TooiDecisionResponse(queue_id=item.id, entity=item.entity,
                                decision=DECISION_LINKED, uri=chosen["uri"])


@router.post("/entities/{entity_id:path}/override", response_model=EntityPinResponse)
async def set_entity_override(
    entity_id: str,
    body: EntityPinIn,
    queue: TriageQueueRepository = Depends(get_triage_queue_repo),
    entity_repo: EntityRepository = Depends(get_entity_repo),
) -> EntityPinResponse:
    """Pin/release an entity from the KG detail panel (Q.5 override toggle).

    Reuses the same override-wins contract as the queue decision but is keyed on
    the entity id so the operator can toggle ``manual_override`` on ANY entity,
    not only queued ones. Pinning (``manual_override=true``) also closes any OPEN
    queue row for the entity so automation no longer refreshes it; releasing
    (``manual_override=false``) leaves the queue untouched (a later pipeline run
    may re-derive and re-queue it).
    """
    if body.status is not None and body.status not in (
        STATUS_ACTIVE,
        STATUS_REFERENCE,
    ):
        raise HTTPException(
            status_code=422,
            detail=f"status must be '{STATUS_ACTIVE}' or '{STATUS_REFERENCE}'",
        )

    wrote = await entity_repo.set_manual_override(
        entity_id, status=body.status, manual_override=body.manual_override
    )
    if not wrote:
        raise HTTPException(status_code=404, detail="entity not found")

    # Close the entity's open queue row(s) on a pin so the queue reflects the
    # operator's decision; a release leaves them open.
    closed = 0
    if body.manual_override:
        for item in await queue.list_open():
            # Status rows only: a pin answers the status question, not whether
            # the entity is a TOOI organisation (migration 83).
            if item.entity == entity_id and item.kind == KIND_STATUS:
                await queue.set_decision(item.id, body.status or "pinned")
                closed += 1

    return EntityPinResponse(
        entity=entity_id,
        status=body.status,
        manual_override=body.manual_override,
        closed_queue_rows=closed,
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
