"""Innamebatches en de innamewachtrij (F2-13, ADR 0019, 0020, plan §5.4).

Batches:

* ``GET /api/v1/intake/batches[/{id}]``: ``pijplijn`` of ``beheerder``;
* ``POST /api/v1/intake/batches``, ``PATCH .../{id}``, ``POST .../{id}/confirm``:
  alleen ``beheerder``.

Wachtrij (Langflow meldt aan en haalt op; de fork roept Langflow niet aan, B5):

* ``POST /api/v1/intake/items``: aanmelden (``pijplijn``);
* ``GET /api/v1/intake/items?status=besloten``: besloten items ophalen
  (``pijplijn``; andere statussen alleen ``beheerder``);
* ``PATCH /api/v1/intake/items/{id}``: status ``verwerkt`` (``pijplijn``);
* ``POST /api/v1/intake/items/{id}/decision``: het besluit (``beheerder``).

``lezen_extern`` en ``lezen_intern`` krijgen op elke route 403.
"""

from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, Response
from shared.models.agents import Right
from shared.models.source_map import IntakeBatch, IntakeItem
from surrealdb_service.repositories.base import RecordNotFoundError, RevConflictError

from app_main.api.agent_auth import AgentKeyContext, require_right
from app_main.api.routers.source_map import fout_naar_http
from app_main.api.schemas import (
    ConfirmRequest,
    IntakeBatchCreate,
    IntakeBatchPatch,
    IntakeBatchResponse,
    IntakeDecisionRequest,
    IntakeItemCreate,
    IntakeItemPatch,
    IntakeItemResponse,
)
from app_main.dependencies import get_intake_service
from app_main.services.inname_binding import BindingError
from app_main.services.intake_service import IntakeConflictError, IntakeService
from app_main.services.source_map_service import AlBevestigdError, VerbodenLocatieError

router = APIRouter(prefix="/v1/intake", tags=["intake"])

_FOUTEN = (RevConflictError, RecordNotFoundError, VerbodenLocatieError, PermissionError,
           IntakeConflictError, AlBevestigdError, BindingError)


def _batch(batch: IntakeBatch) -> IntakeBatchResponse:
    return IntakeBatchResponse(
        **batch.model_dump(exclude={"created", "updated", "created_at"}),
        effective_classification=batch.effectieve_classificatie,
    )


def _item(item: IntakeItem) -> IntakeItemResponse:
    return IntakeItemResponse(
        **item.model_dump(exclude={"created", "updated", "created_at", "registered_by",
                                   "modified_by"}),
        effective_classification=item.effectieve_classificatie,
    )


# -- Batches ----------------------------------------------------------------


@router.get("/batches", response_model=List[IntakeBatchResponse],
            dependencies=[Depends(require_right(Right.PIJPLIJN))])
async def list_batches(service: IntakeService = Depends(get_intake_service)):
    return [_batch(b) for b in await service.list_batches()]


@router.get("/batches/{batch_id}", response_model=IntakeBatchResponse,
            dependencies=[Depends(require_right(Right.PIJPLIJN))])
async def get_batch(batch_id: str, service: IntakeService = Depends(get_intake_service)):
    return _batch(await service.get_batch(batch_id))


@router.post("/batches", status_code=201, response_model=IntakeBatchResponse)
async def create_batch(
    body: IntakeBatchCreate,
    key: AgentKeyContext = Depends(require_right(Right.BEHEERDER)),
    service: IntakeService = Depends(get_intake_service),
):
    try:
        batch = await service.create_batch(
            route=body.route, files=body.files, classification=body.classification,
            source_map_rule=body.source_map_rule_id, bevestig=body.confirm,
            permission=key.permission, person=key.person, key_id=key.key_id,
            reason=body.reason,
        )
    except _FOUTEN as e:
        raise fout_naar_http(e) from None
    return _batch(batch)


@router.patch("/batches/{batch_id}", response_model=IntakeBatchResponse)
async def update_batch(
    batch_id: str,
    body: IntakeBatchPatch,
    key: AgentKeyContext = Depends(require_right(Right.BEHEERDER)),
    service: IntakeService = Depends(get_intake_service),
):
    try:
        batch = await service.update_batch(
            batch_id, body.model_dump(exclude_unset=True, exclude={"rev", "confirm", "reason"}),
            rev=body.rev, bevestig=body.confirm, permission=key.permission,
            person=key.person, key_id=key.key_id, reason=body.reason,
        )
    except _FOUTEN as e:
        raise fout_naar_http(e) from None
    return _batch(batch)


@router.post("/batches/{batch_id}/confirm", response_model=IntakeBatchResponse)
async def confirm_batch(
    batch_id: str,
    body: ConfirmRequest,
    key: AgentKeyContext = Depends(require_right(Right.BEHEERDER)),
    service: IntakeService = Depends(get_intake_service),
):
    try:
        batch = await service.confirm_batch(
            batch_id, rev=body.rev, permission=key.permission, person=key.person,
            key_id=key.key_id, reason=body.reason,
        )
    except _FOUTEN as e:
        raise fout_naar_http(e) from None
    return _batch(batch)


# -- Wachtrij ---------------------------------------------------------------


@router.post("/items", status_code=201, response_model=IntakeItemResponse)
async def register_item(
    body: IntakeItemCreate,
    response: Response,
    key: AgentKeyContext = Depends(require_right(Right.PIJPLIJN)),
    service: IntakeService = Depends(get_intake_service),
):
    try:
        item, nieuw = await service.register_item(
            origin=body.origin, path=body.path, content_hash=body.content_hash,
            gatekeeper=body.gatekeeper.model_dump(exclude_none=True) if body.gatekeeper else None,
            registered_by=key.agent_id,
        )
    except _FOUTEN as e:
        raise fout_naar_http(e) from None
    if not nieuw:
        response.status_code = 200
    return _item(item)


@router.get("/items", response_model=List[IntakeItemResponse])
async def list_items(
    status: Optional[Literal["wacht", "besloten", "verwerkt"]] = None,
    key: AgentKeyContext = Depends(require_right(Right.PIJPLIJN)),
    service: IntakeService = Depends(get_intake_service),
):
    try:
        items = await service.list_items(status=status, permission=key.permission)
    except _FOUTEN as e:
        raise fout_naar_http(e) from None
    return [_item(i) for i in items]


@router.patch("/items/{item_id}", response_model=IntakeItemResponse)
async def mark_processed(
    item_id: str,
    body: IntakeItemPatch,
    key: AgentKeyContext = Depends(require_right(Right.PIJPLIJN)),
    service: IntakeService = Depends(get_intake_service),
):
    try:
        item = await service.mark_processed(
            item_id, rev=body.rev, run_id=body.run_id, door=key.person or key.agent_id,
            key_id=key.key_id,
        )
    except _FOUTEN as e:
        raise fout_naar_http(e) from None
    return _item(item)


@router.post("/items/{item_id}/decision", response_model=IntakeItemResponse)
async def decide_item(
    item_id: str,
    body: IntakeDecisionRequest,
    key: AgentKeyContext = Depends(require_right(Right.BEHEERDER)),
    service: IntakeService = Depends(get_intake_service),
):
    try:
        item = await service.decide_item(
            item_id, rev=body.rev, route=body.route, classification=body.classification,
            permission=key.permission, person=key.person, key_id=key.key_id,
            reason=body.reason,
        )
    except _FOUTEN as e:
        raise fout_naar_http(e) from None
    return _item(item)
