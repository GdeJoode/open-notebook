"""Run-eindpunten voor Langflow (F2-04, ADR 0003, 0022 E, plan §5.4).

* ``POST /api/v1/runs``: maakt een run (``bezig``); 409 als de flow niet in Git
  staat en de invoer intern of onbekend is (de run is dan ``mislukt``).
* ``PATCH /api/v1/runs/{run_id}``: werkt een lopende run bij of sluit hem af;
  409 op een afgesloten run.
* ``GET /api/v1/runs/{run_id}``: de run.

Recht ``pijplijn`` (en ``beheerder``).
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from shared.models.agents import Right
from shared.models.processing_run import ProcessingRun

from app_main.api.agent_auth import require_right
from app_main.api.schemas import (
    RunCreateRequest,
    RunCreateResponse,
    RunPatchRequest,
    RunResponse,
)
from app_main.dependencies import get_processing_run_service
from app_main.services.processing_run_service import (
    FlowNotRegisteredError,
    ProcessingRunService,
    RunClosedError,
)

router = APIRouter(
    prefix="/v1/runs",
    tags=["runs"],
    dependencies=[Depends(require_right(Right.PIJPLIJN))],
)


def _antwoord(run: ProcessingRun) -> RunResponse:
    return RunResponse(**run.model_dump(exclude={"created", "updated"}))


@router.post("", status_code=201, response_model=RunCreateResponse)
async def create_run(
    body: RunCreateRequest,
    service: ProcessingRunService = Depends(get_processing_run_service),
):
    try:
        run = await service.start(
            trigger=body.trigger,
            pipeline_id=body.pipeline_id,
            git_commit=body.pipeline_version.git_commit,
            flow_hash=body.pipeline_version.flow_hash,
            source_map_rule_id=body.source_map_rule_id,
            label_schema=body.label_schema,
        )
    except FlowNotRegisteredError as e:
        return JSONResponse(
            status_code=409,
            content={"detail": "Flow niet vastgelegd in Git", "run_id": e.run_id,
                     "status": "mislukt"},
        )
    return RunCreateResponse(run_id=str(run.id), status=run.status)


@router.patch("/{run_id}", response_model=RunResponse)
async def update_run(
    run_id: str,
    body: RunPatchRequest,
    service: ProcessingRunService = Depends(get_processing_run_service),
) -> RunResponse:
    try:
        run = await service.update(
            run_id,
            status=body.status,
            ended_at=body.ended_at,
            log=[r.model_dump() for r in body.log],
            models=body.models,
            error=body.error,
        )
    except RunClosedError:
        raise HTTPException(status_code=409, detail="Run is al afgesloten") from None
    return _antwoord(run)


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    service: ProcessingRunService = Depends(get_processing_run_service),
) -> RunResponse:
    return _antwoord(await service.get(run_id))
