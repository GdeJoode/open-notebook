"""Per-notebook quality-audit endpoints (Track F.1).

Exposes the always-on, LLM-free audit:

* ``POST /api/notebooks/{notebook_id}/audit`` runs the six checks now and returns
  the fresh run (``run_id`` + findings). Work is bounded graph reads + in-Python
  predicates, so it runs inline (mirrors ``orphans.py``); no job queue, no model.
* ``GET /api/notebooks/{notebook_id}/audit`` returns the most recent run's
  snapshot (empty findings if the notebook has never been audited).

Auth is the notebook-scoped lookup every notebook route uses; a missing notebook
is a 404.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from shared.models.audit import AuditRunResponse

from app_main.dependencies import (
    get_audit_service,
    get_deep_audit_service,
    get_notebook_service,
)
from app_main.services.audit.audit_service import AuditService
from app_main.services.audit.deep_audit_service import DeepAuditService
from app_main.services.notebook_service import NotebookService

router = APIRouter(prefix="/notebooks/{notebook_id}", tags=["audit"])


@router.post("/audit", response_model=AuditRunResponse)
async def run_audit(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    audit_service: AuditService = Depends(get_audit_service),
) -> AuditRunResponse:
    """Run the six LLM-free quality checks over the notebook and persist a run."""
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    try:
        return await audit_service.run_all(str(notebook.id))
    except Exception as e:  # noqa: BLE001 — surface as 500, never a half-run
        logger.error(f"Audit run failed for {notebook_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Audit run failed: {str(e)}")


@router.post("/audit/deep", response_model=AuditRunResponse)
async def run_deep_audit(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    deep_service: DeepAuditService = Depends(get_deep_audit_service),
) -> AuditRunResponse:
    """Run the on-demand deep checks (conflicting facts + provenance gaps).

    Kept off the always-on ``/audit`` path. Enriches the latest run with the two
    deep checks and returns the combined snapshot (cheap + deep). Fail-soft per
    check. See :class:`DeepAuditService` for why this is LLM-free by reuse.
    """
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    try:
        return await deep_service.run_deep(str(notebook.id))
    except Exception as e:  # noqa: BLE001
        logger.error(f"Deep audit failed for {notebook_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Deep audit failed: {str(e)}")


@router.get("/audit", response_model=AuditRunResponse)
async def get_latest_audit(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    audit_service: AuditService = Depends(get_audit_service),
) -> AuditRunResponse:
    """Return the most recent audit run's findings (empty if never run)."""
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    try:
        return await audit_service.latest(str(notebook.id))
    except Exception as e:  # noqa: BLE001
        logger.error(f"Reading audit for {notebook_id} failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Reading audit failed: {str(e)}"
        )
