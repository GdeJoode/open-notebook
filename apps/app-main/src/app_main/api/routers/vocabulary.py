"""Vocabulary router — external-vocabulary refresh + status (K.4).

* ``POST /api/vocabulary/refresh`` (re-)syncs the bulk vocabularies (TOOI) into
  ``reference_entity`` and stamps ``last_validated``. Idempotent — a second
  refresh creates no duplicate rows.
* ``GET /api/vocabulary/status`` reports, per provider, the loaded-row count and
  the most-recent ``last_validated`` timestamp.

Crossref is an on-demand provider (no bulk pre-load), so ``refresh`` reports it
with ``loaded=0``; it is exercised at reconcile time, not here.
"""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app_main.dependencies import get_reference_entity_repo
from shared.vocabulary.crossref_provider import CrossrefProvider
from shared.vocabulary.tooi_provider import TOOIProvider
from surrealdb_service.repositories.reference_entity import (
    ReferenceEntityRepository,
)

router = APIRouter(prefix="/vocabulary", tags=["vocabulary"])


def _crossref_contact() -> str:
    return os.environ.get("CROSSREF_CONTACT_EMAIL", "noreply@open-notebook.local")


def _tooi_source() -> Optional[str]:
    """Optional bulk-file path for TOOI (K-D2). None → bundled verified seed."""
    return os.environ.get("TOOI_BULK_SOURCE") or None


# --- Schemas -----------------------------------------------------------------


class RefreshRequest(BaseModel):
    providers: Optional[List[str]] = Field(
        default=None,
        description="Provider names to refresh; None = all bulk providers (tooi).",
    )


class ProviderRefreshOut(BaseModel):
    name: str
    loaded: int
    on_demand: bool = False


class RefreshResponse(BaseModel):
    results: List[ProviderRefreshOut]


class ProviderStatusOut(BaseModel):
    name: str
    row_count: int
    last_validated: Optional[str]
    on_demand: bool = False


class StatusResponse(BaseModel):
    providers: List[ProviderStatusOut]


# --- Endpoints ---------------------------------------------------------------


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_vocabulary(
    body: RefreshRequest,
    repo: ReferenceEntityRepository = Depends(get_reference_entity_repo),
) -> RefreshResponse:
    """Re-sync bulk vocabularies into ``reference_entity`` (idempotent)."""
    wanted = set(body.providers) if body.providers else None
    results: List[ProviderRefreshOut] = []

    if wanted is None or "tooi" in wanted:
        tooi = TOOIProvider(repo, source=_tooi_source())
        loaded = await tooi.refresh()
        results.append(ProviderRefreshOut(name="tooi", loaded=loaded))

    if wanted is None or "crossref" in wanted:
        # Crossref is on-demand — refresh is a no-op, reported for completeness.
        crossref = CrossrefProvider(contact_email=_crossref_contact())
        loaded = await crossref.refresh()
        results.append(
            ProviderRefreshOut(name="crossref", loaded=loaded, on_demand=True)
        )

    return RefreshResponse(results=results)


@router.get("/status", response_model=StatusResponse)
async def vocabulary_status(
    repo: ReferenceEntityRepository = Depends(get_reference_entity_repo),
) -> StatusResponse:
    """Per-provider loaded-row count + last refresh timestamp."""
    tooi_count = await repo.count(source="tooi")
    tooi_last = await repo.last_validated("tooi")
    return StatusResponse(
        providers=[
            ProviderStatusOut(
                name="tooi", row_count=tooi_count, last_validated=tooi_last
            ),
            ProviderStatusOut(
                name="crossref",
                row_count=0,
                last_validated=None,
                on_demand=True,
            ),
        ]
    )
