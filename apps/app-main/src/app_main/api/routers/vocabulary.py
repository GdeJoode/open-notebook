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
from shared.vocabulary.crossref_provider import CrossrefProvider
from shared.vocabulary.tooi_bulk import TOOIBulkFetcher
from shared.vocabulary.tooi_provider import TOOIProvider
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.reference_entity import (
    ReferenceEntityRepository,
)

from app_main.dependencies import get_entity_repo, get_reference_entity_repo
from app_main.services.entity_resolution.vocabulary_reconcile_service import (
    DEFAULT_TYPES,
    reconcile_active_entities,
)
from app_main.services.entity_resolution.vocabulary_reconciler import (
    VocabularyReconciler,
)

router = APIRouter(prefix="/vocabulary", tags=["vocabulary"])


def _crossref_contact() -> str:
    return os.environ.get("CROSSREF_CONTACT_EMAIL", "noreply@open-notebook.local")


def _tooi_source() -> Optional[str]:
    """Operator bulk-file path for TOOI (K-D2). Set → overrides the remote fetch."""
    return os.environ.get("TOOI_BULK_SOURCE") or None


def _tooi_bulk_fetcher() -> Optional[TOOIBulkFetcher]:
    """The live TOOI registers fetcher — the resolved K-D2 full-vocabulary source.

    Attached by default so ``refresh`` pulls the full government-organisation set
    from ``repository.officiele-overheidspublicaties.nl`` (fail-soft → seed if the
    registers are unreachable). An operator file (``TOOI_BULK_SOURCE``) takes
    priority over it; ``TOOI_DISABLE_REMOTE=1`` opts out of any network entirely
    (seed/file only — e.g. an air-gapped deploy).
    """
    if os.environ.get("TOOI_DISABLE_REMOTE", "").strip() in {"1", "true", "yes"}:
        return None
    return TOOIBulkFetcher(contact_email=_crossref_contact())


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


class ReconcileRequest(BaseModel):
    entity_types: List[str] = Field(default_factory=lambda: list(DEFAULT_TYPES))
    #: Compute the report and write nothing. The default, so the door is safe to
    #: open: a real run needs `dry_run: false` in the body.
    dry_run: bool = True


class ReconcileResponse(BaseModel):
    dry_run: bool
    tooi_refreshed: int = 0
    scanned: int
    by_type: dict
    linked: List[dict]
    candidates: List[dict]


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
        tooi = TOOIProvider(
            repo,
            source=_tooi_source(),
            bulk_fetcher=_tooi_bulk_fetcher(),
        )
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


@router.post("/reconcile", response_model=ReconcileResponse)
async def reconcile_vocabulary(
    body: ReconcileRequest,
    repo: ReferenceEntityRepository = Depends(get_reference_entity_repo),
    entity_repo=Depends(get_entity_repo),
) -> ReconcileResponse:
    """Link active named entities to TOOI (D2 — the production caller K.4 lacked).

    Loads TOOI first when no register row carries lookup keys (an empty table, or
    rows written before migration 81). Returns every link and every candidate the
    precision guard refused, split by type; ``dry_run`` (default) writes nothing.
    """
    refreshed = 0
    keyed = await execute_query(
        "SELECT count() AS c FROM reference_entity WHERE source_vocabulary = 'tooi' "
        "AND array::len(lookup_keys) > 0 GROUP ALL;"
    )
    tooi = TOOIProvider(repo, source=_tooi_source(), bulk_fetcher=_tooi_bulk_fetcher())
    if not keyed or not keyed[0].get("c"):
        refreshed = await tooi.refresh()
    reconciler = VocabularyReconciler(
        [tooi], entity_repo=None if body.dry_run else entity_repo
    )
    report = await reconcile_active_entities(
        reconciler, types=body.entity_types, dry_run=body.dry_run
    )
    return ReconcileResponse(
        dry_run=report.dry_run, tooi_refreshed=refreshed, scanned=report.scanned,
        by_type=report.by_type, linked=report.linked, candidates=report.candidates,
    )
