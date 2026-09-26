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

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from shared.vocabulary.crossref_provider import CrossrefProvider
from shared.vocabulary.tooi_bulk import TOOIBulkFetcher
from shared.vocabulary.tooi_provider import TOOIProvider
from surrealdb_service.repositories.reference_entity import (
    ReferenceEntityRepository,
)

from app_main.dependencies import (
    get_entity_repo,
    get_reference_entity_repo,
    get_triage_queue_repo,
)
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
    #: A subset of the owner's scope; anything else is refused (422).
    entity_types: List[str] = Field(default_factory=lambda: list(DEFAULT_TYPES))
    #: Compute the report and write nothing — no links, no queue rows, no
    #: register load. The default: a real run needs `dry_run: false`.
    dry_run: bool = True

    @field_validator("entity_types")
    @classmethod
    def _in_scope(cls, value: List[str]) -> List[str]:
        outside = sorted(set(value) - set(DEFAULT_TYPES))
        if outside or not value:
            raise ValueError(
                f"entity_types must be a non-empty subset of {list(DEFAULT_TYPES)}; "
                f"got {outside or value}"
            )
        return value


class RegisterStatus(BaseModel):
    rows: int
    keyed: int


class ReconcileResponse(BaseModel):
    dry_run: bool
    tooi_register: RegisterStatus
    scanned: int
    scanned_by: dict
    skipped_no_name: int
    by_type: dict
    linked: List[dict]
    candidates: List[dict]
    no_match: List[dict]
    write_failed: List[dict]
    queued: int


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



#: A register this small is the bundled seed (the ten core ministries), which
#: `refresh()` falls back to when the live registers are unreachable. Reconciling
#: against it would report every municipality as "no match" — silently.
_SEED_SIZE = 10


@router.post("/reconcile", response_model=ReconcileResponse)
async def reconcile_vocabulary(
    body: ReconcileRequest,
    repo: ReferenceEntityRepository = Depends(get_reference_entity_repo),
    entity_repo=Depends(get_entity_repo),
    queue=Depends(get_triage_queue_repo),
) -> ReconcileResponse:
    """Link named entities to TOOI (D2 — the production caller K.4 lacked).

    Does NOT load the register: that is ``POST /vocabulary/refresh``, which a
    reconcile used to trigger itself — inside a request, writing ~2,900 rows even
    on a dry run. Refuses (409) unless the register is loaded in full: every row
    keyed (migration 81) and more than the bundled seed.
    """
    status = await repo.key_status("tooi")
    if status["rows"] <= _SEED_SIZE or status["keyed"] < status["rows"]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"TOOI register not ready ({status['keyed']} of {status['rows']} rows "
                f"keyed; the seed alone is {_SEED_SIZE}). Run POST /vocabulary/refresh."
            ),
        )
    tooi = TOOIProvider(repo)
    reconciler = VocabularyReconciler(
        [tooi], entity_repo=None if body.dry_run else entity_repo
    )
    report = await reconcile_active_entities(
        reconciler, types=body.entity_types, dry_run=body.dry_run,
        queue=None if body.dry_run else queue,
    )
    return ReconcileResponse(
        dry_run=report.dry_run, tooi_register=RegisterStatus(**status),
        scanned=report.scanned, scanned_by=report.scanned_by,
        skipped_no_name=report.skipped_no_name, by_type=report.by_type,
        linked=report.linked, candidates=report.candidates, no_match=report.no_match,
        write_failed=report.write_failed, queued=report.queued,
    )
