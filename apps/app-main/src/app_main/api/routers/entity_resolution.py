"""Entity-resolution router — retroactive canonicalization (K.3) + dedup (K.5).

K.3 endpoints (deterministic-collision merges):

* ``POST /api/entity-resolution/plan`` returns the **dry-run** ``MergePlan``
  (no writes) for an optional notebook scope.
* ``POST /api/entity-resolution/apply`` applies a **reviewed** list of clusters.
  It requires explicit cluster ids — there is intentionally NO "apply all"
  affordance; the caller must echo back the exact clusters from a prior plan.

K.5 endpoints (fuzzy/embedding candidate dedup + user overlay):

* ``GET /api/entity-resolution/candidates`` — the review queue: fuzzy/embedding
  merge proposals partitioned into auto-merge / review bands (read-only).
* ``POST /api/entity-resolution/overlay`` — add a force-merge/force-split rule.
* ``DELETE /api/entity-resolution/overlay/{id}`` — remove a rule.
"""

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from surrealdb_service.connection import ensure_record_id, execute_query

from app_main.services.entity_resolution import (
    CandidateDedupService,
    MergeCluster,
    OverlayService,
    RecanonicalizationService,
    merge_audit,
)

router = APIRouter(prefix="/entity-resolution", tags=["entity-resolution"])


def get_recanonicalization_service() -> RecanonicalizationService:
    return RecanonicalizationService()


def get_candidate_dedup_service() -> CandidateDedupService:
    return CandidateDedupService()


def get_overlay_service() -> OverlayService:
    return OverlayService()


# --- Response/request schemas -------------------------------------------------


class PlanRequest(BaseModel):
    notebook_id: Optional[str] = Field(
        default=None,
        description="Scope the plan to one notebook's entities; None = global.",
    )


class ClusterOut(BaseModel):
    new_canonical: str
    entity_type: str
    winner_id: str
    loser_ids: List[str]
    member_surface_forms: List[str]
    total_source_docs: int
    size: int


class PlanResponse(BaseModel):
    scope: str
    total_active_entities: int
    cluster_count: int
    clusters: List[ClusterOut]


class ApplyClusterIn(BaseModel):
    """A reviewed cluster to apply — echoed back from a prior plan.

    Only the winner + losers + identity are needed; the apply path re-loads the
    losers and is idempotent, so stale ``total_source_docs`` etc. are harmless.
    """

    new_canonical: str
    entity_type: str
    winner_id: str
    loser_ids: List[str]
    member_surface_forms: List[str] = Field(default_factory=list)
    #: PC.5b: who decided this merge. ``curator_approved`` — a human approved it
    #: (a review candidate, or a plan cluster); ``auto_merge`` — the auto band
    #: decided. Required: the server no longer infers it, and it is logged.
    origin: Literal["curator_approved", "auto_merge"]
    #: The method and evidence that produced the proposal, for the audit log.
    method: Optional[str] = None
    score: Optional[float] = None
    evidence: Optional[str] = None
    #: A merge across entity types must say so, and only a curator may make one.
    cross_type: bool = False


class ApplyRequest(BaseModel):
    clusters: List[ApplyClusterIn] = Field(
        ...,
        min_length=1,
        description="Explicit reviewed clusters to merge. No implicit apply-all.",
    )


class ApplyResultOut(BaseModel):
    new_canonical: str
    entity_type: str
    winner_id: str
    merged_loser_ids: List[str]
    relations_repointed: int
    aliases_created: int
    skipped: bool


class ApplyResponse(BaseModel):
    applied: int
    skipped: int
    results: List[ApplyResultOut]


# --- Endpoints ----------------------------------------------------------------


@router.post("/plan", response_model=PlanResponse)
async def plan_merges(
    body: PlanRequest,
    svc: RecanonicalizationService = Depends(get_recanonicalization_service),
) -> PlanResponse:
    """Return the dry-run merge plan (no writes)."""
    plan = await svc.plan_merges(notebook_id=body.notebook_id)
    return PlanResponse(
        scope=plan.scope,
        total_active_entities=plan.total_active_entities,
        cluster_count=plan.cluster_count,
        clusters=[
            ClusterOut(
                new_canonical=c.new_canonical,
                entity_type=c.entity_type,
                winner_id=c.winner_id,
                loser_ids=c.loser_ids,
                member_surface_forms=c.member_surface_forms,
                total_source_docs=c.total_source_docs,
                size=c.size,
            )
            for c in plan.clusters
        ],
    )


async def _validate_clusters(clusters: List[ApplyClusterIn]) -> Dict[str, Dict[str, Any]]:
    """Refuse a merge the server would not make, BEFORE any cluster is applied.

    PC.5b. The route merged whatever it was sent, and the docstring's "only
    ``auto_merge`` candidates" was enforced only by the frontend — which also
    sends review candidates a curator approved. So the server now checks, for
    every cluster, what it can check itself:

    * every id exists; the winner is ``active``; a loser is ``active`` or
      already ``merged`` (a repeat apply is skipped, not refused — idempotency);
    * every member has the cluster's ``entity_type`` unless ``cross_type`` is
      set, and only ``curator_approved`` may set it (the auto band never merges
      across types, PC.2);
    * the winner is not a loser, and no id appears twice in the request.

    Returns ``{id: {"name", "type"}}`` for the audit log. One bad cluster fails
    the request with 422 and nothing is merged.
    """
    seen: set[str] = set()
    for c in clusters:
        if not c.winner_id or not c.loser_ids:
            raise HTTPException(
                status_code=422,
                detail="Each cluster requires a winner_id and at least one loser_id",
            )
        members = [c.winner_id, *c.loser_ids]
        if len(set(members)) != len(members) or seen & set(members):
            raise HTTPException(
                status_code=422,
                detail=f"cluster {c.winner_id}: an id appears twice (winner among "
                "losers, or in two clusters)",
            )
        seen |= set(members)
        if c.cross_type and c.origin != "curator_approved":
            raise HTTPException(
                status_code=422,
                detail=f"cluster {c.winner_id}: only a curator may merge across types",
            )

    try:
        ids = [ensure_record_id(i) for i in seen]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"malformed entity id: {exc}")
    rows = await execute_query(
        "SELECT id, canonical_name, entity_type, status FROM $ids;", {"ids": ids}
    )
    found = {str(r["id"]): r for r in rows or []}
    for c in clusters:
        for eid in [c.winner_id, *c.loser_ids]:
            row = found.get(str(ensure_record_id(eid)))
            if row is None:
                raise HTTPException(status_code=422, detail=f"entity {eid} not found")
            allowed = ("active",) if eid == c.winner_id else ("active", "merged")
            if row.get("status", "active") not in allowed:
                raise HTTPException(
                    status_code=422,
                    detail=f"entity {eid} is {row.get('status')!r}; "
                    f"{'the winner' if eid == c.winner_id else 'a loser'} must be "
                    f"{' or '.join(allowed)}",
                )
            if not c.cross_type and row.get("entity_type") != c.entity_type:
                raise HTTPException(
                    status_code=422,
                    detail=f"entity {eid} is {row.get('entity_type')!r}, the cluster "
                    f"is {c.entity_type!r}; set cross_type to merge across types",
                )
    return {
        eid: {"name": str(r.get("canonical_name") or ""), "type": str(r.get("entity_type") or "")}
        for eid, r in found.items()
    }


@router.post("/apply", response_model=ApplyResponse)
async def apply_merges(
    body: ApplyRequest,
    svc: RecanonicalizationService = Depends(get_recanonicalization_service),
) -> ApplyResponse:
    """Apply an explicit list of merge clusters (mutating).

    Every cluster is validated before any is applied (``_validate_clusters``),
    and every merged pair is recorded in ``resolution_log`` with its origin
    (PC.5b).

    Known limits: validation and apply are not one transaction — a concurrent
    merge between the two can still reach ``apply_merge``, which skips losers
    no longer active. ``origin`` is what the client states; the server checks
    what it constrains (``cross_type``) and logs it, it cannot verify it.
    """
    if not body.clusters:
        raise HTTPException(
            status_code=422, detail="At least one cluster id set is required"
        )
    known = await _validate_clusters(body.clusters)

    results: List[ApplyResultOut] = []
    applied = 0
    skipped = 0
    for c in body.clusters:
        cluster = MergeCluster(
            new_canonical=c.new_canonical,
            entity_type=c.entity_type,
            winner_id=c.winner_id,
            loser_ids=c.loser_ids,
            member_surface_forms=c.member_surface_forms,
            total_source_docs=0,
        )
        result = await svc.apply_merge(cluster)
        if result.skipped:
            skipped += 1
        else:
            applied += 1
            await merge_audit.record_merges(
                winner_id=str(ensure_record_id(c.winner_id)),
                merged_loser_ids=[str(ensure_record_id(i)) for i in result.merged_loser_ids],
                names={k: v["name"] for k, v in known.items()},
                types={k: v["type"] for k, v in known.items()},
                origin=c.origin, method=c.method, score=c.score, evidence=c.evidence,
            )
        results.append(
            ApplyResultOut(
                new_canonical=result.new_canonical,
                entity_type=result.entity_type,
                winner_id=result.winner_id,
                merged_loser_ids=result.merged_loser_ids,
                relations_repointed=result.relations_repointed,
                aliases_created=result.aliases_created,
                skipped=result.skipped,
            )
        )

    return ApplyResponse(applied=applied, skipped=skipped, results=results)


# --- K.5: candidate dedup review queue -----------------------------------------


class CandidateOut(BaseModel):
    id_a: str
    id_b: str
    name_a: str
    name_b: str
    entity_type: str
    #: Why this pair is here when `method` alone does not say: the head run
    #: `containment` removed, or the shape and shared name `surface_parts` found
    #: (D1). See `MergeCandidate.evidence`.
    evidence: str = ""
    #: ``id_b``'s type when the two differ, else "". Non-empty only for the
    #: ``fold_equal_cross_type`` method: without it the card renders the same
    #: name twice with nothing to distinguish the rows.
    entity_type_b: str = ""
    score: float
    band: str
    method: str
    winner_id: str
    loser_id: str


class CandidatesResponse(BaseModel):
    scope: str
    total_active_entities: int
    auto_merge_count: int
    review_count: int
    auto_merge: List[CandidateOut]
    review: List[CandidateOut]


@router.get("/candidates", response_model=CandidatesResponse)
async def list_candidates(
    notebook_id: Optional[str] = None,
    svc: CandidateDedupService = Depends(get_candidate_dedup_service),
) -> CandidatesResponse:
    """Return the fuzzy/embedding merge proposals (read-only review queue).

    The ``review`` band is queued for a human and is NEVER auto-applied. Both
    bands reach ``POST /apply`` — a review candidate once a curator approves it —
    and the request states which (``origin``), which the server logs; what the
    server enforces is in ``_validate_clusters`` (PC.5b).
    """
    report = await svc.propose_candidates(notebook_id=notebook_id)

    def _out(c) -> CandidateOut:
        return CandidateOut(
            id_a=c.id_a,
            id_b=c.id_b,
            name_a=c.name_a,
            name_b=c.name_b,
            entity_type=c.entity_type,
            entity_type_b=c.entity_type_b,
            evidence=c.evidence,
            score=c.score,
            band=c.band,
            method=c.method,
            winner_id=c.winner_id,
            loser_id=c.loser_id,
        )

    return CandidatesResponse(
        scope=report.scope,
        total_active_entities=report.total_active_entities,
        auto_merge_count=report.auto_merge_count,
        review_count=report.review_count,
        auto_merge=[_out(c) for c in report.auto_merge],
        review=[_out(c) for c in report.review],
    )


# --- K.5: user overlay (force-merge / force-split) -----------------------------


class OverlayCreateIn(BaseModel):
    kind: str = Field(..., description="'merge' (force-include) or 'split' (veto)")
    name_a: str = Field(..., min_length=1)
    name_b: str = Field(..., min_length=1)
    entity_type: str = Field(
        default="",
        description="Shared entity type the rule binds (type-discipline).",
    )
    scope: str = Field(default="global", description="'global' or 'notebook'")
    notebook_id: Optional[str] = Field(
        default=None, description="Required when scope='notebook'."
    )


class OverlayCreateResponse(BaseModel):
    id: str


@router.post("/overlay", response_model=OverlayCreateResponse)
async def add_overlay(
    body: OverlayCreateIn,
    svc: OverlayService = Depends(get_overlay_service),
) -> OverlayCreateResponse:
    """Add a force-merge or force-split overlay rule (the human escape hatch)."""
    try:
        rid = await svc.add_rule(
            kind=body.kind,
            name_a=body.name_a,
            name_b=body.name_b,
            entity_type=body.entity_type,
            scope=body.scope,
            notebook_id=body.notebook_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return OverlayCreateResponse(id=rid)


@router.delete("/overlay/{overlay_id:path}")
async def delete_overlay(
    overlay_id: str,
    svc: OverlayService = Depends(get_overlay_service),
) -> dict:
    """Remove an overlay rule by id."""
    ok = await svc.remove_rule(overlay_id)
    if not ok:
        raise HTTPException(status_code=404, detail="overlay rule not found")
    return {"deleted": overlay_id}
