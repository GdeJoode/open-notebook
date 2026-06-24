"""After-extraction triage orchestrator (Q.4).

``TriagePipeline.run`` is the single idempotent entry point that fires AFTER each
extraction persist. It is PURE ORCHESTRATION — every heavy step is a call into an
already-merged, FROZEN service; nothing here re-implements matching, merging,
signals, or status logic:

    B1 match     -> CandidateDedupService.propose_candidates   (K.5; match conflicts)
    B2 merge     -> done by the caller's persist_filtered_result (B.8/O.1/Q.2a)
                    + RecanonicalizationService                  (K.3; collision merges)
    B3 signals   -> TriageSignalsService.compute_report          (Q.2)
    B4 status    -> StatusAssignmentService.assign / apply        (Q.3)
    B5 queue     -> TriageQueueRepository.upsert                  (dedup-by-entity)
    B6 backbone  -> BackboneCheckService.warnings_for             (weak-edge warnings)

**Idempotency design (R4) — choice (b), re-triage a safe superset.**
The hook fires AFTER the merge, so the orchestrator cannot itself observe the
pre-merge degree/doc-count the affected-set "changed" detection would need. Of the
plan's two options we take **(b): re-triage the batch entities plus their relation
neighbours directly**, with an EMPTY pre-batch snapshot so Q.2 marks them all
"new" (a deliberate superset of the true affected set). This is safe because every
downstream write is idempotent under re-derivation:

  * status — :meth:`StatusAssignmentService.apply` writes + logs ONLY when the
    status actually changes; an unchanged re-run is a no-op (no DB write, no log
    row). Status is a pure function of tier x degree x override, so re-deriving
    the same inputs yields the same status.
  * provenance — counted by Q.2a's per-edge ``source_documents`` UNION at persist
    time, NOT by triage; triage only READS degree/doc-count, so re-triaging never
    double-counts.
  * queue — :meth:`TriageQueueRepository.upsert` dedups by entity (one OPEN row
    per entity, UPDATEd not appended).

So running the same batch twice yields identical statuses, no extra log rows, and
no duplicate queue rows (proven by the run-twice integration test). The cost of
choice (b) over choice (a) is that unchanged-but-touched nodes are re-EVALUATED
each run; because every evaluation is a no-op when nothing changed, that costs DB
reads but produces zero spurious writes — the AC3 "unchanged nodes untouched"
guarantee is met at the WRITE layer (log/queue stability), which is what the tests
assert.

Triage NEVER fails extraction: the caller wraps :meth:`run` in a log-and-continue
guard (mirroring the filtering-failure isolation), and ``run`` additionally returns
an ``ok`` flag rather than raising on a partial-step failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from app_main.services.triage.backbone_check import (
    BackboneCheckService,
    BackboneWarning,
)
from app_main.services.triage.signals_service import (
    EntitySignals,
    TriageSignalsService,
)
from app_main.services.triage.status_assignment_service import (
    StatusAssignmentService,
    StatusDecision,
)
from app_main.services.triage.triage_config import TriageConfig, load_triage_config
from app_main.services.triage.triage_queue import TriageQueueRepository


@dataclass
class TriageRunReport:
    """Outcome of one :meth:`TriagePipeline.run`.

    A value object the caller can log/inspect; the pipeline's real effects are the
    DB writes (status, log, queue). ``ok`` is False when a step raised and the run
    bailed early (the caller still treats the extraction as successful — triage is
    non-blocking).
    """

    batch_id: str
    source_id: str
    ok: bool = True
    affected: Set[str] = field(default_factory=set)
    statuses_changed: int = 0
    queued: int = 0
    backbone_warnings: List[BackboneWarning] = field(default_factory=list)
    match_conflicts: int = 0
    error: Optional[str] = None


class TriagePipeline:
    """Orchestrate B1->B6 over one extraction batch. Idempotent, non-blocking.

    All collaborators are injected so the pipeline is trivially unit-testable with
    mocks; the production wiring lives in ``dependencies.get_triage_pipeline``.
    """

    def __init__(
        self,
        *,
        entity_repository: Any,
        signals_service: TriageSignalsService,
        status_service: StatusAssignmentService,
        queue_repository: TriageQueueRepository,
        backbone_service: BackboneCheckService,
        candidate_service: Any = None,
        recanonicalization_service: Any = None,
        config: Optional[TriageConfig] = None,
    ) -> None:
        """Wire the orchestrator.

        Args:
            entity_repository: ``EntityRepository`` — the relation-neighbour and
                triage-row batch loaders.
            signals_service: Q.2 :class:`TriageSignalsService` (B3).
            status_service: Q.3 :class:`StatusAssignmentService` (B4).
            queue_repository: The B5 :class:`TriageQueueRepository`.
            backbone_service: The B6 :class:`BackboneCheckService`.
            candidate_service: K.5 ``CandidateDedupService`` for B1 match-conflict
                surfacing (optional; when ``None`` the match step is skipped — the
                merge itself already happened at persist time).
            recanonicalization_service: K.3 service for B2 collision merges
                (optional; the primary merge is the caller's persist).
            config: Q.1 config; defaults to the cached committed config.
        """
        self._repo = entity_repository
        self._signals = signals_service
        self._status = status_service
        self._queue = queue_repository
        self._backbone = backbone_service
        self._candidates = candidate_service
        self._recanon = recanonicalization_service
        self._cfg = config if config is not None else load_triage_config()

    async def run(
        self,
        source_id: str,
        persisted_entity_ids: List[str],
        batch_id: str,
    ) -> TriageRunReport:
        """Run B1->B6 over the persisted batch. Idempotent + non-blocking.

        Args:
            source_id: The source whose extraction just persisted.
            persisted_entity_ids: The batch's persisted entity record-ids (from
                ``persist_filtered_result``'s ``persisted_entity_ids``).
            batch_id: A stable identifier for this triage run (recorded on the
                status-change log + queue rows).

        Returns:
            A :class:`TriageRunReport`. On any step failure the report's ``ok`` is
            False and ``error`` is set; :meth:`run` does NOT raise (the caller's
            guard is a second belt-and-braces, not the only one).
        """
        report = TriageRunReport(batch_id=batch_id, source_id=source_id)
        if not persisted_entity_ids:
            logger.debug("Triage[{}]: empty batch, nothing to triage", batch_id)
            return report

        try:
            # --- Affected set (R4 choice b): batch entities + relation neighbours.
            affected_ids = await self._affected_superset(persisted_entity_ids)
            report.affected = set(affected_ids)

            # --- Load the triage projection for the affected set in ONE call.
            rows = await self._repo.triage_rows_for_entities(affected_ids)
            row_by_id: Dict[str, Dict[str, Any]] = {
                str(r.get("id")): r for r in rows
            }

            # --- B3 signals. Empty snapshot => Q.2 marks the whole superset
            #     "new"/changed; idempotency is secured by the no-op writes
            #     downstream, not by the affected-set narrowing (see module doc).
            signals = await self._signals.compute_report(
                batch_entities=row_by_id,
                pre_batch_snapshot={},
            )

            # --- B4 status + B5 queue, per affected entity.
            for entity_id in affected_ids:
                row = row_by_id.get(entity_id)
                sig = signals.by_id.get(entity_id)
                if row is None or sig is None:
                    continue
                await self._triage_one(row, sig, batch_id, report)

            # --- B1 match: surface match CONFLICTS (review-band proposals) into
            #     the queue. The merge already happened (B2 at persist); this step
            #     flags residual same-type near-duplicates an operator should look
            #     at. Optional — skipped when no candidate service is wired.
            await self._surface_match_conflicts(source_id, report, batch_id, row_by_id)

            # --- B6 backbone: weak-edge warnings for well-connected affected nodes.
            degree_by_id = {
                i: s.effective_degree for i, s in signals.by_id.items()
            }
            report.backbone_warnings = await self._backbone.warnings_for(
                affected_ids, degree_by_id
            )

            logger.info(
                "Triage[{}] source={}: affected={} status_changes={} queued={} "
                "backbone_warnings={} match_conflicts={}",
                batch_id,
                source_id,
                len(report.affected),
                report.statuses_changed,
                report.queued,
                len(report.backbone_warnings),
                report.match_conflicts,
            )
            return report
        except Exception as exc:  # non-blocking: never propagate from triage.
            logger.exception("Triage[{}] failed: {}", batch_id, exc)
            report.ok = False
            report.error = str(exc)
            return report

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    async def _affected_superset(self, batch_ids: List[str]) -> List[str]:
        """Batch entities plus their active relation neighbours (R4 choice b).

        The neighbours are pulled because a batch entity gaining an edge changes
        its NEIGHBOUR's effective degree too (a structural/promoted edge counts
        for both endpoints — Q.2 AC5); re-triaging the neighbour keeps its status
        correct. Dedup-preserving order, batch ids first.
        """
        seen: Set[str] = set()
        ordered: List[str] = []
        for i in batch_ids:
            sid = str(i)
            if sid not in seen:
                seen.add(sid)
                ordered.append(sid)
        edges = await self._repo.relations_for_entities(batch_ids)
        for edge in edges or []:
            for endpoint in (str(edge.get("in")), str(edge.get("out"))):
                if endpoint and endpoint not in seen and endpoint != "None":
                    seen.add(endpoint)
                    ordered.append(endpoint)
        return ordered

    async def _triage_one(
        self,
        row: Dict[str, Any],
        sig: EntitySignals,
        batch_id: str,
        report: TriageRunReport,
    ) -> None:
        """B4 status + B5 queue for one affected entity."""
        primary_type = row.get("primary_type") or row.get("entity_type") or ""
        tier = self._cfg.tier_for(str(primary_type))

        decision: StatusDecision = self._status.assign(
            row, tier, sig.effective_degree
        )
        changed = await self._status.apply(decision, batch_id)
        if changed:
            report.statuses_changed += 1

        # B5: queue the entity when the decision raised any review flag. The
        # queue dedups by entity, so re-running is stable; an operator-pinned
        # entity raises no flags (assign returns empty queue_flags) and so is
        # never re-queued.
        if decision.queue_flags:
            reason = "; ".join(decision.queue_flags)
            await self._queue.upsert(
                decision.entity_id,
                name=str(row.get("canonical_name") or ""),
                type=str(primary_type),
                structural_degree=sig.effective_degree,
                doc_count=sig.doc_count,
                reason=reason,
                batch_id=batch_id,
            )
            report.queued += 1

    async def _surface_match_conflicts(
        self,
        source_id: str,
        report: TriageRunReport,
        batch_id: str,
        row_by_id: Dict[str, Dict[str, Any]],
    ) -> None:
        """B1: queue review-band match proposals touching this batch.

        Read-only proposal surfacing — NEVER auto-merges (only the K.5
        ``auto_merge`` band may be applied, and that path is the operator's via
        the resolution hub, not triage). A ``review`` candidate whose winner OR
        loser is in this batch is queued as a match-conflict row so the operator
        sees it alongside the other triage edge cases. Dedup-by-entity keeps it
        idempotent across re-runs.
        """
        if self._candidates is None:
            return
        try:
            proposals = await self._candidates.propose_candidates()
        except Exception as exc:
            logger.warning("Triage[{}] match step skipped: {}", batch_id, exc)
            return
        batch_set = set(row_by_id.keys())
        for cand in getattr(proposals, "review", []) or []:
            winner = getattr(cand, "winner_id", "") or getattr(cand, "id_a", "")
            loser = getattr(cand, "loser_id", "") or getattr(cand, "id_b", "")
            target = winner if winner in batch_set else (
                loser if loser in batch_set else None
            )
            if target is None:
                continue
            row = row_by_id.get(target, {})
            reason = (
                f"match conflict: '{getattr(cand, 'name_a', '')}' ~ "
                f"'{getattr(cand, 'name_b', '')}' "
                f"({getattr(cand, 'method', '?')} {getattr(cand, 'score', 0):.2f})"
            )
            # Pass degree/doc_count=0 (we only know an entity is in a fuzzy-match
            # pair, not its degree) with merge_reason=True so this NEVER clobbers
            # a prior status-flag row's real signals/reason for the same entity:
            # the queue preserves a non-zero degree/doc_count under the
            # zero-as-unknown rule and APPENDS the match-conflict reason rather
            # than replacing the status-flag reason (Q.4 review minor).
            await self._queue.upsert(
                target,
                name=str(row.get("canonical_name") or ""),
                type=str(row.get("primary_type") or row.get("entity_type") or ""),
                structural_degree=0,
                doc_count=0,
                reason=reason,
                batch_id=batch_id,
                merge_reason=True,
            )
            report.match_conflicts += 1
