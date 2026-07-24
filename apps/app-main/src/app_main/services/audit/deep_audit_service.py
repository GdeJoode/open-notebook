"""On-demand "deep" audit checks (Track F.3) — checks 7 & 8.

Two heavier, on-demand checks that enrich the F.1 audit. Kept OFF the always-on
path (a separate ``POST …/audit/deep`` route) so the cheap audit stays bounded.

Reuse over reinvention (design decision, deviates from the plan's "2 new LLM
checks")
=========================================================================
* **conflicting_facts (check 7)** — the codebase ALREADY has an LLM contradiction
  judge (:mod:`contradiction_judge_service`, Track Z.2) that judges whether two
  RELATED sources ``contradicts``/``reinforces``/``neutral`` and persists the
  confident ones as ``source_verdict`` edges (migration 69). Re-running an LLM
  here would duplicate that machinery and create a second, drifting contradiction
  path. Instead this check SURFACES the already-judged ``contradicts`` verdicts
  among the notebook's sources as audit findings — no redundant model call.
* **provenance_gaps (check 8)** — relations whose ``source_documents`` is empty
  carry no supporting evidence. This is deterministic (no LLM needed): flag them.

Net effect: the deep audit is LLM-free *by reuse*. The LLM cost was already paid
by Z.2 at ingest; the audit just reads its output. Both checks are fail-soft (a
query error yields a ``check_error`` finding, never a 500) and persist under the
F.1 ``audit_findings`` scheme, appended to the latest run so the F.1 ``GET``
returns cheap + deep together (distinguishable by ``check_id``).
"""

from __future__ import annotations

import json
from typing import List, Optional

from loguru import logger
from shared.models.audit import AuditFinding, AuditRunResponse
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query

from app_main.services.audit.audit_service import AuditService

#: check_ids for the two deep checks.
CHECK_CONFLICTING_FACTS = "conflicting_facts"
CHECK_PROVENANCE_GAPS = "provenance_gaps"

#: A DISTINCT error check_id for deep checks — must NOT collide with the cheap
#: run's ``check_error`` (audit_service), or the deep clear-before-append would
#: delete a legitimate cheap-check failure from the same run.
CHECK_DEEP_ERROR = "deep_check_error"

#: The deep check_ids whose rows a re-run CLEARs before re-appending (so a repeat
#: POST …/audit/deep replaces rather than duplicates). ONLY deep ids — never the
#: cheap ``check_error`` — so the clear can't erase a cheap-check failure.
_DEEP_CHECK_IDS = [CHECK_CONFLICTING_FACTS, CHECK_PROVENANCE_GAPS, CHECK_DEEP_ERROR]

#: Bound the number of findings per check so one pathological notebook can't
#: write thousands of rows.
_MAX_PER_CHECK = 200


class DeepAuditService:
    """Run the two on-demand deep checks and enrich the latest F.1 run (F.3)."""

    def __init__(
        self,
        audit_service: Optional[AuditService] = None,
        config: Optional[SurrealDBConfig] = None,
    ) -> None:
        self.audit_service = audit_service or AuditService(config=config)
        self.config = config

    async def run_deep(self, notebook_id: str) -> AuditRunResponse:
        """Run both deep checks, append to the latest run, return the combined."""
        # Establish a run to attach to: reuse the latest cheap run, else run the
        # cheap checks first so there is a coherent snapshot to enrich.
        latest = await self.audit_service.latest(notebook_id)
        if not latest.run_id:
            latest = await self.audit_service.run_all(notebook_id)
        run_id = latest.run_id

        deep: List[AuditFinding] = []
        for check_id, coro in (
            (CHECK_CONFLICTING_FACTS, self._conflicting_facts(notebook_id)),
            (CHECK_PROVENANCE_GAPS, self._provenance_gaps(notebook_id)),
        ):
            try:
                deep.extend(await coro)
            except Exception as exc:  # noqa: BLE001 — one broken check ≠ failed run
                logger.warning("deep audit {c} failed: {e}", c=check_id, e=exc)
                deep.append(
                    AuditFinding(
                        check_id=CHECK_DEEP_ERROR,
                        severity="warn",
                        title=f"Deep check '{check_id}' could not run",
                    )
                )

        # Clear any deep findings already attached to this run before appending,
        # so a repeat POST …/audit/deep REPLACES rather than duplicates (append is
        # a bare CREATE loop with no dedup). Scoped to the deep check_ids only, so
        # the cheap findings on the same run are untouched. Best-effort: a clear
        # failure must not break the run (worst case is a re-appearing duplicate,
        # not a 500).
        try:
            await execute_query(
                "DELETE audit_findings WHERE notebook = $nb AND run_id = $run "
                "AND check_id IN $ids",
                {
                    "nb": ensure_record_id(notebook_id),
                    "run": run_id,
                    "ids": _DEEP_CHECK_IDS,
                },
                self.config,
            )
        except Exception as exc:  # noqa: BLE001 — clear is best-effort
            logger.warning("deep audit clear-before-append failed: {e}", e=exc)
        await self.audit_service.append_findings(notebook_id, run_id, deep)
        # Return the full enriched snapshot (cheap + deep).
        return await self.audit_service.latest(notebook_id)

    async def _conflicting_facts(self, notebook_id: str) -> List[AuditFinding]:
        """Surface Z.2 ``contradicts`` verdicts among the notebook's sources."""
        source_ids = await self.audit_service.source_repo.load_notebook_source_ids(
            notebook_id
        )
        if not source_ids:
            return []
        ids = [ensure_record_id(s) for s in source_ids]
        rows = await execute_query(
            "SELECT in, out, verdict, confidence FROM source_verdict "
            "WHERE verdict = 'contradicts' AND in IN $ids AND out IN $ids",
            {"ids": ids},
            self.config,
        )
        out: List[AuditFinding] = []
        for r in (rows or [])[:_MAX_PER_CHECK]:
            src_a, src_b = str(r.get("in")), str(r.get("out"))
            out.append(
                AuditFinding(
                    check_id=CHECK_CONFLICTING_FACTS,
                    severity="warn",
                    title=f"Sources {src_a} and {src_b} were judged to contradict",
                    subject=src_a,
                    # json.dumps → valid JSON even when confidence is None (→ null),
                    # and safe against exotic ids (no manual quote interpolation).
                    detail=json.dumps(
                        {"other": src_b, "confidence": r.get("confidence")}
                    ),
                )
            )
        return out

    async def _provenance_gaps(self, notebook_id: str) -> List[AuditFinding]:
        """Relations with no supporting source documents (deterministic)."""
        from shared.models.export import ExportFilter

        permissive = ExportFilter(
            min_connections=0,
            min_confidence=0.0,
            include_orphans=True,
            include_archived=True,
            entity_types=None,
        )
        relations = await self.audit_service.entity_repo.list_relations_for_notebook(
            notebook_id, permissive
        )
        out: List[AuditFinding] = []
        for rel in relations:
            if rel.status not in ("active", "reference"):
                continue
            if not (rel.source_documents or []):
                out.append(
                    AuditFinding(
                        check_id=CHECK_PROVENANCE_GAPS,
                        severity="warn",
                        title=f"Relation '{rel.relation_type}' has no source "
                        f"evidence",
                        subject=str(rel.id),
                    )
                )
                if len(out) >= _MAX_PER_CHECK:
                    break
        return out


__all__ = ["DeepAuditService"]
