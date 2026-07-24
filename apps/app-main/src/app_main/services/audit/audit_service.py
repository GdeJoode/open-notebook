"""LLM-free quality audit for a notebook (Track F.1).

Runs six cheap, always-on checks over a notebook's knowledge graph and persists
the results to ``audit_findings`` (migration 75). No model calls, no job queue —
the work is bounded graph reads plus in-Python predicates, so a ``POST`` runs it
inline (mirrors the orphans dashboard).

Design: the CHECKS are pure functions over already-fetched entity / source /
relation / orphan lists (module-level, unit-testable with plain lists — no DB).
:class:`AuditService` only does the I/O: fetch the notebook's graph, run every
check best-effort (a broken check yields a ``check_error`` finding, never fails
the run), write a snapshot under one ``run_id``, and prune old runs (keep-N,
F-D2 latest-wins).

A per-run **marker** row (``check_id == _RUN_MARKER``) anchors every run — even a
clean one that produced no findings — so :meth:`latest` always reflects the most
recent run and never returns a stale older run's findings. The marker is filtered
out of the returned findings.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from loguru import logger
from shared.models.audit import AuditFinding, AuditRunResponse
from shared.models.entity import Entity, Relation
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories import EntityRepository, SourceRepository

from app_main.config import AuditThresholds, get_audit_thresholds

#: check_id of the per-run marker row (anchors a run; filtered from findings).
_RUN_MARKER = "_run"

#: How many recent runs to keep per notebook before pruning older ones.
_KEEP_RUNS = 5

#: Entity types treated as "fallback"/generic for the schema-drift ratio.
_FALLBACK_TYPES = frozenset({"unknown", "entity", "thing", "misc", "other", ""})


def _detail(payload: Dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# The six checks — pure functions over fetched lists (no I/O, unit-testable).
# ---------------------------------------------------------------------------


def check_citation_completeness(
    entities: Sequence[Entity], _t: AuditThresholds
) -> List[AuditFinding]:
    """Active/reference entities with NO source documents (error)."""
    out: List[AuditFinding] = []
    for e in entities:
        if e.status not in ("active", "reference"):
            continue
        # Guard NONE before len (mirrors the SurrealDB array::len(NONE) rule).
        if not (e.source_documents or []):
            out.append(
                AuditFinding(
                    check_id="citation_completeness",
                    severity="error",
                    title=f"Entity '{e.canonical_name}' has no source citation",
                    subject=str(e.id),
                )
            )
    return out


def check_stale_sources(
    sources: Sequence[Dict], t: AuditThresholds, *, now: datetime
) -> List[AuditFinding]:
    """Sources not updated in ``stale_days`` (info). ``sources`` are dicts with
    ``id`` / ``updated`` / ``title`` (a bounded per-notebook fetch)."""
    out: List[AuditFinding] = []
    cutoff = now.timestamp() - t.stale_days * 86400
    for s in sources:
        updated = s.get("updated")
        ts = _as_timestamp(updated)
        if ts is not None and ts < cutoff:
            out.append(
                AuditFinding(
                    check_id="stale_sources",
                    severity="info",
                    title=f"Source '{s.get('title') or s.get('id')}' is stale "
                    f"(> {t.stale_days}d)",
                    subject=str(s.get("id")),
                )
            )
    return out


def check_low_confidence_survivors(
    entities: Sequence[Entity], t: AuditThresholds
) -> List[AuditFinding]:
    """Entities that barely cleared the extraction filter floor (warn)."""
    lo = t.filter_min_confidence
    hi = t.filter_min_confidence + t.low_confidence_band
    out: List[AuditFinding] = []
    for e in entities:
        if e.status not in ("active", "reference"):
            continue
        if lo <= e.confidence < hi:
            out.append(
                AuditFinding(
                    check_id="low_confidence_survivors",
                    severity="warn",
                    title=f"Entity '{e.canonical_name}' barely survived the filter "
                    f"(confidence {e.confidence:.2f})",
                    subject=str(e.id),
                )
            )
    return out


def check_long_pending_orphans(
    orphans: Sequence[Dict], t: AuditThresholds, *, now: datetime
) -> List[AuditFinding]:
    """Pending orphans older than ``orphan_days`` (warn). ``orphans`` are the
    orphans-dashboard rows (``id`` + ``first_orphaned_at``/``created_at``)."""
    out: List[AuditFinding] = []
    cutoff = now.timestamp() - t.orphan_days * 86400
    for o in orphans:
        ts = _as_timestamp(o.get("first_orphaned_at") or o.get("created_at"))
        # No timestamp → still flag (an undated long-pending orphan is a finding);
        # a dated one only when older than the cutoff.
        if ts is None or ts < cutoff:
            out.append(
                AuditFinding(
                    check_id="long_pending_orphans",
                    severity="warn",
                    title=f"Orphan entity '{o.get('name') or o.get('id')}' has "
                    f"been pending > {t.orphan_days}d",
                    subject=str(o.get("id")),
                )
            )
    return out


def check_community_cohesion(
    entities: Sequence[Entity],
    relations: Sequence[Relation],
    t: AuditThresholds,
) -> List[AuditFinding]:
    """Communities whose internal-edge density is below the floor (info).

    Density = internal edges / C(n, 2) for a community of ``n >= cohesion_min_size``
    members. A low ratio means the community is loosely connected — a possible
    over-merge or a weak cluster worth review.
    """
    community_of: Dict[str, int] = {}
    members: Dict[int, int] = {}
    for e in entities:
        if e.community_id is None:
            continue
        community_of[str(e.id)] = e.community_id
        members[e.community_id] = members.get(e.community_id, 0) + 1

    internal_edges: Dict[int, int] = {}
    for r in relations:
        # Relation surfaces the edge endpoints as in_entity / out_entity
        # (aliases for the RELATE in/out record ids).
        cs = community_of.get(str(r.in_entity)) if r.in_entity else None
        ct = community_of.get(str(r.out_entity)) if r.out_entity else None
        if cs is not None and cs == ct:
            internal_edges[cs] = internal_edges.get(cs, 0) + 1

    out: List[AuditFinding] = []
    for cid, n in members.items():
        if n < t.cohesion_min_size:
            continue
        possible = n * (n - 1) / 2
        density = (internal_edges.get(cid, 0) / possible) if possible else 0.0
        if density < t.cohesion_floor:
            out.append(
                AuditFinding(
                    check_id="community_cohesion",
                    severity="info",
                    title=f"Community {cid} has low cohesion "
                    f"(density {density:.2f}, {n} members)",
                    subject=f"community:{cid}",
                    count=n,
                    detail=_detail({"density": round(density, 4), "members": n}),
                )
            )
    return out


def check_schema_drift(
    entities: Sequence[Entity], t: AuditThresholds
) -> List[AuditFinding]:
    """Flag when the fallback-type entity ratio exceeds ``drift_ratio`` (info)."""
    active = [e for e in entities if e.status in ("active", "reference")]
    if not active:
        return []
    fallback = sum(
        1 for e in active if (e.entity_type or "").strip().lower() in _FALLBACK_TYPES
    )
    ratio = fallback / len(active)
    if ratio > t.drift_ratio:
        return [
            AuditFinding(
                check_id="schema_drift",
                severity="info",
                title=f"Schema drift: {ratio:.0%} of entities use a fallback type "
                f"({fallback}/{len(active)})",
                count=fallback,
                detail=_detail({"ratio": round(ratio, 4), "fallback": fallback,
                                "total": len(active)}),
            )
        ]
    return []


def _as_timestamp(value) -> Optional[float]:
    """Best-effort POSIX timestamp from a datetime / ISO string, else None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


class AuditService:
    """Run + persist the six LLM-free notebook audit checks (F.1)."""

    def __init__(
        self,
        entity_repo: Optional[EntityRepository] = None,
        source_repo: Optional[SourceRepository] = None,
        config: Optional[SurrealDBConfig] = None,
    ) -> None:
        self.entity_repo = entity_repo or EntityRepository()
        self.source_repo = source_repo or SourceRepository()
        self.config = config

    async def run_all(self, notebook_id: str) -> AuditRunResponse:
        """Run every check over the notebook, persist a snapshot, return it."""
        from shared.models.export import ExportFilter

        thresholds = get_audit_thresholds()
        now = datetime.now(timezone.utc)

        # A permissive filter so the audit sees ALL entities (orphans, archived,
        # low-confidence) — the checks decide what is a finding, not the fetch.
        permissive = ExportFilter(
            min_connections=0,
            min_confidence=0.0,
            include_orphans=True,
            include_archived=True,
            entity_types=None,
        )
        entities = await self.entity_repo.list_entities_for_notebook(
            notebook_id, permissive
        )
        relations = await self.entity_repo.list_relations_for_notebook(
            notebook_id, permissive
        )
        orphans = await self._load_orphans(notebook_id)
        sources = await self._load_sources(notebook_id)

        findings: List[AuditFinding] = []
        checks = [
            ("citation_completeness", lambda: check_citation_completeness(entities, thresholds)),
            ("stale_sources", lambda: check_stale_sources(sources, thresholds, now=now)),
            ("low_confidence_survivors", lambda: check_low_confidence_survivors(entities, thresholds)),
            ("long_pending_orphans", lambda: check_long_pending_orphans(orphans, thresholds, now=now)),
            ("community_cohesion", lambda: check_community_cohesion(entities, relations, thresholds)),
            ("schema_drift", lambda: check_schema_drift(entities, thresholds)),
        ]
        for check_id, fn in checks:
            try:
                findings.extend(fn())
            except Exception as exc:  # noqa: BLE001 — one broken check ≠ failed run
                logger.warning("audit check {c} failed: {e}", c=check_id, e=exc)
                findings.append(
                    AuditFinding(
                        check_id="check_error",
                        severity="warn",
                        title=f"Audit check '{check_id}' could not run",
                        detail=_detail({"check_id": check_id, "error": str(exc)}),
                    )
                )

        run_id = uuid.uuid4().hex
        await self._persist(notebook_id, run_id, findings)
        return AuditRunResponse(
            notebook_id=notebook_id,
            run_id=run_id,
            findings=findings,
            created=now.isoformat(),
        )

    async def latest(self, notebook_id: str) -> AuditRunResponse:
        """Return the most recent run's findings (empty if never run)."""
        marker_rows = await execute_query(
            "SELECT run_id, created FROM audit_findings "
            "WHERE notebook = $nb AND check_id = $marker "
            "ORDER BY created DESC LIMIT 1",
            {"nb": ensure_record_id(notebook_id), "marker": _RUN_MARKER},
            self.config,
        )
        if not marker_rows:
            return AuditRunResponse(notebook_id=notebook_id, run_id="", findings=[])
        run_id = str(marker_rows[0].get("run_id"))
        created = marker_rows[0].get("created")
        rows = await execute_query(
            "SELECT * FROM audit_findings "
            "WHERE notebook = $nb AND run_id = $run AND check_id != $marker",
            {
                "nb": ensure_record_id(notebook_id),
                "run": run_id,
                "marker": _RUN_MARKER,
            },
            self.config,
        )
        findings = [
            AuditFinding(
                check_id=r.get("check_id"),
                severity=r.get("severity"),
                title=r.get("title"),
                detail=r.get("detail"),
                subject=str(r["subject"]) if r.get("subject") else None,
                count=r.get("count"),
            )
            for r in (rows or [])
        ]
        # Sort by SEVERITY RANK (error > warn > info), not the raw string — a
        # `ORDER BY severity` in SurrealQL is alphabetical (error < info < warn),
        # which would render info above the more-urgent warn.
        _rank = {"error": 0, "warn": 1, "info": 2}
        findings.sort(key=lambda f: _rank.get(f.severity, 3))
        return AuditRunResponse(
            notebook_id=notebook_id,
            run_id=run_id,
            findings=findings,
            created=str(created) if created else None,
        )

    async def append_findings(
        self, notebook_id: str, run_id: str, findings: List[AuditFinding]
    ) -> None:
        """Append findings to an EXISTING run (no marker, no prune).

        Used by the deep audit (F.3) to enrich the latest cheap run with the
        on-demand checks, so ``latest`` returns the cheap + deep findings under
        one ``run_id`` rather than a deep run superseding the cheap one.
        """
        nb = ensure_record_id(notebook_id)
        for f in findings:
            await execute_query(
                "CREATE audit_findings CONTENT $data",
                {
                    "data": {
                        "notebook": nb,
                        "run_id": run_id,
                        "check_id": f.check_id,
                        "severity": f.severity,
                        "title": f.title,
                        "detail": f.detail,
                        "subject": f.subject,
                        "count": f.count,
                    }
                },
                self.config,
            )

    async def _persist(
        self, notebook_id: str, run_id: str, findings: List[AuditFinding]
    ) -> None:
        """Write the run marker + every finding, then prune old runs (keep-N)."""
        nb = ensure_record_id(notebook_id)
        # Marker anchors the run even when there are zero findings.
        rows: List[Dict] = [
            {
                "notebook": nb,
                "run_id": run_id,
                "check_id": _RUN_MARKER,
                "severity": "info",
                "title": "Audit run",
                "count": len(findings),
            }
        ]
        for f in findings:
            rows.append(
                {
                    "notebook": nb,
                    "run_id": run_id,
                    "check_id": f.check_id,
                    "severity": f.severity,
                    "title": f.title,
                    "detail": f.detail,
                    "subject": f.subject,
                    "count": f.count,
                }
            )
        for row in rows:
            await execute_query(
                "CREATE audit_findings CONTENT $data", {"data": row}, self.config
            )
        await self._prune(notebook_id)

    async def _prune(self, notebook_id: str) -> None:
        """Delete rows from all but the newest ``_KEEP_RUNS`` runs."""
        markers = await execute_query(
            "SELECT run_id, created FROM audit_findings "
            "WHERE notebook = $nb AND check_id = $marker "
            "ORDER BY created DESC",
            {"nb": ensure_record_id(notebook_id), "marker": _RUN_MARKER},
            self.config,
        )
        keep = {str(m.get("run_id")) for m in (markers or [])[:_KEEP_RUNS]}
        stale = [str(m.get("run_id")) for m in (markers or [])[_KEEP_RUNS:]]
        for run_id in stale:
            if run_id in keep:
                continue
            await execute_query(
                "DELETE audit_findings WHERE notebook = $nb AND run_id = $run",
                {"nb": ensure_record_id(notebook_id), "run": run_id},
                self.config,
            )

    async def _load_orphans(self, notebook_id: str) -> List[Dict]:
        try:
            rows = await self.entity_repo.list_orphans_with_status(
                notebook_id, "pending_reconnect"
            )
            return [r if isinstance(r, dict) else r.__dict__ for r in (rows or [])]
        except Exception as exc:  # noqa: BLE001 — best-effort; empty on failure
            logger.warning("audit: orphan fetch failed for {nb}: {e}", nb=notebook_id, e=exc)
            return []

    async def _load_sources(self, notebook_id: str) -> List[Dict]:
        """Load the notebook's sources with just the fields stale_sources needs."""
        source_ids = await self.source_repo.load_notebook_source_ids(notebook_id)
        if not source_ids:
            return []
        rows = await execute_query(
            "SELECT id, updated, title FROM source WHERE id IN $ids",
            {"ids": [ensure_record_id(s) for s in source_ids]},
            self.config,
        )
        return list(rows or [])


__all__ = ["AuditService"]
