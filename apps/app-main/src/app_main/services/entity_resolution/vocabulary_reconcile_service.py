"""Reconcile the graph's named entities against TOOI — the production caller (D2).

`VocabularyReconciler.reconcile_entity` existed since K.4 and had no production
caller; `reference_entity` held zero rows on `staging`. This is the batch door
the owner chose (2026-09-26): explicit, repeatable, nothing runs unasked.

**Scope** (owner decisions, 2026-09-26):

* types — `government_organization`, `organization`, `administrative_area`. The
  last links a place entity (`Roerdalen`, `Gemeente Weert`) to its municipality's
  TOOI record: the owner chose that reading of place-vs-government;
* statuses — `active` and `reference`, the live graph as the audit and source
  code treat it (`organization` has 0 active rows and 120 reference ones);
* an ENDED organisation is a candidate, never a link (see the reconciler);
* a link writes the URI and the record's current names, not its alias history.

**Candidates reach a human.** On a real run every refused match — ambiguous,
one-sided, ended — is appended to the entity's open ``triage_queue`` row (Q.4),
the queue a curator reads, with every candidate URI in the reason. Accepting one
of them is not a queue action yet (a queue decision pins the entity's STATUS);
that door is PC.5's. A dry run writes nothing, queue included.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
from shared.models.entity import Entity
from surrealdb_service.connection import execute_query

from app_main.services.entity_resolution.vocabulary_reconciler import (
    VocabularyReconciler,
)

#: The owner's scope for D2.
DEFAULT_TYPES: Sequence[str] = (
    "government_organization",
    "organization",
    "administrative_area",
)
DEFAULT_STATUSES: Sequence[str] = ("active", "reference")

#: The prefix every reconcile reason carries in `triage_queue`.
QUEUE_REASON_PREFIX = "tooi"


@dataclass
class ReconcileReport:
    dry_run: bool
    scanned: int = 0
    #: type -> status -> rows scanned, including types that scanned none.
    scanned_by: Dict[str, Dict[str, int]] = field(default_factory=dict)
    #: type -> outcome -> count.
    by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    linked: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    no_match: List[Dict[str, Any]] = field(default_factory=list)
    write_failed: List[Dict[str, Any]] = field(default_factory=list)
    skipped_no_name: int = 0
    queued: int = 0


async def _rows(types: Sequence[str], statuses: Sequence[str]) -> List[Dict[str, Any]]:
    return await execute_query(
        "SELECT id, canonical_name, entity_type, status, external_ids, aliases FROM entity "
        "WHERE status IN $statuses AND entity_type IN $types;",
        {"types": list(types), "statuses": list(statuses)},
    )


def _queue_reason(reason: str, candidates: List[Dict[str, Any]]) -> str:
    """One line a curator can act on: why, and every URI with its label and end."""
    parts = [
        f"{c['label']} <{c['uri']}>" + (f" ended {c['valid_until']}" if c["valid_until"] else "")
        for c in candidates
    ]
    return f"{QUEUE_REASON_PREFIX}: {reason}: " + " | ".join(parts)


async def reconcile_active_entities(
    reconciler: VocabularyReconciler,
    *,
    types: Sequence[str] = DEFAULT_TYPES,
    statuses: Sequence[str] = DEFAULT_STATUSES,
    dry_run: bool = True,
    queue: Optional[Any] = None,
    rows: Optional[List[Dict[str, Any]]] = None,
) -> ReconcileReport:
    """Reconcile every entity of ``types`` in ``statuses``; see the module docstring.

    ``reconciler`` must be built WITHOUT a repository for a dry run — it writes
    whenever it has one — and with the entity repository for a real run.
    ``queue`` (a ``TriageQueueRepository``) receives the candidates on a real run.
    ``rows`` replaces the database read (tests).
    """
    rows = rows if rows is not None else await _rows(types, statuses)
    report = ReconcileReport(dry_run=dry_run)
    scanned: Dict[str, Counter] = {t: Counter() for t in types}
    tally: Dict[str, Counter] = defaultdict(Counter)
    batch = f"reconcile-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    for row in rows:
        etype = row.get("entity_type") or ""
        scanned.setdefault(etype, Counter())[row.get("status") or ""] += 1
        name = row.get("canonical_name")
        if not name:
            # 19 `reference` rows on staging carry no name; `Entity` refuses them.
            report.skipped_no_name += 1
            continue
        report.scanned += 1
        ext = row.get("external_ids")
        entity = Entity(
            id=str(row["id"]),
            canonical_name=name,
            entity_type=etype,
            # Before migration 82 staging stored `{}` here on every row.
            external_ids=list(ext) if isinstance(ext, list) else [],
            aliases=list(row.get("aliases") or []),
        )
        result = await reconciler.reconcile_entity(entity)
        base = {"id": str(row["id"]), "name": name, "type": etype, "status": row.get("status")}
        if result.linked:
            m = result.candidates[0]
            entry = {**base, "uri": m.external_uri, "label": m.canonical_name,
                     "matched_on": m.matched_on}
            if result.persisted is False:
                tally[etype]["write_failed"] += 1
                report.write_failed.append(entry)
            else:
                tally[etype]["linked"] += 1
                report.linked.append(entry)
        elif result.candidates:
            tally[etype][result.reason] += 1
            cands = [
                {"uri": c.external_uri, "label": c.canonical_name,
                 "confidence": c.confidence, "matched_on": c.matched_on,
                 "valid_until": c.valid_until}
                for c in result.candidates
            ]
            report.candidates.append({**base, "reason": result.reason, "candidates": cands})
            if not dry_run and queue is not None:
                try:
                    await queue.upsert(
                        entity_id=str(row["id"]), name=name, type=etype,
                        structural_degree=0, doc_count=0,
                        reason=_queue_reason(result.reason, cands),
                        batch_id=batch, merge_reason=True,
                    )
                    report.queued += 1
                except Exception as exc:  # the report still carries the candidate
                    logger.error("reconcile: could not queue {id}: {e}", id=row["id"], e=exc)
        else:
            tally[etype]["no_match"] += 1
            report.no_match.append(base)
    report.scanned_by = {t: dict(c) for t, c in scanned.items()}
    report.by_type = {t: dict(c) for t, c in tally.items()}
    logger.info(
        "reconcile: dry_run={d} scanned={n} linked={l} candidates={c} queued={q} "
        "write_failed={w}",
        d=dry_run, n=report.scanned, l=len(report.linked), c=len(report.candidates),
        q=report.queued, w=len(report.write_failed),
    )
    return report


__all__ = [
    "DEFAULT_STATUSES", "DEFAULT_TYPES", "QUEUE_REASON_PREFIX", "ReconcileReport",
    "reconcile_active_entities",
]
