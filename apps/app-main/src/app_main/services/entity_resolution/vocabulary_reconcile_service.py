"""Reconcile the graph's named entities against TOOI — the production caller (D2).

`VocabularyReconciler.reconcile_entity` existed since K.4 and had no production
caller; `reference_entity` held zero rows on `staging`. This is the batch door
the owner chose (2026-09-26): explicit, repeatable, nothing runs unasked.

Scope, also the owner's: `government_organization`, `organization` and
`administrative_area`. The last is unmeasured territory — whether `Gemeente
Opmeer` is the place or its government, and names two municipalities share
(`Bergen`). The report therefore splits every count by type, and the precision
guard refuses a name that two register rows share, so an ambiguity is recorded,
not resolved.

`dry_run` computes the whole report and writes nothing. A real run writes only
what `reconcile_entity` links — a union onto `external_ids`/`aliases`, never
touching `canonical_name`, `entity_type` or the identity key.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
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


@dataclass
class ReconcileReport:
    dry_run: bool
    scanned: int = 0
    by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    linked: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)


async def _active_entities(types: Sequence[str]) -> List[Dict[str, Any]]:
    return await execute_query(
        "SELECT id, canonical_name, entity_type, external_ids, aliases FROM entity "
        "WHERE status = 'active' AND entity_type IN $types;",
        {"types": list(types)},
    )


async def reconcile_active_entities(
    reconciler: VocabularyReconciler,
    *,
    types: Sequence[str] = DEFAULT_TYPES,
    dry_run: bool = True,
    rows: Optional[List[Dict[str, Any]]] = None,
) -> ReconcileReport:
    """Reconcile every active entity of ``types``; see the module docstring.

    ``reconciler`` must be built WITHOUT a repository for a dry run — the
    reconciler writes whenever it has one — and with the entity repository for a
    real run. ``rows`` replaces the database read (tests).
    """
    rows = rows if rows is not None else await _active_entities(types)
    report = ReconcileReport(dry_run=dry_run, scanned=len(rows))
    tally: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        ext = row.get("external_ids")
        entity = Entity(
            id=str(row["id"]),
            canonical_name=row["canonical_name"],
            entity_type=row["entity_type"],
            # One row on staging stores `{}` here; the repository's union treats
            # it as empty, and so does this.
            external_ids=list(ext) if isinstance(ext, list) else [],
            aliases=list(row.get("aliases") or []),
        )
        result = await reconciler.reconcile_entity(entity)
        etype = row["entity_type"]
        if result.linked:
            tally[etype]["linked"] += 1
            report.linked.append({
                "id": str(row["id"]), "name": row["canonical_name"], "type": etype,
                "uri": result.external_ids[0],
                "matched_on": result.candidates[0].matched_on if result.candidates else "",
            })
        elif result.candidates:
            tally[etype][result.reason] += 1
            report.candidates.append({
                "id": str(row["id"]), "name": row["canonical_name"], "type": etype,
                "reason": result.reason,
                "candidates": [
                    {"uri": c.external_uri, "label": c.canonical_name,
                     "confidence": c.confidence, "matched_on": c.matched_on}
                    for c in result.candidates
                ],
            })
        else:
            tally[etype]["no_match"] += 1
    report.by_type = {t: dict(c) for t, c in tally.items()}
    logger.info(
        "reconcile_active_entities: dry_run={d} scanned={n} linked={l} candidates={c}",
        d=dry_run, n=report.scanned, l=len(report.linked), c=len(report.candidates),
    )
    return report


__all__ = ["DEFAULT_TYPES", "ReconcileReport", "reconcile_active_entities"]
