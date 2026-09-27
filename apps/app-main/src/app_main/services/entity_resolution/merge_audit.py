"""Record every applied merge in ``resolution_log`` (Track PC.5b).

``resolution_log`` (migration 36) had a reader — ``GET /api/resolution-log`` and
the Knowledge Graph's Resolution Log tab — and no writer: 0 rows on staging.
Meanwhile merges were applied by ``POST /entity-resolution/apply`` and, unattended,
by the OKF import, and nothing recorded which ones a curator approved and which a
threshold decided. This writes one row per merged pair.

* ``match_method`` carries the ORIGIN and the method: ``curator_approved:fuzzy``,
  ``auto_merge:fold_equal``, ``curator_approved:recanonicalization_plan``;
* ``status`` is ``accepted`` for both origins. The tab offers accept/reject
  buttons on ``pending`` and ``auto_accepted`` rows, and a log status cannot undo
  a merge — a reject button on an applied merge would promise something that
  does not happen.

Best-effort: the merge has been applied by the time this runs; failing to log it
is reported, never raised.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from loguru import logger
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query

CURATOR_APPROVED = "curator_approved"
AUTO_MERGE = "auto_merge"
ORIGINS = (CURATOR_APPROVED, AUTO_MERGE)


async def record_merges(
    *,
    winner_id: str,
    merged_loser_ids: Iterable[str],
    names: Dict[str, str],
    types: Dict[str, str],
    origin: str,
    method: Optional[str] = None,
    score: Optional[float] = None,
    evidence: Optional[str] = None,
    config: Optional[SurrealDBConfig] = None,
) -> int:
    """One ``resolution_log`` row per merged loser; returns how many were written.

    ``names`` / ``types`` map entity ids to their name and type as they were
    BEFORE the merge (a merged loser's row still exists, soft-merged, but the
    caller has them already).
    """
    written = 0
    for loser in merged_loser_ids:
        try:
            await execute_query(
                "CREATE resolution_log CONTENT $row RETURN NONE;",
                {"row": {
                    "entity_a_text": names.get(winner_id, winner_id),
                    "entity_b_text": names.get(loser, loser),
                    "entity_a_label": types.get(winner_id) or "UNKNOWN",
                    "entity_b_label": types.get(loser) or "UNKNOWN",
                    "match": True,
                    "confidence": float(score) if score is not None else 1.0,
                    "match_method": f"{origin}:{method or 'unspecified'}",
                    "match_reasoning": evidence or "",
                    "status": "accepted",
                }},
                config,
            )
            written += 1
        except Exception as exc:  # the merge stands; say that the log did not
            logger.warning(
                "merge audit: could not log {w} <- {l}: {e}", w=winner_id, l=loser, e=exc
            )
    return written


__all__ = ["AUTO_MERGE", "CURATOR_APPROVED", "ORIGINS", "record_merges"]
