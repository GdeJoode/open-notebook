"""Q.2a backfill: collapse duplicate relation edges into one cross-doc edge.

Before the Q.2a forward fix, ``entity_persistence_service.py`` ran an
UNCONDITIONAL ``RELATE`` per extracted relation, so the SAME edge re-appearing
in another document created a DUPLICATE row (70 live dups confirmed), each
carrying a single document in ``source_documents``. Per-edge cross-document
recurrence was therefore never tracked — which is exactly the signal the
operator's weak-edge promotion model (Q.2) reads (a ``RELATED`` edge that recurs
across documents earns ``active``).

This script collapses those existing duplicates: it groups ACTIVE relations by
``(in, out, relation_type)``, keeps ONE survivor per group, unions every dup's
``source_documents`` onto it (so the survivor carries the cumulative provenance),
keeps the MAX ``confidence``, merges ``properties`` (existing keys retained, dup
keys overlaid — no clobber), and deletes the redundant rows.

Properties:

* **Idempotent** — only groups with MORE THAN ONE edge are collapsed; a second
  run finds every group already singular and changes nothing. Safe to re-run.
* **Non-destructive to non-dups** — a ``(in, out, relation_type)`` group with a
  single edge is never touched. Endpoints (``in``/``out``) and ``relation_type``
  are preserved on the survivor; only redundant duplicates are deleted.
* **Provenance-preserving** — the survivor's ``source_documents`` is the union
  across the whole group, so no document attribution is lost in the collapse.
* **Dry-run** — ``--dry-run`` reports how many groups would collapse and how
  many edges would be deleted, writing NOTHING.
* **Logs progress** — one summary line (groups collapsed / edges deleted).

Usage (run inside the open_notebook app environment, e.g. the app container or
``uv run --project apps/app-main``)::

    python scripts/backfill_relation_merge.py
    python scripts/backfill_relation_merge.py --dry-run   # count only

Args:
    --dry-run   Report duplicate groups + edges that would be deleted; write
                nothing.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from loguru import logger
from surrealdb_service.connection import execute_query

# A group key is (in_id, out_id, relation_type) — the edge identity Q.2a
# enforces one row per.
GroupKey = Tuple[str, str, str]


def _union_preserve_order(*lists: List[Any]) -> List[Any]:
    """Union the lists, dedup, preserve first-seen order.

    Mirrors ``array::union`` semantics so the collapsed survivor's
    ``source_documents`` matches what the forward Q.2a write produces.
    """
    seen: set = set()
    out: List[Any] = []
    for lst in lists:
        for item in lst or []:
            key = str(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


async def _load_active_relations() -> List[Dict[str, Any]]:
    """Load all ACTIVE relation rows with the fields the collapse needs."""
    return await execute_query(
        "SELECT id, in, out, relation_type, source_documents, confidence, "
        "properties FROM relation WHERE status = 'active';"
    )


def _group_duplicates(
    rows: List[Dict[str, Any]],
) -> Dict[GroupKey, List[Dict[str, Any]]]:
    """Group active relations by ``(in, out, relation_type)``.

    Returns only the groups with MORE THAN ONE edge (the true duplicates);
    singular groups are left out so they are never touched.
    """
    grouped: Dict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        in_id = str(row.get("in"))
        out_id = str(row.get("out"))
        rel_type = row.get("relation_type") or ""
        grouped[(in_id, out_id, rel_type)].append(row)
    return {k: v for k, v in grouped.items() if len(v) > 1}


async def _collapse_group(edges: List[Dict[str, Any]]) -> int:
    """Collapse one duplicate group into its survivor; delete the rest.

    Keeps the first edge as the survivor, unions provenance/properties/confidence
    from the whole group onto it, then deletes the redundant edges. Returns the
    number of edges deleted.
    """
    survivor = edges[0]
    redundant = edges[1:]

    merged_sources = _union_preserve_order(
        *[e.get("source_documents") or [] for e in edges]
    )
    # properties: union with first-edge precedence per key but every key
    # surfaced (no clobber of a signal a dup carried). Survivor first so its
    # keys win on conflict; later dups only contribute keys the survivor lacks.
    merged_properties: Dict[str, Any] = {}
    for e in reversed(edges):
        merged_properties.update(e.get("properties") or {})
    merged_confidence = max(
        (e.get("confidence") for e in edges if e.get("confidence") is not None),
        default=survivor.get("confidence"),
    )

    await execute_query(
        "UPDATE type::thing($eid) SET "
        "source_documents = $sources, "
        "confidence = $confidence, "
        "properties = $properties;",
        {
            "eid": str(survivor["id"]),
            "sources": merged_sources,
            "confidence": merged_confidence,
            "properties": merged_properties,
        },
    )
    for e in redundant:
        await execute_query(
            "DELETE type::thing($eid);", {"eid": str(e["id"])}
        )
    return len(redundant)


async def _backfill(dry_run: bool) -> Dict[str, int]:
    """Collapse duplicate relation edges; return a summary dict.

    Returns ``{groups_collapsed, edges_deleted, groups_seen}``.
    """
    rows = await _load_active_relations()
    dup_groups = _group_duplicates(rows)
    groups_seen = len(dup_groups)
    edges_to_delete = sum(len(v) - 1 for v in dup_groups.values())

    logger.info(
        "backfill_relation_merge: {} active relation(s), {} duplicate "
        "(in,out,relation_type) group(s) carrying {} redundant edge(s)",
        len(rows),
        groups_seen,
        edges_to_delete,
    )

    if dry_run:
        for key, edges in list(dup_groups.items())[:10]:
            logger.info(
                "  would collapse {} -> 1 edge for {}",
                len(edges),
                key,
            )
        return {
            "groups_collapsed": 0,
            "edges_deleted": 0,
            "groups_seen": groups_seen,
        }

    if groups_seen == 0:
        logger.info("Nothing to collapse — every edge is already unique.")
        return {"groups_collapsed": 0, "edges_deleted": 0, "groups_seen": 0}

    collapsed = 0
    deleted = 0
    for edges in dup_groups.values():
        deleted += await _collapse_group(edges)
        collapsed += 1

    logger.info(
        "backfill_relation_merge DONE: collapsed {} group(s), deleted {} "
        "redundant edge(s)",
        collapsed,
        deleted,
    )
    return {
        "groups_collapsed": collapsed,
        "edges_deleted": deleted,
        "groups_seen": groups_seen,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse duplicate relation edges (Q.2a)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count duplicate groups + redundant edges; write nothing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_backfill(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
