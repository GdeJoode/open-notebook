"""Populate `entity.name_key` before migration 79 can add its UNIQUE index (PC.3).

Migration 79 refuses on a populated table because the value cannot be computed in
SurrealQL: `normalize_entity_name` is Python — it strips leading articles,
canonicalises documented spelling variants and expands curated org aliases.

**This tool refuses rather than merges.** Two rows whose names normalise to one
key are the duplication PC.3 exists to remove, and merging them is destructive:
it picks a survivor, repoints relations and discards a name. That is a curator's
decision, and PC.2 shipped the door for it — a same-type pair equal under the fold
lands in the auto-merge band of the candidate queue. A backfill that merged
silently would be the failure PC.6 spent five review rounds making unreachable.

So: report the colliding groups, write nothing, exit non-zero. Resolve them
through the curator queue, then run this again.

Usage::

    SURREAL_DATABASE=staging uv run python scripts/backfill_name_key.py          # report only
    SURREAL_DATABASE=staging uv run python scripts/backfill_name_key.py --apply  # write
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from shared.utils.name_normalizer import normalize_entity_name
from surrealdb_service.connection import execute_query

_RECORD_ID = "id"


async def _load_rows() -> List[Dict[str, str]]:
    rows = await execute_query(
        "SELECT id, canonical_name, entity_type, status FROM entity;"
    )
    return [
        {
            "id": str(r[_RECORD_ID]),
            "canonical_name": str(r.get("canonical_name") or ""),
            "entity_type": str(r.get("entity_type") or ""),
            "status": str(r.get("status") or ""),
        }
        for r in (rows or [])
    ]


def plan(rows: List[Dict[str, str]]) -> Tuple[Dict[str, str], List[List[Dict[str, str]]]]:
    """Return `(id -> name_key, colliding groups)`.

    Grouped by `(name_key, entity_type)` because that is the index's shape: two
    rows sharing a key under DIFFERENT types are not a collision, and reporting
    them would send a curator after a non-problem.
    """
    keys: Dict[str, str] = {}
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = normalize_entity_name(row["canonical_name"])
        keys[row["id"]] = key
        groups[(key, row["entity_type"])].append(row)
    collisions = [g for g in groups.values() if len(g) > 1]
    return keys, collisions


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the keys; without it the tool only reports",
    )
    args = parser.parse_args()

    rows = await _load_rows()
    if not rows:
        print("entity table is empty — migration 79 applies with no backfill.")
        return 0

    keys, collisions = plan(rows)
    print(f"{len(rows)} entity rows, {len(set(keys.values()))} distinct keys.")

    if collisions:
        print(
            f"\nREFUSING: {len(collisions)} group(s) normalise to one key. "
            f"Merging them is a curator's decision, not this tool's — resolve "
            f"them in the candidate queue (they appear there as same-type "
            f"`fold_equal` auto-merge proposals), then run this again.\n"
        )
        for group in collisions:
            key = normalize_entity_name(group[0]["canonical_name"])
            print(f"  key {key!r} / type {group[0]['entity_type']!r}:")
            for row in group:
                print(f"      {row['id']}  {row['canonical_name']!r}  [{row['status']}]")
        return 1

    if not args.apply:
        print("\nNo collisions. Re-run with --apply to write the keys.")
        return 0

    written = 0
    for row_id, key in keys.items():
        await execute_query(
            f"UPDATE {row_id} SET name_key = $key;", {"key": key}
        )
        written += 1
    print(f"\nWrote name_key on {written} rows. Migration 79 will now apply.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
