"""O.1 temporary diagnostic: measure relation-endpoint resolution for one source.

Replays the STORED extraction_result relations for a source through the same
endpoint-resolution the persist path uses, tallying per-relation outcomes:

* resolved (both endpoints found) vs skipped
* for skips: which side missed + whether a NAME-ONLY match would have hit
  (i.e. the type filter is what killed it) vs a true name miss.

Read-only: it never RELATEs. Run with the workspace env loaded.

Usage:
    uv run --project apps/app-main python scripts/diag_relation_persist.py source:XXXX
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter

# Load .env so SURREAL_* are set when run outside the container.
from dotenv import load_dotenv

load_dotenv()

from surrealdb_service.connection import execute_query  # noqa: E402

# Mirror the relation-side type resolution used in persist (alias-only path),
# plus the bridge-aware type_by_name the persist builds from this source's
# entities. We import the real helpers to stay faithful.
from app_main.services.entity_persistence_service import (  # noqa: E402
    _normalize_entity_type,
    _resolve_entity_type,
)


async def _name_type_hit(name: str, etype: str | None) -> bool:
    clause = "canonical_name = $name"
    params = {"name": name}
    if etype is not None:
        clause += " AND entity_type = $etype"
        params["etype"] = etype
    rows = await execute_query(
        f"SELECT VALUE id FROM entity WHERE {clause} LIMIT 1;", params
    )
    return bool(rows)


async def main(source_id: str) -> None:
    rows = await execute_query(
        "SELECT entities, relations FROM extraction_result "
        "WHERE source_id = $sid LIMIT 1;",
        {"sid": source_id},
    )
    if not rows:
        print(f"no extraction_result for {source_id}")
        return
    entities = rows[0].get("entities", []) or []
    relations = rows[0].get("relations", []) or []

    # Build the bridge-aware type_by_name exactly as persist does (no schemas on
    # the re-filter path; the LLM label drives it).
    type_by_name = {}
    for e in entities:
        t = e.get("text", "")
        if t.strip():
            type_by_name[t] = _resolve_entity_type(e.get("label", "concept")).entity_type

    outcomes = Counter()
    skip_detail = Counter()
    for rel in relations:
        src = rel.get("source_entity", "")
        tgt = rel.get("target_entity", "")
        if not src or not tgt:
            outcomes["empty_endpoint"] += 1
            continue

        # CURRENT persist logic: rel-carried type wins, else batch map, then
        # alias-normalize.
        src_t = rel.get("source_type") or type_by_name.get(src)
        tgt_t = rel.get("target_type") or type_by_name.get(tgt)
        src_t = _normalize_entity_type(src_t) if src_t else None
        tgt_t = _normalize_entity_type(tgt_t) if tgt_t else None

        src_typed = await _name_type_hit(src, src_t)
        tgt_typed = await _name_type_hit(tgt, tgt_t)
        if src_typed and tgt_typed:
            outcomes["resolved_typed"] += 1
            continue

        outcomes["skipped"] += 1
        # Diagnose WHY each missing side missed.
        for side, name, etype, typed_hit in (
            ("src", src, src_t, src_typed),
            ("tgt", tgt, tgt_t, tgt_typed),
        ):
            if typed_hit:
                continue
            name_only = await _name_type_hit(name, None)
            bridge_t = _resolve_entity_type(name).entity_type if False else None  # noqa
            if name_only:
                skip_detail[f"{side}: TYPE-only miss (name exists, type={etype})"] += 1
            else:
                skip_detail[f"{side}: NAME miss (no entity named, type={etype})"] += 1

    print(f"=== O.1 diagnosis for {source_id} ===")
    print(f"entities stored: {len(entities)}  relations stored: {len(relations)}")
    print("outcomes:", dict(outcomes))
    print("skip detail (per missing endpoint):")
    for k, v in skip_detail.most_common():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "source:052dtl7jrwu1czlpnui4"
    os.environ.setdefault("SURREAL_URL", "ws://localhost:8000")
    asyncio.run(main(sid))
