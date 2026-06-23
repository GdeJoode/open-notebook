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

# Mirror the relation-side type resolution used in persist. Imports follow the
# load_dotenv() above so SURREAL_* are set before the surrealdb pool config is
# read — hence the noqa: E402.
from app_main.services.entity_persistence_service import (  # noqa: E402, I001
    _normalize_entity_type,
    _resolve_entity_type,
)
from surrealdb_service.connection import execute_query  # noqa: E402


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

    from app_main.services.entity_persistence_service import (
        EntityPersistenceService,
    )

    svc = EntityPersistenceService()

    def _endpoint_type(name: str, carried: str | None) -> str | None:
        # The O.1 fixed logic: bridge-resolved batch type wins; else the
        # relation-carried type bridge-resolved; else None.
        t = type_by_name.get(name)
        if t is None and carried:
            t = _resolve_entity_type(carried).entity_type
        return t

    old_outcomes = Counter()
    new_outcomes = Counter()
    skip_detail = Counter()
    name_miss_names = set()
    for rel in relations:
        src = rel.get("source_entity", "")
        tgt = rel.get("target_entity", "")
        if not src or not tgt:
            old_outcomes["empty_endpoint"] += 1
            new_outcomes["empty_endpoint"] += 1
            continue

        # OLD logic: rel-carried type wins, else batch map, then alias-normalize.
        old_s = rel.get("source_type") or type_by_name.get(src)
        old_t = rel.get("target_type") or type_by_name.get(tgt)
        old_s = _normalize_entity_type(old_s) if old_s else None
        old_t = _normalize_entity_type(old_t) if old_t else None
        if await _name_type_hit(src, old_s) and await _name_type_hit(tgt, old_t):
            old_outcomes["resolved"] += 1
        else:
            old_outcomes["skipped"] += 1

        # NEW (fixed) logic: bridge-typed endpoints + name-only fallback via the
        # real service helper.
        ns = _endpoint_type(src, rel.get("source_type"))
        nt = _endpoint_type(tgt, rel.get("target_type"))
        sid = await svc._resolve_endpoint_id(src, ns)
        tid = await svc._resolve_endpoint_id(tgt, nt)
        if sid is not None and tid is not None:
            new_outcomes["resolved"] += 1
        else:
            new_outcomes["skipped"] += 1
            for side, name, eid in (("src", src, sid), ("tgt", tgt, tid)):
                if eid is None:
                    skip_detail[f"{side}: NAME miss (no entity named)"] += 1
                    name_miss_names.add(name)

    print(f"=== O.1 diagnosis for {source_id} ===")
    print(f"entities stored: {len(entities)}  relations stored: {len(relations)}")
    print("OLD (alias-typed, no fallback):", dict(old_outcomes))
    print("NEW (bridge-typed + name-only fallback):", dict(new_outcomes))
    print("NEW skip detail (per missing endpoint):")
    for k, v in skip_detail.most_common():
        print(f"  {v:4d}  {k}")
    if name_miss_names:
        print("sample names with no persisted entity (extraction-consistency, out of O.1 scope):")
        for n in list(name_miss_names)[:8]:
            print(f"    {n!r}")


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "source:052dtl7jrwu1czlpnui4"
    os.environ.setdefault("SURREAL_URL", "ws://localhost:8000")
    asyncio.run(main(sid))
