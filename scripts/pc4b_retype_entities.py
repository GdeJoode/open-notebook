"""Track PC.4b — retype existing entities with PC.4's typing.

PC.4 changed how a label is typed: against every ontology instead of the
document's selection, with 15 ontology types corrected, new NL aliases, literals
on ``other`` and ``Project`` as a programme. New extractions type the new way;
rows written before keep the old type. Identity is ``(name_key, entity_type)``,
so the next extraction that meets such a name writes a SECOND row next to the
old one. This retypes the old rows so that does not happen.

For every active/reference entity with a label (``primary_type``, else
``properties.raw_entity_type``) it recomputes the type with the same function
persistence uses (``_resolve_entity_type`` over ``typing_vocabulary``) and:

* **changes** the type where it differs;
* flags a literal (``Date``, ``Budget``, …) with ``properties.non_type_label``;
* **refuses** a change that would collide: an ACTIVE row with the same
  ``name_key`` already holds the new type (or another change claims it). Two
  such rows are one entity; that is a merge for the curator, not a retype, and
  the unique index ``idx_entity_identity`` would reject the write anyway.

Default is a DRY RUN: it prints what would change and writes nothing. With
``--apply`` it writes, each update guarded on the old type still being there,
and saves a rollback file (id, old type, new type) first.

Notebook schema edits are not projected (typing here has no notebook, like the
re-filter). If any notebook has accepted edits the script refuses unless
``--ignore-notebook-edits`` is given.

Run with SURREAL_* pointing at the target (the WSL venv, env from ``.env``)::

    python scripts/pc4b_retype_entities.py            # dry run
    python scripts/pc4b_retype_entities.py --apply    # write
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app_main.services.entity_persistence_service import (
    _resolve_entity_type,
    load_typing_ontologies,
    typing_vocabulary,
)
from surrealdb_service.connection import ensure_record_id, execute_query

REPO_ROOT = Path(__file__).resolve().parents[1]


async def _notebooks_with_edits() -> list[str]:
    rows = await execute_query(
        "SELECT notebook, accepted_extensions FROM notebook_schema;", {}
    )
    return [
        str(r["notebook"])
        for r in rows or []
        if any(
            isinstance(e, dict) and not str(e.get("type_name", "")).startswith("_")
            for e in (r.get("accepted_extensions") or [])
        )
    ]


async def plan() -> dict:
    vocabulary = typing_vocabulary(None, await load_typing_ontologies())
    rows = await execute_query(
        "SELECT id, canonical_name, name_key, entity_type, status, primary_type, "
        "properties.raw_entity_type AS raw, properties.non_type_label AS flagged "
        "FROM entity WHERE status IN ['active', 'reference'];",
        {},
    )
    occupied = {
        (r.get("name_key"), r.get("entity_type"))
        for r in rows
        if r.get("status") == "active"
    }
    changes, collisions, flags = [], [], []
    for r in rows:
        label = (r.get("primary_type") or r.get("raw") or "").strip()
        if not label:
            continue
        resolved = _resolve_entity_type(label, vocabulary)
        new, old = resolved.entity_type, r.get("entity_type")
        item = {
            "id": str(r["id"]),
            "name": r.get("canonical_name"),
            "label": label,
            "status": r.get("status"),
            "old": old,
            "new": new,
            "non_type_label": resolved.is_noise,
        }
        if new == old:
            if resolved.is_noise and not r.get("flagged"):
                flags.append(item)
            continue
        key = (r.get("name_key"), new)
        if r.get("status") == "active" and key in occupied:
            collisions.append(item)
            continue
        if r.get("status") == "active":
            occupied.add(key)
        changes.append(item)
    return {"changes": changes, "collisions": collisions, "flag_only": flags, "total": len(rows)}


def _print(result: dict) -> None:
    changes, collisions = result["changes"], result["collisions"]
    print(f"entities read: {result['total']}")
    print(f"type changes: {len(changes)}")
    for (old, new), n in Counter((c["old"], c["new"]) for c in changes).most_common():
        print(f"  {old} -> {new}: {n}")
    print("by label:")
    for label, n in Counter(c["label"] for c in changes).most_common(30):
        print(f"  {label}: {n}")
    print(f"literals flagged without a type change: {len(result['flag_only'])}")
    print(f"refused (collision; needs a curator merge): {len(collisions)}")
    for c in collisions:
        print(f"  {c['id']} {c['name']!r} {c['old']} -> {c['new']}")


async def apply(result: dict) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rollback = REPO_ROOT / "claudedocs" / f"pc4b_retype_rollback_{stamp}.json"
    rollback.parent.mkdir(exist_ok=True)
    rollback.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"rollback file: {rollback}")
    written, stale = 0, []
    for c in result["changes"] + result["flag_only"]:
        rows = await execute_query(
            "UPDATE $id SET entity_type = $new, "
            "properties.non_type_label = IF $flag THEN true ELSE properties.non_type_label END "
            "WHERE entity_type = $old RETURN id;",
            {"id": ensure_record_id(c["id"]), "new": c["new"], "old": c["old"],
             "flag": c["non_type_label"]},
        )
        if rows:
            written += 1
        else:
            stale.append(c["id"])  # changed since the plan was made; left alone
    print(f"written: {written}; skipped because the row changed meanwhile: {len(stale)}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write (default: dry run)")
    parser.add_argument("--ignore-notebook-edits", action="store_true")
    args = parser.parse_args()

    edited = await _notebooks_with_edits()
    if edited and not args.ignore_notebook_edits:
        print(f"refusing: notebooks with accepted schema edits: {edited}")
        return 2
    result = await plan()
    _print(result)
    if args.apply:
        await apply(result)
    else:
        print("DRY RUN: nothing written. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
