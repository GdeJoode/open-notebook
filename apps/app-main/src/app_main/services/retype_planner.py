"""Plan a retype of existing entities under a typing vocabulary (PC.4b → OE.1).

Lifted out of ``scripts/pc4b_retype_entities.py`` so the ontology editor's
preview (``ontology_preview_service``) and the retype script count the same way
— one plan, two readers. The script keeps the writing (``--apply`` /
``--rollback``); this module only reads.

For every active/reference entity with a label it types the RAW LLM label
(``properties.raw_entity_type``; else ``primary_type``) exactly as persistence
does (``_resolve_entity_type``) and returns:

* ``changes`` — rows whose type differs;
* ``flag_only`` — literals whose type is already right but lack the
  ``non_type_label`` flag;
* ``collisions`` — changes refused because another active or reference row
  already holds the new ``(name_key, entity_type)`` (one entity: a merge for
  the curator, not a retype; ``upsert_entity`` looks up on that pair without a
  status filter).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from surrealdb_service.connection import execute_query

from app_main.services.entity_persistence_service import _resolve_entity_type

ENTITY_ROWS = (
    "SELECT id, canonical_name, name_key, entity_type, status, primary_type, "
    "properties.raw_entity_type AS raw, properties.non_type_label AS flagged "
    "FROM entity WHERE status IN ['active', 'reference'];"
)


def label_of(row: Dict[str, Any]) -> str:
    """The label persistence typed: the raw LLM label, else ``primary_type``."""
    return (row.get("raw") or row.get("primary_type") or "").strip()


def plan_rows(rows: List[Dict[str, Any]], vocabulary: List[Any]) -> Dict[str, Any]:
    """The plan for ``rows`` (shaped as :data:`ENTITY_ROWS` returns them)."""
    holders: Dict[Any, set] = {}
    for r in rows:
        holders.setdefault((r.get("name_key"), r.get("entity_type")), set()).add(str(r["id"]))
    changes, collisions, flags = [], [], []
    for r in rows:
        label = label_of(r)
        if not label:
            continue
        resolved = _resolve_entity_type(label, vocabulary)
        new, old, rid = resolved.entity_type, r.get("entity_type"), str(r["id"])
        item = {
            "id": rid,
            "name": r.get("canonical_name"),
            "label": label,
            "status": r.get("status"),
            "old": old,
            "new": new,
            "old_flag": r.get("flagged"),
            "new_flag": True if resolved.is_noise else r.get("flagged"),
        }
        if new == old:
            if resolved.is_noise and r.get("flagged") is not True:
                flags.append(item)
            continue
        key = (r.get("name_key"), new)
        if holders.get(key, set()) - {rid}:
            collisions.append({**item, "held_by": sorted(holders[key] - {rid})})
            continue
        holders.setdefault(key, set()).add(rid)
        changes.append(item)
    return {"changes": changes, "collisions": collisions, "flag_only": flags, "total": len(rows)}


async def entity_rows(config: Optional[Any] = None) -> List[Dict[str, Any]]:
    return await execute_query(ENTITY_ROWS, {}, config) or []


async def plan_retype(vocabulary: List[Any], config: Optional[Any] = None) -> Dict[str, Any]:
    """The plan for the live graph under ``vocabulary``."""
    return plan_rows(await entity_rows(config), vocabulary)


__all__ = ["ENTITY_ROWS", "entity_rows", "label_of", "plan_retype", "plan_rows"]
