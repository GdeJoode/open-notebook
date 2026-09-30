"""The documents behind a triage row's *Docs* number (Q-B1, part 1).

The queue shows ``doc_count`` — how many distinct ``source_documents`` the entity
had when triage last ran. A curator hovering that number wants to know WHICH
documents. The ids are already on the entity; this resolves them to titles in
two batched reads, for every row of the queue at once.

The list is read now, while ``doc_count`` is the triage run's snapshot, so the
two can differ after a later extraction; the frontend says so when they do.

Part 2 of Q-B1 — the passage where the entity was found — is not here: the
corpus on staging predates PC.3's ``properties.grounding``, so there is no stored
passage to show (see ``docs/tracks/Q-triage/status.md``, Q-B1).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query


async def sources_for_entities(
    entity_ids: Iterable[str], config: Optional[SurrealDBConfig] = None
) -> Dict[str, List[Dict[str, str]]]:
    """``{entity_id: [{"id": source_id, "title": title}, ...]}``, in the entity's
    own ``source_documents`` order, distinct. An entity with no documents maps to
    ``[]``; a source id with no ``source`` row keeps its id with an empty title."""
    ids = list(dict.fromkeys(str(e) for e in entity_ids if e))
    if not ids:
        return {}
    # `FROM $ids` fetches the records directly; `FROM entity WHERE id IN $ids`
    # scanned the table: 1,418 ms against 99 ms for 1,238 queue rows on staging
    # (Q-B1 review), paid on every queue refetch — i.e. after every curator click.
    rows = await execute_query(
        "SELECT id, source_documents FROM $ids;",
        {"ids": [ensure_record_id(i) for i in ids]},
        config,
    )
    per_entity: Dict[str, List[str]] = {}
    for row in rows or []:
        docs = row.get("source_documents") or []
        per_entity[str(row["id"])] = list(dict.fromkeys(str(d) for d in docs if d is not None))

    source_ids = sorted({s for docs in per_entity.values() for s in docs})
    titles: Dict[str, str] = {}
    if source_ids:
        srows = await execute_query(
            "SELECT id, title FROM source WHERE id IN $ids;",
            {"ids": [ensure_record_id(s) for s in source_ids]},
            config,
        )
        titles = {str(r["id"]): str(r.get("title") or "") for r in srows or []}

    out: Dict[str, List[Dict[str, str]]] = {}
    for eid in ids:
        out[eid] = [{"id": s, "title": titles.get(s, "")} for s in per_entity.get(eid, [])]
    return out


async def entity_types(
    entity_ids: Iterable[str], config: Optional[SurrealDBConfig] = None
) -> Dict[str, str]:
    """``{entity_id: entity_type}``: each row's canonical type, read now (WB.3).

    A queue row's own ``type`` is the display label (``primary_type``) the
    triage run stored; the triage table shows this canonical type first, as
    the Knowledge Graph table does, and the label second. An entity that no
    longer exists is absent.
    """
    ids = list(dict.fromkeys(str(e) for e in entity_ids if e))
    if not ids:
        return {}
    rows = await execute_query(
        "SELECT id, entity_type FROM $ids;",
        {"ids": [ensure_record_id(i) for i in ids]},
        config,
    )
    return {
        str(r["id"]): str(r["entity_type"])
        for r in rows or []
        if r.get("entity_type")
    }


__all__ = ["entity_types", "sources_for_entities"]
