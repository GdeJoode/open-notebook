"""Schema proposals from ontology gaps, grouped by LABEL (Track PC.5a).

An ontology gap is ONE entity whose label (the extractor's rich type word — "Regio
Deal", "Programmalijn") aligned to no type in the applied schemas (N.4d.4). The
evolution agent's auto-propose counted gaps per entity TEXT and, at five, proposed
a new type NAMED AFTER THAT ENTITY: `Gemeente Opmeer` became type `GemeenteOpmeer`
— an instance promoted to a type, the error D-N4 warns against. The owner turned
that off (2026-09-27, `gap_auto_propose_enabled`, default False).

What a curator can act on is the label: "eight entities labelled *Programmalijn*
match no type" is a schema question, and the entities are its evidence. So this
module groups a notebook's gaps by label, and a label with at least
``threshold`` DISTINCT entities becomes a pending extension in that notebook's
queue — the `PendingExtensionsPanel` on the Schema tab, where accept runs the
N.4d.3 placement and reject is remembered (migration 84).

Scope is the notebook: gaps are stored per ontology, not per notebook, so a gap
belongs to a notebook when one of its ``source_documents`` is one of the
notebook's sources (the ``reference`` edge, source → notebook).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query

ORIGIN = "ontology_gap"
_EXAMPLES = 5


async def label_proposals_for_notebook(
    notebook_id: str, *, threshold: int, config: Optional[SurrealDBConfig] = None
) -> List[Dict[str, Any]]:
    """The labels in this notebook's gaps that have ``threshold`` or more distinct
    entities, as ``merge_pending_extensions`` proposals (``type_name`` = label).
    Ignored gaps (a rejected proposal's) are left out."""
    sources = await execute_query(
        "SELECT VALUE in FROM reference WHERE out = $nb;",
        {"nb": ensure_record_id(notebook_id)},
        config,
    )
    notebook_sources = {str(s) for s in sources or []}
    if not notebook_sources:
        return []
    gaps = await execute_query(
        "SELECT entity_text, entity_type_guess, source_documents FROM ontology_gap "
        "WHERE entity_type_guess != NONE AND status != 'ignored';",
        None,
        config,
    )
    # key -> casing variants seen. Rows come back in no guaranteed order, so the
    # form shown is chosen deterministically (see `_display`), not first-seen.
    by_label: Dict[str, Dict[str, set]] = {}
    labels: Dict[str, set] = {}
    for gap in gaps or []:
        docs = {str(d) for d in (gap.get("source_documents") or [])}
        if not docs & notebook_sources:
            continue
        label = str(gap.get("entity_type_guess") or "").strip()
        text = str(gap.get("entity_text") or "").strip()
        if not label or not text:
            continue
        key = label.lower()
        labels.setdefault(key, set()).add(label)
        by_label.setdefault(key, {}).setdefault(text.lower(), set()).add(text)

    proposals: List[Dict[str, Any]] = []
    for key, entities in sorted(by_label.items(), key=lambda kv: -len(kv[1])):
        if len(entities) < threshold:
            continue
        label = _display(labels[key])
        examples = [_display(entities[k]) for k in sorted(entities)][:_EXAMPLES]
        proposals.append({
            "type_name": label,
            "rationale": (
                f"{len(entities)} entities labelled '{label}' matched no type in the "
                f"applied schemas, e.g. {', '.join(examples)}."
            ),
            "examples": examples,
            "evidence_count": len(entities),
            "origin": ORIGIN,
        })
    logger.debug("gap label proposals for {nb}: {n}", nb=notebook_id, n=len(proposals))
    return proposals


def _display(variants: set) -> str:
    """One casing for a set of variants: a capitalised form over an all-lowercase
    one (`Economie` over `economie`), then alphabetical — deterministic whatever
    order the rows arrived in."""
    return sorted(variants, key=lambda v: (v.islower(), v))[0]


__all__ = ["ORIGIN", "label_proposals_for_notebook"]
