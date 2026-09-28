"""What an ontology edit would do, before it takes effect (Track OE.1).

The editor (OE.2) sends a CANDIDATE set — each ontology's own content, keyed
by name — and this compares it with the ACTIVE set on the real graph:

* ``findings`` — :func:`ontology_manager.validation.validate`; ``blocking`` is
  true when the edit introduces a problem;
* ``label_changes`` — every label (declared in either set, or carried by an
  entity) whose canonical type differs, with how many entities carry it;
* ``retype`` — the plan a publish would offer (``retype_planner``, the same
  plan the retype script applies): changes, collisions, flag-only rows, and
  how many of the changes THIS edit causes (the rest were already due);
* ``detection`` — the documents whose ontology selection would change, the
  detection replayed per source with both sets (it is deterministic: content
  overlap and document type, no model).

Read-only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from ontology_manager.schema import Ontology
from ontology_manager.validation import BLOCK, resolve_set, validate
from surrealdb_service.connection import ensure_record_id, execute_query

from app_main.services.entity_persistence_service import (
    _resolve_entity_type,
    typing_vocabulary,
)
from app_main.services.retype_planner import entity_rows, label_of, plan_rows


async def load_own_ontologies() -> Dict[str, Ontology]:
    """The ACTIVE set: every ontology's own content, as DEEP COPIES (the registry
    hands out shared cached objects; a candidate built from them must not
    change what every caller sees)."""
    from ontology_manager.registry import get_ontology_registry

    registry = get_ontology_registry()
    own: Dict[str, Ontology] = {}
    for name in await registry.list_ontologies():
        ontology = await registry.get(name, resolve_inheritance=False)
        if ontology is not None:
            own[name] = ontology.model_copy(deep=True)
    return own


def vocabulary_of(own: Dict[str, Ontology]) -> List[Ontology]:
    """The typing vocabulary a set yields: every ontology, inheritance resolved."""
    merged, _ = resolve_set(own)
    return typing_vocabulary(None, list(merged.values()))


def _declared_labels(own: Dict[str, Ontology]) -> set:
    merged, _ = resolve_set(own)
    labels = set()
    for ontology in merged.values():
        for key, definition in ontology.entity_types.items():
            labels.add(definition.name or key)
            labels.update(definition.aliases or [])
    return labels


def label_changes(
    rows: List[Dict[str, Any]], active: List[Ontology], candidate: List[Ontology],
    declared: set,
) -> List[Dict[str, Any]]:
    carried: Dict[str, int] = {}
    for r in rows:
        label = label_of(r)
        if label:
            carried[label] = carried.get(label, 0) + 1
    out = []
    for label in sorted(set(carried) | declared, key=str.lower):
        before = _resolve_entity_type(label, active).entity_type
        after = _resolve_entity_type(label, candidate).entity_type
        if before != after:
            out.append({"label": label, "before": before, "after": after,
                        "entities": carried.get(label, 0)})
    return out


async def detection_changes(
    active: Dict[str, Ontology], candidate: Dict[str, Ontology]
) -> List[Dict[str, Any]]:
    """Per source, the selection under each set — only the ones that differ."""
    from ontology_extraction.multi_schema_orchestrator import detect_applicable_schemas

    from app_main.services.entity_extraction_service import _applicability_sample

    merged_active = list(resolve_set(active)[0].values())
    merged_candidate = list(resolve_set(candidate)[0].values())
    sources = await execute_query(
        "SELECT id, title, metadata.document_type AS dt FROM source;", {}
    ) or []
    out = []
    for source in sources:
        chunks = await execute_query(
            "SELECT text, order FROM chunk WHERE source = $id ORDER BY order ASC;",
            {"id": ensure_record_id(str(source["id"]))},
        ) or []
        sample = _applicability_sample(chunks)
        picks = []
        for ontologies in (merged_active, merged_candidate):
            detected = await detect_applicable_schemas(
                document_type=source.get("dt"), document_text=sample,
                ontologies=ontologies, top_k=3,
            )
            picks.append([o.metadata.name for o, _ in detected])
        if picks[0] != picks[1]:
            out.append({"source": str(source["id"]), "title": source.get("title"),
                        "before": picks[0], "after": picks[1]})
    return out


async def preview(
    candidate: Dict[str, Ontology],
    active: Optional[Dict[str, Ontology]] = None,
    *,
    rows: Optional[List[Dict[str, Any]]] = None,
    include_detection: bool = True,
) -> Dict[str, Any]:
    """Everything the editor shows before a publish. See the module docstring."""
    if active is None:
        active = await load_own_ontologies()
    findings = validate(candidate, active)
    v_active, v_candidate = vocabulary_of(active), vocabulary_of(candidate)
    if rows is None:
        rows = await entity_rows()

    plan = plan_rows(rows, v_candidate)
    caused = [
        c for c in plan["changes"]
        if _resolve_entity_type(c["label"], v_active).entity_type != c["new"]
    ]
    detection: List[Dict[str, Any]] = []
    if include_detection:
        try:
            detection = await detection_changes(active, candidate)
        except Exception as exc:  # a preview never fails on the replay
            logger.warning(f"ontology preview: detection replay failed ({exc})")
            detection = [{"error": str(exc)}]

    return {
        "findings": [f.__dict__ for f in findings],
        "blocking": any(f.severity == BLOCK for f in findings),
        "label_changes": label_changes(
            rows, v_active, v_candidate, _declared_labels(active) | _declared_labels(candidate)
        ),
        "retype": {
            "changes": len(plan["changes"]),
            "collisions": len(plan["collisions"]),
            "flag_only": len(plan["flag_only"]),
            "caused_by_this_edit": len(caused),
            "plan": plan,
        },
        "detection": detection,
    }


__all__ = [
    "detection_changes", "label_changes", "load_own_ontologies", "preview", "vocabulary_of",
]
