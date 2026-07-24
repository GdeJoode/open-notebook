"""Shared entity/relation projection for the Track-D export family.

OKF-D2 (drift-avoidance): the Obsidian exporter (:mod:`obsidian_export_service`)
and the OKF exporter (:mod:`okf_export_service`) must select and filter the
*same* subset of the knowledge graph — otherwise the two curated-knowledge
surfaces would silently diverge (a relation dropped by one but kept by the
other, an ``archived`` entity leaking into one bundle but not the other).

Rather than re-implement the filter pipeline in each service, the pure,
repo-free core lives here and both exporters call :func:`project_graph`.
The IO (repository fetch) and format-specific rendering (wikilinks vs OKF
markdown links, frontmatter keys, bundle tree) stay in the respective
services; only the *which rows survive* decision is centralised.

The pipeline order is load-order-significant and mirrors the pre-extraction
Obsidian behaviour exactly:

1. ``status`` post-filter — drop ``archived`` / ``merged`` tombstones
   (the D.0 SurrealQL gates on the orthogonal ``orphan_status`` axis, so
   this second axis is applied Python-side).
2. ``min_connections`` post-filter — drop entities whose in+out degree in
   the raw relation set is below the threshold.
3. dropped-relation accounting — a relation whose endpoint didn't survive
   steps 1-2 is counted for the ``ExportReport`` and later silently dropped
   by the renderer (Q-D-4), never emitted as a dangling link.

The full relation list is returned unchanged; endpoint survival is resolved
at render time against the surviving-entity set so each format can account
for its own dangling links.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List

from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter

# Entity statuses that must never reach an exporter regardless of the
# orphan_* lifecycle filter. ``merged`` rows are tombstones pointing at
# canonical successors; ``archived`` rows are user-deleted. Both are noise
# from the analyst's perspective. Public so the D.1c preview router and the
# OKF exporter share one source of truth for "what's excluded".
EXCLUDED_ENTITY_STATUSES = frozenset({"archived", "merged"})


@dataclass
class GraphProjection:
    """Result of :func:`project_graph`.

    ``entities`` are the survivors of the status + min_connections filters.
    ``relations`` is the *unfiltered* relation list (each renderer resolves
    endpoint survival itself against the surviving-entity set). ``dropped_relations``
    is the count of relations whose endpoint fell out of the surviving set —
    surfaced in the ``ExportReport`` so a high-drop run is visible after the fact.
    """

    entities: List[Entity]
    relations: List[Relation]
    dropped_relations: int


def apply_min_connections_filter(
    entities: List[Entity],
    relations: List[Relation],
    min_connections: int,
) -> List[Entity]:
    """Drop entities whose in+out degree is below ``min_connections``.

    ``min_connections <= 0`` short-circuits — keep everyone, no degree
    computation needed. For non-zero thresholds we count in+out relations
    per entity (a self-loop counts as 2, matching graph-theory degree).

    The degree is computed over *all* relations, including those whose other
    endpoint was already dropped by the status filter — this makes the
    threshold a signal of an entity's structural prominence in the raw graph
    rather than an artefact of the local filter set.
    """
    if min_connections <= 0:
        return entities

    degree: Counter[str] = Counter()
    for relation in relations:
        src = str(relation.in_entity) if relation.in_entity else None
        dst = str(relation.out_entity) if relation.out_entity else None
        if src:
            degree[src] += 1
        if dst:
            degree[dst] += 1

    return [e for e in entities if e.id and degree[str(e.id)] >= min_connections]


def project_graph(
    entities: List[Entity],
    relations: List[Relation],
    filter_: ExportFilter,
) -> GraphProjection:
    """Apply the shared export filter pipeline to a raw entity/relation set.

    See the module docstring for the ordered semantics. This function is pure:
    it neither touches a repository nor a clock, so exporters get byte-stable,
    unit-testable projection behaviour.
    """
    survivors = [
        e for e in entities if (e.status or "active") not in EXCLUDED_ENTITY_STATUSES
    ]
    survivors = apply_min_connections_filter(
        survivors, relations, filter_.min_connections
    )

    surviving_ids = {str(e.id) for e in survivors if e.id}
    dropped_relations = 0
    for relation in relations:
        src = str(relation.in_entity) if relation.in_entity else None
        dst = str(relation.out_entity) if relation.out_entity else None
        if not src or not dst:
            dropped_relations += 1
            continue
        if src not in surviving_ids or dst not in surviving_ids:
            dropped_relations += 1

    return GraphProjection(
        entities=survivors,
        relations=relations,
        dropped_relations=dropped_relations,
    )


__all__ = [
    "EXCLUDED_ENTITY_STATUSES",
    "GraphProjection",
    "apply_min_connections_filter",
    "project_graph",
]
