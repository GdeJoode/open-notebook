"""Backbone check (B6) — flag well-connected nodes resting on weak edges (Q.4).

A node that reaches the well-connected band (effective degree >= the config
``well_connected_threshold``, default 3) is structurally important. If a chunk of
that connectivity rests on WEAK (``RELATED``) edges — machine-inferred co-occurrence
rather than a structural relation — the backbone is shaky: the node *looks*
well-connected but the evidence is soft. The backbone check surfaces, per such
node, exactly which incident edges are weak so an operator can judge whether the
connectivity is real.

This is a READ-TIME computation: it reads the config predicate-strength (Q.1) and
the persisted edges; it writes NOTHING and needs NO schema change. Weak-ness is a
property of the predicate in config, never a stored column, so re-tiering a
predicate (config edit) changes the warnings on the next read with no migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set

from loguru import logger

from app_main.services.triage.triage_config import TriageConfig, load_triage_config


@dataclass(frozen=True)
class WeakEdge:
    """One weak incident edge on a well-connected node."""

    other: str
    relation_type: str
    doc_count: int


@dataclass(frozen=True)
class BackboneWarning:
    """A well-connected node and the weak edges its connectivity rests on.

    ``weak_edges`` is non-empty by construction — a node with no weak incident
    edge produces NO warning (Q.4 AC5).
    """

    entity_id: str
    structural_degree: int
    weak_edges: List[WeakEdge]


def _doc_count(source_documents: Optional[Any]) -> int:
    """Distinct-document count from an edge's ``source_documents`` bag."""
    if not source_documents:
        return 0
    return len({str(d) for d in source_documents if d is not None})


class BackboneCheckService:
    """Compute backbone warnings for a set of affected nodes (B6).

    The strong/weak split and the well-connected threshold come from the injected
    (or default-loaded) Q.1 config; nothing here hardcodes ``RELATED`` or ``3``.
    """

    def __init__(
        self,
        entity_repository: Any,
        config: Optional[TriageConfig] = None,
    ) -> None:
        """Wire the repository (the relation seam) and resolve the config.

        Args:
            entity_repository: An ``EntityRepository`` exposing
                ``relations_for_entities`` (the batched incident-edge loader).
            config: An already-loaded :class:`TriageConfig`; defaults to the
                cached committed config.
        """
        self._repo = entity_repository
        self._cfg = config if config is not None else load_triage_config()

    async def warnings_for(
        self,
        affected_ids: List[str],
        degree_by_id: Mapping[str, int],
    ) -> List[BackboneWarning]:
        """Backbone warnings for the affected nodes that are well-connected.

        Only nodes whose effective degree is ``>= well_connected_threshold`` are
        examined (the ``degree_by_id`` map comes from Q.2 signals — no re-query of
        degree here). For each, every incident edge whose predicate is WEAK (per
        config) becomes a :class:`WeakEdge`; a node with none yields no warning.

        Args:
            affected_ids: The affected set (Q.2). Non-well-connected ids are
                skipped up front so the edge query only loads relevant nodes.
            degree_by_id: Effective degree per id (from the Q.2 SignalsReport).

        Returns:
            One :class:`BackboneWarning` per well-connected node WITH weak edges,
            ordered by the input id order for stable output.
        """
        threshold = self._cfg.well_connected_threshold
        well_connected = [
            i for i in affected_ids if int(degree_by_id.get(i, 0)) >= threshold
        ]
        if not well_connected:
            return []

        well_set: Set[str] = {str(i) for i in well_connected}
        edges = await self._repo.relations_for_entities(well_connected)

        # Partition weak edges by the well-connected endpoint(s) they touch.
        weak_by_node: Dict[str, List[WeakEdge]] = {i: [] for i in well_connected}
        for edge in edges or []:
            predicate = str(edge.get("relation_type") or "")
            if self._cfg.predicate_strength(predicate) != "weak":
                continue
            src = str(edge.get("in"))
            tgt = str(edge.get("out"))
            dc = _doc_count(edge.get("source_documents"))
            # Attribute the weak edge to whichever endpoint(s) are in the
            # well-connected set (a weak edge between two well-connected nodes
            # is a warning for both).
            if src in well_set:
                weak_by_node[src].append(
                    WeakEdge(other=tgt, relation_type=predicate, doc_count=dc)
                )
            if tgt in well_set and tgt != src:
                weak_by_node[tgt].append(
                    WeakEdge(other=src, relation_type=predicate, doc_count=dc)
                )

        warnings: List[BackboneWarning] = []
        for node in well_connected:
            weak = weak_by_node.get(node, [])
            if not weak:
                continue
            warnings.append(
                BackboneWarning(
                    entity_id=node,
                    structural_degree=int(degree_by_id.get(node, 0)),
                    weak_edges=weak,
                )
            )
        logger.debug(
            "BackboneCheck: {n} well-connected affected nodes -> {w} warnings",
            n=len(well_connected),
            w=len(warnings),
        )
        return warnings
