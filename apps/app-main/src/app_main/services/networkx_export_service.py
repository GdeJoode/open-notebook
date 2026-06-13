"""NetworkX 7-format graph export service (Phase D.3).

Builds a ``networkx.DiGraph`` from the notebook-scoped entity/relation
projections introduced in D.0 (see ``EntityRepository.list_entities_for_notebook``
/ ``list_relations_for_notebook``) and serialises it to any of seven
formats: ``graphml``, ``gexf``, ``gml``, ``json-tree``, ``edge-list``,
``adjacency-list``, ``pickle``.

Why a dedicated service?
========================
JSONL and Obsidian (D.2 / D.1) emit one row per entity. NetworkX needs
the *whole graph in memory* before any byte goes out -- the writers are
not streaming. This service therefore owns the in-memory graph
construction plus the format dispatch, and the router just hands the
bytes back as ``Response`` (no streaming wrapper).

Attribute flattening contract
=============================
The XML/text serialisers in ``networkx`` (GraphML, GEXF, GML) reject
non-primitive attribute values -- specifically lists and dicts. The
Track-B entity model carries both:

* ``type_tags: List[str]``   -- multi-type tag list from migration 44.
* ``properties: Dict[str, Any]`` -- free-form key/value bag.

To keep all seven formats round-trippable from a single graph, this
service flattens BOTH attributes before serialisation:

* ``type_tags`` -> comma-joined string (e.g. ``["Person", "Researcher"]``
  becomes ``"Person,Researcher"``). Consumers split on ``","`` and trim
  whitespace.
* ``properties`` -> JSON-encoded string (``json.dumps``). Consumers
  ``json.loads`` to rehydrate.

The same flattening applies to edge ``properties``. The flat shape is
applied to *every* format -- not just XML/text -- so a consumer that
loads a graphml export and a pickle export of the same notebook sees
the same attribute shape, simplifying downstream pandas pipelines.

Note on ``json-tree``
=====================
``nx.tree_data`` requires a rooted tree (no cycles, single source). Most
notebook graphs are neither. We fall back to ``nx.node_link_data`` when
``tree_data`` raises -- this preserves a 1:1 lossless JSON of the graph
without forcing the caller to know whether their notebook is
tree-shaped. The mode is reported via the JSON envelope's top-level
``format`` key so downstream tooling can pick the right reader (the
node-link envelope already self-identifies in its key set).

Q-D-8 telemetry constraint
==========================
The telemetry payload carries **counts only** -- never entity or
relation IDs. The ``ExportReport`` model and ``record_metric`` payload
are constructed accordingly.

Archived/merged entities
========================
``EntityRepository.list_entities_for_notebook`` already filters
``orphan_status in {pending_reconnect, archived}`` via ``ExportFilter``
flags. The ``Entity.status`` field (active | merged | archived) is a
*different* axis introduced in B.5b and not currently consulted by the
D.0 SurrealQL. We add a Python-side post-filter here for safety
(``status not in {archived, merged}``) so a merged or archived entity
never lands in the export even if a future repository revision starts
returning them. The cost is one boolean per entity; the safety
margin is worth it.
"""

from __future__ import annotations

import io
import json
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from loguru import logger
from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter, ExportReport, NetworkxExportRequest
from shared.services.metrics import record_metric


# ---------------------------------------------------------------------------
# Repository protocol -- only the two methods we call, so tests can pass
# AsyncMocks without spinning up a real EntityRepository.
# ---------------------------------------------------------------------------


@dataclass
class _GraphBuildSummary:
    """Internal accountancy for a single ``_build_graph`` call.

    Kept separate from ``ExportReport`` so the public ``export`` method
    can compose the final report -- duration, format, etc -- without
    over-coupling the graph builder to telemetry concerns.
    """

    graph: nx.DiGraph
    entities_written: int
    relations_written: int
    dropped_relations: int


# Entity statuses that should never reach an exporter regardless of the
# orphan_* lifecycle filter. ``merged`` rows are tombstones pointing at
# their canonical successor; ``archived`` rows are user-deleted. Both
# are noise from the analyst's perspective.
_EXCLUDED_ENTITY_STATUSES = frozenset({"archived", "merged"})


class NetworkxExportService:
    """Builds + serialises a ``networkx.DiGraph`` for a notebook.

    The service is intentionally stateless aside from the injected
    repositories -- one instance can handle concurrent requests because
    every method takes the notebook id at call time.
    """

    def __init__(
        self,
        entity_repository: Any,
        relation_repository: Optional[Any] = None,
    ) -> None:
        """Wire repository dependencies.

        Args:
            entity_repository: Object with
                ``list_entities_for_notebook(notebook_id, filter)`` and
                ``list_relations_for_notebook(notebook_id, filter)``.
                In production this is ``EntityRepository`` (both methods
                live on the entity repo per D.0).
            relation_repository: Reserved for a future split where
                relations migrate to a dedicated repository. Today both
                methods live on the entity repo, so callers usually pass
                ``None`` and the service uses ``entity_repository`` for
                both lookups.
        """
        self._entity_repo = entity_repository
        self._relation_repo = relation_repository or entity_repository

    async def export(
        self,
        notebook_id: str,
        request: NetworkxExportRequest,
    ) -> Tuple[bytes, ExportReport]:
        """Build the graph + serialise + return ``(bytes, report)``.

        The router uses the bytes for the response body and the report
        for both the telemetry payload and the response headers (a
        future revision could surface counts in ``X-Export-*`` headers,
        but D.3 keeps the body-only contract for simplicity).

        Args:
            notebook_id: SurrealDB record ID of the notebook
                (e.g. ``"notebook:abc"``).
            request: ``NetworkxExportRequest`` carrying the format
                discriminator and the shared ``ExportFilter``.

        Returns:
            Tuple of ``(serialised_bytes, ExportReport)``.
        """
        started = time.monotonic()

        summary = await self._build_graph(notebook_id, request.filter)
        payload = self._serialize(summary.graph, request.format)

        duration_ms = int((time.monotonic() - started) * 1000)

        report = ExportReport(
            entities_written=summary.entities_written,
            relations_written=summary.relations_written,
            files_written=1,
            bytes_written=len(payload),
            duration_ms=duration_ms,
            filter_applied=request.filter,
            metadata={
                "dropped_relations": summary.dropped_relations,
            } if summary.dropped_relations else None,
        )

        # Telemetry -- counts only, no IDs per Q-D-8.
        await record_metric(
            "export.networkx",
            {**report.model_dump(mode="json"), "format": request.format},
            source=None,
            notebook=notebook_id,
        )

        logger.info(
            "networkx export: notebook={nb} format={fmt} entities={ents} "
            "relations={rels} bytes={bytes} duration_ms={ms}",
            nb=notebook_id,
            fmt=request.format,
            ents=summary.entities_written,
            rels=summary.relations_written,
            bytes=len(payload),
            ms=duration_ms,
        )

        return payload, report

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    async def _build_graph(
        self,
        notebook_id: str,
        filter_: ExportFilter,
    ) -> _GraphBuildSummary:
        """Project repository rows into a ``networkx.DiGraph``.

        Steps:

        1. Fetch entities via the D.0 projection method (no embeddings).
        2. Post-filter by ``Entity.status`` (extra safety net -- see
           module docstring).
        3. Fetch relations via the D.0 projection method (already filtered
           to in/out entities in the same notebook).
        4. Build nodes with FLAT attributes (lists/dicts JSON-encoded).
        5. Build edges, dropping those whose endpoints didn't make it
           into the node set (Q-D-4 silent-drop semantics).

        Args:
            notebook_id: Notebook record ID.
            filter_: Filter snapshot to propagate to both repository calls.

        Returns:
            ``_GraphBuildSummary`` carrying the graph + counters.
        """
        entities: List[Entity] = await self._entity_repo.list_entities_for_notebook(
            notebook_id, filter_
        )
        # Post-filter on Entity.status for archived/merged rows. The D.0
        # SurrealQL gates on orphan_status (a different axis). Doing this
        # in Python costs one boolean per row; cheaper than another DB
        # round-trip for the few callers that need it.
        entities = [
            e for e in entities if (e.status or "active") not in _EXCLUDED_ENTITY_STATUSES
        ]

        relations: List[Relation] = await self._relation_repo.list_relations_for_notebook(
            notebook_id, filter_
        )

        graph = nx.DiGraph()

        # Build node attribute dict and remember the id-set so the edge
        # loop can drop dangling endpoints (Q-D-4).
        node_ids: set[str] = set()
        for entity in entities:
            if not entity.id:
                continue
            node_id = str(entity.id)
            graph.add_node(node_id, **self._node_attrs(entity))
            node_ids.add(node_id)

        # Build edges. The D.0 list_relations_for_notebook query already
        # restricts in/out to the filtered entity-id set, so we expect
        # 100% inclusion -- but we still gate on node_ids to honour the
        # status post-filter (an edge whose endpoint we just dropped
        # because status=merged would otherwise leak through).
        dropped = 0
        relations_written = 0
        for relation in relations:
            src = str(relation.in_entity) if relation.in_entity else None
            dst = str(relation.out_entity) if relation.out_entity else None
            if not src or not dst:
                dropped += 1
                continue
            if src not in node_ids or dst not in node_ids:
                dropped += 1
                continue
            graph.add_edge(src, dst, **self._edge_attrs(relation))
            relations_written += 1

        return _GraphBuildSummary(
            graph=graph,
            entities_written=len(node_ids),
            relations_written=relations_written,
            dropped_relations=dropped,
        )

    @staticmethod
    def _node_attrs(entity: Entity) -> Dict[str, Any]:
        """Project ``Entity`` -> flat NetworkX node attribute dict.

        Lists/dicts are flattened per the module docstring. ``None`` is
        coerced to an empty string for ``primary_type`` because GraphML
        rejects ``None`` attribute values.
        """
        return {
            "canonical_name": entity.canonical_name or "",
            "entity_type": entity.entity_type or "",
            # Comma-joined string per the flattening contract.
            "type_tags": ",".join(entity.type_tags or []),
            "primary_type": entity.primary_type or "",
            "confidence": float(entity.confidence or 0.0),
            # JSON-encoded so dicts of arbitrary depth survive the XML
            # serialisers without losing structure.
            "properties": json.dumps(entity.properties or {}, sort_keys=True),
        }

    @staticmethod
    def _edge_attrs(relation: Relation) -> Dict[str, Any]:
        """Project ``Relation`` -> flat NetworkX edge attribute dict."""
        return {
            "relation_type": relation.relation_type or "",
            "confidence": float(relation.confidence or 0.0),
            "properties": json.dumps(relation.properties or {}, sort_keys=True),
        }

    # ------------------------------------------------------------------
    # Format dispatch
    # ------------------------------------------------------------------

    def _serialize(self, graph: nx.DiGraph, fmt: str) -> bytes:
        """Dispatch to the right writer for ``fmt``.

        All writers go through a ``BytesIO`` so the public surface is
        always ``bytes`` -- the FastAPI ``Response`` doesn't care about
        the wire format, just the byte string and the Content-Type.

        Args:
            graph: The pre-flattened ``DiGraph``.
            fmt: One of the seven literals in ``NetworkxFormat``.

        Returns:
            Serialised bytes.

        Raises:
            ValueError: If ``fmt`` is not one of the seven known
                literals. Pydantic validation on the request body should
                catch this earlier; the raise here is defence in depth.
        """
        if fmt == "graphml":
            buf = io.BytesIO()
            nx.write_graphml(graph, buf)
            return buf.getvalue()

        if fmt == "gexf":
            buf = io.BytesIO()
            nx.write_gexf(graph, buf)
            return buf.getvalue()

        if fmt == "gml":
            # GML's writer accepts a path or a file-like object opened in
            # binary mode; ``BytesIO`` works directly.
            buf = io.BytesIO()
            nx.write_gml(graph, buf)
            return buf.getvalue()

        if fmt == "json-tree":
            return self._serialize_json_tree(graph)

        if fmt == "edge-list":
            buf = io.BytesIO()
            nx.write_edgelist(graph, buf)
            return buf.getvalue()

        if fmt == "adjacency-list":
            buf = io.BytesIO()
            nx.write_adjlist(graph, buf)
            return buf.getvalue()

        if fmt == "pickle":
            # ``pickle.dumps`` returns bytes directly; no BytesIO needed.
            return pickle.dumps(graph)

        raise ValueError(f"Unknown NetworkX format: {fmt!r}")

    @staticmethod
    def _serialize_json_tree(graph: nx.DiGraph) -> bytes:
        """JSON serialisation with ``tree_data`` -> ``node_link_data`` fallback.

        ``nx.tree_data`` requires a rooted tree -- exactly one source,
        no cycles. Most notebook graphs fail that. We try ``tree_data``
        first (root = alphabetically-first canonical_name to keep
        deterministic output), and on any failure fall back to
        ``node_link_data`` which works for any DiGraph.

        The fallback envelope is self-identifying (``node_link_data``
        emits keys like ``directed``, ``multigraph``, ``nodes``, ``links``),
        so downstream code can dispatch on shape if it cares.
        """
        if graph.number_of_nodes() == 0:
            # Empty graph -- emit an empty node-link envelope. tree_data
            # raises on empty input; node_link_data handles it cleanly.
            return json.dumps(nx.node_link_data(graph, edges="links")).encode("utf-8")

        try:
            # Pick a stable root: the node with the smallest
            # ``canonical_name`` (alphabetical). Ties resolved by node id
            # so the output is deterministic across runs.
            root = min(
                graph.nodes,
                key=lambda n: (
                    graph.nodes[n].get("canonical_name", ""),
                    str(n),
                ),
            )
            tree = nx.tree_data(graph, root=root)
            return json.dumps(tree).encode("utf-8")
        except (nx.NetworkXError, nx.NetworkXNotImplemented, TypeError) as exc:
            # ``tree_data`` raises ``NetworkXError`` for disconnected
            # graphs and ``NetworkXNotImplemented`` for ones with cycles.
            # We catch both and fall back to the universal node-link
            # representation. Logged at INFO -- this is expected for any
            # notebook with more than one connected component.
            logger.info(
                "json-tree: falling back to node_link_data ({exc_type}: {exc})",
                exc_type=type(exc).__name__,
                exc=exc,
            )
            return json.dumps(nx.node_link_data(graph, edges="links")).encode("utf-8")


__all__ = ["NetworkxExportService"]
