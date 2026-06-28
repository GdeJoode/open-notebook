"""Unified read-only graph-edge loader across the three KG edge tables.

The knowledge graph spreads its edges over three RELATE tables with different
endpoint types and different per-edge metadata:

- ``relation`` (entity -> entity, migration 39) — the semantic KG edge.
- ``mentions`` (source -> entity, migration 66) — document salience edges.
- ``cites``    (source -> source, migration 67) — document citation edges.

The per-table loaders (``EntityRepository.load_mentions_edges`` /
``SourceRepository.load_cites_edges`` / ``get_entity_detail``'s relation query)
each project their OWN shape. A graph-traversal consumer (the MCP ``related``
tool) wants one uniform shape regardless of which table an edge came from, and
wants every edge incident on a node whether the node is the edge's ``in`` OR its
``out`` endpoint. This module is that single seam: it normalises all three
tables to ``{id, source, target, edge_type, metadata}`` and unions both
directions for a node, so a caller never has to know which table holds which
relationship. Read-only — it never writes; the canonical edge tables are only
ever mutated by their own dedicated write helpers.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query

# The three KG edge tables and the per-edge data fields each carries (every
# DEFINE FIELD beyond the structural in/out/id). Anything in this list is rolled
# into the uniform ``metadata`` bag so no provenance is lost in the projection.
_EDGE_TABLES: Dict[str, List[str]] = {
    "relation": [
        "relation_type",
        "confidence",
        "status",
        "properties",
        "source_documents",
    ],
    "mentions": [
        "weight",
        "concept_name",
        "concept_type",
        "document_frequency",
    ],
    "cites": [
        "confidence",
        "reference_text",
        "match_method",
    ],
}


class GraphRepository:
    """Read-only unified loader over the ``relation``/``mentions``/``cites`` edges."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        self.config = config

    async def load_all_edges(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Load every edge incident on ``node_id`` across the KG edge tables.

        Returns a uniform shape per edge regardless of source table::

            {
                "id":        "<edge record id>",
                "source":    "<in endpoint id>",
                "target":    "<out endpoint id>",
                "edge_type": "relation" | "mentions" | "cites",
                "metadata":  { ...per-table data fields... },
            }

        Both directions are covered: an edge counts whether ``node_id`` is its
        ``in`` (the node is the source) OR its ``out`` (the node is the target),
        so a single call gives a node's full neighbourhood. A ``relation`` edge
        only ever touches entities, a ``cites`` edge only sources, a ``mentions``
        edge a source on ``in`` and an entity on ``out`` — so querying every
        table for one id is safe (a mismatched table simply returns nothing for
        that id) and keeps the caller ignorant of which table is relevant.

        Args:
            node_id: Full record id of the node (``entity:...`` / ``source:...``
                / etc.). Malformed ids return ``[]`` rather than raising — this
                is a read path a tool calls with caller-supplied ids.
            edge_types: Optional subset of ``{"relation", "mentions", "cites"}``
                to restrict the scan to. ``None`` loads all three. Unknown names
                are ignored.

        Returns:
            A flat list of uniform edge dicts (both directions, all requested
            tables). Empty when the node has no incident edges, on a bad id, or
            on a query failure (each table is loaded independently, so one
            table's failure does not suppress the others).
        """
        if not node_id:
            return []
        try:
            rid = ensure_record_id(node_id)
        except Exception as e:
            logger.error(f"load_all_edges: bad node_id {node_id!r}: {e}")
            return []

        if edge_types is None:
            tables = list(_EDGE_TABLES)
        else:
            requested = set(edge_types)
            tables = [t for t in _EDGE_TABLES if t in requested]

        edges: List[Dict[str, Any]] = []
        for table in tables:
            data_fields = _EDGE_TABLES[table]
            projection = ", ".join(
                ["id", "in AS source", "out AS target", *data_fields]
            )
            try:
                rows = await execute_query(
                    f"SELECT {projection} FROM {table} "
                    "WHERE in = $id OR out = $id",
                    {"id": rid},
                    self.config,
                )
            except Exception as e:
                logger.error(f"load_all_edges: load {table} failed: {e}")
                continue
            for row in rows or []:
                metadata = {
                    field: row.get(field)
                    for field in data_fields
                    if row.get(field) is not None
                }
                edges.append(
                    {
                        "id": str(row.get("id")) if row.get("id") is not None else None,
                        "source": str(row.get("source"))
                        if row.get("source") is not None
                        else None,
                        "target": str(row.get("target"))
                        if row.get("target") is not None
                        else None,
                        "edge_type": table,
                        "metadata": metadata,
                    }
                )
        return edges
