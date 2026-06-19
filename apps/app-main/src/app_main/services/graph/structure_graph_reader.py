"""Reads a source's structure subgraph for the API (Phase I.F).

Returns a Sigma/graphology-friendly ``{nodes, edges}`` payload bounded by a
page-limit so a 10K-page document never streams its whole 100K-node graph in one
response (Risk 3). The ``dn_source`` index (migration 49) makes the per-source
filter cheap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from surrealdb_service.connection import ensure_record_id, execute_query

# Hard cap on ``page_limit`` (Q-I-F-2). The API rejects anything above this.
MAX_PAGE_LIMIT = 500
DEFAULT_PAGE_LIMIT = 100


async def read_structure_graph(
    source_id: str,
    *,
    page: Optional[int] = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    config: Any = None,
) -> Dict[str, Any]:
    """Load the structure subgraph for ``source_id``.

    Args:
        source_id: Source record id (e.g. ``"source:abc"``).
        page: Optional physical-page filter. When given, only doc_nodes on that
            page (plus their section ancestors, which carry no page) are
            returned; edges are restricted to the returned node set.
        page_limit: Max number of doc_node rows to return. Capped at
            :data:`MAX_PAGE_LIMIT` by the caller (the router raises 422 above
            the cap before reaching this function).
        config: Optional SurrealDB config override (tests/seam).

    Returns:
        ``{"nodes": [...], "edges": [...], "total_nodes": int,
        "truncated": bool}``. ``edges`` entries carry ``source``, ``target`` and
        ``type`` (parent_of | next_node | derived_from).
    """
    sid = ensure_record_id(source_id)
    limit = min(max(int(page_limit), 1), MAX_PAGE_LIMIT)

    where = "source = $source_id"
    params: Dict[str, Any] = {"source_id": sid, "limit": limit}
    if page is not None:
        # Section nodes have a NONE page; keep them so the tree stays connected.
        where += " AND (page = $page OR page = NONE)"
        params["page"] = int(page)

    try:
        nodes = await execute_query(
            "SELECT id, self_ref, element_type, text, page, level, sequence, "
            f"bbox FROM doc_node WHERE {where} "
            "ORDER BY sequence ASC LIMIT $limit",
            params,
            config,
        )
    except Exception as e:
        logger.error(f"Failed to read structure graph for {source_id}: {e}")
        return {"nodes": [], "edges": [], "total_nodes": 0, "truncated": False}

    nodes = nodes or []
    if not nodes:
        return {"nodes": [], "edges": [], "total_nodes": 0, "truncated": False}

    node_ids = [n["id"] for n in nodes]

    edges: List[Dict[str, Any]] = []
    for table in ("parent_of", "next_node"):
        try:
            rows = await execute_query(
                "SELECT in AS source, out AS target "
                f"FROM {table} "
                "WHERE in INSIDE $ids AND out INSIDE $ids",
                {"ids": node_ids},
                config,
            )
        except Exception as e:
            logger.error(f"Failed to read {table} for {source_id}: {e}")
            rows = []
        for row in rows or []:
            edges.append(
                {
                    "source": str(row["source"]),
                    "target": str(row["target"]),
                    "type": table,
                }
            )

    # derived_from runs chunk -> doc_node; only the doc_node endpoint is in our
    # node set, so filter on ``out`` only. The chunk id is surfaced so the UI can
    # map a clicked node back to its chunk for the bbox highlight.
    try:
        derived = await execute_query(
            "SELECT in AS source, out AS target FROM derived_from "
            "WHERE out INSIDE $ids",
            {"ids": node_ids},
            config,
        )
    except Exception as e:
        logger.error(f"Failed to read derived_from for {source_id}: {e}")
        derived = []
    for row in derived or []:
        edges.append(
            {
                "source": str(row["source"]),
                "target": str(row["target"]),
                "type": "derived_from",
            }
        )

    # Attach the owning chunk id onto each leaf node (drives the UI highlight).
    chunk_by_node = {
        str(row["target"]): str(row["source"])
        for row in (derived or [])
    }
    shaped_nodes = [
        {
            "id": str(n["id"]),
            "self_ref": n.get("self_ref"),
            "element_type": n.get("element_type"),
            "text": n.get("text"),
            "page": n.get("page"),
            "level": n.get("level"),
            "sequence": n.get("sequence"),
            "bbox": n.get("bbox"),
            "chunk_id": chunk_by_node.get(str(n["id"])),
        }
        for n in nodes
    ]

    return {
        "nodes": shaped_nodes,
        "edges": edges,
        "total_nodes": len(shaped_nodes),
        "truncated": len(shaped_nodes) >= limit,
    }
