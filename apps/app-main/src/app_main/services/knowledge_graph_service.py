"""
Knowledge graph service - business logic for entity browsing and graph visualization.

Wraps the EntityRepository and GraphLayoutService to provide paginated entity
access, search, and pre-computed graph layouts for the frontend.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from surrealdb_service.repositories.entity import EntityRepository


class KnowledgeGraphService:
    """Service for knowledge graph browsing and visualization."""

    def __init__(self, entity_repo: EntityRepository):
        self.entity_repo = entity_repo

    async def list_entities(
        self,
        limit: int = 50,
        offset: int = 0,
        entity_type: Optional[str] = None,
        status: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List entities with pagination + optional type/status/source filters.

        Joins the Q.2 triage signals onto each page row so the KG table can show
        per-entity ``structural_degree`` and ``doc_count`` (Q.5 AC2):

        * ``doc_count`` is the distinct-document count from ``source_documents``.
        * ``structural_degree`` is the effective degree (structural edges +
          promoted weak edges), computed in ONE batched relation query for the
          whole page rather than per row.

        The signals join is best-effort: any failure (e.g. a missing relation
        table in a fresh DB) degrades to omitting the two fields, never breaking
        the listing.
        """
        rows = await self.entity_repo.list_entities(
            limit=limit,
            offset=offset,
            entity_type=entity_type,
            status=status,
            source_id=source_id,
        )
        return await self._attach_signals(rows)

    async def count_entities(
        self,
        entity_type: Optional[str] = None,
        status: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> int:
        """Count entities with optional type/status/source filters."""
        return await self.entity_repo.count_entities(
            entity_type=entity_type, status=status, source_id=source_id
        )

    async def _attach_signals(
        self, rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Annotate a page of entity rows with degree + doc-count (Q.5).

        One batched effective-degree query covers the whole page. Imported lazily
        so the repository-only callers of this service stay free of the triage
        config dependency, and wrapped so a signals failure never breaks the
        listing (the rows still render, just without the two extra fields).
        """
        if not rows:
            return rows
        try:
            from app_main.services.triage.signals_service import (
                TriageSignalsService,
            )

            signals = TriageSignalsService(self.entity_repo)
            ids = [str(r["id"]) for r in rows if r.get("id") is not None]
            degree_by_id = await signals._batch_effective_degree(ids)
            for row in rows:
                rid = str(row.get("id"))
                row["structural_degree"] = int(degree_by_id.get(rid, 0))
                row["doc_count"] = signals.doc_count(row)
        except Exception as e:  # pragma: no cover - defensive degrade
            logger.warning(f"list_entities: signals join skipped: {e}")
        return rows

    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get a single entity with its relations and source metadata.

        Joins the Q.2 triage signals (``structural_degree`` + ``doc_count``) so
        the detail panel can show connectivity alongside the ``status`` /
        ``manual_override`` already carried by the ``SELECT *`` detail row (Q.5).
        """
        detail = await self.entity_repo.get_entity_detail(entity_id)
        if detail is None:
            return None
        (annotated,) = await self._attach_signals([detail])
        return annotated

    async def get_entity_types_summary(self) -> List[Dict[str, Any]]:
        """Get entity counts grouped by type."""
        return await self.entity_repo.get_entity_types_summary()

    async def search_entities(
        self, query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search entities by name text."""
        return await self.entity_repo.search_entities(query=query, limit=limit)

    async def get_graph_data(
        self,
        entity_type: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """Get graph data (nodes + edges) for visualization.

        Returns raw nodes and edges suitable for client-side rendering.
        Layout computation is handled by the frontend graph library.

        ``limit`` is a soft cap on the *node* count. The repository is
        edge-first: it selects the active relation core, derives nodes
        from the relation endpoints (so edges always render), then pads
        the node budget with the highest-weight active entities.
        """
        raw = await self.entity_repo.get_all_entities_and_relations(
            entity_type=entity_type, limit=limit
        )
        return _to_graph_json(raw)


def _to_graph_json(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw entity/relation data to graph-friendly JSON."""
    nodes = []
    for node in raw.get("nodes", []):
        nodes.append({
            "id": node["id"],
            "label": node.get("name", ""),
            "entity_type": node.get("entity_type", ""),
            "weight": node.get("weight", 1),
        })

    edges = []
    for edge in raw.get("edges", []):
        edges.append({
            "source": edge.get("source", edge.get("in", "")),
            "target": edge.get("target", edge.get("out", "")),
            "label": edge.get("relation_type", ""),
        })

    return {"nodes": nodes, "edges": edges}
