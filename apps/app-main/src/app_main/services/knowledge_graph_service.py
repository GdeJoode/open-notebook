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
    ) -> List[Dict[str, Any]]:
        """List entities with pagination and optional type filter."""
        return await self.entity_repo.list_entities(
            limit=limit, offset=offset, entity_type=entity_type
        )

    async def count_entities(self, entity_type: Optional[str] = None) -> int:
        """Count entities with optional type filter."""
        return await self.entity_repo.count_entities(entity_type=entity_type)

    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get a single entity with its relations and source metadata."""
        return await self.entity_repo.get_entity_detail(entity_id)

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
        limit: int = 5000,
    ) -> Dict[str, Any]:
        """Get graph data (nodes + edges) for visualization.

        Returns raw nodes and edges suitable for client-side rendering.
        Layout computation is handled by the frontend graph library.
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
