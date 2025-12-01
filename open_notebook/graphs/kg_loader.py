"""
Knowledge Graph Loader

Loads knowledge graph data from SurrealDB into NetworkX for analysis.
Supports:
- Full graph loading
- Incremental updates
- Subgraph extraction for large graphs

See docs/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md Phase 3.2 for documentation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from loguru import logger

from open_notebook.database.repository import repo_query
from open_notebook.domain.knowledge_graph import (
    Entity,
    EntityType,
    Claim,
    ClaimType,
)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class LoaderConfig:
    """Configuration for graph loading."""

    # Include node types
    include_entities: bool = True
    include_sources: bool = True
    include_claims: bool = True

    # Include edge types
    include_mentions: bool = True  # Source -> Entity
    include_entity_relations: bool = True  # Entity -> Entity
    include_supports: bool = True  # Source -> Claim
    include_same_as: bool = True  # Entity -> Entity (deduplication)

    # Filtering
    min_entity_mentions: int = 0  # Minimum mentions to include entity
    entity_types: Optional[List[EntityType]] = None  # Filter by type

    # Performance
    batch_size: int = 1000


# =============================================================================
# GRAPH LOADER
# =============================================================================


class KnowledgeGraphLoader:
    """
    Loads knowledge graph from SurrealDB into NetworkX.

    The graph structure:
    - Nodes: entities, sources (chunks/documents), claims
    - Edges: mentions, relations, supports, same_as

    Node attributes:
    - node_type: 'entity', 'source', 'claim'
    - (type-specific attributes)

    Edge attributes:
    - edge_type: 'mentions', 'relates_to', 'supports', 'same_as'
    - weight: edge weight for algorithms
    - (type-specific attributes)
    """

    def __init__(self, config: Optional[LoaderConfig] = None):
        self.config = config or LoaderConfig()
        self._graph: Optional[nx.DiGraph] = None
        self._node_index: Dict[str, Dict[str, Any]] = {}

    @property
    def graph(self) -> Optional[nx.DiGraph]:
        """Get the loaded graph."""
        return self._graph

    async def load_full_graph(self) -> nx.DiGraph:
        """
        Load the complete knowledge graph from SurrealDB.

        Returns:
            NetworkX DiGraph with all nodes and edges
        """
        logger.info("Loading full knowledge graph from SurrealDB...")

        self._graph = nx.DiGraph()
        self._node_index = {}

        # Load nodes
        if self.config.include_entities:
            await self._load_entities()

        if self.config.include_sources:
            await self._load_sources()

        if self.config.include_claims:
            await self._load_claims()

        # Load edges
        if self.config.include_mentions:
            await self._load_mentions()

        if self.config.include_entity_relations:
            await self._load_entity_relations()

        if self.config.include_supports:
            await self._load_supports()

        if self.config.include_same_as:
            await self._load_same_as()

        logger.info(
            f"Loaded graph with {self._graph.number_of_nodes()} nodes "
            f"and {self._graph.number_of_edges()} edges"
        )

        return self._graph

    async def load_subgraph(
        self,
        center_nodes: List[str],
        max_hops: int = 2,
    ) -> nx.DiGraph:
        """
        Load a subgraph centered on specific nodes.

        Useful for large graphs where full loading is impractical.

        Args:
            center_nodes: Node IDs to center the subgraph on
            max_hops: Maximum distance from center nodes to include

        Returns:
            NetworkX DiGraph containing the subgraph
        """
        logger.info(f"Loading subgraph centered on {len(center_nodes)} nodes, max_hops={max_hops}")

        # First, find all nodes within max_hops
        node_ids = await self._expand_neighborhood(center_nodes, max_hops)

        # Load those specific nodes and their edges
        self._graph = nx.DiGraph()
        self._node_index = {}

        await self._load_nodes_by_ids(node_ids)
        await self._load_edges_for_nodes(node_ids)

        logger.info(
            f"Loaded subgraph with {self._graph.number_of_nodes()} nodes "
            f"and {self._graph.number_of_edges()} edges"
        )

        return self._graph

    async def load_entity_subgraph(
        self,
        entity_ids: List[str],
        include_sources: bool = True,
        max_source_hops: int = 1,
    ) -> nx.DiGraph:
        """
        Load subgraph starting from specific entities.

        Optimized for HippoRAG retrieval where we start from query entities.

        Args:
            entity_ids: Entity IDs to start from
            include_sources: Include source nodes connected to entities
            max_source_hops: How many hops to follow from entities to sources

        Returns:
            NetworkX DiGraph
        """
        logger.info(f"Loading entity-centered subgraph for {len(entity_ids)} entities")

        self._graph = nx.DiGraph()
        self._node_index = {}

        # Load the seed entities
        await self._load_entities_by_ids(entity_ids)

        # Load related entities (same_as, relations)
        related_entities = await self._get_related_entities(entity_ids)
        if related_entities:
            await self._load_entities_by_ids(related_entities)

        all_entity_ids = set(entity_ids) | set(related_entities)

        # Load sources if requested
        if include_sources:
            source_ids = await self._get_sources_for_entities(list(all_entity_ids))
            if source_ids:
                await self._load_sources_by_ids(source_ids)

            # Load edges between entities and sources
            await self._load_mentions_for_entities(list(all_entity_ids))

        # Load entity-entity relations
        await self._load_relations_for_entities(list(all_entity_ids))

        # Load same_as links
        await self._load_same_as_for_entities(list(all_entity_ids))

        logger.info(
            f"Loaded entity subgraph with {self._graph.number_of_nodes()} nodes "
            f"and {self._graph.number_of_edges()} edges"
        )

        return self._graph

    async def update_incremental(
        self,
        since_timestamp: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        Incrementally update the graph with new/modified data.

        Args:
            since_timestamp: Only load data modified after this timestamp

        Returns:
            Tuple of (nodes_added, edges_added)
        """
        if self._graph is None:
            logger.warning("No graph loaded, performing full load instead")
            await self.load_full_graph()
            return self._graph.number_of_nodes(), self._graph.number_of_edges()

        initial_nodes = self._graph.number_of_nodes()
        initial_edges = self._graph.number_of_edges()

        # Load new entities
        if self.config.include_entities:
            await self._load_entities(since=since_timestamp)

        # Load new sources
        if self.config.include_sources:
            await self._load_sources(since=since_timestamp)

        # Load new edges
        if self.config.include_mentions:
            await self._load_mentions(since=since_timestamp)

        if self.config.include_entity_relations:
            await self._load_entity_relations(since=since_timestamp)

        nodes_added = self._graph.number_of_nodes() - initial_nodes
        edges_added = self._graph.number_of_edges() - initial_edges

        logger.info(f"Incremental update: +{nodes_added} nodes, +{edges_added} edges")

        return nodes_added, edges_added

    # =========================================================================
    # NODE LOADING
    # =========================================================================

    async def _load_entities(self, since: Optional[str] = None) -> None:
        """Load entity nodes."""
        query = "SELECT * FROM entity"
        params = {}

        if since:
            query += " WHERE updated_at > $since"
            params["since"] = since

        if self.config.entity_types:
            type_values = [t.value for t in self.config.entity_types]
            if since:
                query += " AND entity_type IN $types"
            else:
                query += " WHERE entity_type IN $types"
            params["types"] = type_values

        results = await repo_query(query, params)

        for r in results:
            node_id = r.get("id")
            if node_id:
                self._add_entity_node(node_id, r)

        logger.debug(f"Loaded {len(results)} entities")

    async def _load_entities_by_ids(self, entity_ids: List[str]) -> None:
        """Load specific entities by ID."""
        if not entity_ids:
            return

        # Batch load
        for i in range(0, len(entity_ids), self.config.batch_size):
            batch = entity_ids[i:i + self.config.batch_size]
            results = await repo_query(
                "SELECT * FROM entity WHERE id IN $ids",
                {"ids": batch}
            )
            for r in results:
                node_id = r.get("id")
                if node_id:
                    self._add_entity_node(node_id, r)

    async def _load_sources(self, since: Optional[str] = None) -> None:
        """Load source nodes (chunks/documents)."""
        query = "SELECT id, title, source_type, created_at FROM source"
        params = {}

        if since:
            query += " WHERE updated_at > $since"
            params["since"] = since

        results = await repo_query(query, params)

        for r in results:
            node_id = r.get("id")
            if node_id:
                self._add_source_node(node_id, r)

        logger.debug(f"Loaded {len(results)} sources")

    async def _load_sources_by_ids(self, source_ids: List[str]) -> None:
        """Load specific sources by ID."""
        if not source_ids:
            return

        for i in range(0, len(source_ids), self.config.batch_size):
            batch = source_ids[i:i + self.config.batch_size]
            results = await repo_query(
                "SELECT id, title, source_type, created_at FROM source WHERE id IN $ids",
                {"ids": batch}
            )
            for r in results:
                node_id = r.get("id")
                if node_id:
                    self._add_source_node(node_id, r)

    async def _load_claims(self, since: Optional[str] = None) -> None:
        """Load claim nodes."""
        query = "SELECT * FROM claim"
        params = {}

        if since:
            query += " WHERE updated_at > $since"
            params["since"] = since

        results = await repo_query(query, params)

        for r in results:
            node_id = r.get("id")
            if node_id:
                self._add_claim_node(node_id, r)

        logger.debug(f"Loaded {len(results)} claims")

    async def _load_nodes_by_ids(self, node_ids: Set[str]) -> None:
        """Load nodes by ID, detecting type from ID prefix."""
        entity_ids = [nid for nid in node_ids if nid.startswith("entity:")]
        source_ids = [nid for nid in node_ids if nid.startswith("source:")]
        claim_ids = [nid for nid in node_ids if nid.startswith("claim:")]

        await self._load_entities_by_ids(entity_ids)
        await self._load_sources_by_ids(source_ids)
        # Claims if needed

    def _add_entity_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """Add an entity node to the graph."""
        self._graph.add_node(
            node_id,
            node_type="entity",
            name=data.get("name", ""),
            entity_type=data.get("entity_type", "other"),
            description=data.get("description"),
            aliases=data.get("aliases", []),
        )
        self._node_index[node_id] = data

    def _add_source_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """Add a source node to the graph."""
        self._graph.add_node(
            node_id,
            node_type="source",
            title=data.get("title", ""),
            source_type=data.get("source_type", ""),
        )
        self._node_index[node_id] = data

    def _add_claim_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """Add a claim node to the graph."""
        self._graph.add_node(
            node_id,
            node_type="claim",
            statement=data.get("statement", ""),
            claim_type=data.get("claim_type", "factual"),
            confidence=data.get("confidence", 0.5),
        )
        self._node_index[node_id] = data

    # =========================================================================
    # EDGE LOADING
    # =========================================================================

    async def _load_mentions(self, since: Optional[str] = None) -> None:
        """Load mention edges (source -> entity)."""
        query = "SELECT * FROM mentions"
        params = {}

        if since:
            query += " WHERE created_at > $since"
            params["since"] = since

        results = await repo_query(query, params)

        for r in results:
            source_id = r.get("in")  # Source node
            entity_id = r.get("out")  # Entity node

            if source_id and entity_id:
                # Only add if both nodes exist
                if source_id in self._graph and entity_id in self._graph:
                    self._graph.add_edge(
                        source_id,
                        entity_id,
                        edge_type="mentions",
                        weight=r.get("confidence", 1.0),
                        context=r.get("context"),
                    )

        logger.debug(f"Loaded {len(results)} mention edges")

    async def _load_mentions_for_entities(self, entity_ids: List[str]) -> None:
        """Load mentions for specific entities."""
        if not entity_ids:
            return

        results = await repo_query(
            "SELECT * FROM mentions WHERE out IN $entity_ids",
            {"entity_ids": entity_ids}
        )

        for r in results:
            source_id = r.get("in")
            entity_id = r.get("out")

            if source_id and entity_id:
                if source_id in self._graph and entity_id in self._graph:
                    self._graph.add_edge(
                        source_id,
                        entity_id,
                        edge_type="mentions",
                        weight=r.get("confidence", 1.0),
                    )

    async def _load_entity_relations(self, since: Optional[str] = None) -> None:
        """Load entity-entity relation edges."""
        query = "SELECT * FROM relates_to"
        params = {}

        if since:
            query += " WHERE created_at > $since"
            params["since"] = since

        results = await repo_query(query, params)

        for r in results:
            from_id = r.get("in")
            to_id = r.get("out")

            if from_id and to_id:
                if from_id in self._graph and to_id in self._graph:
                    self._graph.add_edge(
                        from_id,
                        to_id,
                        edge_type="relates_to",
                        relation_type=r.get("relation_type", "related"),
                        weight=r.get("confidence", 1.0),
                    )

        logger.debug(f"Loaded {len(results)} relation edges")

    async def _load_relations_for_entities(self, entity_ids: List[str]) -> None:
        """Load relations for specific entities."""
        if not entity_ids:
            return

        results = await repo_query(
            "SELECT * FROM relates_to WHERE in IN $ids OR out IN $ids",
            {"ids": entity_ids}
        )

        for r in results:
            from_id = r.get("in")
            to_id = r.get("out")

            if from_id and to_id:
                if from_id in self._graph and to_id in self._graph:
                    self._graph.add_edge(
                        from_id,
                        to_id,
                        edge_type="relates_to",
                        relation_type=r.get("relation_type", "related"),
                        weight=r.get("confidence", 1.0),
                    )

    async def _load_supports(self, since: Optional[str] = None) -> None:
        """Load support edges (source -> claim)."""
        query = "SELECT * FROM supports"
        params = {}

        if since:
            query += " WHERE created_at > $since"
            params["since"] = since

        results = await repo_query(query, params)

        for r in results:
            source_id = r.get("in")
            claim_id = r.get("out")

            if source_id and claim_id:
                if source_id in self._graph and claim_id in self._graph:
                    self._graph.add_edge(
                        source_id,
                        claim_id,
                        edge_type="supports",
                        weight=r.get("strength", 1.0),
                    )

        logger.debug(f"Loaded {len(results)} support edges")

    async def _load_same_as(self, since: Optional[str] = None) -> None:
        """Load same_as edges (entity deduplication links)."""
        query = "SELECT * FROM same_as"
        params = {}

        if since:
            query += " WHERE created_at > $since"
            params["since"] = since

        results = await repo_query(query, params)

        for r in results:
            entity1 = r.get("in")
            entity2 = r.get("out")

            if entity1 and entity2:
                if entity1 in self._graph and entity2 in self._graph:
                    self._graph.add_edge(
                        entity1,
                        entity2,
                        edge_type="same_as",
                        weight=r.get("similarity", 1.0),
                    )

        logger.debug(f"Loaded {len(results)} same_as edges")

    async def _load_same_as_for_entities(self, entity_ids: List[str]) -> None:
        """Load same_as links for specific entities."""
        if not entity_ids:
            return

        results = await repo_query(
            "SELECT * FROM same_as WHERE in IN $ids OR out IN $ids",
            {"ids": entity_ids}
        )

        for r in results:
            entity1 = r.get("in")
            entity2 = r.get("out")

            if entity1 and entity2:
                if entity1 in self._graph and entity2 in self._graph:
                    self._graph.add_edge(
                        entity1,
                        entity2,
                        edge_type="same_as",
                        weight=r.get("similarity", 1.0),
                    )

    async def _load_edges_for_nodes(self, node_ids: Set[str]) -> None:
        """Load all edges where both endpoints are in node_ids."""
        # This is a simplified version - in production you'd want
        # to query edges more efficiently
        await self._load_mentions()
        await self._load_entity_relations()
        await self._load_supports()
        await self._load_same_as()

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    async def _expand_neighborhood(
        self,
        center_nodes: List[str],
        max_hops: int,
    ) -> Set[str]:
        """
        Find all nodes within max_hops of center_nodes using BFS in SurrealDB.
        """
        node_ids = set(center_nodes)
        current_frontier = set(center_nodes)

        for hop in range(max_hops):
            if not current_frontier:
                break

            # Find neighbors of current frontier
            # This query finds nodes connected via any edge type
            results = await repo_query(
                """
                SELECT out FROM mentions WHERE in IN $nodes
                UNION
                SELECT in FROM mentions WHERE out IN $nodes
                UNION
                SELECT out FROM relates_to WHERE in IN $nodes
                UNION
                SELECT in FROM relates_to WHERE out IN $nodes
                UNION
                SELECT out FROM same_as WHERE in IN $nodes
                UNION
                SELECT in FROM same_as WHERE out IN $nodes
                """,
                {"nodes": list(current_frontier)}
            )

            next_frontier = set()
            for r in results:
                neighbor = r.get("out") or r.get("in")
                if neighbor and neighbor not in node_ids:
                    next_frontier.add(neighbor)
                    node_ids.add(neighbor)

            current_frontier = next_frontier

        return node_ids

    async def _get_related_entities(self, entity_ids: List[str]) -> List[str]:
        """Get entities related to the given entities."""
        results = await repo_query(
            """
            SELECT out FROM relates_to WHERE in IN $ids
            UNION
            SELECT in FROM relates_to WHERE out IN $ids
            UNION
            SELECT out FROM same_as WHERE in IN $ids
            UNION
            SELECT in FROM same_as WHERE out IN $ids
            """,
            {"ids": entity_ids}
        )

        related = []
        for r in results:
            eid = r.get("out") or r.get("in")
            if eid and eid not in entity_ids:
                related.append(eid)

        return related

    async def _get_sources_for_entities(self, entity_ids: List[str]) -> List[str]:
        """Get source IDs that mention the given entities."""
        results = await repo_query(
            "SELECT in FROM mentions WHERE out IN $entity_ids",
            {"entity_ids": entity_ids}
        )

        return [r.get("in") for r in results if r.get("in")]

    def get_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the original data for a node."""
        return self._node_index.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """Get all node IDs of a specific type."""
        if self._graph is None:
            return []

        return [
            node
            for node, data in self._graph.nodes(data=True)
            if data.get("node_type") == node_type
        ]

    def get_entity_nodes(self) -> List[str]:
        """Get all entity node IDs."""
        return self.get_nodes_by_type("entity")

    def get_source_nodes(self) -> List[str]:
        """Get all source node IDs."""
        return self.get_nodes_by_type("source")


# =============================================================================
# SCORE CACHING
# =============================================================================


class ScoreCache:
    """
    Caches computed scores (PPR, centrality, etc.) back to SurrealDB.

    This avoids recomputing expensive metrics on every query.
    """

    def __init__(self):
        pass

    async def cache_ppr_scores(
        self,
        scores: Dict[str, float],
        query_id: str,
    ) -> None:
        """
        Cache PPR scores for a specific query.

        Args:
            scores: Dict mapping node_id to PPR score
            query_id: Identifier for the query (for later retrieval)
        """
        # Store as a single document
        await repo_query(
            """
            CREATE ppr_cache SET
                query_id = $query_id,
                scores = $scores,
                created_at = time::now()
            """,
            {"query_id": query_id, "scores": scores}
        )

    async def get_cached_ppr(self, query_id: str) -> Optional[Dict[str, float]]:
        """Retrieve cached PPR scores."""
        results = await repo_query(
            "SELECT scores FROM ppr_cache WHERE query_id = $query_id",
            {"query_id": query_id}
        )

        if results:
            return results[0].get("scores")
        return None

    async def cache_centrality_scores(
        self,
        node_id: str,
        scores: Dict[str, float],
    ) -> None:
        """
        Cache centrality scores on a node.

        Args:
            node_id: The node to update
            scores: Dict with 'degree', 'pagerank', etc.
        """
        await repo_query(
            """
            UPDATE $node_id SET
                centrality_degree = $degree,
                centrality_pagerank = $pagerank,
                centrality_updated_at = time::now()
            """,
            {
                "node_id": node_id,
                "degree": scores.get("degree", 0),
                "pagerank": scores.get("pagerank", 0),
            }
        )

    async def cache_community_assignments(
        self,
        communities: Dict[str, int],
    ) -> None:
        """
        Cache community assignments on nodes.

        Args:
            communities: Dict mapping node_id to community_id
        """
        # Batch update nodes with their community
        for node_id, community_id in communities.items():
            await repo_query(
                "UPDATE $node_id SET community_id = $community_id",
                {"node_id": node_id, "community_id": community_id}
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def load_knowledge_graph(
    config: Optional[LoaderConfig] = None,
) -> nx.DiGraph:
    """Convenience function to load the full knowledge graph."""
    loader = KnowledgeGraphLoader(config)
    return await loader.load_full_graph()


async def load_entity_neighborhood(
    entity_ids: List[str],
    include_sources: bool = True,
) -> nx.DiGraph:
    """Convenience function to load subgraph around entities."""
    loader = KnowledgeGraphLoader()
    return await loader.load_entity_subgraph(entity_ids, include_sources)
