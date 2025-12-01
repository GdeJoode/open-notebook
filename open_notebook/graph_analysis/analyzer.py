"""
High-level graph analysis interface.

This module provides the main GraphAnalyzer class that:
- Loads graph data from SurrealDB
- Manages caching and synchronization
- Provides HippoRAG-style retrieval
- Exposes graph analysis algorithms
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from loguru import logger

from open_notebook.database.repository import repo_query
from open_notebook.graph_analysis.base import (
    CentralityMethod,
    CommunityAlgorithm,
    CommunityResult,
    EdgeInfo,
    GraphBackend,
    NodeInfo,
    PPRResult,
)
from open_notebook.graph_analysis.networkx_backend import NetworkXBackend


class GraphAnalyzer:
    """
    High-level interface for graph analysis operations.

    Manages:
    - Loading graph data from SurrealDB
    - Caching and synchronization
    - Algorithm execution via pluggable backend
    - Result caching back to SurrealDB

    Example usage:
        analyzer = GraphAnalyzer()
        await analyzer.load_full_graph()

        # HippoRAG-style retrieval
        results = await analyzer.hipporag_retrieve(
            query_embedding=query_vec,
            fact_scores={"entity:abc": 0.9, "entity:xyz": 0.7},
            top_k=20
        )

        # Centrality analysis
        scores = await analyzer.compute_centrality(CentralityMethod.PAGERANK)

        # Community detection
        communities = await analyzer.detect_communities()
    """

    def __init__(
        self,
        backend_class: Type[GraphBackend] = NetworkXBackend,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Initialize the GraphAnalyzer.

        Args:
            backend_class: Graph backend implementation to use
            cache_ttl_seconds: Time-to-live for cached graph data
        """
        self._backend: GraphBackend = backend_class()
        self._cache_ttl = cache_ttl_seconds
        self._last_sync: Optional[float] = None
        self._node_id_to_surreal_id: Dict[str, str] = {}

    @property
    def backend(self) -> GraphBackend:
        """Access underlying backend for advanced operations."""
        return self._backend

    @property
    def is_loaded(self) -> bool:
        """Check if graph data has been loaded."""
        return self._last_sync is not None

    @property
    def node_count(self) -> int:
        """Get number of nodes in the graph."""
        return self._backend.get_node_count()

    @property
    def edge_count(self) -> int:
        """Get number of edges in the graph."""
        return self._backend.get_edge_count()

    def _is_cache_valid(self) -> bool:
        """Check if cached graph is still valid."""
        if not self._last_sync:
            return False
        return time.time() - self._last_sync < self._cache_ttl

    async def load_full_graph(self, force_reload: bool = False) -> None:
        """
        Load complete graph from SurrealDB.

        For large graphs, consider using load_subgraph() instead.

        Args:
            force_reload: Force reload even if cache is valid
        """
        if not force_reload and self._is_cache_valid():
            logger.debug("Using cached graph")
            return

        logger.info("Loading full graph from SurrealDB...")

        nodes: List[NodeInfo] = []
        edges: List[EdgeInfo] = []

        # Load all node types in parallel
        source_query = repo_query("SELECT id, source_type, title FROM source")
        entity_query = repo_query("SELECT id, entity_type, name FROM entity")
        person_query = repo_query("SELECT id, name FROM person")
        organization_query = repo_query("SELECT id, name, org_type FROM organization")
        topic_query = repo_query("SELECT id, name, level FROM topic")
        claim_query = repo_query("SELECT id, claim_type, statement FROM claim")

        sources = await source_query
        entities = await entity_query
        persons = await person_query
        organizations = await organization_query
        topics = await topic_query
        claims = await claim_query

        # Process nodes
        for s in sources:
            node_id = self._normalize_id(s.get("id"))
            nodes.append(
                NodeInfo(
                    id=node_id,
                    node_type="source",
                    attributes={
                        "source_type": s.get("source_type"),
                        "title": s.get("title"),
                    },
                )
            )

        for e in entities:
            node_id = self._normalize_id(e.get("id"))
            nodes.append(
                NodeInfo(
                    id=node_id,
                    node_type="entity",
                    attributes={
                        "entity_type": e.get("entity_type"),
                        "name": e.get("name"),
                    },
                )
            )

        for p in persons:
            node_id = self._normalize_id(p.get("id"))
            nodes.append(
                NodeInfo(
                    id=node_id,
                    node_type="person",
                    attributes={"name": p.get("name")},
                )
            )

        for o in organizations:
            node_id = self._normalize_id(o.get("id"))
            nodes.append(
                NodeInfo(
                    id=node_id,
                    node_type="organization",
                    attributes={"name": o.get("name"), "org_type": o.get("org_type")},
                )
            )

        for t in topics:
            node_id = self._normalize_id(t.get("id"))
            nodes.append(
                NodeInfo(
                    id=node_id,
                    node_type="topic",
                    attributes={"name": t.get("name"), "level": t.get("level")},
                )
            )

        for c in claims:
            node_id = self._normalize_id(c.get("id"))
            nodes.append(
                NodeInfo(
                    id=node_id,
                    node_type="claim",
                    attributes={
                        "claim_type": c.get("claim_type"),
                        "statement": c.get("statement"),
                    },
                )
            )

        # Load all edge types
        edge_queries = {
            "cites": "SELECT in, out FROM cites",
            "mentions": "SELECT in, out, confidence FROM mentions",
            "same_as": "SELECT in, out, similarity FROM same_as",
            "authored_by": "SELECT in, out, role FROM authored_by",
            "affiliated_with": "SELECT in, out FROM affiliated_with",
            "discusses": "SELECT in, out, relevance FROM discusses",
            "supports": "SELECT in, out, strength FROM supports",
            "contradicts": "SELECT in, out, strength FROM contradicts",
            "broader_than": "SELECT in, out FROM broader_than",
            "related_to_topic": "SELECT in, out, strength FROM related_to_topic",
            "implements": "SELECT in, out FROM implements",
            "supersedes": "SELECT in, out FROM supersedes",
            "leads_to": "SELECT in, out FROM leads_to",
        }

        for edge_type, query in edge_queries.items():
            try:
                results = await repo_query(query)
                for r in results:
                    source_id = self._normalize_id(r.get("in"))
                    target_id = self._normalize_id(r.get("out"))

                    # Determine weight based on edge type
                    weight = self._get_edge_weight(edge_type, r)

                    edges.append(
                        EdgeInfo(
                            source_id=source_id,
                            target_id=target_id,
                            edge_type=edge_type,
                            weight=weight,
                            attributes={
                                k: v
                                for k, v in r.items()
                                if k not in ("in", "out", "id")
                            },
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to load {edge_type} edges: {e}")

        # Load into backend
        self._backend.load_from_edges(edges, nodes)
        self._last_sync = time.time()

        logger.info(
            f"Graph loaded: {self._backend.get_node_count()} nodes, "
            f"{self._backend.get_edge_count()} edges"
        )

    def _normalize_id(self, surreal_id: Any) -> str:
        """Normalize SurrealDB record ID to string."""
        if surreal_id is None:
            return ""
        # Handle RecordId objects
        if hasattr(surreal_id, "id"):
            return f"{surreal_id.table}:{surreal_id.id}"
        return str(surreal_id)

    def _get_edge_weight(self, edge_type: str, edge_data: Dict[str, Any]) -> float:
        """Determine edge weight based on type and attributes."""
        if edge_type == "mentions":
            return edge_data.get("confidence", 1.0)
        elif edge_type == "same_as":
            return edge_data.get("similarity", 0.8)
        elif edge_type == "discusses":
            return edge_data.get("relevance", 1.0)
        elif edge_type in ("supports", "contradicts"):
            strength = edge_data.get("strength", "moderate")
            return {"weak": 0.3, "moderate": 0.6, "strong": 1.0}.get(strength, 0.6)
        elif edge_type == "related_to_topic":
            return edge_data.get("strength", 1.0)
        else:
            return 1.0

    async def load_subgraph(
        self,
        seed_ids: List[str],
        hops: int = 2,
        edge_types: Optional[List[str]] = None,
    ) -> None:
        """
        Load a subgraph starting from seed nodes.

        More efficient for large graphs when you only need local analysis.

        Args:
            seed_ids: Starting node IDs
            hops: Number of hops to traverse from seeds
            edge_types: Filter to specific edge types
        """
        # Load full graph first, then extract subgraph
        await self.load_full_graph()

        # Get neighbors from loaded graph
        all_nodes = set(seed_ids)
        for seed in seed_ids:
            neighbors = self._backend.get_neighbors(
                seed, hops=hops, edge_types=edge_types
            )
            all_nodes.update(neighbors)

        # Extract subgraph
        self._backend = self._backend.get_subgraph(list(all_nodes))

        logger.info(
            f"Subgraph loaded: {self._backend.get_node_count()} nodes, "
            f"{self._backend.get_edge_count()} edges"
        )

    async def hipporag_retrieve(
        self,
        query_embedding: np.ndarray,
        fact_scores: Dict[str, float],
        top_k: int = 20,
        damping: float = 0.5,
        passage_weight: float = 0.05,
    ) -> List[Tuple[str, float]]:
        """
        HippoRAG-style retrieval using Personalized PageRank.

        This implements the core HippoRAG retrieval algorithm:
        1. Use fact scores as seed weights for entities
        2. Run PPR from entity seeds to find relevant passages
        3. Return top-k source documents

        Args:
            query_embedding: Query vector for dense fallback
            fact_scores: Dict of entity_id -> relevance score from fact matching
            top_k: Number of results to return
            damping: PPR damping factor (HippoRAG uses 0.5 for aggressive teleportation)
            passage_weight: Weight multiplier for passage/source nodes

        Returns:
            List of (source_id, score) tuples ranked by relevance
        """
        await self.load_full_graph()

        if not fact_scores:
            # Fallback to dense retrieval when no entity matches
            logger.info("No fact scores, falling back to dense retrieval")
            return await self._dense_fallback(query_embedding, top_k)

        # Build reset probability vector (teleportation distribution)
        reset_prob: Dict[str, float] = {}

        # Add entity weights from fact scores
        for entity_id, score in fact_scores.items():
            if self._backend.has_node(entity_id):
                reset_prob[entity_id] = score

        # Add passage weights (lower weight as per HippoRAG paper)
        source_nodes = self._backend.get_nodes_by_type("source")
        for source_id in source_nodes:
            # Could incorporate dense similarity here for hybrid approach
            reset_prob[source_id] = passage_weight

        if not reset_prob:
            logger.warning("No valid seed nodes found in graph")
            return await self._dense_fallback(query_embedding, top_k)

        # Run Personalized PageRank
        ppr_result = self._backend.personalized_pagerank(
            reset_prob=reset_prob, damping=damping
        )

        # Filter to only source nodes and return top-k
        source_ppr = ppr_result.filter_by_prefix("source:")
        return source_ppr.top_k(top_k)

    async def _dense_fallback(
        self, query_embedding: np.ndarray, top_k: int
    ) -> List[Tuple[str, float]]:
        """Dense retrieval fallback when graph retrieval fails."""
        # Use SurrealDB vector search
        results = await repo_query(
            """
            SELECT id, vector::similarity::cosine(embedding, $query_emb) AS score
            FROM source
            WHERE embedding != NONE
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"query_emb": query_embedding.tolist(), "limit": top_k},
        )
        return [(self._normalize_id(r["id"]), r["score"]) for r in results]

    async def compute_centrality(
        self,
        method: CentralityMethod = CentralityMethod.PAGERANK,
        node_type: Optional[str] = None,
        cache_results: bool = True,
    ) -> Dict[str, float]:
        """
        Compute centrality scores for nodes.

        Args:
            method: Centrality algorithm to use
            node_type: Filter to specific node type (e.g., "source", "person")
            cache_results: Whether to cache results back to SurrealDB

        Returns:
            Dict mapping node_id to centrality score
        """
        await self.load_full_graph()

        scores = self._backend.get_centrality(method)

        # Filter by node type if specified
        if node_type:
            scores = {
                k: v
                for k, v in scores.items()
                if self._backend.get_node_type(k) == node_type
            }

        # Cache to SurrealDB
        if cache_results:
            await self._cache_scores_to_surreal(scores, f"centrality_{method.value}")

        return scores

    async def _cache_scores_to_surreal(
        self, scores: Dict[str, float], score_type: str
    ) -> None:
        """Cache computed scores back to SurrealDB."""
        # Group by table type
        table_scores: Dict[str, Dict[str, float]] = {}
        for node_id, score in scores.items():
            if ":" in node_id:
                table, record_id = node_id.split(":", 1)
                if table not in table_scores:
                    table_scores[table] = {}
                table_scores[table][node_id] = score

        # Update each table
        for table, node_scores in table_scores.items():
            for node_id, score in node_scores.items():
                try:
                    await repo_query(
                        f"""
                        UPDATE {node_id} SET cached_scores.{score_type} = $score
                        """,
                        {"score": score},
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache score for {node_id}: {e}")

    async def detect_communities(
        self,
        algorithm: CommunityAlgorithm = CommunityAlgorithm.LOUVAIN,
        cache_results: bool = True,
    ) -> CommunityResult:
        """
        Detect communities in the graph.

        Args:
            algorithm: Community detection algorithm
            cache_results: Whether to cache community assignments

        Returns:
            CommunityResult with partition and modularity
        """
        await self.load_full_graph()

        result = self._backend.detect_communities(algorithm)

        logger.info(
            f"Found {len(result.communities)} communities "
            f"(modularity: {result.modularity:.3f})"
        )

        return result

    async def find_experts(
        self,
        topic_id: str,
        min_publications: int = 2,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find experts on a topic using graph analysis.

        Combines:
        - Publication count on topic
        - Citation centrality
        - Co-authorship network position

        Args:
            topic_id: Topic to find experts for
            min_publications: Minimum publications required
            top_k: Number of experts to return

        Returns:
            List of expert info dicts with scores
        """
        # Use SurrealDB query for complex traversal
        query = """
        SELECT
            person.id AS person_id,
            person.name AS name,
            count(->authored_by<-source) AS publications,
            count(->authored_by<-source<-cites) AS citations,
            person.h_index AS h_index,
            person.expertise_areas AS expertise
        FROM person
        WHERE
            ->authored_by<-source->discusses->topic CONTAINS $topic_id
        GROUP BY person.id
        ORDER BY citations DESC
        LIMIT $limit
        """

        try:
            results = await repo_query(
                query, {"topic_id": topic_id, "limit": top_k}
            )
            return [
                {
                    "person_id": self._normalize_id(r.get("person_id")),
                    "name": r.get("name"),
                    "publications": r.get("publications", 0),
                    "citations": r.get("citations", 0),
                    "h_index": r.get("h_index"),
                    "expertise": r.get("expertise", []),
                }
                for r in results
                if r.get("publications", 0) >= min_publications
            ]
        except Exception as e:
            logger.error(f"Expert search query failed: {e}")
            return []

    async def get_citation_network(
        self, source_id: str, depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get citation network around a source.

        Args:
            source_id: Central source document
            depth: Number of citation hops to include

        Returns:
            Dict with network statistics and nodes
        """
        await self.load_full_graph()

        # Get neighbors via cites edges
        citing = self._backend.get_neighbors(
            source_id, hops=depth, edge_types=["cites"], direction="in"
        )
        cited = self._backend.get_neighbors(
            source_id, hops=depth, edge_types=["cites"], direction="out"
        )

        return {
            "source_id": source_id,
            "citing_sources": citing,
            "cited_sources": cited,
            "in_citations": len(citing),
            "out_citations": len(cited),
            "total_network_size": len(citing) + len(cited) + 1,
        }

    async def find_similar_entities(
        self, entity_id: str, threshold: float = 0.8, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find entities similar to a given entity using same_as relations.

        Args:
            entity_id: Entity to find similar entities for
            threshold: Minimum similarity score
            limit: Maximum results

        Returns:
            List of similar entity info
        """
        query = """
        SELECT
            out AS entity_id,
            similarity,
            out.name AS name,
            out.entity_type AS entity_type
        FROM same_as
        WHERE in = $entity_id AND similarity >= $threshold
        ORDER BY similarity DESC
        LIMIT $limit
        """

        try:
            results = await repo_query(
                query,
                {"entity_id": entity_id, "threshold": threshold, "limit": limit},
            )
            return [
                {
                    "entity_id": self._normalize_id(r.get("entity_id")),
                    "name": r.get("name"),
                    "entity_type": r.get("entity_type"),
                    "similarity": r.get("similarity"),
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Similar entity search failed: {e}")
            return []

    def invalidate_cache(self) -> None:
        """Force graph reload on next operation."""
        self._last_sync = None

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded graph."""
        if not self.is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "node_count": self._backend.get_node_count(),
            "edge_count": self._backend.get_edge_count(),
            "last_sync": self._last_sync,
            "cache_valid": self._is_cache_valid(),
        }
