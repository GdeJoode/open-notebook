"""
Knowledge Graph Analysis Algorithms

Implements HippoRAG-style graph analysis:
- Personalized PageRank (PPR) with custom reset probabilities
- Centrality calculations (degree, betweenness, eigenvector)
- Community detection
- Path finding utilities

See docs/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md Phase 3 for documentation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
from loguru import logger


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PPRConfig:
    """Configuration for Personalized PageRank."""

    # Damping factor (probability of following an edge vs teleporting)
    alpha: float = 0.85

    # Maximum iterations for convergence
    max_iter: int = 100

    # Convergence tolerance
    tol: float = 1e-6

    # Weight attribute name on edges
    weight_attr: str = "weight"


@dataclass
class AnalysisConfig:
    """Configuration for graph analysis."""

    ppr: PPRConfig = field(default_factory=PPRConfig)

    # Centrality settings
    centrality_normalized: bool = True

    # Community detection settings
    community_resolution: float = 1.0  # Higher = more communities

    # Path finding settings
    max_path_length: int = 5


# =============================================================================
# PERSONALIZED PAGERANK
# =============================================================================


class PersonalizedPageRank:
    """
    Personalized PageRank implementation for HippoRAG-style retrieval.

    PPR computes node importance relative to a set of seed nodes,
    which makes it ideal for finding relevant passages given query entities.
    """

    def __init__(self, config: Optional[PPRConfig] = None):
        self.config = config or PPRConfig()

    def compute(
        self,
        graph: nx.Graph,
        seed_nodes: Union[List[str], Dict[str, float]],
        alpha: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute Personalized PageRank scores.

        Args:
            graph: NetworkX graph (directed or undirected)
            seed_nodes: Either a list of seed node IDs (uniform distribution)
                       or a dict mapping node IDs to personalization weights
            alpha: Damping factor (overrides config if provided)

        Returns:
            Dict mapping node IDs to PPR scores
        """
        if not graph.nodes():
            return {}

        alpha = alpha or self.config.alpha

        # Build personalization vector
        if isinstance(seed_nodes, list):
            # Uniform distribution over seed nodes
            personalization = {
                node: 1.0 / len(seed_nodes)
                for node in seed_nodes
                if node in graph
            }
        else:
            # Use provided weights (normalize them)
            total = sum(w for n, w in seed_nodes.items() if n in graph)
            if total > 0:
                personalization = {
                    n: w / total for n, w in seed_nodes.items() if n in graph
                }
            else:
                personalization = {}

        if not personalization:
            logger.warning("No valid seed nodes found in graph")
            return {}

        try:
            # Use NetworkX's pagerank implementation
            scores = nx.pagerank(
                graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                weight=self.config.weight_attr,
            )
            return scores

        except nx.PowerIterationFailedConvergence:
            logger.warning("PPR failed to converge, returning partial results")
            # Try with more iterations
            try:
                scores = nx.pagerank(
                    graph,
                    alpha=alpha,
                    personalization=personalization,
                    max_iter=self.config.max_iter * 2,
                    tol=self.config.tol * 10,
                    weight=self.config.weight_attr,
                )
                return scores
            except Exception:
                return {}

    def compute_from_query_entities(
        self,
        graph: nx.Graph,
        entity_ids: List[str],
        entity_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute PPR starting from query entities.

        This is the main entry point for HippoRAG-style retrieval:
        1. Extract entities from query
        2. Use those entities as PPR seeds
        3. Rank passages by PPR score

        Args:
            graph: Knowledge graph with entity and passage nodes
            entity_ids: List of entity node IDs from query
            entity_weights: Optional weights for entities (e.g., from embedding similarity)

        Returns:
            Dict mapping all node IDs to PPR scores
        """
        if entity_weights:
            seed_nodes = {
                eid: entity_weights.get(eid, 1.0)
                for eid in entity_ids
            }
        else:
            seed_nodes = entity_ids

        return self.compute(graph, seed_nodes)

    def get_top_nodes(
        self,
        scores: Dict[str, float],
        k: int = 10,
        node_type: Optional[str] = None,
        graph: Optional[nx.Graph] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k nodes by PPR score.

        Args:
            scores: PPR scores from compute()
            k: Number of top nodes to return
            node_type: Optional filter by node type attribute
            graph: Required if node_type filter is used

        Returns:
            List of (node_id, score) tuples sorted by score descending
        """
        if node_type and graph:
            filtered_scores = {
                node: score
                for node, score in scores.items()
                if graph.nodes.get(node, {}).get("node_type") == node_type
            }
        else:
            filtered_scores = scores

        sorted_nodes = sorted(
            filtered_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_nodes[:k]


# =============================================================================
# CENTRALITY CALCULATIONS
# =============================================================================


class CentralityCalculator:
    """
    Computes various centrality measures for knowledge graph nodes.

    Centrality helps identify:
    - Important entities (high degree/betweenness)
    - Authoritative sources (high eigenvector)
    - Bridge concepts (high betweenness)
    """

    def __init__(self, normalized: bool = True):
        self.normalized = normalized

    def degree_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Compute degree centrality (number of connections).

        High degree = well-connected entity or frequently cited source.
        """
        return nx.degree_centrality(graph)

    def betweenness_centrality(
        self,
        graph: nx.Graph,
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute betweenness centrality (how often node is on shortest paths).

        High betweenness = bridge between different topics/communities.

        Args:
            graph: The graph
            k: Sample size for approximation (None = exact, slower)
        """
        return nx.betweenness_centrality(
            graph,
            k=k,
            normalized=self.normalized,
        )

    def eigenvector_centrality(
        self,
        graph: nx.Graph,
        max_iter: int = 100,
    ) -> Dict[str, float]:
        """
        Compute eigenvector centrality (connection to important nodes).

        High eigenvector = connected to other important entities.
        """
        try:
            return nx.eigenvector_centrality(
                graph,
                max_iter=max_iter,
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed to converge")
            return {}

    def pagerank(
        self,
        graph: nx.Graph,
        alpha: float = 0.85,
    ) -> Dict[str, float]:
        """
        Compute PageRank centrality (global importance).

        Similar to eigenvector but handles disconnected graphs better.
        """
        try:
            return nx.pagerank(graph, alpha=alpha)
        except Exception as e:
            logger.warning(f"PageRank failed: {e}")
            return {}

    def compute_all(
        self,
        graph: nx.Graph,
        include_expensive: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute multiple centrality measures.

        Args:
            graph: The graph
            include_expensive: Include betweenness (O(n*m) complexity)

        Returns:
            Dict with keys 'degree', 'eigenvector', 'pagerank', 'betweenness'
        """
        results = {
            "degree": self.degree_centrality(graph),
            "pagerank": self.pagerank(graph),
        }

        # Eigenvector only works on connected graphs
        if nx.is_connected(graph.to_undirected() if graph.is_directed() else graph):
            results["eigenvector"] = self.eigenvector_centrality(graph)

        if include_expensive:
            # Approximate for large graphs
            k = min(100, len(graph.nodes())) if len(graph.nodes()) > 500 else None
            results["betweenness"] = self.betweenness_centrality(graph, k=k)

        return results


# =============================================================================
# COMMUNITY DETECTION
# =============================================================================


class CommunityDetector:
    """
    Detects communities/clusters in the knowledge graph.

    Communities can represent:
    - Topic clusters
    - Related entity groups
    - Research areas
    """

    def __init__(self, resolution: float = 1.0):
        """
        Args:
            resolution: Higher values = more communities (Louvain parameter)
        """
        self.resolution = resolution

    def detect_louvain(
        self,
        graph: nx.Graph,
        resolution: Optional[float] = None,
    ) -> Dict[str, int]:
        """
        Detect communities using Louvain algorithm.

        Args:
            graph: The graph (will be converted to undirected)
            resolution: Override default resolution

        Returns:
            Dict mapping node IDs to community IDs
        """
        resolution = resolution or self.resolution

        # Louvain requires undirected graph
        if graph.is_directed():
            undirected = graph.to_undirected()
        else:
            undirected = graph

        try:
            # NetworkX 3.0+ has community module
            communities = nx.community.louvain_communities(
                undirected,
                resolution=resolution,
            )

            # Convert to node -> community_id mapping
            node_to_community = {}
            for community_id, nodes in enumerate(communities):
                for node in nodes:
                    node_to_community[node] = community_id

            return node_to_community

        except Exception as e:
            logger.error(f"Louvain community detection failed: {e}")
            return {}

    def detect_label_propagation(
        self,
        graph: nx.Graph,
    ) -> Dict[str, int]:
        """
        Detect communities using label propagation (faster, less accurate).

        Good for large graphs where Louvain is too slow.
        """
        if graph.is_directed():
            undirected = graph.to_undirected()
        else:
            undirected = graph

        try:
            communities = nx.community.label_propagation_communities(undirected)

            node_to_community = {}
            for community_id, nodes in enumerate(communities):
                for node in nodes:
                    node_to_community[node] = community_id

            return node_to_community

        except Exception as e:
            logger.error(f"Label propagation failed: {e}")
            return {}

    def get_community_summary(
        self,
        graph: nx.Graph,
        communities: Dict[str, int],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get summary statistics for each community.

        Returns:
            Dict mapping community_id to stats dict with:
            - size: number of nodes
            - nodes: list of node IDs
            - density: internal edge density
        """
        # Group nodes by community
        community_nodes: Dict[int, List[str]] = {}
        for node, comm_id in communities.items():
            if comm_id not in community_nodes:
                community_nodes[comm_id] = []
            community_nodes[comm_id].append(node)

        summaries = {}
        for comm_id, nodes in community_nodes.items():
            subgraph = graph.subgraph(nodes)
            n = len(nodes)
            m = subgraph.number_of_edges()

            # Density = actual edges / possible edges
            max_edges = n * (n - 1) / 2 if not graph.is_directed() else n * (n - 1)
            density = m / max_edges if max_edges > 0 else 0

            summaries[comm_id] = {
                "size": n,
                "nodes": nodes,
                "edges": m,
                "density": density,
            }

        return summaries


# =============================================================================
# PATH FINDING
# =============================================================================


class PathFinder:
    """
    Path finding utilities for knowledge graph exploration.

    Useful for:
    - Finding connections between entities
    - Tracing claim provenance
    - Discovering relationship paths
    """

    def __init__(self, max_path_length: int = 5):
        self.max_path_length = max_path_length

    def shortest_path(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        weight: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes.

        Args:
            graph: The graph
            source: Source node ID
            target: Target node ID
            weight: Edge attribute to use as weight (None = unweighted)

        Returns:
            List of node IDs in path, or None if no path exists
        """
        try:
            return nx.shortest_path(graph, source, target, weight=weight)
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    def all_simple_paths(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        max_length: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Find all simple paths between two nodes.

        Args:
            graph: The graph
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length (default: self.max_path_length)

        Returns:
            List of paths (each path is a list of node IDs)
        """
        max_length = max_length or self.max_path_length

        try:
            paths = list(nx.all_simple_paths(
                graph, source, target, cutoff=max_length
            ))
            return paths
        except nx.NodeNotFound:
            return []

    def paths_through_node(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        through: str,
        max_length: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Find paths that pass through a specific node.

        Useful for understanding how an entity connects two concepts.
        """
        max_length = max_length or self.max_path_length

        # Find paths from source to through
        paths_to = self.all_simple_paths(graph, source, through, max_length // 2)

        # Find paths from through to target
        paths_from = self.all_simple_paths(graph, through, target, max_length // 2)

        # Combine paths
        full_paths = []
        for p1 in paths_to:
            for p2 in paths_from:
                # p2[0] == through, so skip it to avoid duplication
                full_path = p1 + p2[1:]
                if len(full_path) <= max_length:
                    full_paths.append(full_path)

        return full_paths

    def find_connecting_entities(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        max_length: int = 3,
    ) -> Set[str]:
        """
        Find all entities that lie on paths between source and target.

        Useful for discovering bridging concepts.
        """
        paths = self.all_simple_paths(graph, source, target, max_length)

        connecting = set()
        for path in paths:
            # Exclude source and target
            for node in path[1:-1]:
                connecting.add(node)

        return connecting

    def neighborhood(
        self,
        graph: nx.Graph,
        node: str,
        radius: int = 1,
    ) -> Set[str]:
        """
        Get all nodes within a certain distance of a node.

        Args:
            graph: The graph
            node: Center node ID
            radius: Maximum distance

        Returns:
            Set of node IDs within radius
        """
        if node not in graph:
            return set()

        # BFS to find all nodes within radius
        neighbors = set()
        current_level = {node}
        seen = {node}

        for _ in range(radius):
            next_level = set()
            for n in current_level:
                for neighbor in graph.neighbors(n):
                    if neighbor not in seen:
                        next_level.add(neighbor)
                        seen.add(neighbor)
            neighbors.update(next_level)
            current_level = next_level

        return neighbors


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def compute_ppr(
    graph: nx.Graph,
    seed_nodes: Union[List[str], Dict[str, float]],
    alpha: float = 0.85,
) -> Dict[str, float]:
    """Convenience function for PPR computation."""
    ppr = PersonalizedPageRank(PPRConfig(alpha=alpha))
    return ppr.compute(graph, seed_nodes)


def get_centralities(
    graph: nx.Graph,
    include_expensive: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Convenience function for centrality computation."""
    calc = CentralityCalculator()
    return calc.compute_all(graph, include_expensive)


def detect_communities(
    graph: nx.Graph,
    resolution: float = 1.0,
) -> Dict[str, int]:
    """Convenience function for community detection."""
    detector = CommunityDetector(resolution)
    return detector.detect_louvain(graph, resolution)


def find_path(
    graph: nx.Graph,
    source: str,
    target: str,
) -> Optional[List[str]]:
    """Convenience function for shortest path."""
    finder = PathFinder()
    return finder.shortest_path(graph, source, target)
