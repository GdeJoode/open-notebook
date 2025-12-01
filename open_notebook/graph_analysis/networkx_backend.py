"""
NetworkX implementation of the GraphBackend interface.

NetworkX provides a pure-Python graph library with extensive algorithm support.
While slower than igraph for large graphs, it offers easier debugging and
a more Pythonic API suitable for development and medium-sized graphs.
"""

from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from open_notebook.graph_analysis.base import (
    CentralityMethod,
    CommunityAlgorithm,
    CommunityResult,
    EdgeInfo,
    GraphBackend,
    NodeInfo,
    PPRResult,
)


class NetworkXBackend(GraphBackend):
    """NetworkX implementation of GraphBackend."""

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._node_types: Dict[str, str] = {}  # node_id -> type
        self._edge_types: Dict[tuple, str] = {}  # (src, tgt) -> type

    def load_from_edges(
        self, edges: List[EdgeInfo], nodes: Optional[List[NodeInfo]] = None
    ) -> None:
        """Load graph from edge list."""
        self._graph = nx.DiGraph()
        self._node_types = {}
        self._edge_types = {}

        # Add nodes first if provided
        if nodes:
            for node in nodes:
                self.add_node(node)

        # Add edges
        for edge in edges:
            self.add_edge(edge)

    def add_node(self, node: NodeInfo) -> None:
        """Add a single node."""
        self._graph.add_node(node.id, **node.attributes)
        self._node_types[node.id] = node.node_type

    def add_edge(self, edge: EdgeInfo) -> None:
        """Add a single edge."""
        # Ensure nodes exist
        if edge.source_id not in self._graph:
            self._graph.add_node(edge.source_id)
        if edge.target_id not in self._graph:
            self._graph.add_node(edge.target_id)

        self._graph.add_edge(
            edge.source_id, edge.target_id, weight=edge.weight, **edge.attributes
        )
        self._edge_types[(edge.source_id, edge.target_id)] = edge.edge_type

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its edges."""
        if node_id in self._graph:
            self._graph.remove_node(node_id)
            self._node_types.pop(node_id, None)
            # Clean up edge types
            self._edge_types = {
                k: v for k, v in self._edge_types.items() if node_id not in k
            }

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        return node_id in self._graph

    def get_node_count(self) -> int:
        return self._graph.number_of_nodes()

    def get_edge_count(self) -> int:
        return self._graph.number_of_edges()

    def get_node_type(self, node_id: str) -> Optional[str]:
        """Get the type of a node."""
        return self._node_types.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """Get all nodes of a specific type."""
        return [nid for nid, ntype in self._node_types.items() if ntype == node_type]

    def personalized_pagerank(
        self,
        reset_prob: Dict[str, float],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> PPRResult:
        """
        Compute Personalized PageRank using NetworkX.

        HippoRAG uses damping=0.5 by default for more aggressive teleportation
        back to seed entities.
        """
        if not reset_prob:
            # Standard PageRank if no personalization
            scores = nx.pagerank(self._graph, alpha=damping, max_iter=max_iter, tol=tol)
        else:
            # Normalize reset probabilities
            total = sum(reset_prob.values())
            if total > 0:
                personalization = {k: v / total for k, v in reset_prob.items()}
            else:
                personalization = None

            scores = nx.pagerank(
                self._graph,
                alpha=damping,
                personalization=personalization,
                max_iter=max_iter,
                tol=tol,
            )

        node_ids = list(scores.keys())
        score_array = np.array([scores[n] for n in node_ids])

        return PPRResult(node_ids=node_ids, scores=score_array)

    def get_centrality(self, method: CentralityMethod, **kwargs) -> Dict[str, float]:
        """Compute centrality scores."""
        if method == CentralityMethod.DEGREE:
            # Normalize by n-1
            return dict(nx.degree_centrality(self._graph))

        elif method == CentralityMethod.BETWEENNESS:
            return dict(
                nx.betweenness_centrality(
                    self._graph, weight=kwargs.get("weight", "weight"), normalized=True
                )
            )

        elif method == CentralityMethod.EIGENVECTOR:
            try:
                return dict(
                    nx.eigenvector_centrality(
                        self._graph,
                        max_iter=kwargs.get("max_iter", 100),
                        weight=kwargs.get("weight", "weight"),
                    )
                )
            except nx.PowerIterationFailedConvergence:
                # Fallback to numpy-based computation
                return dict(
                    nx.eigenvector_centrality_numpy(
                        self._graph, weight=kwargs.get("weight", "weight")
                    )
                )

        elif method == CentralityMethod.PAGERANK:
            return dict(
                nx.pagerank(
                    self._graph,
                    alpha=kwargs.get("alpha", 0.85),
                    weight=kwargs.get("weight", "weight"),
                )
            )

        elif method == CentralityMethod.CLOSENESS:
            return dict(nx.closeness_centrality(self._graph))

        else:
            raise ValueError(f"Unknown centrality method: {method}")

    def detect_communities(
        self, algorithm: CommunityAlgorithm, **kwargs
    ) -> CommunityResult:
        """Detect communities using specified algorithm."""
        # Convert to undirected for community detection
        undirected = self._graph.to_undirected()

        if algorithm == CommunityAlgorithm.LOUVAIN:
            try:
                import community as community_louvain

                partition = community_louvain.best_partition(
                    undirected,
                    weight=kwargs.get("weight", "weight"),
                    resolution=kwargs.get("resolution", 1.0),
                )
                modularity = community_louvain.modularity(partition, undirected)
            except ImportError:
                # Fallback to greedy modularity if python-louvain not installed
                return self.detect_communities(
                    CommunityAlgorithm.GREEDY_MODULARITY, **kwargs
                )

        elif algorithm == CommunityAlgorithm.LABEL_PROPAGATION:
            communities = list(nx.community.label_propagation_communities(undirected))
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
            modularity = nx.community.modularity(undirected, communities)

        elif algorithm == CommunityAlgorithm.GREEDY_MODULARITY:
            communities = list(
                nx.community.greedy_modularity_communities(
                    undirected, weight=kwargs.get("weight", "weight")
                )
            )
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
            modularity = nx.community.modularity(undirected, communities)

        else:
            raise ValueError(f"Unknown community algorithm: {algorithm}")

        # Build communities dict
        communities_dict: Dict[int, List[str]] = {}
        for node, comm_id in partition.items():
            if comm_id not in communities_dict:
                communities_dict[comm_id] = []
            communities_dict[comm_id].append(node)

        return CommunityResult(
            node_to_community=partition,
            communities=communities_dict,
            modularity=modularity,
        )

    def shortest_path(
        self, source_id: str, target_id: str, weight: Optional[str] = None
    ) -> Optional[List[str]]:
        """Find shortest path."""
        try:
            return nx.shortest_path(
                self._graph, source=source_id, target=target_id, weight=weight
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_neighbors(
        self,
        node_id: str,
        hops: int = 1,
        edge_types: Optional[List[str]] = None,
        direction: str = "both",
    ) -> List[str]:
        """Get neighbors within n hops."""
        if node_id not in self._graph:
            return []

        visited = {node_id}
        current_level = {node_id}

        for _ in range(hops):
            next_level = set()
            for node in current_level:
                # Outgoing edges
                if direction in ("out", "both"):
                    for neighbor in self._graph.successors(node):
                        if neighbor not in visited:
                            # Filter by edge type if specified
                            if edge_types:
                                edge_type = self._edge_types.get((node, neighbor))
                                if edge_type not in edge_types:
                                    continue
                            next_level.add(neighbor)

                # Incoming edges
                if direction in ("in", "both"):
                    for neighbor in self._graph.predecessors(node):
                        if neighbor not in visited:
                            if edge_types:
                                edge_type = self._edge_types.get((neighbor, node))
                                if edge_type not in edge_types:
                                    continue
                            next_level.add(neighbor)

            visited.update(next_level)
            current_level = next_level

        visited.discard(node_id)  # Remove starting node
        return list(visited)

    def get_subgraph(
        self, node_ids: List[str], include_edges_between: bool = True
    ) -> "NetworkXBackend":
        """Extract subgraph."""
        subgraph = NetworkXBackend()

        node_set = set(node_ids)

        # Add nodes
        for node_id in node_ids:
            if node_id in self._graph:
                subgraph._graph.add_node(node_id, **self._graph.nodes[node_id])
                if node_id in self._node_types:
                    subgraph._node_types[node_id] = self._node_types[node_id]

        # Add edges
        if include_edges_between:
            for src, tgt, data in self._graph.edges(data=True):
                if src in node_set and tgt in node_set:
                    subgraph._graph.add_edge(src, tgt, **data)
                    if (src, tgt) in self._edge_types:
                        subgraph._edge_types[(src, tgt)] = self._edge_types[(src, tgt)]

        return subgraph

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "nodes": [
                {
                    "id": n,
                    "type": self._node_types.get(n, "unknown"),
                    "attributes": dict(self._graph.nodes[n]),
                }
                for n in self._graph.nodes()
            ],
            "edges": [
                {
                    "source": src,
                    "target": tgt,
                    "type": self._edge_types.get((src, tgt), "unknown"),
                    "weight": data.get("weight", 1.0),
                    "attributes": {k: v for k, v in data.items() if k != "weight"},
                }
                for src, tgt, data in self._graph.edges(data=True)
            ],
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        self._graph = nx.DiGraph()
        self._node_types = {}
        self._edge_types = {}

        for node_data in data.get("nodes", []):
            self._graph.add_node(node_data["id"], **node_data.get("attributes", {}))
            self._node_types[node_data["id"]] = node_data.get("type", "unknown")

        for edge_data in data.get("edges", []):
            self._graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                weight=edge_data.get("weight", 1.0),
                **edge_data.get("attributes", {}),
            )
            self._edge_types[(edge_data["source"], edge_data["target"])] = edge_data.get(
                "type", "unknown"
            )

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._graph.clear()
        self._node_types.clear()
        self._edge_types.clear()

    # NetworkX-specific convenience methods

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to underlying NetworkX graph for advanced operations."""
        return self._graph

    def get_in_degree(self, node_id: str) -> int:
        """Get in-degree of a node."""
        return self._graph.in_degree(node_id) if node_id in self._graph else 0

    def get_out_degree(self, node_id: str) -> int:
        """Get out-degree of a node."""
        return self._graph.out_degree(node_id) if node_id in self._graph else 0

    def get_edge_data(
        self, source_id: str, target_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get edge attributes."""
        if self._graph.has_edge(source_id, target_id):
            data = dict(self._graph.edges[source_id, target_id])
            data["edge_type"] = self._edge_types.get(
                (source_id, target_id), "unknown"
            )
            return data
        return None

    def to_undirected(self) -> "NetworkXBackend":
        """Convert to undirected graph."""
        undirected = NetworkXBackend()
        undirected._graph = self._graph.to_undirected()
        undirected._node_types = self._node_types.copy()
        # For undirected, store both directions
        for (src, tgt), etype in self._edge_types.items():
            undirected._edge_types[(src, tgt)] = etype
            undirected._edge_types[(tgt, src)] = etype
        return undirected
