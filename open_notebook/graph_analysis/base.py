"""
Abstract base classes for graph analysis backends.

This module defines the interface that all graph backends must implement,
allowing for future swapping between NetworkX and igraph without changing
the high-level API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CentralityMethod(str, Enum):
    """Available centrality algorithms."""

    DEGREE = "degree"
    BETWEENNESS = "betweenness"
    EIGENVECTOR = "eigenvector"
    PAGERANK = "pagerank"
    CLOSENESS = "closeness"


class CommunityAlgorithm(str, Enum):
    """Available community detection algorithms."""

    LOUVAIN = "louvain"
    LABEL_PROPAGATION = "label_propagation"
    GREEDY_MODULARITY = "greedy_modularity"


@dataclass
class NodeInfo:
    """Information about a node in the graph."""

    id: str
    node_type: str  # "source", "entity", "person", "organization", "topic", "claim"
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeInfo:
    """Information about an edge in the graph."""

    source_id: str
    target_id: str
    edge_type: str  # "cites", "mentions", "same_as", "authored_by", etc.
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PPRResult:
    """Result of Personalized PageRank computation."""

    node_ids: List[str]
    scores: np.ndarray

    def top_k(self, k: int) -> List[Tuple[str, float]]:
        """Return top-k nodes by score."""
        indices = np.argsort(self.scores)[::-1][:k]
        return [(self.node_ids[i], float(self.scores[i])) for i in indices]

    def filter_by_prefix(self, prefix: str) -> "PPRResult":
        """Filter results to only nodes with given ID prefix."""
        mask = [nid.startswith(prefix) for nid in self.node_ids]
        filtered_ids = [nid for nid, m in zip(self.node_ids, mask) if m]
        filtered_scores = self.scores[mask]
        return PPRResult(node_ids=filtered_ids, scores=filtered_scores)


@dataclass
class CommunityResult:
    """Result of community detection."""

    node_to_community: Dict[str, int]
    communities: Dict[int, List[str]]
    modularity: float

    def get_community(self, node_id: str) -> Optional[int]:
        """Get community ID for a node."""
        return self.node_to_community.get(node_id)

    def get_community_members(self, community_id: int) -> List[str]:
        """Get all members of a community."""
        return self.communities.get(community_id, [])


class GraphBackend(ABC):
    """
    Abstract base class for graph analysis backends.

    Implementations: NetworkXBackend, IGraphBackend (future)

    This abstraction allows swapping graph libraries without changing
    the high-level API. NetworkX is used initially for ease of development,
    with igraph available as an optimization for large graphs.
    """

    @abstractmethod
    def load_from_edges(
        self, edges: List[EdgeInfo], nodes: Optional[List[NodeInfo]] = None
    ) -> None:
        """
        Load graph from edge list.

        Args:
            edges: List of EdgeInfo objects defining the graph structure
            nodes: Optional list of NodeInfo objects with node attributes
        """
        pass

    @abstractmethod
    def add_node(self, node: NodeInfo) -> None:
        """Add a single node to the graph."""
        pass

    @abstractmethod
    def add_edge(self, edge: EdgeInfo) -> None:
        """Add a single edge to the graph."""
        pass

    @abstractmethod
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        pass

    @abstractmethod
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        pass

    @abstractmethod
    def get_node_count(self) -> int:
        """Return number of nodes in the graph."""
        pass

    @abstractmethod
    def get_edge_count(self) -> int:
        """Return number of edges in the graph."""
        pass

    @abstractmethod
    def get_node_type(self, node_id: str) -> Optional[str]:
        """Get the type of a node."""
        pass

    @abstractmethod
    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """Get all nodes of a specific type."""
        pass

    @abstractmethod
    def personalized_pagerank(
        self,
        reset_prob: Dict[str, float],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> PPRResult:
        """
        Compute Personalized PageRank.

        Args:
            reset_prob: Dict mapping node_id to reset probability (teleportation vector)
            damping: Probability of following edges (vs teleporting). HippoRAG uses 0.5.
            max_iter: Maximum iterations for convergence
            tol: Convergence tolerance

        Returns:
            PPRResult with node scores
        """
        pass

    @abstractmethod
    def get_centrality(self, method: CentralityMethod, **kwargs) -> Dict[str, float]:
        """
        Compute centrality scores for all nodes.

        Args:
            method: Centrality algorithm to use
            **kwargs: Algorithm-specific parameters

        Returns:
            Dict mapping node_id to centrality score
        """
        pass

    @abstractmethod
    def detect_communities(
        self, algorithm: CommunityAlgorithm, **kwargs
    ) -> CommunityResult:
        """
        Detect communities in the graph.

        Args:
            algorithm: Community detection algorithm to use
            **kwargs: Algorithm-specific parameters

        Returns:
            CommunityResult with partition and modularity
        """
        pass

    @abstractmethod
    def shortest_path(
        self, source_id: str, target_id: str, weight: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes.

        Args:
            source_id: Starting node
            target_id: Destination node
            weight: Edge attribute to use as weight (None for unweighted)

        Returns:
            List of node IDs in path, or None if no path exists
        """
        pass

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        hops: int = 1,
        edge_types: Optional[List[str]] = None,
        direction: str = "both",
    ) -> List[str]:
        """
        Get neighbors within n hops.

        Args:
            node_id: Starting node
            hops: Number of hops to traverse
            edge_types: Filter to specific edge types (None for all)
            direction: "in", "out", or "both"

        Returns:
            List of neighbor node IDs (excluding starting node)
        """
        pass

    @abstractmethod
    def get_subgraph(
        self, node_ids: List[str], include_edges_between: bool = True
    ) -> "GraphBackend":
        """
        Extract subgraph containing specified nodes.

        Args:
            node_ids: Nodes to include in subgraph
            include_edges_between: Include edges between specified nodes

        Returns:
            New GraphBackend instance with the subgraph
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary for storage/transfer."""
        pass

    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load graph from dictionary."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        pass
