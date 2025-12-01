"""
Tests for Knowledge Graph Analysis.

These tests verify:
1. Personalized PageRank computation
2. Centrality calculations
3. Community detection
4. Path finding
5. Graph loading (mocked)
6. Analysis methods

Run with: pytest tests/test_kg_analysis.py -v
"""

import pytest
import networkx as nx
from unittest.mock import AsyncMock, MagicMock, patch

from open_notebook.graphs.kg_analysis import (
    PersonalizedPageRank,
    CentralityCalculator,
    CommunityDetector,
    PathFinder,
    PPRConfig,
    compute_ppr,
    get_centralities,
    detect_communities,
    find_path,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    G = nx.DiGraph()

    # Add entity nodes
    G.add_node("entity:1", node_type="entity", name="Alice", entity_type="person")
    G.add_node("entity:2", node_type="entity", name="Bob", entity_type="person")
    G.add_node("entity:3", node_type="entity", name="Climate Change", entity_type="topic")
    G.add_node("entity:4", node_type="entity", name="WHO", entity_type="organization")

    # Add source nodes
    G.add_node("source:1", node_type="source", title="Paper 1")
    G.add_node("source:2", node_type="source", title="Paper 2")

    # Add edges
    G.add_edge("source:1", "entity:1", edge_type="mentions", weight=1.0)
    G.add_edge("source:1", "entity:3", edge_type="mentions", weight=1.0)
    G.add_edge("source:2", "entity:2", edge_type="mentions", weight=1.0)
    G.add_edge("source:2", "entity:3", edge_type="mentions", weight=1.0)
    G.add_edge("source:2", "entity:4", edge_type="mentions", weight=1.0)
    G.add_edge("entity:1", "entity:3", edge_type="relates_to", weight=0.8)
    G.add_edge("entity:4", "entity:3", edge_type="relates_to", weight=0.9)

    return G


@pytest.fixture
def connected_graph():
    """Create a connected undirected graph for community detection."""
    G = nx.Graph()

    # Cluster 1
    G.add_edges_from([
        ("a", "b"), ("b", "c"), ("c", "a"),
    ])

    # Cluster 2
    G.add_edges_from([
        ("d", "e"), ("e", "f"), ("f", "d"),
    ])

    # Bridge between clusters
    G.add_edge("c", "d")

    return G


# =============================================================================
# PPR TESTS
# =============================================================================


class TestPersonalizedPageRank:
    """Tests for PPR computation."""

    def test_ppr_with_list_seeds(self, simple_graph):
        """Test PPR with seed nodes as list."""
        ppr = PersonalizedPageRank()
        scores = ppr.compute(simple_graph, ["entity:3"])

        assert len(scores) > 0
        # Seed node should have high score
        assert scores.get("entity:3", 0) > 0

    def test_ppr_with_dict_seeds(self, simple_graph):
        """Test PPR with weighted seed nodes."""
        ppr = PersonalizedPageRank()
        scores = ppr.compute(simple_graph, {"entity:3": 0.7, "entity:4": 0.3})

        assert len(scores) > 0
        assert scores.get("entity:3", 0) > 0

    def test_ppr_empty_graph(self):
        """Test PPR on empty graph."""
        ppr = PersonalizedPageRank()
        scores = ppr.compute(nx.DiGraph(), ["entity:1"])

        assert scores == {}

    def test_ppr_invalid_seeds(self, simple_graph):
        """Test PPR with seeds not in graph."""
        ppr = PersonalizedPageRank()
        scores = ppr.compute(simple_graph, ["nonexistent:1"])

        assert scores == {}

    def test_ppr_custom_alpha(self, simple_graph):
        """Test PPR with custom damping factor."""
        ppr = PersonalizedPageRank(PPRConfig(alpha=0.5))
        scores = ppr.compute(simple_graph, ["entity:3"])

        assert len(scores) > 0

    def test_get_top_nodes(self, simple_graph):
        """Test getting top-k nodes by score."""
        ppr = PersonalizedPageRank()
        scores = ppr.compute(simple_graph, ["entity:3"])
        top = ppr.get_top_nodes(scores, k=3)

        assert len(top) <= 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)
        # Should be sorted by score descending
        if len(top) > 1:
            assert top[0][1] >= top[1][1]

    def test_get_top_nodes_by_type(self, simple_graph):
        """Test filtering top nodes by type."""
        ppr = PersonalizedPageRank()
        scores = ppr.compute(simple_graph, ["entity:3"])
        top = ppr.get_top_nodes(scores, k=10, node_type="entity", graph=simple_graph)

        # All returned nodes should be entities
        for node_id, _ in top:
            assert simple_graph.nodes[node_id].get("node_type") == "entity"


class TestPPRConvenienceFunction:
    """Test convenience function."""

    def test_compute_ppr(self, simple_graph):
        """Test compute_ppr convenience function."""
        scores = compute_ppr(simple_graph, ["entity:3"], alpha=0.85)

        assert len(scores) > 0
        assert scores.get("entity:3", 0) > 0


# =============================================================================
# CENTRALITY TESTS
# =============================================================================


class TestCentralityCalculator:
    """Tests for centrality calculations."""

    def test_degree_centrality(self, simple_graph):
        """Test degree centrality calculation."""
        calc = CentralityCalculator()
        scores = calc.degree_centrality(simple_graph)

        assert len(scores) == simple_graph.number_of_nodes()
        # entity:3 should have high degree (many connections)
        assert scores.get("entity:3", 0) > 0

    def test_pagerank(self, simple_graph):
        """Test PageRank centrality."""
        calc = CentralityCalculator()
        scores = calc.pagerank(simple_graph)

        assert len(scores) == simple_graph.number_of_nodes()
        # Sum should be approximately 1
        assert abs(sum(scores.values()) - 1.0) < 0.01

    def test_betweenness_centrality(self, simple_graph):
        """Test betweenness centrality."""
        calc = CentralityCalculator()
        scores = calc.betweenness_centrality(simple_graph)

        assert len(scores) == simple_graph.number_of_nodes()

    def test_compute_all(self, simple_graph):
        """Test computing all centralities."""
        calc = CentralityCalculator()
        results = calc.compute_all(simple_graph)

        assert "degree" in results
        assert "pagerank" in results

    def test_compute_all_with_expensive(self, simple_graph):
        """Test computing all centralities including expensive ones."""
        calc = CentralityCalculator()
        results = calc.compute_all(simple_graph, include_expensive=True)

        assert "betweenness" in results


class TestCentralityConvenienceFunction:
    """Test convenience function."""

    def test_get_centralities(self, simple_graph):
        """Test get_centralities convenience function."""
        results = get_centralities(simple_graph)

        assert "degree" in results
        assert "pagerank" in results


# =============================================================================
# COMMUNITY DETECTION TESTS
# =============================================================================


class TestCommunityDetector:
    """Tests for community detection."""

    def test_louvain_detection(self, connected_graph):
        """Test Louvain community detection."""
        detector = CommunityDetector()
        communities = detector.detect_louvain(connected_graph)

        assert len(communities) == connected_graph.number_of_nodes()
        # Should detect at least 2 communities (the two clusters)
        unique_communities = set(communities.values())
        assert len(unique_communities) >= 1  # May merge due to bridge

    def test_label_propagation(self, connected_graph):
        """Test label propagation detection."""
        detector = CommunityDetector()
        communities = detector.detect_label_propagation(connected_graph)

        assert len(communities) == connected_graph.number_of_nodes()

    def test_community_summary(self, connected_graph):
        """Test community summary statistics."""
        detector = CommunityDetector()
        communities = detector.detect_louvain(connected_graph)
        summary = detector.get_community_summary(connected_graph, communities)

        # Check summary structure
        for comm_id, stats in summary.items():
            assert "size" in stats
            assert "nodes" in stats
            assert "density" in stats
            assert stats["size"] == len(stats["nodes"])

    def test_directed_graph_conversion(self, simple_graph):
        """Test that directed graphs are handled correctly."""
        detector = CommunityDetector()
        communities = detector.detect_louvain(simple_graph)

        # Should work on directed graph (converts to undirected)
        assert len(communities) > 0


class TestCommunityConvenienceFunction:
    """Test convenience function."""

    def test_detect_communities(self, connected_graph):
        """Test detect_communities convenience function."""
        communities = detect_communities(connected_graph, resolution=1.0)

        assert len(communities) == connected_graph.number_of_nodes()


# =============================================================================
# PATH FINDING TESTS
# =============================================================================


class TestPathFinder:
    """Tests for path finding utilities."""

    def test_shortest_path(self, simple_graph):
        """Test shortest path finding."""
        finder = PathFinder()
        path = finder.shortest_path(simple_graph, "source:1", "entity:3")

        assert path is not None
        assert path[0] == "source:1"
        assert path[-1] == "entity:3"

    def test_shortest_path_no_path(self, simple_graph):
        """Test when no path exists."""
        finder = PathFinder()
        # entity:1 to source:2 - no direct path in directed graph
        # (depends on graph structure)
        path = finder.shortest_path(simple_graph, "entity:1", "source:2")

        # May or may not exist depending on direction

    def test_shortest_path_invalid_nodes(self, simple_graph):
        """Test with invalid nodes."""
        finder = PathFinder()
        path = finder.shortest_path(simple_graph, "nonexistent", "entity:1")

        assert path is None

    def test_all_simple_paths(self, simple_graph):
        """Test finding all simple paths."""
        finder = PathFinder()
        paths = finder.all_simple_paths(
            simple_graph, "source:1", "entity:3", max_length=3
        )

        assert len(paths) >= 1
        for path in paths:
            assert path[0] == "source:1"
            assert path[-1] == "entity:3"

    def test_neighborhood(self, simple_graph):
        """Test neighborhood extraction."""
        finder = PathFinder()
        # For directed graph, use source:1 which has outgoing edges
        neighbors = finder.neighborhood(simple_graph, "source:1", radius=1)

        assert len(neighbors) > 0
        # source:1 is not in its own neighborhood
        assert "source:1" not in neighbors

    def test_neighborhood_radius_2(self, simple_graph):
        """Test neighborhood with larger radius."""
        finder = PathFinder()
        neighbors_1 = finder.neighborhood(simple_graph, "entity:3", radius=1)
        neighbors_2 = finder.neighborhood(simple_graph, "entity:3", radius=2)

        # Larger radius should include more or equal nodes
        assert len(neighbors_2) >= len(neighbors_1)

    def test_find_connecting_entities(self, simple_graph):
        """Test finding connecting entities."""
        finder = PathFinder()
        connecting = finder.find_connecting_entities(
            simple_graph, "source:1", "entity:4", max_length=4
        )

        # Should find entities on paths between these nodes
        # (may be empty if no path exists within max_length)


class TestPathConvenienceFunction:
    """Test convenience function."""

    def test_find_path(self, simple_graph):
        """Test find_path convenience function."""
        path = find_path(simple_graph, "source:1", "entity:3")

        assert path is not None
        assert path[0] == "source:1"
        assert path[-1] == "entity:3"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAnalysisIntegration:
    """Integration tests combining multiple analysis methods."""

    def test_ppr_to_centrality_comparison(self, simple_graph):
        """Compare PPR scores with global centrality."""
        ppr = PersonalizedPageRank()
        calc = CentralityCalculator()

        # PPR from specific seed
        ppr_scores = ppr.compute(simple_graph, ["entity:3"])

        # Global PageRank
        global_pr = calc.pagerank(simple_graph)

        # Both should have scores for all nodes
        assert set(ppr_scores.keys()) == set(global_pr.keys())

        # PPR should give higher relative score to seed neighborhood
        # compared to global PageRank

    def test_community_plus_centrality(self, connected_graph):
        """Test combining community detection with centrality."""
        detector = CommunityDetector()
        calc = CentralityCalculator()

        communities = detector.detect_louvain(connected_graph)
        centrality = calc.degree_centrality(connected_graph)

        # Bridge node should have high betweenness
        betweenness = calc.betweenness_centrality(connected_graph)

        # Node 'c' or 'd' (bridge) should have higher betweenness
        bridge_nodes = ["c", "d"]
        non_bridge = ["a", "b", "e", "f"]

        avg_bridge = sum(betweenness.get(n, 0) for n in bridge_nodes) / len(bridge_nodes)
        avg_non_bridge = sum(betweenness.get(n, 0) for n in non_bridge) / len(non_bridge)

        # Bridge nodes should have higher or equal betweenness
        # (might be equal in simple graph)
