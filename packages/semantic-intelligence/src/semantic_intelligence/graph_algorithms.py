"""
Graph algorithms for the semantic intelligence layer.

Exports the SurrealDB entity/relation graph into NetworkX, runs standard
graph algorithms, and writes computed scores back to entity records.

Algorithms:
- PageRank: importance based on incoming relation structure
- Betweenness centrality: bridging nodes between communities
- Community detection: Louvain method for grouping related entities
- Link prediction: Jaccard and Adamic-Adar for predicting missing relations
- Shortest path: find the path between two entities through the graph

Usage:
    from semantic_intelligence.graph_algorithms import (
        build_graph, compute_pagerank, compute_communities, write_scores,
    )

    graph = await build_graph()
    scores = compute_pagerank(graph)
    communities = compute_communities(graph)
    await write_scores(scores, communities)

    python -m semantic_intelligence.graph_algorithms
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from loguru import logger


# ============================================================================
# Graph construction from SurrealDB
# ============================================================================


async def build_graph(
    entity_filter: str = "",
    include_properties: bool = True,
) -> nx.DiGraph:
    """Export the SurrealDB entity/relation graph into a NetworkX DiGraph.

    Each entity becomes a node with its properties as attributes.
    Each relation becomes a directed edge with relation_type and confidence.

    Args:
        entity_filter: Optional SurrealQL WHERE clause (e.g. "entity_type = 'Gemeente'").
        include_properties: Whether to include entity properties as node attributes.

    Returns:
        NetworkX directed graph.
    """
    from surrealdb_service.connection import execute_query

    G = nx.DiGraph()

    # Load entities
    query = "SELECT * FROM entity WHERE status = 'active'"
    if entity_filter:
        query += f" AND {entity_filter}"

    entities = await execute_query(query)
    for ent in entities:
        eid = str(ent.get("id", ""))
        attrs = {
            "canonical_name": ent.get("canonical_name", ""),
            "entity_type": ent.get("entity_type", ""),
            "confidence": ent.get("confidence", 1.0),
        }
        if include_properties:
            attrs["properties"] = ent.get("properties", {})
        G.add_node(eid, **attrs)

    # Load relations
    relations = await execute_query("SELECT * FROM relation WHERE status = 'active'")
    for rel in relations:
        from_id = str(rel.get("in", ""))
        to_id = str(rel.get("out", ""))
        if from_id in G and to_id in G:
            G.add_edge(
                from_id,
                to_id,
                relation_type=rel.get("relation_type", "related_to"),
                confidence=rel.get("confidence", 1.0),
                id=str(rel.get("id", "")),
            )

    logger.info(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ============================================================================
# PageRank
# ============================================================================


def compute_pagerank(
    graph: nx.DiGraph,
    alpha: float = 0.85,
    weight: Optional[str] = "confidence",
) -> Dict[str, float]:
    """Compute PageRank scores for all nodes.

    Args:
        graph: NetworkX directed graph.
        alpha: Damping factor (0.85 is standard).
        weight: Edge attribute to use as weight. None for unweighted.

    Returns:
        Dict mapping node_id to PageRank score.
    """
    if graph.number_of_nodes() == 0:
        return {}
    scores = nx.pagerank(graph, alpha=alpha, weight=weight)
    logger.info(f"PageRank: computed for {len(scores)} nodes")
    return scores


# ============================================================================
# Betweenness centrality
# ============================================================================


def compute_betweenness(
    graph: nx.DiGraph,
    k: Optional[int] = None,
    weight: Optional[str] = "confidence",
) -> Dict[str, float]:
    """Compute betweenness centrality for all nodes.

    Identifies bridge nodes that connect different parts of the graph.

    Args:
        graph: NetworkX directed graph.
        k: Sample size for approximation (None = exact, slow for large graphs).
        weight: Edge attribute for weighted shortest paths.

    Returns:
        Dict mapping node_id to betweenness centrality score.
    """
    if graph.number_of_nodes() == 0:
        return {}

    # For large graphs, sample k nodes for approximation
    if k is None and graph.number_of_nodes() > 5000:
        k = min(500, graph.number_of_nodes())

    scores = nx.betweenness_centrality(graph, k=k, weight=weight)
    logger.info(f"Betweenness centrality: computed for {len(scores)} nodes")
    return scores


# ============================================================================
# Community detection (Louvain)
# ============================================================================


def compute_communities(
    graph: nx.DiGraph,
    resolution: float = 1.0,
) -> Dict[str, int]:
    """Detect communities using the Louvain method.

    Converts the directed graph to undirected for community detection,
    as Louvain operates on undirected graphs.

    Args:
        graph: NetworkX directed graph.
        resolution: Louvain resolution parameter. Higher = more communities.

    Returns:
        Dict mapping node_id to community_id (int).
    """
    if graph.number_of_nodes() == 0:
        return {}

    # Convert to undirected (Louvain requires undirected)
    undirected = graph.to_undirected()

    try:
        import community as community_louvain

        partition = community_louvain.best_partition(undirected, resolution=resolution)
    except ImportError:
        # Fallback: use networkx built-in greedy modularity
        logger.warning("python-louvain not installed, using greedy_modularity_communities")
        communities = nx.community.greedy_modularity_communities(undirected)
        partition = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                partition[node] = idx

    num_communities = len(set(partition.values()))
    logger.info(f"Community detection: {num_communities} communities found")
    return partition


# ============================================================================
# Link prediction
# ============================================================================


def predict_links(
    graph: nx.DiGraph,
    node_a: str,
    node_b: str,
) -> Dict[str, float]:
    """Predict the likelihood of a link between two nodes.

    Uses multiple metrics for a comprehensive score.

    Args:
        graph: NetworkX directed graph.
        node_a: Source node ID.
        node_b: Target node ID.

    Returns:
        Dict with prediction scores: jaccard, adamic_adar, common_neighbors.
    """
    undirected = graph.to_undirected()

    if node_a not in undirected or node_b not in undirected:
        return {"jaccard": 0.0, "adamic_adar": 0.0, "common_neighbors": 0}

    # Common neighbors
    cn = list(nx.common_neighbors(undirected, node_a, node_b))

    # Jaccard coefficient
    jaccard = list(nx.jaccard_coefficient(undirected, [(node_a, node_b)]))
    jaccard_score = jaccard[0][2] if jaccard else 0.0

    # Adamic-Adar
    aa = list(nx.adamic_adar_index(undirected, [(node_a, node_b)]))
    aa_score = aa[0][2] if aa else 0.0

    return {
        "jaccard": jaccard_score,
        "adamic_adar": aa_score,
        "common_neighbors": len(cn),
    }


def predict_links_batch(
    graph: nx.DiGraph,
    pairs: Optional[List[Tuple[str, str]]] = None,
    top_k: int = 20,
) -> List[Tuple[str, str, float]]:
    """Predict missing links for all non-connected pairs or a given list.

    Args:
        graph: NetworkX directed graph.
        pairs: Specific pairs to evaluate. If None, evaluates all non-edges.
        top_k: Number of top predictions to return.

    Returns:
        List of (node_a, node_b, score) sorted by score descending.
    """
    undirected = graph.to_undirected()

    if pairs:
        predictions = list(nx.jaccard_coefficient(undirected, pairs))
    else:
        # Generate candidates: non-connected pairs with shared neighbors
        candidates = nx.non_edges(undirected)
        predictions = list(nx.jaccard_coefficient(undirected, candidates))

    # Sort by score descending
    predictions.sort(key=lambda x: x[2], reverse=True)
    return [(str(u), str(v), p) for u, v, p in predictions[:top_k]]


# ============================================================================
# Shortest path
# ============================================================================


def find_shortest_path(
    graph: nx.DiGraph,
    source: str,
    target: str,
    weight: Optional[str] = None,
) -> List[str]:
    """Find the shortest path between two nodes.

    Args:
        graph: NetworkX directed graph.
        source: Source node ID.
        target: Target node ID.
        weight: Edge attribute for weighted shortest path.

    Returns:
        List of node IDs from source to target. Empty if no path exists.
    """
    try:
        return nx.shortest_path(graph, source, target, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def find_all_paths(
    graph: nx.DiGraph,
    source: str,
    target: str,
    max_depth: int = 5,
) -> List[List[str]]:
    """Find all simple paths between two nodes up to a given depth.

    Args:
        graph: NetworkX directed graph.
        source: Source node ID.
        target: Target node ID.
        max_depth: Maximum path length.

    Returns:
        List of paths (each path is a list of node IDs).
    """
    try:
        return list(nx.all_simple_paths(graph, source, target, cutoff=max_depth))
    except nx.NodeNotFound:
        return []


# ============================================================================
# Write scores back to SurrealDB
# ============================================================================


async def write_scores(
    pagerank: Optional[Dict[str, float]] = None,
    betweenness: Optional[Dict[str, float]] = None,
    communities: Optional[Dict[str, int]] = None,
) -> int:
    """Write computed graph scores back to entity records in SurrealDB.

    Args:
        pagerank: Dict mapping entity_id to PageRank score.
        betweenness: Dict mapping entity_id to betweenness centrality.
        communities: Dict mapping entity_id to community_id.

    Returns:
        Number of entities updated.
    """
    from surrealdb_service.connection import execute_query

    # Collect all entity IDs that need updating
    all_ids = set()
    if pagerank:
        all_ids.update(pagerank.keys())
    if betweenness:
        all_ids.update(betweenness.keys())
    if communities:
        all_ids.update(communities.keys())

    updated = 0
    for eid in all_ids:
        fields = []
        params: Dict[str, Any] = {"id": eid}

        if pagerank and eid in pagerank:
            fields.append("pagerank = $pr")
            params["pr"] = pagerank[eid]
        if betweenness and eid in betweenness:
            fields.append("betweenness = $bt")
            params["bt"] = betweenness[eid]
        if communities and eid in communities:
            fields.append("community_id = $cid")
            params["cid"] = communities[eid]

        if fields:
            fields.append("updated_at = time::now()")
            query = f"UPDATE $id SET {', '.join(fields)}"
            await execute_query(query, params)
            updated += 1

    logger.info(f"Wrote graph scores to {updated} entities")
    return updated


# ============================================================================
# Convenience: run all algorithms and write back
# ============================================================================


async def run_all(entity_filter: str = "") -> Dict[str, Any]:
    """Run all graph algorithms and write scores to SurrealDB.

    Returns:
        Summary dict with node/edge counts, community count, top entities.
    """
    graph = await build_graph(entity_filter)

    if graph.number_of_nodes() == 0:
        logger.warning("Empty graph — no entities found")
        return {"nodes": 0, "edges": 0}

    pr = compute_pagerank(graph)
    bt = compute_betweenness(graph)
    comm = compute_communities(graph)
    await write_scores(pagerank=pr, betweenness=bt, communities=comm)

    # Top entities by PageRank
    top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    top_entities = [
        {
            "id": eid,
            "name": graph.nodes[eid].get("canonical_name", ""),
            "pagerank": score,
            "community": comm.get(eid),
        }
        for eid, score in top_pr
    ]

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "communities": len(set(comm.values())),
        "top_entities": top_entities,
    }


# ============================================================================
# CLI entry point
# ============================================================================


async def _demo():
    """Demo: run all algorithms on the entity graph."""
    result = await run_all()
    logger.info(f"Graph analysis complete: {result}")

    if result.get("top_entities"):
        logger.info("Top entities by PageRank:")
        for ent in result["top_entities"]:
            logger.info(
                f"  {ent['name']}: PR={ent['pagerank']:.4f}, "
                f"community={ent['community']}"
            )


if __name__ == "__main__":
    asyncio.run(_demo())
