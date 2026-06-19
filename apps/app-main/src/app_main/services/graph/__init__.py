"""Document structure graph services (Track I, Phase I.F).

Builds and persists a structural graph (doc_node + parent_of / next_node /
derived_from edges) from a source's chunks. See ``doc_graph_builder``.
"""

from app_main.services.graph.doc_graph_builder import (
    ComputedGraph,
    DocGraphBuilder,
    GraphEdge,
    GraphNode,
    compute_graph,
)

__all__ = [
    "ComputedGraph",
    "DocGraphBuilder",
    "GraphEdge",
    "GraphNode",
    "compute_graph",
]
