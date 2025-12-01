"""
Knowledge Graph Analysis Module.

This module provides graph analysis capabilities for the knowledge graph,
with a pluggable backend architecture allowing for future optimization
(e.g., switching from NetworkX to igraph for performance).
"""

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
from open_notebook.graph_analysis.analyzer import GraphAnalyzer

__all__ = [
    # Base classes and types
    "GraphBackend",
    "NodeInfo",
    "EdgeInfo",
    "PPRResult",
    "CommunityResult",
    "CentralityMethod",
    "CommunityAlgorithm",
    # Backend implementations
    "NetworkXBackend",
    # High-level interface
    "GraphAnalyzer",
]
